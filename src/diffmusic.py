"""
diffMUSIC: Differentiable MUSIC for DoA Estimation with Hardware Impairment Learning

This module implements the diffMUSIC algorithm as described in the paper:
"Physically Parameterized Differentiable MUSIC for DoA Estimation with Uncalibrated Arrays"

Key Features:
- Learnable antenna positions and complex gains
- Differentiable steering matrix computation
- Softmax-based peak finding for end-to-end learning
- Support for both supervised and unsupervised learning
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy as sc

from src.subspace_method import SubspaceMethod
from src.signal_creation import SystemModel
from src.utils import *
from src.metrics import SpectrumLoss, UnsupervisedSpectrumLoss, RMSPELoss
# from src.metrics import RMSPELoss


class DiffMUSIC(SubspaceMethod):
    """
    Differentiable MUSIC (diffMUSIC) and classical MUSIC implementation for DoA estimation with hardware impairment learning.
    The algorithm choice (diffMUSIC vs MUSIC) is determined by system_model_params.model_type.
    
    This implementation is focused on:
    - Far-field scenarios only
    - Non-coherent sources  
    - Narrowband signals
    
    Key features:
    - Learnable antenna positions (nn.Parameter)
    - Learnable complex gains (nn.Parameter) 
    - Differentiable steering matrix computation
    - Softmax-based peak finding for differentiability (diffMUSIC)
    - Hard peak finding for classical MUSIC
    """

    def __init__(self, system_model_params, N: int, model_order_estimation: str = None,
                 physical_array: torch.Tensor = None, physical_gains: torch.Tensor = None):
        """
        Initialize diffMUSIC
        
        Args:
            system_model_params: System model parameters object
            N: Number of antennas
            model_order_estimation: Model order estimation method
            physical_array: Physical array positions (optional)
            physical_gains: Physical antenna gains (optional)
        """
        system_model = SystemModel(system_model_params)
        super().__init__(system_model, model_order_estimation=model_order_estimation,
                         physical_array=physical_array, physical_gains=physical_gains)
        
        self.params = system_model.params
        self.N = N
        self.wavelength = self.params.wavelength
        self.window_size = self.params.softmax_window_size
        
        # Store model type for algorithm selection
        self.model_type = system_model_params.model_type.lower() if hasattr(system_model_params, 'model_type') else 'diffmusic'
        
        # Initialize learnable parameters
        self._init_learnable_parameters(physical_array, physical_gains)
        
        # Initialize DoA grid for far-field
        self._init_angle_grid()
        
        # Precompute steering matrix on grid
        self._precompute_steering_grid()
        
        self.music_spectrum = None
        self.noise_subspace = None

    def _init_learnable_parameters(self, physical_array=None, physical_gains=None):
        """Initialize learnable antenna positions and complex gains"""
        
        # Initialize antenna positions
        if physical_array is not None:
            # Use provided physical array positions
            if isinstance(physical_array, np.ndarray):
                initial_positions = torch.from_numpy(physical_array).to(torch.float64)
            else:
                initial_positions = physical_array.to(torch.float64)
        else:
            # Default: nominal ULA with half-wavelength spacing
            initial_positions = torch.arange(self.N, dtype=torch.float64) * (self.wavelength / 2)
        
        self.antenna_positions = nn.Parameter(initial_positions)
        
        # Initialize complex gains
        if physical_gains is not None:
            # Use provided physical gains
            if isinstance(physical_gains, np.ndarray):
                initial_gains = torch.from_numpy(physical_gains).to(torch.complex64)
            else:
                initial_gains = physical_gains.to(torch.complex64)
        else:
            # Default: unit gains (real=1, imag=0)
            gains_real = torch.ones(self.N, dtype=torch.float64)
            gains_imag = torch.zeros(self.N, dtype=torch.float64)
            initial_gains = torch.complex(gains_real, gains_imag).to(torch.complex64)
        
        self.complex_gain = nn.Parameter(initial_gains)

    def _init_angle_grid(self):
        """Initialize angle grid for DOA estimation"""
        angle_range = np.deg2rad(self.params.doa_range)
        angle_resolution = np.deg2rad(self.params.doa_resolution / 2) # Higher resolution by 2 than the original grid.
        angle_decimals = int(np.ceil(np.log10(1 / angle_resolution))) # Formula to determine floating point accuracy based on the resolution.
        
        self.angles_grid = torch.arange(-angle_range, angle_range + angle_resolution, 
                                      angle_resolution, dtype=torch.float64)
        self.angles_grid = torch.round(self.angles_grid, decimals=angle_decimals)

    def get_angles_grid(self) -> torch.Tensor:
        """
        Get the angle grid for DoA estimation
        
        Returns:
            Tensor of angles in radians, shape (num_angles,)
        """
        return self.angles_grid.to(self.device)

    def _precompute_steering_grid(self):
        """Precompute steering vectors for the angular grid"""
        # This will be computed dynamically during forward pass since parameters are learnable
        pass

    def get_array_learnable_parameters(self, learnable = True, normalized_gain: bool = False) -> tuple:
        """
        Get the learnable antenna positions and complex gains
        
        Args:
            learnable: If True, return nn.Parameters; if False, return detached numpy arrays
            normalized_gain: If True, normalize gains to L2 norm of 1 (only works when learnable=False)
        
        Returns:
            tuple: (antenna_positions, complex_gains)
        """
        if normalized_gain and learnable:
            raise ValueError("normalized_gain=True can only be used when learnable=False")
            
        if learnable:
            return self.antenna_positions, self.complex_gain
        else:
            # CRITICAL FIX: Ensure proper deep copy to avoid memory sharing issues
            # Use .clone() before .detach() to ensure we get a completely separate tensor
            positions = self.antenna_positions.clone().detach().cpu().numpy()
            gains = self.complex_gain.clone().detach().cpu().numpy()
            
            if normalized_gain:
                # Normalize gains to L2 norm of 1
                gains_norm = np.linalg.norm(gains)
                gains /= (gains_norm + 1e-8)  # Prevent division by zero
            
            return positions, gains
    
    def set_window_size(self, window_size):
        """
        Dynamically update window size for diffMUSIC
        
        Args:
            window_size: New window size (int for absolute, float 0-1 for relative)
        """
        if isinstance(window_size, float) and 0 < window_size < 1:
            # Relative to grid size
            self.window_size = int(window_size * len(self.angles_grid))
        else:
            # Absolute size
            self.window_size = int(window_size)
        

    def compute_steering_matrix(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable steering matrix for given angles
        
        Args:
            angles: Tensor of angles in radians, shape (num_angles,)
            
        Returns:
            Complex steering matrix of shape (N, num_angles)
        """
        if angles.dim() == 0:
            angles = angles.unsqueeze(0)
        
        # Ensure angles have the same dtype as antenna_positions (float64)
        angles = angles.to(torch.float64)
        
        # Get complex gains - use normalized version when not in training mode
        if self.training:
            antenna_positions, complex_gains = self.get_array_learnable_parameters(learnable=True)
        else:
            # Use normalized gains during evaluation
            antenna_positions_np, complex_gains_np = self.get_array_learnable_parameters(
                learnable=False, normalized_gain=True
            )
            # Convert back to torch tensors
            antenna_positions = torch.from_numpy(antenna_positions_np).to(self.antenna_positions.device)
            complex_gains = torch.from_numpy(complex_gains_np).to(self.complex_gain.device)
        
        complex_gains = complex_gains.to(torch.complex128)  # Ensure complex gains are in complex128 format
        
        # Compute steering vectors: a(θ) = g ⊙ exp(-j * 2π * p * sin(θ) / λ)
        # where ⊙ is element-wise multiplication
        
        # Phase computation: (N, 1) * (1, num_angles) -> (N, num_angles)
        phase_delays = antenna_positions.unsqueeze(1) @ torch.sin(angles).unsqueeze(0)
        
        # Steering matrix without gains
        steering_base = torch.exp(-2j * torch.pi * phase_delays / self.wavelength)
        
        # Apply complex gains: (N, N) * (N, num_angles) -> (N, num_angles)
        steering_matrix = torch.diag(complex_gains) @ steering_base
        
        # # Normalization (Note it is detached, the gradients were coupled and it made the learning a mess)
        # norm_factor = 1 / torch.linalg.norm(complex_gains.detach(), ord=2)
        # steering_matrix = norm_factor * steering_matrix
        
        return steering_matrix.to(torch.complex64)
    
#NOTE: from now on, the first dimension is always batch size, in our case it's dummy for compatibility. It'll be always 1.

    def forward(self, cov: torch.Tensor, number_of_sources: int, known_angles=None):
        """
        Forward pass of diffMUSIC
        
        Args:
            cov: Covariance matrix of shape (BATCH_SIZE, N, N). We leave the batch dimension although for now it's going to be dummy just for the sake of compatibility.
            number_of_sources: Number of sources to estimate, if None, will estimate the number of sources
            known_angles: Not used in far-field case (kept for compatibility)
            known_distances: Not used in far-field case (kept for compatibility)
            
        Returns:
            tuple: (estimated_angles, source_estimation, eigen_regularization)
        """
        # Ensure covariance is complex
        cov = cov.to(torch.complex128)
        
        # Subspace decomposition
        _, noise_subspace, source_estimation, eigen_regularization = self.subspace_separation(cov, number_of_sources)
        self.noise_subspace = noise_subspace.to(self.device)
        
        # Compute MUSIC spectrum
        inverse_spectrum = self._compute_inverse_spectrum(self.noise_subspace)
        self.music_spectrum = 1 / (inverse_spectrum + 1e-10)
        
        # Peak finding (differentiable during training, hard during inference)
        estimated_angles, peaks_masks = self._peak_finder(number_of_sources)
        
        return estimated_angles, peaks_masks, source_estimation, eigen_regularization #eigen_regularization is not used in this implementation

    def _compute_inverse_spectrum(self, noise_subspace: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse MUSIC spectrum using current learnable parameters
        
        Args:
            noise_subspace: Noise subspace of shape (batch_size, N, N-M)
            
        Returns:
            Inverse spectrum of shape (batch_size, num_angles)
        """
        # Compute steering matrix for all angles in the grid
        steering_grid = self.compute_steering_matrix(self.angles_grid.to(self.device))  # (N, num_angles)
        
        # Compute inverse spectrum: ||U_N^H * A(θ)||²
        # steering_grid: (N, num_angles), noise_subspace: (batch_size, N, N-M)
        var1 = torch.einsum("na, bnm -> bam", 
                           steering_grid.conj(), 
                           noise_subspace.to(torch.complex64))
        # Computes the matrix multiplication using Einstein notation, just a fancier way to write it.
        
        inverse_spectrum = torch.norm(var1, dim=2) ** 2  # (batch_size, num_angles)
        
        return inverse_spectrum

    def _peak_finder(self, number_of_sources: int) -> torch.Tensor:
        """
        Peak finding - soft for diffMUSIC, hard for MUSIC
        
        Args:
            number_of_sources: Number of peaks to find
            
        Returns:
            Estimated angles in radians
        """
        if self.model_type == "diffmusic":
            return self._differentiable_peak_finder(number_of_sources)
        else:  # MUSIC or other types
            return self._hard_peak_finder(number_of_sources, return_angles=True)

    def _hard_peak_finder(self, number_of_sources: int, return_angles = True) -> torch.Tensor:
        """
        Non-differentiable peak finding for inference
        
        Args:
            number_of_sources: Number of peaks to find
            return_angles: Wheter to return angles or peaks_indices for DiffMUSIC's internal use
            
        Returns:
            Estimated angles in radians, shape (batch_size, number_of_sources)
        """
        batch_size = self.music_spectrum.shape[0]
        peaks = torch.zeros(batch_size, number_of_sources, dtype=torch.int64, device=self.device)
        
        for batch in range(batch_size):
            spectrum = self.music_spectrum[batch].cpu().detach().numpy()
            
            # Find peaks using scipy
            peaks_indices = sc.signal.find_peaks(spectrum, threshold=0.0)[0]
            
            if len(peaks_indices) < number_of_sources:
                warnings.warn(f"diffMUSIC: Not enough peaks found, trying to find another {number_of_sources - len(peaks_indices)} peaks by top amplitude.")
                # Use highest values instead
                additional_peaks = torch.topk(torch.from_numpy(spectrum), 
                                            number_of_sources - len(peaks_indices), 
                                            largest=True).indices.numpy()
                peaks_indices = np.concatenate([peaks_indices, additional_peaks])
            
            # Sort by amplitude and take top peaks
            sorted_peaks = peaks_indices[np.argsort(spectrum[peaks_indices])[::-1]]
            peaks[batch] = torch.from_numpy(sorted_peaks[:number_of_sources]).to(self.device)
        
        if not return_angles:
            # Return peak indices directly
            return peaks
        else:
            # Convert indices to angles
            estimated_angles = torch.gather(
                self.angles_grid.unsqueeze(0).repeat(batch_size, 1).to(self.device), 
                1, peaks
            )
            
            return estimated_angles, None # Here for compatibility with the differentiable peak finder, we return None for peaks_masks.

    def _differentiable_peak_finder(self, number_of_sources: int) -> torch.Tensor:
        """
        Differentiable peak finding using softmax (Algorithm 2 from paper)
        
        Args:
            number_of_sources: Number of sources to estimate
            
        Returns:
            Estimated angles in radians, shape (batch_size, number_of_sources)
        """
        batch_size = self.music_spectrum.shape[0]
        estimated_angles = torch.zeros(batch_size, number_of_sources, 
                                     dtype=torch.float64, device=self.device)
        
        peaks_indices = self._hard_peak_finder(number_of_sources, return_angles=False) # torch.Tensor of (batch_size, number_of_sources)
        peaks_masks = [] # a list of lists, each containing indices of peaks for each batch, needed for the unsupervised loss.
        
        for batch in range(batch_size):
            batch_masks = []
            spectrum = self.music_spectrum[batch]
            
            # Find initial peaks (non-differentiable, but gradients will flow through softmax)
            peaks_indices_per_batch = peaks_indices[batch]
            
            # For each peak, apply differentiable refinement
            for source_idx in range(number_of_sources):
                peak_idx = peaks_indices_per_batch[source_idx]
                
                # Create angular mask around peak
                mask_indices = self._create_angular_mask(peak_idx, spectrum.shape[0])
                batch_masks.append(mask_indices)  # Store for unsupervised loss
                
                # Extract spectrum values in the mask
                masked_spectrum = spectrum[mask_indices]
                
                # Apply softmax to get weights
                weights = torch.softmax(masked_spectrum, dim=0)
                
                # Compute weighted average of angles (Equation 13 from paper)
                masked_angles = self.angles_grid[mask_indices].to(self.device)
                estimated_angles[batch, source_idx] = torch.sum(weights * masked_angles)

            peaks_masks.append(batch_masks)  # Store masks for unsupervised loss
        
        return estimated_angles, peaks_masks


    def _create_angular_mask(self, center_idx: int, spectrum_length: int) -> torch.Tensor:
        """
        Create angular mask around peak center (ΠL operation from paper)
        
        Args:
            center_idx: Center index of the peak
            spectrum_length: Length of the spectrum
            
        Returns:
            Indices for the angular mask
        """
        half_window = self.window_size // 2
        
        # Create mask indices around center
        start_idx = max(0, center_idx - half_window)
        end_idx = min(spectrum_length, center_idx + half_window + 1)
        
        mask_indices = torch.arange(start_idx, end_idx, device=self.device, dtype=torch.long)
        
        return mask_indices


    def set_nominal_parameters(self, positions: torch.Tensor = None, gains: torch.Tensor = None):
        """
        Set parameters to nominal values (useful for initialization or comparison)
        
        Args:
            positions: Nominal antenna positions (if None, use half-wavelength spacing)
            gains: Nominal complex gains (if None, use unit gains)
        """
        with torch.no_grad():
            if positions is None:
                # Half-wavelength spacing
                positions = torch.arange(self.N, dtype=torch.float64) * (self.wavelength / 2)
            
            if gains is None:
                # Unit gains
                gains = torch.ones(self.N, dtype=torch.complex64)
            
            self.antenna_positions.copy_(positions)
            self.complex_gain.copy_(gains)

    def plot_spectrum(self, highlight_angles: torch.Tensor = None, 
                     path_to_save: str | None = False, batch_idx: int = 0):
        """
        Plot the MUSIC spectrum
        
        Args:
            batch_idx: Which batch element to plot
            highlight_angles: True angles to highlight on the plot
            save: Whether to save the plot
            title: Plot title
        """
        if self.music_spectrum is None:
            warnings.warn("No spectrum computed yet. Run forward pass first.")
            return
        
        angles_deg = torch.rad2deg(self.angles_grid).cpu().numpy()
        spectrum = self.music_spectrum[batch_idx].cpu().detach().numpy()
        
        plt.figure(figsize=(10, 6))
        plt.plot(angles_deg, spectrum, 'b-', linewidth=2, label=f'{self.model_type.upper()} Spectrum')
        
        if highlight_angles is not None:
            highlight_deg = torch.rad2deg(highlight_angles).cpu().numpy()
            for i, angle in enumerate(highlight_deg):
                plt.axvline(x=angle, color='r', linestyle='--', alpha=0.7, 
                           label='True DoA' if i == 0 else "")
        
        plt.xlabel('Angle [degrees]')
        plt.ylabel('Spectrum Power')
        plt.title(self.model_type.upper() + " Spectrum")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if path_to_save is not None:
            plt.savefig(path_to_save / f"Spectrum.png", dpi=300)
        plt.show()



class DiffMUSICLoss(nn.Module):
    """
    Wrapper loss class for diffMUSIC training
    Supports different loss strategies: RMSPE, spectrum, and unsupervised
    """
    
    def __init__(self, loss_type: str = "rmspe", gains_reg_coeff: float = 0.0, 
                 normalized_target_gains_norm: float = 1.0, **kwargs):
        """
        Args:
            loss_type: "rmspe" for LSL,θ, "spectrum" for LSL,P, or "unsupervised" for LUL
            gains_reg_coeff: Coefficient for gains regularization. If 0, regularization is disabled
            normalized_target_gains_norm: Target normalized norm of the gains vector (default: 1.0)
            **kwargs: Additional arguments for specific loss functions
        """
        super(DiffMUSICLoss, self).__init__()
        self.loss_type = loss_type
        self.gains_reg_coeff = gains_reg_coeff
        self.normalized_target_gains_norm = normalized_target_gains_norm
        
        # Import loss functions from metrics
        from src.metrics import RMSPELoss, SpectrumLoss, UnsupervisedSpectrumLoss
        
        if loss_type == "rmspe":
            self.loss_fn = RMSPELoss(**kwargs)
        elif loss_type == "spectrum":
            self.loss_fn = SpectrumLoss(**kwargs)
        elif loss_type == "unsupervised":
            self.loss_fn = UnsupervisedSpectrumLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _compute_gains_regularization(self, algorithm):
        """
        Compute gains regularization term: gains_reg_coeff * (||learned_gains||_L2 / sqrt(len(learned_gains)) - normalized_target_gains_norm)^2
        
        Args:
            algorithm: DiffMUSIC algorithm instance with learnable gains
            
        Returns:
            regularization_loss: Computed regularization loss (tensor)
        """
        if self.gains_reg_coeff == 0 or not hasattr(algorithm, 'complex_gain'):
            return torch.tensor(0.0, device=algorithm.complex_gain.device, requires_grad=True)
        
        # Get the learned gains (not normalized)
        _, learned_gains = algorithm.get_array_learnable_parameters(learnable=True)
        
        # Debug: Check if gradients are enabled
        if not learned_gains.requires_grad:
            print(f"WARNING: learned_gains.requires_grad = {learned_gains.requires_grad}")
        
        # Compute L2 norm of gains and normalize by sqrt(length)
        gains_l2_norm = torch.linalg.norm(learned_gains, ord=2)
        sqrt_length = torch.sqrt(torch.tensor(len(learned_gains), device=learned_gains.device, dtype=gains_l2_norm.dtype))
        normalized_gains_norm = gains_l2_norm / sqrt_length
        
        # Convert target to tensor on correct device
        target_norm = torch.tensor(self.normalized_target_gains_norm, 
                                   device=learned_gains.device, dtype=normalized_gains_norm.dtype)
        
        # Use squared difference to avoid gradient issues at zero and ensure proper gradient flow
        difference = normalized_gains_norm - target_norm
        regularization_loss = self.gains_reg_coeff * (difference ** 2)
        
        return regularization_loss
    
    def forward(self, algorithm=None, return_components=False, **kwargs):
        """
        Forward pass - delegates to appropriate loss function and adds regularization
        
        Args:
            algorithm: DiffMUSIC algorithm instance (for regularization)
            return_components: If True, return dictionary with loss components
            **kwargs: Arguments for the specific loss function
        """
        # Compute primary loss
        if self.loss_type == "rmspe":
            primary_loss = self.loss_fn(kwargs['predictions'], kwargs['targets'])
        elif self.loss_type == "spectrum":
            primary_loss = self.loss_fn(kwargs['spectrum'], kwargs['targets'], kwargs['angles_grid'])
        elif self.loss_type == "unsupervised":
            primary_loss = self.loss_fn(kwargs['spectrum'], kwargs['peak_masks'])
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Compute regularization loss
        regularization_loss = self._compute_gains_regularization(algorithm) if algorithm is not None else torch.tensor(0.0)
        
        # Total loss
        total_loss = primary_loss + regularization_loss
        
        if return_components:
            return {
                'total_loss': total_loss,
                'primary_loss': primary_loss,
                'regularization_loss': regularization_loss
            }
        else:
            return total_loss