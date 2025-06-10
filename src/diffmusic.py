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
# from src.metrics import RMSPELoss


class DiffMUSIC(SubspaceMethod):
    """
    Differentiable MUSIC (diffMUSIC) implementation for DoA estimation with hardware impairment learning.
    
    This implementation is focused on:
    - Far-field scenarios only
    - Non-coherent sources  
    - Narrowband signals
    
    Key features:
    - Learnable antenna positions (nn.Parameter)
    - Learnable complex gains (nn.Parameter) 
    - Differentiable steering matrix computation
    - Softmax-based peak finding for differentiability
    """

    def __init__(self, system_model_params, N: int, 
                 window_size: int = 21, model_order_estimation: str = None):
        """
        Initialize diffMUSIC
        
        Args:
            system_model: System model object (kept for compatibility)
            N: Number of antennas
            wavelength: Signal wavelength  
            window_size: Size of angular window for softmax peak finding
            model_order_estimation: Model order estimation method
        """
        system_model = SystemModel(system_model_params)
        super().__init__(system_model, model_order_estimation=model_order_estimation)
        
        self.params = system_model.params
        self.N = N
        self.wavelength = self.params.wavelength
        self.window_size = window_size # For now it is inserted directly
        
        # Initialize learnable parameters
        self._init_learnable_parameters()
        
        # Initialize DoA grid for far-field
        self._init_angle_grid()
        
        # Precompute steering matrix on grid
        self._precompute_steering_grid()
        
        self.music_spectrum = None
        self.noise_subspace = None

    def _init_learnable_parameters(self):
        """Initialize learnable antenna positions and complex gains"""
        
        # Initialize antenna positions - nominal ULA with half-wavelength spacing
        nominal_positions = torch.arange(self.N, dtype=torch.float64) * (self.wavelength / 2)
        self.antenna_positions = nn.Parameter(nominal_positions)
        
        # Initialize complex gains - start with unit gains (real=1, imag=0)
        gains_real = torch.ones(self.N, dtype=torch.float64)
        gains_imag = torch.zeros(self.N, dtype=torch.float64)
        self.gains_real = nn.Parameter(gains_real)
        self.gains_imag = nn.Parameter(gains_imag)
        self.complex_gain = torch.complex(gains_real, gains_imag, dtype=torch.complex64)

    def _init_angle_grid(self):
        """Initialize angle grid for DOA estimation"""
        angle_range = np.deg2rad(self.params.doa_range)
        angle_resolution = np.deg2rad(self.params.doa_resolution / 2) # Higher resolution by 2 than the original grid.
        angle_decimals = int(np.ceil(np.log10(1 / angle_resolution))) # Formula to determine floating point accuracy based on the resolution.
        
        self.angles_dict = torch.arange(-angle_range, angle_range + angle_resolution, 
                                      angle_resolution, dtype=torch.float64)
        self.angles_dict = torch.round(self.angles_dict, decimals=angle_decimals)

    def _precompute_steering_grid(self):
        """Precompute steering vectors for the angular grid"""
        # This will be computed dynamically during forward pass since parameters are learnable
        pass



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
        
        # Get complex gains
        complex_gains = self.complex_gain
        
        # Compute steering vectors: a(θ) = g ⊙ exp(-j * 2π * p * sin(θ) / λ)
        # where ⊙ is element-wise multiplication
        
        # Phase computation: (N, 1) * (1, num_angles) -> (N, num_angles)
        phase_delays = self.antenna_positions.unsqueeze(1) @ torch.sin(angles).unsqueeze(0)
        
        # Steering matrix without gains
        steering_base = torch.exp(-2j * torch.pi * phase_delays / self.wavelength)
        
        # Apply complex gains: (N, N) * (N, num_angles) -> (N, num_angles)
        steering_matrix = torch.diag(complex_gains) @ steering_base
        
        # Normalization (as in paper equation 2)
        norm_factor = 1 / torch.linalg.norm(complex_gains, ord=2)
        steering_matrix = norm_factor * steering_matrix
        
        return steering_matrix.to(torch.complex64)
    
#TODO: Read up to the peak_finder inside the forward-pass. Looks fine, need to understand the peak_finder. 
# If the hard decision is fine, we can use this also for MUSIC. Keep reading and notice the first dimension 
# of cov is the batch_dim. Another thing is to change system_model_params to also include the training_params 
# (all params in general). For now, the gains real and imag parts are parameters, change just 
# the whole complex gain to be a parameter, and use it in the steering matrix. 

    def forward(self, cov: torch.Tensor, number_of_sources: int, known_angles=None):
        """
        Forward pass of diffMUSIC
        
        Args:
            cov: Covariance matrix of shape (BATCH_SIZE, N, N)
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
        estimated_angles = self._peak_finder(number_of_sources)
        
        return estimated_angles, source_estimation, eigen_regularization #eigen_regularization is not used in this implementation

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
                           noise_subspace)  # (batch_size, num_angles, N-M)
        # Computes the matrix multiplication using Einstein notation, just a fancier way to write it.
        
        inverse_spectrum = torch.norm(var1, dim=2) ** 2  # (batch_size, num_angles)
        
        return inverse_spectrum

    def _peak_finder(self, number_of_sources: int) -> torch.Tensor:
        """
        Peak finding - differentiable during training, hard during inference
        
        Args:
            number_of_sources: Number of peaks to find
            
        Returns:
            Estimated angles in radians
        """
        if self.training: #This is built-in since it is a Module, just use model.train() or model.eval()
            return self._differentiable_peak_finder(number_of_sources)
        else:
            return self._hard_peak_finder(number_of_sources)

    def _hard_peak_finder(self, number_of_sources: int) -> torch.Tensor:
        """
        Non-differentiable peak finding for inference
        
        Args:
            number_of_sources: Number of peaks to find
            
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
                warnings.warn("diffMUSIC: Not enough peaks found, using highest values")
                # Use highest values instead
                additional_peaks = torch.topk(torch.from_numpy(spectrum), 
                                            number_of_sources - len(peaks_indices), 
                                            largest=True).indices.numpy()
                peaks_indices = np.concatenate([peaks_indices, additional_peaks])
            
            # Sort by amplitude and take top peaks
            sorted_peaks = peaks_indices[np.argsort(spectrum[peaks_indices])[::-1]]
            peaks[batch] = torch.from_numpy(sorted_peaks[:number_of_sources]).to(self.device)
        
        # Convert indices to angles
        estimated_angles = torch.gather(
            self.angles_grid.unsqueeze(0).repeat(batch_size, 1).to(self.device), 
            1, peaks
        )
        
        return estimated_angles

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
        
        for batch in range(batch_size):
            spectrum = self.music_spectrum[batch]
            
            # Find initial peaks (non-differentiable, but gradients will flow through softmax)
            peaks_indices = self._find_initial_peaks(spectrum, number_of_sources)
            
            # For each peak, apply differentiable refinement
            for source_idx in range(number_of_sources):
                peak_idx = peaks_indices[source_idx]
                
                # Create angular mask around peak
                mask_indices = self._create_angular_mask(peak_idx, spectrum.shape[0])
                
                # Extract spectrum values in the mask
                masked_spectrum = spectrum[mask_indices]
                
                # Apply softmax to get weights
                weights = torch.softmax(masked_spectrum, dim=0)
                
                # Compute weighted average of angles (Equation 13 from paper)
                masked_angles = self.angles_grid[mask_indices].to(self.device)
                estimated_angles[batch, source_idx] = torch.sum(weights * masked_angles)
        
        return estimated_angles

    def _find_initial_peaks(self, spectrum: torch.Tensor, number_of_sources: int) -> torch.Tensor:
        """Find initial peak locations (non-differentiable but provides starting points)"""
        spectrum_np = spectrum.cpu().detach().numpy()
        peaks_indices = sc.signal.find_peaks(spectrum_np, threshold=0.0)[0]
        
        if len(peaks_indices) < number_of_sources:
            # Use highest values if not enough peaks
            additional_peaks = torch.topk(spectrum, 
                                        number_of_sources - len(peaks_indices), 
                                        largest=True).indices.cpu().numpy()
            peaks_indices = np.concatenate([peaks_indices, additional_peaks])
        
        # Sort by amplitude and take top peaks
        sorted_peaks = peaks_indices[np.argsort(spectrum_np[peaks_indices])[::-1]]
        return torch.from_numpy(sorted_peaks[:number_of_sources]).to(self.device)

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
        
        mask_indices = torch.arange(start_idx, end_idx, device=self.device)
        
        return mask_indices

    def get_learned_parameters(self):
        """
        Get the learned array parameters
        
        Returns:
            dict: Dictionary containing learned positions and gains
        """
        return {
            'antenna_positions': self.antenna_positions.detach().cpu().numpy(),
            'complex_gains': self.get_complex_gains().detach().cpu().numpy(),
            'gains_magnitude': torch.abs(self.get_complex_gains()).detach().cpu().numpy(),
            'gains_phase': torch.angle(self.get_complex_gains()).detach().cpu().numpy()
        }

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
            self.gains_real.copy_(gains.real.to(torch.float64))
            self.gains_imag.copy_(gains.imag.to(torch.float64))

    def plot_spectrum(self, batch_idx: int = 0, highlight_angles: torch.Tensor = None, 
                     save: bool = False, title: str = "diffMUSIC Spectrum"):
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
        plt.plot(angles_deg, spectrum, 'b-', linewidth=2, label='diffMUSIC Spectrum')
        
        if highlight_angles is not None:
            highlight_deg = torch.rad2deg(highlight_angles).cpu().numpy()
            for i, angle in enumerate(highlight_deg):
                plt.axvline(x=angle, color='r', linestyle='--', alpha=0.7, 
                           label='True DoA' if i == 0 else "")
        
        plt.xlabel('Angle [degrees]')
        plt.ylabel('Spectrum Power')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save:
            plt.savefig('diffmusic_spectrum.pdf')
        plt.show()

    def plot_learned_array(self, save: bool = False):
        """
        Plot the learned antenna array geometry
        
        Args:
            save: Whether to save the plot
        """
        learned_params = self.get_learned_parameters()
        positions = learned_params['antenna_positions']
        gains_mag = learned_params['gains_magnitude'] 
        gains_phase = learned_params['gains_phase']
        
        # Nominal positions for comparison
        nominal_positions = np.arange(self.N) * (self.wavelength / 2)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Antenna positions
        ax1.scatter(nominal_positions, np.zeros_like(nominal_positions), 
                   marker='o', s=100, alpha=0.5, label='Nominal', color='blue')
        ax1.scatter(positions, np.zeros_like(positions), 
                   marker='s', s=100, label='Learned', color='red')
        ax1.set_xlabel('Position [wavelengths]')
        ax1.set_title('Antenna Positions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Gain magnitudes
        antenna_indices = np.arange(self.N)
        ax2.bar(antenna_indices - 0.2, np.ones(self.N), width=0.4, 
               alpha=0.7, label='Nominal', color='blue')
        ax2.bar(antenna_indices + 0.2, gains_mag, width=0.4, 
               alpha=0.7, label='Learned', color='red')
        ax2.set_xlabel('Antenna Index')
        ax2.set_ylabel('Gain Magnitude')
        ax2.set_title('Complex Gain Magnitudes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Gain phases
        ax3.bar(antenna_indices, gains_phase, width=0.6, alpha=0.7, color='green')
        ax3.set_xlabel('Antenna Index')
        ax3.set_ylabel('Gain Phase [radians]')
        ax3.set_title('Complex Gain Phases')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('diffmusic_learned_array.pdf')
        plt.show()

    def test_step(self, batch, batch_idx, model: nn.Module = None):
        """
        Test step compatible with the existing framework
        
        Args:
            batch: Test batch (x, sources_num, label)
            batch_idx: Batch index
            model: Model (not used, kept for compatibility)
            
        Returns:
            tuple: (rmspe, accuracy, test_length)
        """
        x, sources_num, label = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        test_length = x.shape[0]
        x = x.to(self.device)
        angles = label.to(self.device)
        
        # Check if sources number is consistent
        if (sources_num != sources_num[0]).any():
            raise Exception("diffMUSIC test_step: Inconsistent number of sources in batch")
        
        sources_num = sources_num[0]
        
        # Compute covariance
        if self.system_model.params.signal_nature == "non-coherent":
            Rx = self.pre_processing(x, mode="sample")
        else:
            Rx = self.pre_processing(x, mode="sample")
        
        # Run diffMUSIC
        predictions, sources_num_estimation, _ = self(Rx, number_of_sources=sources_num)
        
        # Compute RMSPE 
        criterion = RMSPELoss(balance_factor=1.0)
        rmspe = criterion(predictions, angles).sum().item()
        
        # Compute accuracy
        acc = self.source_estimation_accuracy(sources_num, sources_num_estimation)
        
        return rmspe, acc, test_length

    def __str__(self):
        return "diffMUSIC"

    def _get_name(self):
        return "diffMUSIC"


# class DiffMUSICLoss(nn.Module):
#     """
#     Loss functions for diffMUSIC training
#     Implements both supervised learning strategies from the paper:
#     - LSL,θ: RMSPE on estimated DoAs
#     - LSL,P: Maximize spectrum amplitude at true DoA locations
#     """
    
#     def __init__(self, loss_type: str = "rmspe"):
#         """
#         Args:
#             loss_type: "rmspe" for LSL,θ or "spectrum" for LSL,P
#         """
#         super().__init__()
#         self.loss_type = loss_type
#         self.rmspe_loss = RMSPELoss(balance_factor=1.0)
    
#     def forward(self, predictions, targets, spectrum=None, angles_grid=None):
#         """
#         Compute loss based on specified type
        
#         Args:
#             predictions: Predicted DoAs (for RMSPE loss)
#             targets: True DoAs
#             spectrum: MUSIC spectrum (for spectrum loss)
#             angles_grid: Angular grid (for spectrum loss)
            
#         Returns:
#             Loss value
#         """
#         if self.loss_type == "rmspe":
#             return self.rmspe_loss(predictions, targets)
        
#         elif self.loss_type == "spectrum":
#             if spectrum is None or angles_grid is None:
#                 raise ValueError("Spectrum and angles_grid required for spectrum loss")
            
#             return self._spectrum_loss(targets, spectrum, angles_grid)
        
#         else:
#             raise ValueError(f"Unknown loss type: {self.loss_type}")
    
#     def _spectrum_loss(self, true_angles, spectrum, angles_grid):
#         """
#         Spectrum-based loss (LSL,P from paper)
#         Maximizes spectrum amplitude at true DoA locations
#         """
#         batch_size = spectrum.shape[0]
#         total_loss = 0
        
#         for batch in range(batch_size):
#             for angle in true_angles[batch]:
#                 # Find closest angle in grid
#                 angle_idx = torch.argmin(torch.abs(angles_grid - angle))
#                 # Negative spectrum value (to maximize)
#                 total_loss -= spectrum[batch, angle_idx]
        
#         return total_loss / (batch_size * true_angles.shape[1])


# class UnsupervisedDiffMUSIC(DiffMUSIC):
#     """
#     Unsupervised diffMUSIC using Jain's Index (LUL from paper)
#     Maximizes spectrum sharpness without requiring true DoA labels
#     """
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.jains_index_loss = JainsIndexLoss()
    
#     def compute_unsupervised_loss(self):
#         """
#         Compute unsupervised loss using Jain's Index on spectrum peaks
        
#         Returns:
#             Loss value encouraging sharp peaks
#         """
#         if self.music_spectrum is None:
#             raise ValueError("No spectrum computed. Run forward pass first.")
        
#         total_loss = 0
#         batch_size = self.music_spectrum.shape[0]
        
#         for batch in range(batch_size):
#             spectrum = self.music_spectrum[batch]
            
#             # Find initial peaks to create masks
#             peaks_indices = self._find_initial_peaks(spectrum, self.system_model.params.M)
            
#             # Apply Jain's index to each peak region
#             for peak_idx in peaks_indices:
#                 mask_indices = self._create_angular_mask(peak_idx, spectrum.shape[0])
#                 masked_spectrum = spectrum[mask_indices]
                
#                 # Jain's index encourages sharp peaks
#                 jains_loss = self.jains_index_loss(masked_spectrum)
#                 total_loss += jains_loss
        
#         return total_loss / batch_size


# class JainsIndexLoss(nn.Module):
#     """
#     Jain's Index loss for unsupervised learning
#     Encourages sharp, concentrated peaks in the spectrum
#     """
    
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, x):
#         """
#         Compute Jain's Index: J(x) = (sum(x))^2 / (n * sum(x^2))
        
#         Args:
#             x: Input tensor (spectrum values)
            
#         Returns:
#             Jain's index value (higher = more concentrated)
#         """
#         n = x.shape[0]
#         sum_x = torch.sum(x)
#         sum_x_squared = torch.sum(x ** 2)
        
#         jains_index = (sum_x ** 2) / (n * sum_x_squared + 1e-8)
        
#         # Return negative to minimize (we want to maximize Jain's index)
#         return -jains_index