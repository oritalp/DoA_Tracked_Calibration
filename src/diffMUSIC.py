import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy as sc

from src.signal_creation import SystemModel, SystemModelParams
from src.utils import set_unified_seed
from scipy.ndimage import maximum_filter
from scipy.ndimage import label
from scipy.ndimage import find_objects

#TODO: Haven't checked this method yet.
def find_k_highest_peaks(matrix, k):
    """
    Find the k highest peaks in a 2D matrix using SciPy tools. A peak is defined as a
    local maximum surrounded by smaller values.
    """
    # Apply maximum filter to find local maxima
    neighborhood = maximum_filter(matrix, size=21, mode='constant', cval=-np.inf)
    local_max = (matrix == neighborhood)

    # Label the connected components of local maxima
    labeled, num_features = label(local_max)
    slices = find_objects(labeled)

    # Extract peak positions and values
    peaks = []
    try:
        for sl in slices:
            row = int((sl[0].start + sl[0].stop - 1) / 2)
            col = int((sl[1].start + sl[1].stop - 1) / 2)
            value = matrix[row, col]
            peaks.append((row, col, value))
    except Exception as e:
        pass

    # Sort peaks by value (descending) and select the top k
    peaks = sorted(peaks, key=lambda x: x[2], reverse=True)[:k]
    if len(peaks) < k:
        warnings.warn(f"find_k_highest_peaks: Less than {k} peaks found.")
        # add random peaks
        x_random = np.random.randint(0, matrix.shape[0], (k - len(peaks),))
        y_random = np.random.randint(0, matrix.shape[1], (k - len(peaks),))
        for i in range(k - len(peaks)):
            peaks.append((x_random[i], y_random[i], matrix[x_random[i], y_random[i]]))

    return peaks


class DiffMUSIC(nn.Module):
    """
    Differentiable MUSIC implementation with learnable antenna gains and positions.
    Integrates with the existing SystemModel architecture for far-field, non-coherent, narrowband scenarios.
    """
    
    def __init__(self, 
                 system_model_params: SystemModelParams,
                 init_antenna_positions: torch.Tensor = None,
                 init_antenna_gains: torch.Tensor = None,
                 gain_constraint: str = "positive",  # "positive", "normalized", "none"
                 temperature: float = 1.0,
                 cell_size_coeff: float = 0.2 # between 0 and 1, determines the size of tthe window for softmax peak
                                              # finding, the size is computed as len(grid) * cell_size_coeff
                ):
        """
        Initialize DiffMUSIC with learnable antenna parameters.
        
        Args:
            system_model_params: SystemModelParams instance
            init_antenna_positions: Initial antenna positions [N] for ULA case
            init_antenna_gains: Initial antenna gains [N]
            gain_constraint: Type of gain constraint
            temperature: Temperature for soft peak finding
            cell_size_coeff: Coefficient for cell size in soft peak finding
        """
        super().__init__()
        
        self.params = system_model_params
        self.N = self.params.N  # Number of antennas
        self.M = self.params.M  # Number of sources
        self.gain_constraint = gain_constraint
        self.temperature = temperature
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize antenna positions as learnable parameters
        if init_antenna_positions is None:
            init_antenna_positions = torch.arange(self.N, dtype=torch.float64) * (self.params.wavelength / 2)
        
        self.antenna_positions = nn.Parameter(init_antenna_positions.clone())
        assert init_antenna_positions.requires_grad is True, "Initial antenna positions must be a differentiable tensor."

        # Initialize antenna gains as learnable parameters
        if init_antenna_gains is None:
            init_antenna_gains = torch.ones(self.N, dtype=torch.complex64)
        self.antenna_gains = nn.Parameter(init_antenna_gains.clone())
        
        # Initialize angle grid for far-field DOA estimation
        self._init_angle_grid()
        
        # Initialize cell size for soft peak finding
        self.cell_size = int(self.angles_dict.shape[0] * cell_size_coeff)
        if self.cell_size % 2 == 0:
            self.cell_size += 1
            
        # Store current MUSIC spectrum for analysis
        self.music_spectrum = None
        
    def _init_angle_grid(self):
        """Initialize angle grid for DOA estimation"""
        angle_range = np.deg2rad(self.params.doa_range)
        angle_resolution = np.deg2rad(self.params.doa_resolution / 2) # Higher resolution by 2 than the original grid.
        angle_decimals = int(np.ceil(np.log10(1 / angle_resolution))) #Formula to determine floating point accuracy based on the resolution.
        
        self.angles_dict = torch.arange(-angle_range, angle_range + angle_resolution, 
                                      angle_resolution, dtype=torch.float64)
        self.angles_dict = torch.round(self.angles_dict, decimals=angle_decimals)
    
    def _apply_position_constraints(self):
        """Reorders the antenna positions, ensuring we don't get stucked if some replace positions.
           I don't think we need this because it is mathematically fine that thy will replace order.
           In fact, this replacement may occur some incontinuity in the loss.
           We do need to keep it in mind though.
           """
        # with torch.no_grad():
        #     if self.position_constraint == "ula":
        #         sorted_positions, _ = torch.sort(self.antenna_positions)
        #         self.antenna_positions.data = sorted_positions
        pass
    
    def _apply_gain_constraints(self):
        """Prevent extrme gains from future random walk. For now i leave commented."""
        # with torch.no_grad():
        #     if self.gain_constraint == "positive":
        #         # Just clamp to reasonable ranges to prevent numerical issues
        #         real_part = torch.clamp(self.antenna_gains.real, min=0.1, max=10.0)
        #         imag_part = torch.clamp(self.antenna_gains.imag, min=-2.0, max=2.0)
        #         self.antenna_gains.data = torch.complex(real_part, imag_part, dtype=torch.complex64)
        pass
    
    def get_constrained_parameters(self):
        """Get antenna parameters without any sorting - order doesn't matter mathematically"""
        positions = self.antenna_positions
        gains = self.antenna_gains
        return positions, gains
    
#NOTE: ORI - I read up to this point.
    
    def compute_steering_matrix(self, angles: torch.Tensor):
        """
        Compute steering matrix for far-field sources using current antenna configuration.
        Follows the pattern from your SystemModel.steering_vec_far_field method.
        
        Args:
            angles: DOA angles in radians [num_angles]
            
        Returns:
            Complex steering matrix [N, num_angles]
        """
        positions, gains = self.get_constrained_parameters()
        
        # Convert angles to proper shape for broadcasting
        if angles.dim() == 1:
            angles = angles.unsqueeze(0)  # [1, num_angles]
        
        # Reshape positions for broadcasting: [N, 1]
        positions = positions.view(-1, 1)
        
        # Compute phase delays: [N, 1] * sin([1, num_angles]) -> [N, num_angles]
        time_delay = positions @ torch.sin(angles)
        
        # Compute steering vectors (following your formula)
        steering_matrix = torch.exp(-2j * torch.pi * time_delay / self.params.wavelength)
        steering_matrix = steering_matrix.to(torch.complex64)
        
        # Apply antenna gains
        gains_diag = torch.diag(gains)
        
        # Normalize gains (following your normalization pattern)
        normalization_factor = 1 / torch.linalg.norm(gains, ord=2)
        steering_matrix = normalization_factor * gains_diag @ steering_matrix
        
        return steering_matrix
    
    def sample_covariance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute sample covariance matrix (following your pattern).
        
        Args:
            x: Input samples [batch_size, N, T] or [N, T]
            
        Returns:
            Covariance matrices [batch_size, N, N]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size, sensor_number, samples_number = x.shape
        Rx = torch.einsum("bmt, btl -> bml", x, torch.conj(x).transpose(1, 2)) / samples_number
        return Rx
    
    def subspace_separation(self, cov: torch.Tensor, number_of_sources: int):
        """
        Perform eigendecomposition and separate signal/noise subspaces.
        
        Args:
            cov: Covariance matrix [batch_size, N, N]
            number_of_sources: Number of sources
            
        Returns:
            signal_subspace, noise_subspace, source_estimation, eigen_regularization
        """
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Sort in descending order
        sorted_indices = torch.argsort(eigenvalues, dim=-1, descending=True)
        eigenvalues = torch.gather(eigenvalues, -1, sorted_indices)
        eigenvectors = torch.gather(eigenvectors, -1, sorted_indices.unsqueeze(-2).expand_as(eigenvectors))
        
        # Separate subspaces
        signal_subspace = eigenvectors[:, :, :number_of_sources]
        noise_subspace = eigenvectors[:, :, number_of_sources:]
        
        # Source estimation (simplified)
        source_estimation = number_of_sources
        
        # Eigen regularization
        eigen_regularization = torch.mean(eigenvalues[:, number_of_sources:])
        
        return signal_subspace, noise_subspace, source_estimation, eigen_regularization
    
    def get_inverse_spectrum(self, noise_subspace: torch.Tensor):
        """
        Compute inverse MUSIC spectrum using current antenna configuration.
        
        Args:
            noise_subspace: Noise subspace [batch_size, N, N-M]
            
        Returns:
            Inverse spectrum [batch_size, num_angles]
        """
        # Compute steering matrix for all angles
        steering_dict = self.compute_steering_matrix(self.angles_dict)
        steering_dict = steering_dict[:noise_subspace.shape[1]].to(self.device)
        
        # Compute projection onto noise subspace
        var1 = torch.einsum("an, bnm -> bam", 
                           steering_dict.conj().transpose(0, 1)[:, :noise_subspace.shape[1]],
                           noise_subspace)
        inverse_spectrum = torch.norm(var1, dim=2) ** 2
        
        return inverse_spectrum
    
    def soft_peak_finding_1d(self, spectrum: torch.Tensor, search_space: torch.Tensor, num_sources: int):
        """
        Differentiable 1D peak finding using temperature-scaled softmax.
        Adapted from your __maskpeak_1d method.
        
        Args:
            spectrum: MUSIC spectrum [batch_size, num_points]
            search_space: Search grid values [num_points]
            num_sources: Number of sources to find
            
        Returns:
            Estimated parameters [batch_size, num_sources]
        """
        batch_size = spectrum.shape[0]
        
        # Find hard peaks for initialization
        peaks = torch.zeros(batch_size, num_sources, dtype=torch.int64, device=self.device)
        for batch in range(batch_size):
            music_spectrum_np = spectrum[batch].cpu().detach().numpy().squeeze()
            # Find spectrum peaks
            peaks_tmp = sc.signal.find_peaks(music_spectrum_np, threshold=0.0)[0]
            if len(peaks_tmp) < num_sources:
                # Take top values if not enough peaks
                random_peaks = torch.topk(torch.from_numpy(music_spectrum_np), 
                                        num_sources - peaks_tmp.shape[0], largest=True).indices.cpu().detach().numpy()
                peaks_tmp = np.concatenate((peaks_tmp, random_peaks))
            # Sort by amplitude
            sorted_peaks = peaks_tmp[np.argsort(music_spectrum_np[peaks_tmp])[::-1]]
            peaks[batch] = torch.from_numpy(sorted_peaks[0:num_sources]).to(self.device)
        
        # Soft peak finding (differentiable)
        soft_decision = torch.zeros(batch_size, num_sources, dtype=torch.float64, device=self.device)
        top_indxs = peaks.to(self.device)

        for source in range(num_sources):
            # Create cell around each peak
            cell_idx = (top_indxs[:, source][:, None]
                       - self.cell_size
                       + torch.arange(2 * self.cell_size + 1, dtype=torch.long, device=self.device))
            
            # Handle boundaries
            out_of_bounds_mask = (cell_idx < 0) | (cell_idx >= spectrum.shape[1])
            cell_idx[out_of_bounds_mask] = top_indxs[:, source].unsqueeze(1).expand_as(cell_idx)[out_of_bounds_mask]
            cell_idx = cell_idx.reshape(batch_size, -1, 1)
            
            # Extract spectrum values in cell
            metrix_thr = torch.gather(spectrum.unsqueeze(-1).expand(-1, -1, cell_idx.size(-1)), 1,
                                    cell_idx).requires_grad_(True)
            
            # Apply temperature-scaled softmax
            soft_max = torch.softmax(metrix_thr / self.temperature, dim=1)
            
            # Compute weighted average
            soft_decision[:, source] = torch.einsum("bms, bms -> bs", 
                                                  search_space[cell_idx.cpu()].to(self.device), 
                                                  soft_max).squeeze()

        return soft_decision
    
    def forward(self, x: torch.Tensor, number_of_sources: int = None):
        """
        Forward pass of DiffMUSIC.
        
        Args:
            x: Input samples [batch_size, N, T] or [N, T]
            number_of_sources: Number of sources (defaults to self.M)
            
        Returns:
            Estimated DOA angles, source estimation, eigen regularization
        """
        if number_of_sources is None:
            number_of_sources = self.M
            
        # Apply constraints (for monitoring during training)
        if self.training:
            self._apply_position_constraints()
            self._apply_gain_constraints()
        
        # Compute covariance matrix
        cov = self.sample_covariance(x)
        
        # Subspace separation
        _, noise_subspace, source_estimation, eigen_regularization = self.subspace_separation(cov, number_of_sources)
        
        # Compute inverse spectrum
        inverse_spectrum = self.get_inverse_spectrum(noise_subspace)
        
        # Compute MUSIC spectrum
        self.music_spectrum = 1 / (inverse_spectrum + 1e-10)
        
        # Differentiable peak finding
        params = self.soft_peak_finding_1d(self.music_spectrum, self.angles_dict, number_of_sources)
        
        return params, source_estimation, eigen_regularization
    
    def plot_spectrum(self, batch: int = 0, true_angles: torch.Tensor = None, save: bool = False):
        """Plot MUSIC spectrum"""
        if self.music_spectrum is None:
            print("No spectrum available. Run forward pass first.")
            return
            
        x = np.rad2deg(self.angles_dict.detach().cpu().numpy())
        y = self.music_spectrum[batch].detach().cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="DiffMUSIC Spectrum", linewidth=2)
        
        if true_angles is not None:
            true_angles_deg = np.rad2deg(true_angles.detach().cpu().numpy())
            for i, angle in enumerate(true_angles_deg):
                plt.axvline(angle, color='red', linestyle='--', alpha=0.7, 
                           label=f'True DOA {i+1}' if i == 0 else "")
        
        plt.xlabel('Angle [degrees]')
        plt.ylabel('Spectrum Power')
        plt.title('DiffMUSIC Spectrum')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save:
            plt.savefig('diffmusic_spectrum.pdf')
        plt.show()
    
    def plot_array_geometry(self, save: bool = False):
        """Plot current antenna array geometry"""
        positions, gains = self.get_constrained_parameters()
        positions_np = positions.detach().cpu().numpy()
        gains_np = torch.abs(gains).detach().cpu().numpy()
        
        plt.figure(figsize=(12, 4))
        
        # Plot antenna positions
        plt.scatter(positions_np, np.zeros_like(positions_np), 
                   c=gains_np, s=100, cmap='viridis', 
                   edgecolors='black', linewidth=1, marker='s')
        plt.colorbar(label='Antenna Gain Magnitude')
        
        # Add antenna numbers
        for i, pos in enumerate(positions_np):
            plt.annotate(f'{i}', (pos, 0.01), 
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom')
        
        plt.xlabel('Position [meters]')
        plt.ylabel('')
        plt.title('DiffMUSIC Antenna Array Geometry')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 0.1)
        
        # Add wavelength reference
        plt.axhline(0, color='black', linewidth=0.5)
        spacing_text = f'λ/2 = {self.params.wavelength/2:.3f}m'
        plt.text(0.02, 0.98, spacing_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            plt.savefig('diffmusic_array_geometry.pdf')
        plt.show()
    
    def get_antenna_info(self):
        """Get current antenna configuration"""
        positions, gains = self.get_constrained_parameters()
        return {
            'positions': positions.detach().cpu().numpy(),
            'gains': gains.detach().cpu().numpy(),
            'position_constraint': self.position_constraint,
            'gain_constraint': self.gain_constraint,
            'N': self.N,
            'wavelength': self.params.wavelength
        }


class DiffMUSICTrainer:
    """Training wrapper for DiffMUSIC integrated with your simulation framework"""
    
    def __init__(self, diffmusic_model: DiffMUSIC, training_params: dict):
        self.model = diffmusic_model
        self.training_params = training_params
        
        # Initialize optimizer
        if training_params["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=training_params["learning_rate"],
                weight_decay=training_params.get("weight_decay", 0)
            )
        elif training_params["optimizer"] == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=training_params["learning_rate"],
                weight_decay=training_params.get("weight_decay", 0)
            )
        
        # Initialize scheduler
        if training_params["scheduler"] == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10
            )
        elif training_params["scheduler"] == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=training_params.get("step_size", 50), 
                gamma=0.5
            )
    
    def train_step(self, samples_batch: torch.Tensor, targets_batch: torch.Tensor, num_sources: int = None):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions, _, eigen_reg = self.model(samples_batch, num_sources)
        
        # Compute RMSE loss
        loss = torch.sqrt(torch.mean((predictions - targets_batch) ** 2))
        
        # Add regularization terms
        reg_loss = 0.01 * eigen_reg  # Eigenvalue regularization
        
        # Add antenna position regularization (encourage smooth spacing)
        positions, _ = self.model.get_constrained_parameters()
        if len(positions) > 1:
            pos_diff = positions[1:] - positions[:-1]
            target_spacing = self.model.params.wavelength / 2
            spacing_reg = 0.001 * torch.mean((pos_diff - target_spacing) ** 2)
        else:
            spacing_reg = 0.0
        
        total_loss = loss + reg_loss + spacing_reg
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return total_loss.item(), loss.item(), reg_loss.item(), spacing_reg
    
    def validate(self, val_samples: torch.Tensor, val_targets: torch.Tensor, num_sources: int = None):
        """Validation step"""
        self.model.eval()
        
        with torch.no_grad():
            predictions, _, _ = self.model(val_samples, num_sources)
            loss = torch.sqrt(torch.mean((predictions - val_targets) ** 2))
        
        if hasattr(self.scheduler, 'step'):
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(loss)
            else:
                self.scheduler.step()
        
        return loss.item()


# Integration function for your simulation framework
def create_diffmusic_model(system_model_params: SystemModelParams, model_config: dict):
    """
    Create DiffMUSIC model for integration with your simulation framework.
    
    Args:
        system_model_params: Your SystemModelParams instance
        model_config: Model configuration dictionary
        
    Returns:
        DiffMUSIC model instance
    """
    model_params = model_config.get("model_params", {})
    
    diffmusic = DiffMUSIC(
        system_model_params=system_model_params,
        position_constraint=model_params.get("position_constraint", "ula"),
        gain_constraint=model_params.get("gain_constraint", "positive"),
        temperature=model_params.get("temperature", 0.1),
        cell_size_coeff=model_params.get("cell_size_coeff", 0.2)
    )
    
    return diffmusic


# Example usage function compatible with your framework
def run_diffmusic_simulation(system_model_params: SystemModelParams, 
                           training_params: dict,
                           samples: torch.Tensor,
                           true_angles: torch.Tensor):
    """
    Example function showing how to use DiffMUSIC with your simulation framework.
    
    Args:
        system_model_params: Your SystemModelParams instance
        training_params: Training parameters dictionary
        samples: Signal samples [batch_size, N, T] or [N, T]
        true_angles: True DOA angles [batch_size, M] or [M]
        
    Returns:
        Trained DiffMUSIC model and final loss
    """
    # Create model
    model_config = {
        "model_type": "diffMUSIC",
        "model_params": {
            "position_constraint": "ula",
            "gain_constraint": "positive",
            "temperature": 0.1
        }
    }
    
    diffmusic = create_diffmusic_model(system_model_params, model_config)
    trainer = DiffMUSICTrainer(diffmusic, training_params)
    
    # Move to device
    device = training_params["device"]
    diffmusic = diffmusic.to(device)
    samples = samples.to(device)
    true_angles = true_angles.to(device)
    
    # Training loop
    num_epochs = training_params["epochs"]
    batch_size = training_params["batch_size"]
    
    # Simple training (you can adapt this to your batch loading pattern)
    for epoch in range(num_epochs):
        total_loss, data_loss, reg_loss, spacing_reg = trainer.train_step(
            samples, true_angles, system_model_params.M
        )
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Loss = {total_loss:.6f}, Data Loss = {data_loss:.6f}")
    
    # Validation
    val_loss = trainer.validate(samples, true_angles, system_model_params.M)
    print(f"Final Validation Loss: {val_loss:.6f}")
    
    return diffmusic, val_loss