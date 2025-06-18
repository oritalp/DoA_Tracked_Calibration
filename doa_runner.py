"""
DoA Algorithm Runner - Handles training and evaluation of DoA estimation algorithms
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
from datetime import datetime

from src.utils import sample_covariance


class AlgorithmFactory:
    """
    Factory class for creating DoA algorithms and training components
    Encapsulated within DoARunner to avoid circular imports and organize related functionality
    """
    
    def __init__(self, device):
        self.device = device
    
    def create_algorithm(self, model_type: str, system_model_params, **kwargs):
        """
        Create DoA estimation algorithm based on model type
        
        Args:
            model_type: Type of algorithm ("diffMUSIC", "MUSIC", etc.)
            system_model_params: System model parameters
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Algorithm instance
        """
        from src.diffmusic import DiffMUSIC
        
        if model_type.lower() == "diffmusic":
            algorithm = DiffMUSIC(
                system_model_params=system_model_params,
                N=system_model_params.N,
                model_order_estimation=kwargs.get('model_order_estimation', None),
                physical_array=kwargs.get('physical_array', None),
                physical_gains=kwargs.get('physical_gains', None)
            )
            algorithm.train()
        elif model_type.lower() == "music":
            # For now, use DiffMUSIC in eval mode for classical MUSIC
            algorithm = DiffMUSIC(
                system_model_params=system_model_params,
                N=system_model_params.N,
                model_order_estimation=kwargs.get('model_order_estimation', None),
                physical_array=kwargs.get('physical_array', None),
                physical_gains=kwargs.get('physical_gains', None)
            )
            algorithm.eval()  # Set to evaluation mode for classical MUSIC behavior
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return algorithm.to(self.device)
    
    def create_loss_function(self, loss_type: str, **kwargs):
        """
        Create loss function based on type
        
        Args:
            loss_type: Type of loss ("rmspe", "spectrum", "unsupervised")
            **kwargs: Additional loss-specific parameters
            
        Returns:
            Loss function instance
        """
        from src.diffmusic import DiffMUSICLoss
        
        return DiffMUSICLoss(loss_type=loss_type, **kwargs)
    
    def create_optimizer(self, algorithm, optimizer_type: str, learning_rate: float, **kwargs):
        """
        Create optimizer for algorithm parameters
        
        Args:
            algorithm: Algorithm instance with learnable parameters
            optimizer_type: Type of optimizer ("Adam", "SGD")
            learning_rate: Learning rate
            **kwargs: Additional optimizer parameters
            
        Returns:
            Optimizer instance
        """
        if optimizer_type.lower() == "adam":
            return optim.Adam(
                algorithm.parameters(), 
                lr=learning_rate,
                weight_decay=kwargs.get('weight_decay', 0)
            )
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(
                algorithm.parameters(), 
                lr=learning_rate,
                momentum=kwargs.get('momentum', 0),
                weight_decay=kwargs.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def create_scheduler(self, optimizer, scheduler_type: str, **kwargs):
        """
        Create learning rate scheduler
        
        Args:
            optimizer: Optimizer instance
            scheduler_type: Type of scheduler ("StepLR", "ReduceLROnPlateau")
            **kwargs: Additional scheduler parameters
            
        Returns:
            Scheduler instance or None
        """
        if scheduler_type is None:
            return None
        
        if scheduler_type.lower() == "steplr":
            return lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 50),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type.lower() == "reducelronplateau":
            return lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class DoARunner:
    """
    Main runner class for DoA estimation algorithms
    Handles both training and evaluation phases
    """
    
    def __init__(self, system_model_params, data_dict: Dict[str, Any]):
        """
        Initialize DoA Runner
        
        Args:
            system_model_params: SystemModelParams object containing all configuration
            data_dict: Dictionary containing:
                - 'measurements': Measurement data (N, T)
                - 'true_angles': True DoA angles (M,)
                - 'steering_matrix': Steering matrix (N, M)
                - 'noise': Noise data (N, T)
                - 'physical_array': Physical array positions (N,)
                - 'physical_antennas_gains': Physical antenna gains (N,)
        """
        self.system_params = system_model_params
        self.data_dict = data_dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Precompute covariance matrix once (reused across all trials)
        measurements = self.data_dict['measurements'].to(self.device).unsqueeze(0)
        self.cov_matrix = sample_covariance(measurements)
        
        # Create factory for algorithm and training components
        self.factory = AlgorithmFactory(self.device)
        
        # Create algorithm (may be recreated during window size optimization)
        self.algorithm = self.factory.create_algorithm(
            model_type=system_model_params.model_type,
            system_model_params=system_model_params
        )
        
        # Initialize training components if needed
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.results = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        if self._needs_training():
            self._setup_training()
    
    def _needs_training(self) -> bool:
        """Check if algorithm needs training"""
        return self.algorithm.training

    
    def _setup_training(self):
        """Setup training components (loss, optimizer, scheduler)"""
        # Determine loss type based on system parameters
        if self.system_params.loss_type is None:
            warnings.warn("Loss type not specified, defaulting to 'rmspe'.")
            self.system_params.loss_type = 'rmspe'
        loss_type = getattr(self.system_params, 'loss_type', 'rmspe')

        
        # Create loss function
        self.loss_fn = self.factory.create_loss_function(loss_type)
        
        # Create optimizer
        self.optimizer = self.factory.create_optimizer(
            algorithm=self.algorithm,
            optimizer_type=self.system_params.optimizer,
            learning_rate=self.system_params.learning_rate,
            weight_decay=getattr(self.system_params, 'weight_decay', 0)
        )
        
        # Create scheduler
        self.scheduler = self.factory.create_scheduler(
            optimizer=self.optimizer,
            scheduler_type=self.system_params.scheduler,
            step_size=getattr(self.system_params, 'step_size', 50),
            patience=getattr(self.system_params, 'patience', 10)
        )
    
    def _setup_wandb(self, window_size, trial_idx=None):
        """
        Setup wandb logging if enabled
        
        Args:
            window_size: Current window size being used
            trial_idx: Trial index for optimization mode (None for single mode)
        """
        if not getattr(self.system_params, 'use_wandb', False):
            return
        
        try:
            import wandb
            
            # Create project name with model type and timestamp
            dt_string = getattr(self.system_params, 'dt_string_for_save', 
                              datetime.now().strftime("%d_%m_%Y_%H_%M"))
            project_name = f"{self.system_params.model_type}_{dt_string}"
            
            # Create run name
            if trial_idx is not None:
                run_name = f"trial_{trial_idx+1}_ws_{window_size}_{self.system_params.loss_type}"
            else:
                run_name = f"ws_{window_size}_{self.system_params.loss_type}"
            
            # Initialize wandb
            wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "model_type": self.system_params.model_type,
                    "loss_type": self.system_params.loss_type,
                    "window_size": window_size,
                    "learning_rate": self.system_params.learning_rate,
                    "epochs": self.system_params.epochs,
                    "N": self.system_params.N,
                    "M": self.system_params.M,
                    "T": self.system_params.T,
                    "snr": self.system_params.snr,
                    "optimizer": self.system_params.optimizer,
                    "scheduler": self.system_params.scheduler,
                },
                reinit=True  # Allow multiple runs in same script
            )
            
        except ImportError:
            warnings.warn("wandb not installed. Install with 'pip install wandb' to enable logging.")
        except Exception as e:
            warnings.warn(f"Failed to initialize wandb: {e}")
    
    def _close_wandb(self):
        """Close wandb run if active"""
        if getattr(self.system_params, 'use_wandb', False):
            try:
                import wandb
                wandb.finish()
            except:
                pass
    
    def run(self) -> Dict[str, Any]:
        """
        Main run method - handles both training and evaluation
        Supports single window size or window size optimization
        
        Returns:
            Dictionary with results including estimated DoAs and metrics
        """
        # Normalize window_size to always be a list/array for unified handling
        window_size = getattr(self.system_params, 'softmax_window_size', 21)
        if not hasattr(window_size, '__len__'):
            window_sizes = [window_size]  # Single case
            optimization_mode = False
        else:
            window_sizes = window_size  # Multiple cases
            optimization_mode = len(window_sizes) > 1
        
        if optimization_mode:
            print(f"Starting window size optimization over {len(window_sizes)} configurations...")
        else:
            print("Running single configuration...")
            
        results = self._run_with_window_optimization(window_sizes, optimization_mode)
        
        self.results = results
        return results
    
    def _create_fresh_algorithm(self, window_size):
        """
        Create a new algorithm instance with fresh parameters
        
        Args:
            window_size: Window size (int, float, or relative float)
            
        Returns:
            Fresh algorithm instance with specified window size
        """
        algorithm = self.factory.create_algorithm(
            model_type=self.system_params.model_type,
            system_model_params=self.system_params
        )
        
        # Set the specific window size
        if isinstance(window_size, float) and 0 < window_size < 1:
            # Case 2: relative to grid size
            algorithm.window_size = int(window_size * len(algorithm.angles_grid))
        else:
            # Case 1: absolute size
            algorithm.window_size = int(window_size)
        
        return algorithm.to(self.device)
    
    def _run_with_window_optimization(self, window_sizes, optimization_mode=True) -> Dict[str, Any]:
        """
        Unified method for both single and multiple window size configurations
        
        Args:
            window_sizes: List of window sizes to try
            optimization_mode: Whether to print optimization-specific messages
            
        Returns:
            Results from best configuration (or single configuration) plus stats
        """
        import time
        
        best_rmspe = float('inf')
        best_results = None
        best_window_size = None
        
        total_start_time = time.time()
        trial_times = []
        
        for i, window_size in enumerate(window_sizes):
            trial_start_time = time.time()
            
            if optimization_mode:
                print(f"Trial {i+1}/{len(window_sizes)}: window_size={window_size}")
            else:
                print(f"Running with window_size={window_size}")
            
            # Create fresh algorithm instance with reset parameters
            self.algorithm = self._create_fresh_algorithm(window_size)
            
            # Setup fresh training components and wandb
            if self._needs_training():
                self._setup_training()
                self._setup_wandb(window_size, trial_idx=i if optimization_mode else None)
            
            # Run complete training + evaluation cycle
            results = {}
            if self._needs_training():
                train_results = self._run_training()
                results.update(train_results)
            
            eval_results = self._run_evaluation()
            results.update(eval_results)
            
            # Close wandb for this trial
            self._close_wandb()
            
            trial_end_time = time.time()
            trial_duration = trial_end_time - trial_start_time
            trial_times.append(trial_duration)
            
            # Check if this is the best configuration
            if results['rmspe'] < best_rmspe:
                best_rmspe = results['rmspe']
                best_results = results.copy()
                best_window_size = window_size
                status_msg = f"New best RMSPE: {best_rmspe:.6f}" if optimization_mode else f"RMSPE: {best_rmspe:.6f}"
                print(f"  {status_msg} (Time: {trial_duration:.2f}s)")
            else:
                print(f"  RMSPE: {results['rmspe']:.6f} (Time: {trial_duration:.2f}s)")
        
        total_time = np.sum(np.array(trial_times))
        
        if optimization_mode:
            average_duration = np.mean(trial_times)
            print(f"\nOptimization complete!")
            print(f"Best window_size: {best_window_size}, RMSPE: {best_rmspe:.6f}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average time per trial: {average_duration:.2f}s")
            
            # Add optimization stats
            best_results['optimal_window_size'] = best_window_size
            best_results['optimization_stats'] = {
                'total_time': total_time,
                'average_time_per_trial': average_duration,
                'trial_times': trial_times,
                'num_trials': len(window_sizes)
            }
        else:
            print(f"Completed in {total_time:.2f}s")
            best_results['execution_time'] = total_time
        
        return best_results
    
    def _run_training(self) -> Dict[str, Any]:
        """
        Run training phase using precomputed covariance matrix
        
        Returns:
            Training results
        """
        
        # Extract data
        true_angles = self.data_dict['true_angles'].to(self.device).unsqueeze(0)    # (1, M)
        M = len(self.data_dict['true_angles'])
        
        # Reset training history for fresh instance
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

        print(f"Training for {self.system_params.epochs} epochs...")
        
        for epoch in tqdm(range(self.system_params.epochs), desc="Training"):
            epoch_loss = self._train_epoch(self.cov_matrix, true_angles, M)
            
            # Store training history
            self.training_history['train_loss'].append(epoch_loss)
            if self.optimizer:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.training_history['learning_rates'].append(current_lr)
            
            # Log to wandb if enabled
            if getattr(self.system_params, 'use_wandb', False):
                try:
                    import wandb
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": epoch_loss,
                        "learning_rate": current_lr if self.optimizer else 0
                    })
                except:
                    pass
            
            # Step scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(epoch_loss)
                else:
                    self.scheduler.step()
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.system_params.epochs}, Loss: {epoch_loss:.6f}")
        
        return {
            'training_history': self.training_history,
            'final_train_loss': epoch_loss
        }
    
    def _train_epoch(self, cov_matrix: torch.Tensor, true_angles: torch.Tensor, M: int) -> float:
        """
        Train for one epoch
        
        Args:
            cov_matrix: Covariance matrix (1, N, N)
            true_angles: True angles (1, M)
            M: Number of sources
            
        Returns:
            Epoch loss
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        algorithm_result = self.algorithm(cov_matrix, M)
        if self.system_params.model_type.lower() == "diffmusic":
            estimated_angles, peaks_masks, _ , _ = algorithm_result
        if peaks_masks is None:
            warnings.warn("No peaks masks returned from algorithm, something is weird and you need to check that.")
        
        # Compute loss based on loss type
        loss_kwargs = {
            'predictions': estimated_angles,
            'targets': true_angles
        }
        
        # Add additional arguments for spectrum-based losses
        if hasattr(self.loss_fn, 'loss_type') and self.loss_fn.loss_type in ['spectrum', 'unsupervised']:
            loss_kwargs['spectrum'] = self.algorithm.music_spectrum
            loss_kwargs['angles_grid'] = self.algorithm.angles_grid
            
            if self.loss_fn.loss_type == 'unsupervised':

                loss_kwargs['peak_masks'] = peaks_masks
        
        # Compute loss
        loss = self.loss_fn(**loss_kwargs)
        
        # Handle different loss return types
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _run_evaluation(self) -> Dict[str, Any]:
        """
        Run evaluation phase using precomputed covariance matrix
        
        Returns:
            Evaluation results
        """
        
        with torch.no_grad():
            # Extract data
            true_angles = self.data_dict['true_angles'].to(self.device).unsqueeze(0)    # (1, M)
            M = len(self.data_dict['true_angles'])
            
            # Forward pass with precomputed covariance
            algorithm_result = self.algorithm(self.cov_matrix, M)
            if "music" in self.system_params.model_type.lower(): # Handle MUSIC and DiffMUSIC
                estimated_angles, _, _, _ = algorithm_result
            
            # Remove batch dimension for output
            estimated_angles = estimated_angles.squeeze(0)  # (M,)
            true_angles = true_angles.squeeze(0)  # (M,)

            learned_antenna_positions, learned_coplex_gains = self.algorithm.get_array_learnable_parameters(learnable=False)

            
            # Compute RMSPE for evaluation
            from src.metrics import RMSPELoss
            rmspe_fn = RMSPELoss()
            rmspe = rmspe_fn(estimated_angles.unsqueeze(0), true_angles.unsqueeze(0))

                   
        
        return {
            'estimated_angles': np.sort(estimated_angles.cpu().numpy()),
            'true_angles': true_angles.cpu().numpy(),
            'rmspe': rmspe.item(),
            "learned_antenna_positions": learned_antenna_positions,
            "learned_coplex_gains": learned_coplex_gains,
            'music_spectrum': self.algorithm.music_spectrum.cpu().numpy() if self.algorithm.music_spectrum is not None else None,
            'angles_grid': self.algorithm.angles_grid.cpu().numpy() if hasattr(self.algorithm, 'angles_grid') else None
        }
    
    def save_model(self, path: Path):
        """Save trained model"""
        if self._needs_training():
            torch.save({
                'model_state_dict': self.algorithm.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_history': self.training_history,
                'system_params': self.system_params
            }, path)
    
    def load_model(self, path: Path):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.algorithm.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']

    def plot_graphs(self, path: Path):
        """
        Plot specified graphs by case
        """
        
        if self.results is None:
            raise ValueError("No results available. Run the algorithm first.")
        true_angles = torch.from_numpy(self.results['true_angles']).to(self.device)

        if "music" in self.system_params.model_type.lower() and self.system_params.plot_results:
                    if self.algorithm.music_spectrum is None:
                        raise ValueError("Music spectrum is None the algorithm hasn't runned yet.")
                    self.algorithm.plot_spectrum(
                        true_angles,
                        path
                    )
        return None