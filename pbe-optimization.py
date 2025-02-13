import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, LinearOperator
import multiprocessing as mp
from functools import partial
from typing import Tuple, Optional
import logging
import warnings

class RobustPBESolver:
    def __init__(self, 
                 grid_size: int, 
                 boundary_conditions: dict,
                 dielectric_constant: float,
                 ionic_strength: float,
                 temperature: float = 298.15):
        """
        Initialize PBE solver with validation and error checking
        
        Args:
            grid_size: Number of grid points in each dimension
            boundary_conditions: Dictionary specifying boundary conditions
            dielectric_constant: Dielectric constant of the medium
            ionic_strength: Ionic strength in mol/L
            temperature: Temperature in Kelvin
        
        Raises:
            ValueError: If input parameters are invalid
        """
        # Input validation
        self._validate_inputs(grid_size, boundary_conditions, 
                            dielectric_constant, ionic_strength, temperature)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Store validated parameters
        self.grid_size = grid_size
        self.boundary_conditions = boundary_conditions
        self.dielectric = dielectric_constant
        self.ionic_strength = ionic_strength
        self.temperature = temperature
        
        # Physical constants
        self.kB = 1.380649e-23  # Boltzmann constant
        self.e0 = 1.602176634e-19  # Elementary charge
        self.NA = 6.02214076e23  # Avogadro constant
        
        # Initialize solver state
        self.converged = False
        self.setup_grid()
        
    def _validate_inputs(self, grid_size, boundary_conditions, 
                        dielectric_constant, ionic_strength, temperature):
        """Validate all input parameters"""
        if grid_size <= 0 or not isinstance(grid_size, int):
            raise ValueError("Grid size must be a positive integer")
            
        if dielectric_constant <= 0:
            raise ValueError("Dielectric constant must be positive")
            
        if ionic_strength < 0:
            raise ValueError("Ionic strength cannot be negative")
            
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
            
        required_bc = {'left', 'right', 'top', 'bottom', 'front', 'back'}
        if not all(key in boundary_conditions for key in required_bc):
            raise ValueError(f"Missing boundary conditions. Required: {required_bc}")

    def setup_grid(self):
        """Initialize computational grid with error checking"""
        try:
            # Create adaptive mesh with memory checks
            memory_estimate = self.grid_size**3 * 8  # bytes for double precision
            available_memory = self._get_available_memory()
            
            if memory_estimate > 0.8 * available_memory:
                warnings.warn("Grid size may exceed available memory")
            
            self.dx = 1.0 / self.grid_size
            self.x = np.linspace(0, 1, self.grid_size)
            self.y = np.linspace(0, 1, self.grid_size)
            self.z = np.linspace(0, 1, self.grid_size)
            
        except MemoryError:
            raise MemoryError("Insufficient memory for grid initialization")
            
    def _get_available_memory(self) -> int:
        """Estimate available system memory"""
        try:
            import psutil
            return psutil.virtual_memory().available
        except ImportError:
            self.logger.warning("psutil not available, skipping memory check")
            return float('inf')

    def multigrid_preconditioner(self, residual: np.ndarray) -> np.ndarray:
        """
        Implement robust multigrid preconditioner with error checking
        
        Args:
            residual: Current residual vector
            
        Returns:
            Preconditioned residual
        """
        try:
            # Ensure residual is properly shaped
            residual = np.asarray(residual).reshape(self.grid_size, 
                                                  self.grid_size, 
                                                  self.grid_size)
            
            # Check for NaN/Inf values
            if not np.all(np.isfinite(residual)):
                raise ValueError("Non-finite values in residual")
            
            # Restriction operator with conservative interpolation
            restricted = self._restrict(residual)
            
            # Solve coarse grid problem
            coarse_solution = self._solve_coarse_grid(restricted)
            
            # Prolongation with monotonicity preservation
            prolonged = self._prolong(coarse_solution)
            
            return prolonged
            
        except Exception as e:
            self.logger.error(f"Multigrid preconditioner failed: {str(e)}")
            # Fall back to diagonal preconditioner
            return residual / self.build_matrix().diagonal()

    def solve(self, 
             max_iterations: int = 1000, 
             tolerance: float = 1e-6,
             checkpoint_freq: int = 100) -> Tuple[np.ndarray, dict]:
        """
        Main solver with robust error handling and convergence monitoring
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            checkpoint_freq: Frequency of solution checkpointing
            
        Returns:
            Tuple of (solution array, convergence info dictionary)
        """
        try:
            # Initialize solution and convergence tracking
            solution = np.zeros((self.grid_size, self.grid_size, self.grid_size))
            matrix = self.build_matrix()
            convergence_history = []
            
            # Set up preconditioner
            M = LinearOperator((matrix.shape[0], matrix.shape[1]),
                             matvec=self.multigrid_preconditioner)
            
            for iter in range(max_iterations):
                # Calculate residual with stability checks
                residual = self.calculate_residual(solution)
                residual_norm = np.linalg.norm(residual)
                convergence_history.append(residual_norm)
                
                # Check for divergence
                if not np.isfinite(residual_norm):
                    raise RuntimeError("Solution diverged")
                
                # Apply preconditioner with fallback options
                try:
                    preconditioned = M.matvec(residual)
                except Exception as e:
                    self.logger.warning(f"Preconditioner failed, using fallback: {str(e)}")
                    preconditioned = residual
                
                # Update solution
                solution = spsolve(matrix, preconditioned)
                
                # Checkpoint solution periodically
                if iter % checkpoint_freq == 0:
                    self._save_checkpoint(solution, iter)
                
                # Check convergence
                if residual_norm < tolerance:
                    self.converged = True
                    break
                    
            # Prepare return information
            convergence_info = {
                'iterations': iter + 1,
                'converged': self.converged,
                'final_residual': residual_norm,
                'convergence_history': convergence_history
            }
            
            return solution, convergence_info
            
        except Exception as e:
            self.logger.error(f"Solver failed: {str(e)}")
            raise

    def _save_checkpoint(self, solution: np.ndarray, iteration: int):
        """Save solution checkpoint for recovery"""
        try:
            np.save(f"pbe_checkpoint_{iteration}.npy", solution)
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {str(e)}")

    def validate_solution(self, solution: np.ndarray) -> bool:
        """
        Validate physical constraints of the solution
        
        Returns:
            bool: Whether solution meets physical constraints
        """
        try:
            # Check charge conservation
            total_charge = np.sum(solution * self.dx**3)
            if abs(total_charge - self.total_system_charge) > 1e-10:
                return False
            
            # Check boundary conditions
            if not self._check_boundary_conditions(solution):
                return False
            
            # Check for physical values
            if np.any(solution > self.max_potential) or np.any(solution < self.min_potential):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Solution validation failed: {str(e)}")
            return False