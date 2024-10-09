from typing import Callable, Optional, Union
from qiskit.primitives import BaseEstimator
from .partial_unitary_projection_optimizer import PartialUnitaryProjectionOptimizer
from qiskit_algorithms.eigensolvers import Eigensolver
from qiskit_algorithms.exceptions import AlgorithmError
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
import torch
import numpy as np

from .opt_orb_eigensolver import OptOrbEigensolver, OptOrbEigensolverResult

class OptOrbSSVQE(OptOrbEigensolver):

    def __init__(self,
        num_spin_orbitals: int,
        excited_states_solver: Eigensolver,
        mapper: QubitMapper,
        estimator: BaseEstimator,
        partial_unitary_optimizer: PartialUnitaryProjectionOptimizer,
        problem: ElectronicStructureProblem = None,
        integral_tensors: Optional[Union[tuple[torch.Tensor, torch.Tensor], tuple[np.ndarray, np.ndarray]]] = None,
        initial_partial_unitary: Optional[Union[torch.Tensor, np.ndarray]] = None,
        maxiter: int = 10,
        stopping_tolerance: float = 10**-5,
        spin_conserving: bool = False,
        wavefuntion_real: bool = False,
        outer_loop_callback: Optional[Callable] = None,
        partial_unitary_random_perturbation: Optional[float] = None,
        eigensolver_random_perturbation: Optional[float] = None,
        RDM_ops_batchsize: Optional[int] = None,
        weight_vector: Optional[Union[list, np.ndarray]] = None):

        """
        
        Args:
            num_spin_orbitals: The number of spin-orbitals to use for the active space.
            excited_states_solver: An instance of VQE to use for the wavefunction optimization.
            mapper: A QubitMapper to use for the RDM calculations.
            partial_unitary_optimizer: An instance of PartialUnitaryProjectionOptimizer to use for the basis optimization.
            problem: The ElectronicStructureProblem from which molecule information such as one and two body integrals is obtained.
            integral_tensors: A tuple storing the one and two body integrals of the full orbital space. The first
                entry stores the one body integrals and the second entry stores the two body integrals in the Physicist's notation in
                the dense spin-orbital representation, represented as either :class:`torch.Tensor` or :class:`np.ndarray`.
            initial_partial_unitary: The initial guess for the orbital rotation matrix. If ``None``, then a permutation matrix
                selecting the spatial orbitals with the lowest energy will be generated.
            maxiter: The maximum number of outerloop iterations. (The number of times the wavefunction optimization is run.)
            stopping tolerance: The stopping tolerance used to determine if the algorithm should be stopped.
            spin_conserving: A boolean flag that indicates whether or not we are assuming that spin is conserved
                in the system being studied.  Setting to True will skip the calculation of RDM element operators
                which do not conserve spin, automatically setting these elements to zero.
            wavefunction_real: A boolean flag that indicates whether or not we are assuming that the wavefunction is real.
            outer_loop_callback: A callback function that tracks the outerloop progress.
                It takes the outerloop iteration, latest VQE results, and latest outer loop results as arguments.
                This can be used to save the most recent results to a file after each outer loop iteration.
            partial_unitary_random_perturbation: A float representing the standard deviation of a normal distribution from which
                the matrix elements of a perturbation matrix will be sampled. This perturbation matrix will be added to the initial
                partial unitary (and re-orthonormalized) at the beginning of each orbital optimization run.
            eigensolver_random_perturbation: A float representation the standard deviation of a normal distribution from which
                the elements of a set of perturbation parameters will be sampled. These perturbation elements will be added
                to the initial parameters of the eigensolver at each outer loop iteration.
            RDM_ops_batchsize: The number of Pauli ops to store in an :class:`Estimator` at any given time before deleting
                the Estimator and replacing it with a blank copy. Increasing this number will lead to increased memory consumption. Setting
                this number to be too low will hinder runtime performance.
            weight_vector: A list or 1-D array of positive-real valued weights for the weighted sum orbital optimization
                objective function whose length is equal to the number of excited states.
        """

        super().__init__(num_spin_orbitals=num_spin_orbitals,
                         excited_states_solver=excited_states_solver,
                         mapper=mapper,
                         estimator=estimator,
                         partial_unitary_optimizer=partial_unitary_optimizer,
                         problem=problem,
                         integral_tensors=integral_tensors,
                         initial_partial_unitary=initial_partial_unitary,
                         maxiter=maxiter,
                         stopping_tolerance=stopping_tolerance,
                         spin_conserving=spin_conserving,
                         wavefuntion_real=wavefuntion_real,
                         outer_loop_callback=outer_loop_callback,
                         partial_unitary_random_perturbation=partial_unitary_random_perturbation,
                         RDM_ops_batchsize=RDM_ops_batchsize,
                         weight_vector=weight_vector)

        if excited_states_solver.__class__.__name__ != 'SSVQE':

            raise AlgorithmError(f"The ground state solver needs to be of type SSVQE, not {excited_states_solver.__class__.__name__}")

        self.eigensolver_random_perturbation = eigensolver_random_perturbation

    def parameter_update_rule(self, result: OptOrbEigensolverResult,
                                    iteration: int):
        
        if self.eigensolver_random_perturbation == None or self.eigensolver_random_perturbation == 0.0:
            self._excited_states_solver_list[iteration].initial_point = result.optimal_point
        else:
            print('Perturb SSVQE parameters!')
            self._excited_states_solver_list[iteration].initial_point = result.optimal_point + np.random.normal(loc=0.0,
                                                scale=self.eigensolver_random_perturbation, size=result.optimal_point.size)

class OptOrbSSVQEResult(OptOrbEigensolverResult):

    def __init__(self) -> None:
        super().__init__()

