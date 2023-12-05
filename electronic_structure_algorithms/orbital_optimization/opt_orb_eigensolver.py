from typing import Callable, Optional, Union, Dict, List
from qiskit.primitives import BaseEstimator
from qiskit_algorithms.eigensolvers import Eigensolver, EigensolverResult
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
import torch
import copy
import numpy as np
from electronic_structure_algorithms.excited_states_eigensolvers import MCVQE, SSVQE

from .base_opt_orb_solver import BaseOptOrbSolver, BaseOptOrbResult
from .partial_unitary_projection_optimizer import PartialUnitaryProjectionOptimizer

class OptOrbEigensolver(BaseOptOrbSolver):

    def __init__(self,
        problem: Optional[ElectronicStructureProblem],
        integral_tensors: Optional[Union[tuple[torch.Tensor, torch.Tensor], tuple[np.ndarray, np.ndarray]]],
        num_spin_orbitals: int,
        excited_states_solver: Eigensolver,
        mapper: QubitMapper,
        estimator: BaseEstimator,
        partial_unitary_optimizer: PartialUnitaryProjectionOptimizer,
        initial_partial_unitary: Optional[Union[torch.Tensor, np.ndarray]] = None,
        maxiter: int = 10,
        stopping_tolerance: float = 10**-5,
        spin_conserving: bool = False,
        wavefuntion_real: bool = False,
        outer_loop_callback: Optional[Callable] = None,
        partial_unitary_random_perturbation: Optional[float] = None,
        RDM_ops_batchsize: Optional[int] = None,
        weight_vector: Optional[Union[list, np.ndarray]] = None):
        
        """
        
        Args:
            problem: The ElectronicStructureProblem from which molecule information such as one and two body integrals is obtained.
            integral_tensors: A tuple storing the one and two body integrals of the full orbital space. The first
                entry stores the one body integrals and the second entry stores the two body integrals in the Physicist's notation in
                the dense spin-orbital representation, represented as either :class:`torch.Tensor` or :class:`np.ndarray`.
            num_spin_orbitals: The number of spin-orbitals to use for the active space.
            excited_states_solver: An instance of :class:`Eigensolver` for the excited states solver subroutine.
            mapper: A QubitMapper to use for the RDM calculations.
            partial_unitary_optimizer: An instance of PartialUnitaryProjectionOptimizer to use for the basis optimization.
            initial_partial_unitary: The initial guess for the orbital rotation matrix. If ``None``, then a permutation matrix
                selecting the spatial orbitals with the lowest energy will be generated.
            maxiter: The maximum number of outerloop iterations. (The number of times the wavefunction optimization is run.)
            stopping tolerance: The stopping tolerance used to determine if the algorithm should be stopped.
            spin_conserving: A boolean flag that indicates whether or not we are assuming that spin is conserved
                in the system being studied.  Setting to True will skip the calculation of RDM element operators
                which do not conserve spin, automatically setting these elements to zero.
            wavefunction_real: A boolean flag that indicates whether or not we are assuming that the wavefunction is real.
            outer_loop_callback: A callback function that tracks the outerloop progress.
                It takes the outerloop iteration, latest eigensolver results, and latest outer loop results as arguments.
                This can be used to save the most recent results to a file after each outer loop iteration.
            partial_unitary_random_perturbation: A float representing the standard deviation of a normal distribution from which
                the matrix elements of a perturbation matrix will be sampled. This perturbation matrix will be added to the initial
                partial unitary (and re-orthonormalized) at the beginning of each orbital optimization run.
            RDM_ops_batchsize: The number of Pauli ops to store in an :class:`Estimator` at any given time before deleting
                the Estimator and replacing it with a blank copy. Increasing this number will lead to increased memory consumption. Setting
                this number to be too low will hinder runtime performance.
            weight_vector: A list or 1-D array of positive-real valued weights for the weighted sum orbital optimization
                objective function whose length is equal to the number of excited states.

        """
        super().__init__(problem=problem,
                         integral_tensors=integral_tensors,
                         num_spin_orbitals=num_spin_orbitals,
                         mapper=mapper,
                         estimator=estimator,
                         partial_unitary_optimizer=partial_unitary_optimizer,
                         initial_partial_unitary=initial_partial_unitary,
                         maxiter=maxiter,
                         stopping_tolerance=stopping_tolerance,
                         spin_conserving=spin_conserving,
                         wavefuntion_real=wavefuntion_real,
                         outer_loop_callback=outer_loop_callback,
                         partial_unitary_random_perturbation=partial_unitary_random_perturbation,
                         RDM_ops_batchsize=RDM_ops_batchsize)
        
        # generate copies of the eigensolver instance to use for every outerloop iteration.
        self._excited_states_solver_list = [copy.deepcopy(excited_states_solver) for n in range(int(maxiter+1))]

        self.num_states = excited_states_solver.k
        if weight_vector is not None:
            self.weight_vector = weight_vector
        elif hasattr(excited_states_solver, 'weight_vector'):
            self.weight_vector = excited_states_solver.weight_vector
        else:
            self.weight_vector = [self.num_states - n for n in range(self.num_states)]
        self._energy_convergence_list = []
        self._pauli_ops_expectation_values_dict_list = None

    @property
    def energy_convergence_list(self) -> List[float]:
        """Returns the list of outerloop iteration energy values."""
        return self._energy_convergence_list

    @energy_convergence_list.setter
    def energy_convergence_list(self, energy_list: List[float]) -> None:
        """Sets the list of outerloop iteration energy values."""
        self._energy_convergence_list = energy_list

    @property
    def pauli_ops_expectation_values_dict_list(self) -> list[Dict]:
        """Returns the dictionary containing all of the expectation values
            of all the Pauli string operators necessary for calculating
            the RDMs with respect to a particular wavefunction."""
        return self._pauli_ops_expectation_values_dict

    @pauli_ops_expectation_values_dict_list.setter
    def pauli_ops_expectation_values_dict_list(self, some_dict_list: list[Dict]) -> None:
        """Sets the dictionary containing all of the expectation values
            of all the Pauli string operators necessary for calculating
            the RDMs with respect to a particular wavefunction."""
        self._pauli_ops_expectation_values_dict_list = some_dict_list

    def stopping_condition(self, iteration) -> bool:

        """Evaluates whether or not the stopping condition is True.
        Returns True if the algorithm should be stopped, otherwise returns False.
        """

        if len(self._energy_convergence_list) >= 2:
            if iteration == self.maxiter or np.abs(self._energy_convergence_list[-1] - self._energy_convergence_list[-2]) < self.stopping_tolerance:
                return True
            else:
                return False
        
        else:
            return False

    def compute_rotated_weighted_energy_sum(self,
                                            partial_unitary: torch.Tensor,
                                            oneRDM: list[torch.Tensor],
                                            twoRDM: list[torch.Tensor],
                                            one_body_integrals,
                                            two_body_integrals):

        weighted_energy_sum = 0
        state_index = 0

        for one_RDM, two_RDM in zip(oneRDM, twoRDM):

            weighted_energy_sum += torch.tensor(self.weight_vector[state_index],
                                                dtype=torch.float64)*self.compute_rotated_energy(partial_unitary=partial_unitary,
                                                oneRDM=one_RDM,
                                                twoRDM=two_RDM,
                                                one_body_integrals=one_body_integrals,
                                                two_body_integrals=two_body_integrals)
            state_index += 1
        
        return weighted_energy_sum

    def compute_energies(self) -> EigensolverResult:
        
        self.outer_loop_iteration = 0

        optorb_result = OptOrbEigensolverResult()
        self._hamiltonian = self.get_rotated_hamiltonian(self._current_partial_unitary)
        
        self._pauli_op_dict = self.construct_pauli_op_dict(mapper=self.mapper)

        while self.stopping_condition(self.outer_loop_iteration) == False:
            
            result = self._excited_states_solver_list[self.outer_loop_iteration].compute_eigenvalues(operator=self._hamiltonian)
            energies = np.real(result.eigenvalues)
            opt_params = result.optimal_parameters

            # update the optorb result to hold the most recent VQE value.
            optorb_result.eigenvalues = energies

            # update the optorb result to hold the most recent VQE parameters.
            optorb_result.optimal_parameters = opt_params

            if isinstance(opt_params, dict):
                opt_params = [copy.deepcopy(opt_params)] * self.num_states

            # update the optorb result to hold the most recent partial unitary basis transformation.
            optorb_result.optimal_partial_unitary = self._current_partial_unitary
            optorb_result.num_vqe_evaluations += 1

            if self.outer_loop_callback is not None:
                self.outer_loop_callback(self.outer_loop_iteration, result, optorb_result)

            self._energy_convergence_list.append(np.dot(self.weight_vector, energies))
            if isinstance(self._excited_states_solver_list[self.outer_loop_iteration], SSVQE) or isinstance(self._excited_states_solver_list[self.outer_loop_iteration], MCVQE):
                states = [self._excited_states_solver_list[self.outer_loop_iteration].initial_states[n].compose(self._excited_states_solver_list[self.outer_loop_iteration].ansatz).bind_parameters(opt_params[n]) for n in range(self.num_states)]
            else:
                states = [self._excited_states_solver_list[self.outer_loop_iteration].ansatz[n].bind_parameters(opt_params[n]) for n in range(self.num_states)]
            
            if self.stopping_condition(self.outer_loop_iteration) == True:
                break

            string_op_tuple_list = [(key, self._pauli_op_dict[key]) for key in self._pauli_op_dict]
            
            results = [[] for n in range(self.num_states)]
            for n in range(self.num_states):
                ops_counter = 1
                num_ops = len(string_op_tuple_list)
                for op_tuple in string_op_tuple_list:
                    
                    results[n].append(self.estimator_list[self.outer_loop_iteration].run([states[n]], [op_tuple[1]]).result().values[0])

                    if self.RDM_ops_batchsize is not None:
                        if ops_counter % self.RDM_ops_batchsize == 0:
                            self.estimator_list[self.outer_loop_iteration] = copy.deepcopy(self.estimator)
                        
                    ops_counter += 1
            
            self._pauli_ops_expectation_values_dict_list = [dict(zip([op_tuple[0] for op_tuple in string_op_tuple_list], results[n])) for n in range(self.num_states)]

            self.estimator_list[self.outer_loop_iteration] = None

            oneRDM_list = [self.get_one_RDM_tensor(mapper=self.mapper, expectval_dict=self._pauli_ops_expectation_values_dict_list[n]) for n in range(self.num_states)]
            twoRDM_list = [self.get_two_RDM_tensor(mapper=self.mapper, expectval_dict=self._pauli_ops_expectation_values_dict_list[n]) for n in range(self.num_states)]

            if self.partial_unitary_random_perturbation is not None:

                initial_partial_unitary = self.orth(self._current_partial_unitary + torch.Tensor(np.random.normal(loc=0.0,
                                                scale=self.partial_unitary_random_perturbation, size=(self._current_partial_unitary.size()[0],
                                                self._current_partial_unitary.size()[1]))))
            else:

                initial_partial_unitary = self._current_partial_unitary

            oneRDM_list = [oneRDM.to(self._partial_unitary_optimizer_list[self.outer_loop_iteration].device) for oneRDM in oneRDM_list]
            twoRDM_list = [twoRDM.to(self._partial_unitary_optimizer_list[self.outer_loop_iteration].device) for twoRDM in twoRDM_list]
            self.one_body_integrals = self.one_body_integrals.to(self._partial_unitary_optimizer_list[self.outer_loop_iteration].device)
            self.two_body_integrals = self.two_body_integrals.to(self._partial_unitary_optimizer_list[self.outer_loop_iteration].device)
            self._current_partial_unitary = self._partial_unitary_optimizer_list[self.outer_loop_iteration].compute_optimal_rotation(fun=self.compute_rotated_weighted_energy_sum,
                                                                                                                     oneRDM=oneRDM_list,
                                                                                                                     twoRDM=twoRDM_list,
                                                                                                                     one_body_integrals=self.one_body_integrals,
                                                                                                                     two_body_integrals=self.two_body_integrals,
                                                                                                                     initial_partial_unitary=initial_partial_unitary)[0]
            oneRDM_list = [oneRDM.to('cpu') for oneRDM in oneRDM_list]
            twoRDM_list = [twoRDM.to('cpu') for twoRDM in twoRDM_list]
            del oneRDM_list
            del twoRDM_list
            del string_op_tuple_list
            self.one_body_integrals = self.one_body_integrals.to('cpu')
            self.two_body_integrals = self.two_body_integrals.to('cpu')
            
            self._hamiltonian = self.get_rotated_hamiltonian(self._current_partial_unitary)
            self.outer_loop_iteration += 1

            self.parameter_update_rule(result, self.outer_loop_iteration)

            self._partial_unitary_optimizer_list[self.outer_loop_iteration - 1] = None
            self._excited_states_solver_list[self.outer_loop_iteration - 1] = None

        return optorb_result

class OptOrbEigensolverResult(BaseOptOrbResult, EigensolverResult):

    def __init__(self) -> None:
        super().__init__()
        self._num_vqe_evaluations = 0
        self._optimal_partial_unitary = None
    
    @property
    def num_vqe_evaluations(self) -> int:
        """Returns the number of times the eigensolver was run."""
        return self._num_vqe_evaluations

    @num_vqe_evaluations.setter
    def num_vqe_evaluations(self, some_int: int) -> None:
        """Sets the number of times the eigensolver was run."""
        self._num_vqe_evaluations = some_int

    @property
    def optimal_partial_unitary(self) -> torch.Tensor:
        """Returns the optimal partial unitary basis transformation."""
        return self._optimal_partial_unitary

    @optimal_partial_unitary.setter
    def optimal_partial_unitary(self, some_tensor: torch.Tensor) -> None:
        """Sets the optimal partial unitary basis transformation."""
        self._optimal_partial_unitary = some_tensor


