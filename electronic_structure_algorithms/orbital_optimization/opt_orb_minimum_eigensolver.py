from typing import Callable, Optional, Union, Dict, List
from abc import abstractmethod
from qiskit.primitives import BaseEstimator
from .partial_unitary_projection_optimizer import PartialUnitaryProjectionOptimizer
from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolverResult, MinimumEigensolver
from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolverResult
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
import torch
import copy
import numpy as np

from .base_opt_orb_solver import BaseOptOrbSolver, BaseOptOrbResult

class OptOrbMinimumEigensolver(BaseOptOrbSolver):

    def __init__(self,
        num_spin_orbitals: int,
        ground_state_solver: MinimumEigensolver,
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
        RDM_ops_batchsize: Optional[int] = None):
        
        """
        
        Args:
            num_spin_orbitals: The number of spin-orbitals to use for the active space.
            ground_state_solver: A :class:`MinimumEigensolver` used for the ground state solver subroutine.
            mapper: A QubitMapper to use for the RDM calculations.
            partial_unitary_optimizer: An instance of :class:`PartialUnitaryProjectionOptimizer` to use for
                the basis optimization subroutine.
            problem: An ElectronicStructureProblem from which molecule information such as one and two body integrals is obtained.
            integral_tensors: A tuple storing the one and two body integrals of the full orbital space. The first
                entry stores the one body integrals and the second entry stores the two body integrals in the Physicist's notation in
                the dense spin-orbital representation, represented as either :class:`torch.Tensor` or :class:`np.ndarray`.
            initial_partial_unitary: The initial guess for the orbital rotation matrix. If ``None``, then a permutation matrix
                selecting the spatial orbitals with the lowest energy will be generated.
            maxiter: The maximum number of outerloop iterations. (The number of times the wavefunction optimization is run.)
            stopping tolerance: The stopping tolerance used to determine if the algorithm should be stopped.
            spin_conserving: A boolean flag that indicates whether or not we are assuming that spin is conserved
                in the system being studied. Setting to ``True`` will skip the calculation of RDM element operators
                which do not conserve spin, automatically setting these elements to zero. This should only be
                set to ``True`` if it is known that the circuit used to construct the wavefunction in the eigensolver
                subroutine conserves the spin-magnetization quantum number (e.g. UCCSD).
            wavefunction_real: A boolean flag that indicates whether or not we are assuming that the wavefunction is real.
                This allows for the exploitation of symmetries
                in the RDM calculation to reduce runtime, but should only be set to ``True``
                if it is known that the circuit used in the eigensolver subroutine produces wavefunction coefficients
                which are real-valued.
            outer_loop_callback: A callback function that tracks the outerloop progress.
                It takes the outerloop iteration, latest eigensolver results, and latest outer loop results as arguments.
                This can be used to save the most recent results to a file after each outer loop iteration.
            partial_unitary_random_perturbation: A float representing the standard deviation of a normal distribution from which
                the matrix elements of a perturbation matrix will be sampled. This perturbation matrix will be added to the initial
                partial unitary (and re-orthonormalized) at the beginning of each orbital optimization run.
            RDM_ops_batchsize: The number of Pauli ops to store in an :class:`Estimator` at any given time before deleting
                the Estimator and replacing it with a blank copy. Increasing this number will lead to increased memory consumption. Setting
                this number to be too low will hinder runtime performance.

        """
       
        super().__init__(num_spin_orbitals=num_spin_orbitals,
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
                         RDM_ops_batchsize=RDM_ops_batchsize)

        self._ground_state_solver_list = [copy.deepcopy(ground_state_solver) for n in range(int(maxiter+1))]
        self._energy_convergence_list = []
        self._pauli_ops_expectation_values_dict = None

    @property
    def ground_state_solver_list(self) -> List[MinimumEigensolver]:
        """Returns the list of ground state solver instances."""
        return self._ground_state_solver_list

    @ground_state_solver_list.setter
    def ground_state_solver_list(self, instance_list: List[MinimumEigensolver]) -> None:
        """Sets the list of ground state solver instances."""
        self._ground_state_solver_list = instance_list

    @property
    def energy_convergence_list(self) -> List[float]:
        """Returns the list of outerloop iteration energy values."""
        return self._energy_convergence_list

    @energy_convergence_list.setter
    def energy_convergence_list(self, energy_list: List[float]) -> None:
        """Sets the list of outerloop iteration energy values."""
        self._energy_convergence_list = energy_list

    @property
    def pauli_ops_expectation_values_dict(self) -> Dict:
        """Returns the dictionary containing all of the expectation values
            of all the Pauli string operators necessary for calculating
            the RDMs with respect to a particular wavefunction."""
        return self._pauli_ops_expectation_values_dict

    @pauli_ops_expectation_values_dict.setter
    def pauli_ops_expectation_values_dict(self, some_dict: Dict) -> None:
        """Sets the dictionary containing all of the expectation values
            of all the Pauli string operators necessary for calculating
            the RDMs with respect to a particular wavefunction."""
        self._pauli_ops_expectation_values_dict = some_dict

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

    @abstractmethod            
    def parameter_update_rule(self, result: MinimumEigensolverResult, iteration: int):

        raise NotImplementedError("Minimum eigensolver needs to implement a way to update parameters after each orbital optimization.")

    @abstractmethod
    def return_RDM_circuit(self, result: MinimumEigensolverResult, iteration: int):

        NotImplementedError("Minimum eigensolver needs to implement a way to return the circuit used to calculate the one and two RDM.")

    def compute_minimum_energy(self) -> MinimumEigensolverResult:

        outer_loop_iteration = 0

        optorb_result = OptOrbMinimumEigensolverResult()
        self._hamiltonian = self.get_rotated_hamiltonian(self._current_partial_unitary)
        
        self._pauli_op_dict = self.construct_pauli_op_dict(mapper=self.mapper)

        while self.stopping_condition(outer_loop_iteration) == False:
            
            result = self._ground_state_solver_list[outer_loop_iteration].compute_minimum_eigenvalue(operator=self._hamiltonian)
            energy = np.real(result.eigenvalue)
            opt_params = result.optimal_parameters

            # update the optorb result to hold the most recent ground state energy.
            optorb_result.eigenvalue = energy

            # update the optorb result to hold the most recent ground state parameters.
            optorb_result.optimal_parameters = opt_params

            # update the optorb result to hold the most recent partial unitary basis transformation.
            optorb_result.optimal_partial_unitary = self._current_partial_unitary
            optorb_result.num_vqe_evaluations += 1
            optorb_result.optimal_circuit = result.optimal_circuit
            optorb_result.optimal_point = result.optimal_point
            optorb_result.optimal_value = result.optimal_value

            if self.outer_loop_callback is not None:
                self.outer_loop_callback(outer_loop_iteration, result, optorb_result)

            self._energy_convergence_list.append(energy)
            state = copy.deepcopy(result.optimal_circuit).assign_parameters(opt_params)
            
            if self.stopping_condition(outer_loop_iteration) == True:
                break

            string_op_tuple_list = [(key, self._pauli_op_dict[key]) for key in self._pauli_op_dict]
            
            results = []
            ops_counter = 1
            num_ops = len(string_op_tuple_list)
            for op_tuple in string_op_tuple_list:
                
                results.append(self.estimator_list[outer_loop_iteration].run([state], [op_tuple[1]]).result().values[0])

                if self.RDM_ops_batchsize is not None:
                    if ops_counter % self.RDM_ops_batchsize == 0:
                        self.estimator_list[outer_loop_iteration] = copy.deepcopy(self.estimator)
                        
                ops_counter += 1

            self._pauli_ops_expectation_values_dict = dict(zip([op_tuple[0] for op_tuple in string_op_tuple_list], results))
            

            self.estimator_list[outer_loop_iteration] = None

            oneRDM = self.get_one_RDM_tensor(mapper=self.mapper, expectval_dict=self._pauli_ops_expectation_values_dict)
            twoRDM = self.get_two_RDM_tensor(mapper=self.mapper, expectval_dict=self._pauli_ops_expectation_values_dict)

            if self.partial_unitary_random_perturbation is not None:

                initial_partial_unitary = self.orth(self._current_partial_unitary + torch.Tensor(np.random.normal(loc=0.0,
                                                scale=self.partial_unitary_random_perturbation, size=(self._current_partial_unitary.size()[0],
                                                self._current_partial_unitary.size()[1]))))
            else:

                initial_partial_unitary = self._current_partial_unitary

            oneRDM = oneRDM.to(self._partial_unitary_optimizer_list[outer_loop_iteration].device)
            twoRDM = twoRDM.to(self._partial_unitary_optimizer_list[outer_loop_iteration].device)
            self.one_body_integrals = self.one_body_integrals.to(self._partial_unitary_optimizer_list[outer_loop_iteration].device)
            self.two_body_integrals = self.two_body_integrals.to(self._partial_unitary_optimizer_list[outer_loop_iteration].device)
            self._current_partial_unitary = self._partial_unitary_optimizer_list[outer_loop_iteration].compute_optimal_rotation(fun=self.compute_rotated_energy,
                                                                                                                     oneRDM=oneRDM,
                                                                                                                     twoRDM=twoRDM,
                                                                                                                     one_body_integrals=self.one_body_integrals,
                                                                                                                     two_body_integrals=self.two_body_integrals,
                                                                                                                     initial_partial_unitary=initial_partial_unitary)[0]
            oneRDM = oneRDM.to('cpu')
            twoRDM = twoRDM.to('cpu')
            del oneRDM
            del twoRDM
            del string_op_tuple_list
            self.one_body_integrals = self.one_body_integrals.to('cpu')
            self.two_body_integrals = self.two_body_integrals.to('cpu')
            
            self._hamiltonian = self.get_rotated_hamiltonian(self._current_partial_unitary)
            outer_loop_iteration += 1

            self.parameter_update_rule(result, outer_loop_iteration)
            self._ground_state_solver_list[outer_loop_iteration].initial_point = result.optimal_point

            self._partial_unitary_optimizer_list[outer_loop_iteration - 1] = None
            self._ground_state_solver_list[outer_loop_iteration - 1] = None

        return optorb_result

class OptOrbMinimumEigensolverResult(BaseOptOrbResult, MinimumEigensolverResult):

    def __init__(self) -> None:
        super().__init__()


