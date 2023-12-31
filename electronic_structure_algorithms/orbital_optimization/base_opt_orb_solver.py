from typing import Callable, Optional, Union, Dict, List
from qiskit.primitives import BaseEstimator
from .partial_unitary_projection_optimizer import PartialUnitaryProjectionOptimizer
from qiskit_algorithms.variational_algorithm import VariationalResult
from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.operators.tensor_ordering import to_physicist_ordering

import torch
import copy
import numpy as np

class BaseOptOrbSolver():

    def __init__(self,
        num_spin_orbitals: int,
        mapper: QubitMapper,
        estimator: BaseEstimator,
        partial_unitary_optimizer: PartialUnitaryProjectionOptimizer,
        problem: Optional[ElectronicStructureProblem] = None,
        integral_tensors: Optional[Union[tuple[torch.Tensor, torch.Tensor], tuple[np.ndarray, np.ndarray]]] = None,
        initial_partial_unitary: Optional[Union[torch.Tensor, np.ndarray]] = None,
        maxiter: int = 10,
        stopping_tolerance: float = 10**-5,
        spin_conserving: bool = False,
        wavefuntion_real: bool = False,
        outer_loop_callback: Optional[Callable] = None,
        partial_unitary_random_perturbation: Optional[float] = None,
        RDM_ops_batchsize: Optional[int] = 100):
        
        """
        
        Args:
            num_spin_orbitals: The number of spin-orbitals to use for the active space.
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
            stopping_tolerance: The stopping tolerance used to determine if the algorithm should be stopped.
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

        self.mapper = mapper

        # generate copies of the PartialUnitaryProjectionOptimizer instance to use for every outerloop iteration.
        self._partial_unitary_optimizer_list = [copy.deepcopy(partial_unitary_optimizer) for n in range(int(maxiter+1))]

        if integral_tensors is not None:

            if isinstance(integral_tensors[0], torch.Tensor) and isinstance(integral_tensors[1], torch.Tensor):

                self.one_body_integrals, self.two_body_integrals = integral_tensors

            elif isinstance(integral_tensors[0], np.ndarray) and isinstance(integral_tensors[1], np.ndarray):

                self.one_body_integrals, self.two_body_integrals = torch.from_numpy(integral_tensors[0]), torch.from_numpy(integral_tensors[1])
        
        elif problem is not None:

            self.one_body_integrals = torch.from_numpy(np.asarray(problem.hamiltonian.electronic_integrals.second_q_coeffs()["+-"].to_dense()))
            self.two_body_integrals = torch.from_numpy(np.asarray(-1*to_physicist_ordering(problem.hamiltonian.electronic_integrals.second_q_coeffs()["++--"].to_dense())))
            del problem
        
        if initial_partial_unitary is None:
            
            num_original_spin_orbitals = self.one_body_integrals.size()[0]
            num_original_molecular_orbitals = int(num_original_spin_orbitals/2)
            num_molecular_orbitals = int(num_spin_orbitals/2)
            
            initial_partial_unitary_guess = torch.zeros(size=(num_original_molecular_orbitals, num_molecular_orbitals), dtype=torch.float64)
            for n in range(int(num_molecular_orbitals)):
                initial_partial_unitary_guess[n,n] = 1.0

            self.initial_partial_unitary = initial_partial_unitary_guess

        else:

            if isinstance(initial_partial_unitary, torch.Tensor):

                self.initial_partial_unitary = initial_partial_unitary

            elif isinstance(initial_partial_unitary, np.ndarray):

                self.initial_partial_unitary = torch.from_numpy(initial_partial_unitary)
        
        self.estimator = estimator
        self.estimator_list = [copy.deepcopy(estimator) for n in range(maxiter)]
        self.maxiter = maxiter
        self.spin_conserving = spin_conserving
        self.wavefunction_real = wavefuntion_real
        self.outer_loop_callback = outer_loop_callback

        self._hamiltonian = None
        self.stopping_tolerance = stopping_tolerance
        self._current_partial_unitary = self.initial_partial_unitary
        self._pauli_op_dict = None
        self.num_spin_orbitals = num_spin_orbitals
        self.partial_unitary_random_perturbation = partial_unitary_random_perturbation
        self.RDM_ops_batchsize = RDM_ops_batchsize
    
    @property
    def partial_unitary_optimizer_list(self) -> List[PartialUnitaryProjectionOptimizer]:
        """Returns the list of partial unitary optimizers used for each outer loop iteration."""
        return self._partial_unitary_optimizer_list

    @partial_unitary_optimizer_list.setter
    def partial_unitary_optimizer_list(self, optimizer_list: List[PartialUnitaryProjectionOptimizer]) -> None:
        """Sets the list of partial unitary optimizers used for each outer loop iteration."""
        self._partial_unitary_optimizer_list = optimizer_list

    @property
    def current_partial_unitary(self) -> torch.Tensor:
        """Returns the current basis set rotation partial unitary matrix."""
        return self._current_partial_unitary

    @current_partial_unitary.setter
    def current_partial_unitary(self, unitary: torch.Tensor) -> None:
        """Sets the current basis set rotation partial unitary matrix."""
        self._current_partial_unitary = unitary

    @property
    def hamiltonian(self) -> SparsePauliOp:
        """Returns the Hamiltonian in the current basis."""
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, op: SparsePauliOp) -> None:
        """Sets the Hamiltonian in the current basis."""
        self._hamiltonian = op

    @property
    def pauli_op_dict(self) -> Dict:
        """Returns the dictonary containing all of the Pauli string operators
            necessary for calculating the RDMs."""
        return self._pauli_op_dict

    @pauli_op_dict.setter
    def pauli_op_dict(self, some_dict: Dict) -> None:
        """Sets the dictionary containing all of the Pauli string operators
            necessary for calculating the RDMs."""
        self._pauli_op_dict = some_dict

    def is_2body_op_spin_conserving(self, P,Q,R,S) -> bool:

        """Determines whether or not the two body fermionic excitation operator
            involved in the 2-RDM element indexed by (P,Q,R,S) conserves spin or not.
        
        Args:
        
            P,Q,R,S: the RDM element index.

        Returns:
            True if fermionic operator conserves spin, False if it does not conserve spin.
            
        """

        N = self.num_spin_orbitals

        spin_change = 0
        if 0 <= P <= N/2 - 1:
            spin_change += 1
        else:
            spin_change -= 1

        if 0 <= Q <= N/2 - 1:
            spin_change += 1
        else:
            spin_change -= 1

        if 0 <= R <= N/2 - 1:
            spin_change -= 1
        else:
            spin_change += 1

        if 0 <= S <= N/2 - 1:
            spin_change -= 1
        else:
            spin_change += 1

        if spin_change == 0:
                return True
        else:
            return False

    def is_1body_op_spin_conserving(self, P,Q) -> bool:

        """Determines whether or not the one body fermionic excitation operator
            involved in the 1-RDM element indexed by (P,Q) conserves spin or not.
        
        Args:
            
            P,Q: The index of the 1-RDM.

        Returns:
            True if the fermionic operator conserves spin, False if it does not conserve
                spin.
        
        """

        N = self.num_spin_orbitals
            
        spin_change = 0
        if 0 <= P <= N/2 - 1:
            spin_change += 1
        else:
            spin_change -= 1

        if 0 <= Q <= N/2 - 1:
            spin_change -= 1
        else:
            spin_change += 1

        if spin_change == 0:
            return True
        else:
            return False

    def construct_pauli_op_dict(self, mapper: QubitMapper) -> Dict:

        """Constructs a dictionary of all the Pauli string operators necessary for computing the RDMs.
            The dictionary key/value pairs are of the form str(pauli_op): pauli_op. The uniqueness of
            python dictionary keys ensures that no redundant operator evaluations are done.
            
        Args:
            state: The state with respect to which the RDMs are being calculated.
            qubit_converter: The QubitConverter used to map the fermionic excitation
                operators to qubit operators.

        Returns:

            The dictionary consisting of all the Pauli string operators necessary for calculating
                The RDMs.

        """

        N = self.num_spin_orbitals
        pauli_op_dict = {}
    
        def oneRDM_add_pauli_ops_to_set(p,q):

            op = mapper.map(FermionicOp(data={f'+_{p} -_{q}': 1.0},
                num_spin_orbitals=N))

            if op.equiv(op.adjoint()):

                pauli_op_dict[str(op)] = op

            else:

                pauli_string_list = op.to_list()
                for op_tuple in pauli_string_list:
                    #pauli_op_dict[str(op_tuple[0])] = PauliSumOp(SparsePauliOp(op_tuple[0]))
                    pauli_op_dict[str(op_tuple[0])] = SparsePauliOp(op_tuple[0])

            return None

        def twoRDM_add_pauli_ops_to_set(p,q,r,s):

            op = mapper.map(FermionicOp(data={f'+_{p} +_{q} -_{s} -_{r}': 1.0},
                num_spin_orbitals=N))
            
            if op.equiv(op.adjoint()):

                pauli_op_dict[str(op)] = op

            else:

                pauli_string_list = op.to_list()
                for op_tuple in pauli_string_list:
                    #pauli_op_dict[str(op_tuple[0])] = PauliSumOp(SparsePauliOp(op_tuple[0]))
                    pauli_op_dict[str(op_tuple[0])] = SparsePauliOp(op_tuple[0])

            return None

  

        # if element is True, then we still need to generate the pauli operators relevant to this fermionic operator.
        # if element is False, then we do not need to generate any additional pauli operators for this fermionic operator.
        # needs_evaluation_array keeps track of this to avoid redundancy.
        oneRDM_needs_evaluation_array = np.full((N,N), fill_value=True, dtype=bool)
        twoRDM_needs_evaluation_array = np.full((N,N,N,N),fill_value=True, dtype=bool)

        for p in range(N):
            for q in range(N):
                for r in range(N):
                        for s in range(N):
                                    
                            if twoRDM_needs_evaluation_array[p,q,r,s] == True:

                                if p == q or r == s or (self.spin_conserving == True and self.is_2body_op_spin_conserving(p,q,r,s) == False):
                                        
                                        # we do not need to evaluate these operators as these entries will be zero in the 2-RDM.
                                    twoRDM_needs_evaluation_array[p,q,r,s] = False

                                else:
                                        
                                        # we add the relevant pauli ops to the set, then we set this entry to False
                                        # so we can set other elements to False to take advantage of symmetries in the 2RDM
                                        # to avoid redundant evaluations.
                                    twoRDM_add_pauli_ops_to_set(p,q,r,s)
                                    twoRDM_needs_evaluation_array[p,q,r,s] = False
                                        
                                twoRDM_needs_evaluation_array[q,p,r,s] = twoRDM_needs_evaluation_array[p,q,r,s]
                                twoRDM_needs_evaluation_array[p,q,s,r] = twoRDM_needs_evaluation_array[p,q,r,s]
                                twoRDM_needs_evaluation_array[q,p,s,r] = twoRDM_needs_evaluation_array[p,q,r,s]

                                    

                                twoRDM_needs_evaluation_array[r,s,p,q] = twoRDM_needs_evaluation_array[p,q,r,s]
                                twoRDM_needs_evaluation_array[r,s,q,p] = twoRDM_needs_evaluation_array[q,p,r,s]
                                twoRDM_needs_evaluation_array[s,r,p,q] = twoRDM_needs_evaluation_array[p,q,s,r]
                                twoRDM_needs_evaluation_array[s,r,q,p] = twoRDM_needs_evaluation_array[q,p,s,r]

        for p in range(N):
            for q in range(N):
                    
                if oneRDM_needs_evaluation_array[p,q] == True:

                    if self.spin_conserving == True and self.is_1body_op_spin_conserving(p,q) == False:

                            # we don't need the pauli operators from these terms because
                            # they are just zero in the 1-RDM
                        oneRDM_needs_evaluation_array[p,q] = False

                    else:
                    
                        oneRDM_add_pauli_ops_to_set(p,q)
                        oneRDM_needs_evaluation_array[p,q] = False
                        oneRDM_needs_evaluation_array[q,p] = oneRDM_needs_evaluation_array[p,q]

        return pauli_op_dict

    def get_two_RDM_tensor(self, expectval_dict: dict, mapper: QubitMapper) -> torch.Tensor:

        """Constructs and returns the 2-RDM tensor. The class attribute pauli_ops_expectation_values_dict stores the expectation values
            of all the Pauli operators necessary for this calculation. get_two_RDM_tensor simply retrieves these expectation values
            and constructs the 2-RDM.
        
        Args:

            state: The state with respect to which the 2-RDM is being calculated.
            qubit_converter: The QubitConverter used for mapping fermionic operators to qubit operators.

        Returns:
            The 2-RDM with respect to the given state.        
        
        """
        
        #N = state.num_qubits # Associating this number with the number of qubits could fail if symmetry reductions for some mappings is used.
        N = self.num_spin_orbitals                    # This should change if this is something we want to do eventually.
        global two_RDM_found_complex_value_flag
        two_RDM_found_complex_value_flag = False
        def get_two_RDM_element(p: int,q: int,r: int,s: int, mapper: QubitMapper):

            global two_RDM_found_complex_value_flag

            op = mapper.map(FermionicOp(data={f'+_{p} +_{q} -_{s} -_{r}': 1.0},
                num_spin_orbitals=N))
            
            if op.equiv(op.adjoint()):
                
                    mean = expectval_dict[str(op)]
            else:
                    pauli_string_list = op.to_list()
                    mean = 0
                    for op_tuple in pauli_string_list:
                        
                        mean += op_tuple[1]*expectval_dict[str(op_tuple[0])]
            
            if not np.isclose(np.imag(mean), 0.0, rtol=10e-12, atol=10e-12):
                
                two_RDM_found_complex_value_flag = True

            if self.wavefunction_real == True:
                return np.real(mean)
            else:
                return mean

        if self.wavefunction_real == True:
            tensor = np.full(shape=(N,N,N,N), fill_value=None, dtype=np.float64)
        else:
            tensor = np.full(shape=(N,N,N,N), fill_value=None, dtype=np.complex128)
        
        for p in range(N):
                for q in range(N):
                        for r in range(N):
                                for s in range(N):
                                    
                                    if np.isnan(tensor[p,q,r,s]):

                                        if p == q or r == s or (self.spin_conserving == True and self.is_2body_op_spin_conserving(p,q,r,s) == False):

                                            tensor[p,q,r,s] = 0

                                        else:

                                            tensor[p,q,r,s] = get_two_RDM_element(p=p,q=q,r=r,s=s, mapper=mapper)
                                        
                                        tensor[q,p,r,s] = -1*tensor[p,q,r,s]
                                        tensor[p,q,s,r] = -1*tensor[p,q,r,s]
                                        tensor[q,p,s,r] = tensor[p,q,r,s]

                                        if self.wavefunction_real == True:
                                            
                                            tensor[r,s,p,q] = tensor[p,q,r,s]
                                            tensor[r,s,q,p] = tensor[q,p,r,s]
                                            tensor[s,r,p,q] = tensor[p,q,s,r]
                                            tensor[s,r,q,p] = tensor[q,p,s,r]

                                        elif self.wavefunction_real == False:

                                            tensor[r,s,p,q] = np.conj(tensor[p,q,r,s])
                                            tensor[r,s,q,p] = np.conj(tensor[q,p,r,s])
                                            tensor[s,r,p,q] = np.conj(tensor[p,q,s,r])
                                            tensor[s,r,q,p] = np.conj(tensor[q,p,s,r])

        if two_RDM_found_complex_value_flag == False:

            tensor = np.real(tensor)

        tensor = torch.from_numpy(tensor)
        tensor.requires_grad = False

        return tensor

    def get_one_RDM_tensor(self, expectval_dict: dict, mapper: QubitMapper) -> torch.Tensor:
        
        """Constructs and returns the 1-RDM tensor. The class attribute pauli_ops_expectation_values_dict stores the expectation values
            of all the Pauli operators necessary for this calculation. get_one_RDM_tensor simply retrieves these expectation values
            as needed and constructs the 1-RDM.

        Args:
            state: The state with respect to which the 1-RDM is being calculated.
            qubit_converter: The QubitConverter used to map fermionic operators to qubit operators.

        Returns:
            The 1-RDM tensor.
        
        """

        global one_RDM_found_complex_value_flag
        one_RDM_found_complex_value_flag = False
        N = self.num_spin_orbitals
        def get_one_RDM_element(p: int,q: int, mapper: QubitMapper) -> torch.Tensor:
            
            global one_RDM_found_complex_value_flag

            op = mapper.map(FermionicOp(data={f'+_{p} -_{q}': 1.0},
                num_spin_orbitals=N))
            
            if op.equiv(op.adjoint()):
                
                    mean = expectval_dict[str(op)]

            else:
                    pauli_string_list = op.to_list()
                    mean = 0
                    for op_tuple in pauli_string_list:
                        
                            mean += op_tuple[1]*expectval_dict[str(op_tuple[0])]
            
            if not np.isclose(np.imag(mean), 0.0, rtol=10e-12, atol=10e-12):

                one_RDM_found_complex_value_flag = True

            if self.wavefunction_real == True:
                return np.real(mean)
            else:
                return mean

        if self.wavefunction_real == True:
            tensor = np.full(shape=(N,N), fill_value=None, dtype=np.float64)
        else:
            tensor = np.full(shape=(N,N), fill_value=None, dtype=np.complex128)
        
        for p in range(N):
                for q in range(N):
                    
                    if np.isnan(tensor[p,q]):

                        if self.spin_conserving == True and self.is_1body_op_spin_conserving(p,q) == False:

                            tensor[p,q] = 0

                        else:
                    
                            tensor[p,q] = get_one_RDM_element(p=p,q=q, mapper=mapper)

                        if self.wavefunction_real == True:
                            
                            tensor[q,p] = tensor[p,q]
                        
                        else:

                            tensor[q,p] = np.conj(tensor[p,q])

        if one_RDM_found_complex_value_flag == False:
            tensor = np.real(tensor)
            
        tensor = torch.from_numpy(tensor)
        tensor.requires_grad = False

        return tensor

    def compute_rotated_energy(self, partial_unitary: torch.Tensor,
                                     oneRDM: torch.Tensor,
                                     twoRDM: torch.Tensor,
                                     one_body_integrals: torch.Tensor,
                                     two_body_integrals: torch.Tensor) -> float:
        """
        Calculates the energy functional with varied U, but fixed wavefunction.

        Args:
            partial_unitary: The partial unitary matrix U.

        Returns:
            P(U), the energy functional for a given rotation U.
        """

        partial_unitary = torch.block_diag(partial_unitary, partial_unitary)


        if self.wavefunction_real == True or (oneRDM.dtype !=torch.complex128 and twoRDM.dtype != torch.complex128):
        
            energy = torch.einsum('pq,pi,qj,ij', one_body_integrals,
                                          partial_unitary,
                                          partial_unitary,
                                          oneRDM)
            energy += torch.einsum('pqrs,pi,qj,rk,sl,ijkl', two_body_integrals,
                                                       partial_unitary,
                                                       partial_unitary,
                                                       partial_unitary, 
                                                       partial_unitary, 
                                                       twoRDM)

        else:

            partial_unitary = partial_unitary.cdouble()
            temp_one_body_integrals = one_body_integrals.cdouble()
            temp_two_body_integrals = two_body_integrals.cdouble()

            energy = torch.einsum('pq,pi,qj,ij', temp_one_body_integrals,
                                          partial_unitary,
                                          partial_unitary,
                                          oneRDM)
            energy -= torch.einsum('pqrs,pi,qj,rk,sl,ijkl', temp_two_body_integrals,
                                                       partial_unitary,
                                                       partial_unitary,
                                                       partial_unitary, 
                                                       partial_unitary, 
                                                       twoRDM)
        
        return np.real(energy)

    def get_rotated_hamiltonian(self, partial_unitary: torch.Tensor) -> SparsePauliOp:

        """Transforms the one and two body integrals from the initial larger basis and transforms them according to
            a partial unitary matrix U. The transformed Hamiltonian is then constructed from these new integrals.

        Args:
            partial_unitary: The partial unitary transformation U.

        Returns:
            The transformed Hamiltonian.
        
        """

        partial_unitary = torch.block_diag(partial_unitary, partial_unitary)
        
        rotated_one_body_integrals = torch.einsum('pq,pi,qj->ij',
                                     self.one_body_integrals,
                                     partial_unitary, partial_unitary)
        rotated_two_body_integrals = torch.einsum('pqrs,pi,qj,rk,sl->ijkl',
                                     self.two_body_integrals,
                                     partial_unitary, partial_unitary, partial_unitary, partial_unitary)

        num_MO = int(self.num_spin_orbitals/2)
        electronic_energy_from_ints = ElectronicEnergy.from_raw_integrals(h1_a=rotated_one_body_integrals.detach().numpy()[0:num_MO, 0:num_MO],
                        h2_aa=-2*rotated_two_body_integrals.detach().numpy()[0:num_MO, 0:num_MO, 0:num_MO, 0:num_MO])
        
        fermionic_op = electronic_energy_from_ints.second_q_op().normal_order()

        return self.mapper.map(fermionic_op)
    
    def orth(self, V: torch.Tensor) -> torch.Tensor:
        """
        Generate the orthonormal projection of the matrix V.

        Args:
            V: The matrix to be orthonormalized.
                
        Returns:
            orth(V), the orthogonal projection of the matrix V.
        """
        L, Q = torch.linalg.eigh(torch.t(V) @ V)
        result = V @ Q @ (torch.float_power(torch.inverse(torch.diag(L)), 0.5)) @ torch.t(Q).double()
        return result

class BaseOptOrbResult(VariationalResult):

    def __init__(self) -> None:
        super().__init__()
        self._num_vqe_evaluations = 0
        self._optimal_partial_unitary = None
    
    @property
    def num_vqe_evaluations(self) -> int:
        """Returns the number of times VQE was run in OptOrbVQE."""
        return self._num_vqe_evaluations

    @num_vqe_evaluations.setter
    def num_vqe_evaluations(self, some_int: int) -> None:
        """Sets the number of times VQE was run in OptOrbVQE."""
        self._num_vqe_evaluations = some_int

    @property
    def optimal_partial_unitary(self) -> torch.Tensor:
        """Returns the optimal partial unitary basis transformation."""
        return self._optimal_partial_unitary

    @optimal_partial_unitary.setter
    def optimal_partial_unitary(self, some_tensor: torch.Tensor) -> None:
        """Sets the optimal partial unitary basis transformation."""
        self._optimal_partial_unitary = some_tensor




    