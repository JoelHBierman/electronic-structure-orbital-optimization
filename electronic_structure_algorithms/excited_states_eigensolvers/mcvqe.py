# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The multi-configuration Variational Quantum Eigensolver algorithm.

See https://arxiv.org/abs/1810.09434. This implementation is derived
from Qiskit's implemtation of VQE.
"""

from __future__ import annotations

import logging
import warnings
from time import time
from collections.abc import Callable, Sequence

import numpy as np

from qiskit_algorithms.gradients import BaseEstimatorGradient
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms.utils import algorithm_globals
from qiskit.quantum_info import Statevector

from qiskit_algorithms.exceptions import AlgorithmError
from qiskit_algorithms.list_or_dict import ListOrDict
from qiskit_algorithms.optimizers import Optimizer, Minimizer, OptimizerResult
from qiskit_algorithms.eigensolvers.eigensolver import Eigensolver, EigensolverResult

from qiskit_algorithms.observables_evaluator import estimate_observables
from .ssvqe import SSVQE, SSVQEResult
from ..initializations.configuration_interaction_states import get_CIS_states, get_CISD_states

logger = logging.getLogger(__name__)

class MCVQE(SSVQE):

    r"""The Multi-Configuration Variational Quantum Eigensolver algorithm.
    `MCVQE <https://arxiv.org/abs/1810.09434>`__ is a hybrid quantum-classical
    algorithm that uses a variational technique to find the low-lying eigenvalues
    of the Hamiltonian :math:`H` of a given system. MCVQE can be seen as a
    special case of the Subspace Search Variational Quantum Eigensolver (SSVQE).
    The primary difference is that MCVQE chooses all of the weights in the
    objective function to be equal. Thus, its global minimia consist of
    states which span the low-lying eigenspace of the Hamiltonian in question
    rather than the eigenvectors themselves. Because of this, an extra
    post-processing step involving the diagonalization of the contracted
    Hamiltonian in this subspace is required to obtain the eigenstates themselves.
    If the initial states are chosen to be a set of mutually orthogonal states
    whose configuration interaction (CI) coefficients are known, then
    the off-diagonal elements of this contracted Hamiltonian can be computed without
    (potentially expensive) Hamard test-style circuits for computing inner product
    terms. It is worth notating that mutually orthogonal sets of Slater
    determinants fulfill this criteria.

    An instance of MCVQE requires defining four algorithmic sub-components:

    An :attr:`estimator` to compute the expectation values of operators, an integer ``k`` denoting
    the number of eigenstates that the algorithm will attempt to find, an ansatz which is a
    :class:`QuantumCircuit`, and one of the classical :mod:`~qiskit.algorithms.optimizers`.

    The ansatz is varied, via its set of parameters, by the optimizer, such that it works towards
    a set of mutually orthogonal states which span the low-lying eigenspace of
    the Hamiltonian :math:`H`. An optional array of parameter values, via the
    ``initial_point``, may be provided as the starting point for the search of the
    low-lying eigenvalues. This feature is particularly useful such as
    when there are reasons to believe that the solution point is close to a particular
    point. The length of the ``initial_point`` list value must match the number of the
    parameters expected by the ansatz being used. If the ``initial_point`` is left at the
    default of ``None``, then MCVQE will look to the ansatz for a preferred value, based
    on its given initial state. If the ansatz returns ``None``, then a random point will
    be generated within the parameter bounds set, as per above. If the ansatz provides
    ``None`` as the lower bound, then MCVQE will default it to :math:`-2\pi`; similarly,
    if the ansatz returns ``None`` as the upper bound, the default value will be :math:`2\pi`.

    There are two ways to provide initial states to MCVQE. The first is to provide values
    for ``excitations``, ``one_body_integrals``, and ``two_body_integrals``, and
    ``num_particles``. The one and two body integrals must be the same array
    instance as those used to calculate the Hamiltonian and the two body integrals must be
    in the Physicist notation. These integrals are used to compute the matrix elements of a
    contracted configuration interaction Hamiltonian to be diagonalized, yielding CI
    statevectors. The value assigned to ``excitations`` determins the excitation level
    of this CI problem. Setting ``excitations`` to ``s`` will use configuration interaction
    singles (CIS) states and setting it to ``sd`` will use configuration interaction singles
    and doubles (CISD). Similarly, ``num_spin_orbitals`` and ``num_particles`` are also used
    in this CI statevector construction and must match the number of spin-orbitals and
    particles of the problem in question.

    An optional list of initial states, via the ``initial_states``, may also be provided
    if the method described above is not used. Choosing these states appropriately
    is a critical part of the algorithm. They must be mutually orthogonal because this
    is how the algorithm enforces the mutual orthogonality of the solution states.
    Additionally, if one wishes to find the low-lying excited states of a molecular
    Hamiltonian, then we expect the output states to belong to a particular
    particle-number subspace. If an ansatz that preserves particle number such as
    :class:`UCCSD` is used, then states belonging to the incorrect particle
    number subspace will be returned if the ``initial_states`` are not in the correct
    particle number subspace. A similar statement can often be made for the
    spin-magnetization quantum number.

    The following attributes can be set via the initializer but can also be read and
    updated once the MCVQE object has been constructed.

    Attributes:
            estimator (BaseEstimator): The primitive instance used to perform the expectation
                estimation of observables.
            k (int): The number of eigenstates that SSVQE will attempt to find.
            ansatz (QuantumCircuit): A parameterized circuit used as an ansatz for the
                wave function.
            optimizer (Optimizer): A classical optimizer, which can be either a Qiskit optimizer
                or a callable that takes an array as input and returns a Qiskit or SciPy optimization
                result.
            gradient (BaseEstimatorGradient | None): An optional estimator gradient to be used with the
                optimizer.
            initial_states (Sequence[QuantumCircuit]): An optional list of mutually orthogonal
                initial states. If ``None``, then MCVQE will set these to be a list of mutually
                orthogonal computational basis states.
            callback (Callable[[int, np.ndarray, Sequence[float], dict[str, Any]], None] | None): A
                function that can access the intermediate data at each optimization step. These data are
                the evaluation count, the optimizer parameters for the ansatz, the evaluated mean
                energies, and the metadata dictionary.
            check_input_states_orthogonality: A boolean that sets whether or not to check
                that the value of initial_states passed consists of a mutually orthogonal
                set of states. If ``True``, then MCVQE will check that these states are mutually
                orthogonal and return an :class:`AlgorithmError` if they are not.
                This is set to ``True`` by default, but setting this to ``False`` may be desirable
                for larger numbers of qubits to avoid exponentially large computational overhead.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        num_particles: tuple | None = None,
        one_body_integrals: np.ndarray | None = None,
        two_body_integrals: np.ndarray | None = None,
        k: int | None = 2,
        ansatz: QuantumCircuit | None = None,
        optimizer: Optimizer | Minimizer | None = None,
        initial_point: Sequence[float] | None = None,
        gradient: BaseEstimatorGradient | None = None,
        callback: Callable[[int, np.ndarray, Sequence[float], float], None] | None = None,
        check_input_states_orthogonality: bool = True,
        excitations: str = None,
        initial_states: list[QuantumCircuit] = None,
    ) -> None:
        """
        Args:
            estimator: The estimator primitive.
            num_particles: A tuple whose entries denote the number of alpha and beta electrons
                in the problem Hamiltonian.
            one_body_integrals: The one body integrals of the problem Hamiltonian.
            two_body_integrals: The two body integrals of the problem Hamiltonian in the
                Physicist's notation.
            k: The number of eigenstates that the algorithm will attempt to find.
            ansatz: A parameterized circuit used as Ansatz for the wave function.
            optimizer: A classical optimizer. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            gradient: An optional gradient function or operator for optimizer.
            callback: A callback that can access the intermediate data at each optimization step.
                These data are: the evaluation count, the optimizer ansatz parameters,
                the evaluated mean energies, and the metadata dictionary.
            check_input_states_orthogonality: A boolean that sets whether or not to check
                that the value of ``initial_states`` passed consists of a mutually orthogonal
                set of states. If ``True``, then SSVQE will check that these states are mutually
                orthogonal and return an error if they are not. This is set to ``True`` by default,
                but setting this to ``False`` may be desirable for larger numbers of qubits to avoid
                exponentially large computational overhead before the simulation even starts.
            excitations: One of `s` or `sd` to determine whether CIS or CISD states will be used for
            initialization.
            initial_states: An optional list of mutually orthogonal initial states.
        """

        super().__init__(estimator=estimator,
                         k=k,
                         ansatz=ansatz,
                         optimizer=optimizer,
                         initial_point=initial_point,
                         gradient=gradient,
                         callback=callback,
                         check_input_states_orthogonality=check_input_states_orthogonality)

        self.k = k
        self.weight_vector = [1 for n in range(k)]
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.initial_point = initial_point
        self.gradient = gradient
        self.callback = callback
        self.estimator = estimator
        self.check_initial_states_orthogonal = check_input_states_orthogonality
        self._excitations = excitations
        self._initial_states = initial_states
        self._one_body_integrals = one_body_integrals
        self._two_body_integrals = two_body_integrals
        self._off_diagonal_plus_states = None
        self._off_diagonal_minus_states = None
        self.num_particles = num_particles

    @property
    def initial_states(self) -> Sequence[QuantumCircuit] | np.ndarray | None:

        return self._initial_states
        
    @initial_states.setter
    def initial_states(self, states) -> None:

        self._initial_states = states

    @property
    def excitations(self) -> str | None:

        return self._excitations
        
    @excitations.setter
    def excitations(self, excitations_string) -> None:

        self._excitations = excitations_string

    @property
    def one_body_integrals(self) -> np.ndarray | None:

        return self._one_body_integrals

    @one_body_integrals.setter
    def one_body_integrals(self, integrals) -> None:

        self._one_body_integrals = integrals

    @property
    def two_body_integrals(self) -> np.ndarray | None:

        return self._two_body_integrals

    @two_body_integrals.setter
    def two_body_integrals(self, integrals) -> None:

        self._two_body_integrals = integrals 

    def initialize_mcvqe(self) -> None:

        if self.excitations is None and self.initial_states is not None:
            
            for n, state in enumerate(initial_states):

                if isinstance(state, QuantumCircuit):

                    initial_states[n] = np.asarray(Statevector(state))

            self.initial_states = initial_states

        if self.excitations == 's':
            
            num_spin_orbitals = self.one_body_integrals.shape[0]
            initial_states = get_CIS_states(one_body_integrals=self.one_body_integrals,
                                            two_body_integrals=self.two_body_integrals,
                                            num_particles=self.num_particles,
                                            state_representation='dense')[:self.k]

        if self.excitations == 'sd':
            
            num_spin_orbitals = self.one_body_integrals.shape[0]
            initial_states = get_CISD_states(one_body_integrals=self.one_body_integrals,
                                            two_body_integrals=self.two_body_integrals,
                                            num_particles=self.num_particles,
                                            num_spin_orbitals=num_spin_orbitals,
                                            state_representation='dense')[:self.k]

        self.initial_states = [QuantumCircuit(self.ansatz.num_qubits) for n in range(self.k)]
        for n in range(self.k):
            self.initial_states[n].initialize(initial_states[n], self.initial_states[n].qubits)

        self._off_diagonal_plus_states = np.full(fill_value=None, shape=(self.k,self.k))
        self._off_diagonal_minus_states = np.full(fill_value=None, shape=(self.k,self.k))

        for i in range(self.k):
            for j in range(i):
                if i == j:
                    pass
                else:
                    plus_state = QuantumCircuit(self.ansatz.num_qubits)
                    minus_state = QuantumCircuit(self.ansatz.num_qubits)

                    plus_statevector = (initial_states[i] + initial_states[j])/np.sqrt(2)
                    plus_statevector = plus_statevector/np.sqrt(sum(amplitude**2 for amplitude in plus_statevector))
                    
                    minus_statevector = (initial_states[i] - initial_states[j])/np.sqrt(2)
                    minus_statevector = minus_statevector/np.sqrt(sum(amplitude**2 for amplitude in minus_statevector))
            
                    self._off_diagonal_plus_states[i,j] = plus_state
                    self._off_diagonal_plus_states[i,j].initialize(plus_statevector, plus_state.qubits)

                    self._off_diagonal_minus_states[i,j] = minus_state
                    self._off_diagonal_minus_states[i,j].initialize(minus_statevector, minus_state.qubits)

    
    def compute_eigenvalues(
        self,
        operator: BaseOperator | PauliSumOp,
        aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None,
    ) -> EigensolverResult:

        ansatz = self._check_operator_ansatz(operator)

        initial_point = _validate_initial_point(self.initial_point, ansatz)

        self.initialize_mcvqe()

        initial_states = self._check_operator_initial_states(self.initial_states, operator)

        bounds = _validate_bounds(ansatz)

        initialized_ansatz_list = [initial_states[n].compose(ansatz) for n in range(self.k)]

        self.weight_vector = self._check_weight_vector(self.weight_vector)

        evaluate_weighted_energy_sum = self._get_evaluate_weighted_energy_sum(
            initialized_ansatz_list, operator
        )

        if self.gradient is not None:  # need to implement _get_evaluate_gradient
            evaluate_gradient = self._get_evalute_gradient(initialized_ansatz_list, operator)
        else:
            evaluate_gradient = None

        if aux_operators:
            zero_op = PauliSumOp.from_list([("I" * self.ansatz.num_qubits, 0)])

            # Convert the None and zero values when aux_operators is a list.
            # Drop None and convert zero values when aux_operators is a dict.
            if isinstance(aux_operators, list):
                key_op_iterator = enumerate(aux_operators)
                converted = [zero_op] * len(aux_operators)
            else:
                key_op_iterator = aux_operators.items()
                converted = {}
            for key, op in key_op_iterator:
                if op is not None:
                    converted[key] = zero_op if op == 0 else op

            aux_operators = converted

        else:
            aux_operators = None

        start_time = time()

        if callable(self.optimizer):
            optimizer_result = self.optimizer(
                fun=evaluate_weighted_energy_sum,
                x0=initial_point,
                jac=evaluate_gradient,
                bounds=bounds,
            )
        else:
            optimizer_result = self.optimizer.minimize(
                fun=evaluate_weighted_energy_sum,
                x0=initial_point,
                jac=evaluate_gradient,
                bounds=bounds,
            )

        optimizer_time = time() - start_time

        logger.info(
            "Optimization complete in %s seconds.\nFound opt_params %s",
            optimizer_time,
            optimizer_result.x,
        )

        if aux_operators is not None:
            bound_ansatz_list = [
                initialized_ansatz_list[n].bind_parameters(optimizer_result.x)
                for n in range(self.k)
            ]

            aux_values_list = [
                estimate_observables(
                    self.estimator,
                    bound_ansatz_list[n],
                    aux_operators,
                )
                for n in range(self.k)
            ]
        else:
            aux_values_list = None

        return self._build_mcvqe_result(
            optimizer_result, aux_values_list, optimizer_time, operator, initialized_ansatz_list
        )

    def _build_mcvqe_result(
        self,
        optimizer_result: OptimizerResult,
        aux_operators_evaluated: ListOrDict[tuple[complex, tuple[complex, int]]],
        optimizer_time: float,
        operator: BaseOperator | PauliSumOp,
        initialized_ansatz_list: list[QuantumCircuit],
    ) -> MCVQEResult:
        result = MCVQEResult()

        for i in range(self.k):
            for j in range(i):
                if i == j:
                    pass
                else:
                    
                    self._off_diagonal_plus_states[i,j] = self._off_diagonal_plus_states[i,j].compose(self.ansatz)
                    self._off_diagonal_minus_states[i,j] = self._off_diagonal_minus_states[i,j].compose(self.ansatz)

        contracted_Hamiltonian = np.empty(shape=(self.k, self.k))
        for i in range(self.k):
            for j in range(i):

                try:

                    matrix_element_result = self.estimator.run([self._off_diagonal_plus_states[i,j], self._off_diagonal_minus_states[i,j]],
                                                               [operator]*2,
                                                               [optimizer_result.x]*2
                    ).result().values
                    
                    contracted_Hamiltonian[i,j] = 0.5*(matrix_element_result[0] - matrix_element_result[1])
                    if np.isclose(contracted_Hamiltonian[i,j].imag, 0.0):
                        contracted_Hamiltonian[j,i] = contracted_Hamiltonian[i,j]
                    else:
                        contracted_Hamiltonian[j,i] = np.conjugate(contracted_Hamiltonian[i,j])

                except Exception as exc:
                    raise AlgorithmError("The primitive job to evaluate the eigenvalues failed!") from exc
        try:
            diagonal_elements = (
                self.estimator.run(
                    initialized_ansatz_list, [operator] * self.k, [optimizer_result.x] * self.k
                )
                .result()
                .values
            )

            for i in range(self.k):

                contracted_Hamiltonian[i,i] = diagonal_elements[i]
    
        except Exception as exc:
            raise AlgorithmError("The primitive job to evaluate the eigenvalues failed!") from exc
        
        result.eigenvalues = np.linalg.eigh(np.asarray(contracted_Hamiltonian))[0]

        result.cost_function_evals = optimizer_result.nfev
        result.optimal_point = optimizer_result.x
        result.optimal_parameters = dict(zip(self.ansatz.parameters, optimizer_result.x))
        result.optimal_value = optimizer_result.fun
        result.optimizer_time = optimizer_time
        result.aux_operators_evaluated = aux_operators_evaluated
        result.optimizer_result = optimizer_result

        return result

class MCVQEResult(SSVQEResult):
    """MCVQE Result."""

    def __init__(self) -> None:
        super().__init__()
        

def _validate_initial_point(point, ansatz):
    expected_size = ansatz.num_parameters

    # try getting the initial point from the ansatz
    if point is None and hasattr(ansatz, "preferred_init_points"):
        point = ansatz.preferred_init_points
    # if the point is None choose a random initial point

    if point is None:
        # get bounds if ansatz has them set, otherwise use [-2pi, 2pi] for each parameter
        bounds = getattr(ansatz, "parameter_bounds", None)
        if bounds is None:
            bounds = [(-2 * np.pi, 2 * np.pi)] * expected_size

        # replace all Nones by [-2pi, 2pi]
        lower_bounds = []
        upper_bounds = []
        for lower, upper in bounds:
            lower_bounds.append(lower if lower is not None else -2 * np.pi)
            upper_bounds.append(upper if upper is not None else 2 * np.pi)

        # sample from within bounds
        point = algorithm_globals.random.uniform(lower_bounds, upper_bounds)

    elif len(point) != expected_size:
        raise ValueError(
            f"The dimension of the initial point ({len(point)}) does not match the "
            f"number of parameters in the circuit ({expected_size})."
        )

    return point


def _validate_bounds(ansatz):
    if hasattr(ansatz, "parameter_bounds") and ansatz.parameter_bounds is not None:
        bounds = ansatz.parameter_bounds
        if len(bounds) != ansatz.num_parameters:
            raise ValueError(
                f"The number of bounds ({len(bounds)}) does not match the number of "
                f"parameters in the circuit ({ansatz.num_parameters})."
            )
    else:
        bounds = [(None, None)] * ansatz.num_parameters

    return bounds

        