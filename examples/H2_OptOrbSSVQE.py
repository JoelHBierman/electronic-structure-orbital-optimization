import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms.optimizers import L_BFGS_B, COBYLA
from qiskit_aer.primitives import Estimator
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.operators.tensor_ordering import to_physicist_ordering

from electronic_structure_algorithms.orbital_optimization import PartialUnitaryProjectionOptimizer, OptOrbSSVQE
from electronic_structure_algorithms.excited_states_eigensolvers import SSVQE
from electronic_structure_algorithms.initializations import get_HF_permutation_matrix, get_CIS_states

from time import perf_counter

estimator = Estimator(approximation=True)
mapper=JordanWignerMapper()

driver = PySCFDriver(atom=f'H 0 0 0; H 0 0 {0.735}',
                     charge=0,
                     spin=0,
                     unit=DistanceUnit.ANGSTROM,
                     basis='6-31G')

q_molecule = driver.run()
num_particles = q_molecule.num_particles
num_original_spin_orbitals = q_molecule.num_spin_orbitals

l_bfgs_b = L_BFGS_B(maxfun=10**6, maxiter=10**6)
cobyla = COBYLA(maxiter=10**6)

num_reduced_qubits = 4
num_states = 2

ansatz = UCCSD(qubit_mapper=mapper,
               num_spatial_orbitals=int(num_reduced_qubits/2),
               num_particles=num_particles,
               reps=2)

one_body_integrals = torch.from_numpy(np.asarray(q_molecule.hamiltonian.electronic_integrals.second_q_coeffs()["+-"].to_dense()))
two_body_integrals = torch.from_numpy(np.asarray(-1*to_physicist_ordering(q_molecule.hamiltonian.electronic_integrals.second_q_coeffs()["++--"].to_dense())))

initial_partial_unitary = get_HF_permutation_matrix(num_original_spin_orbitals=num_original_spin_orbitals,
                                                    num_spin_orbitals=num_reduced_qubits)

transformed_one_body_integrals = torch.einsum('pq,pi,qj->ij',
                                                  one_body_integrals,
                                                  torch.block_diag(initial_partial_unitary, initial_partial_unitary),
                                                  torch.block_diag(initial_partial_unitary, initial_partial_unitary)).detach().numpy()

transformed_two_body_integrals = torch.einsum('pqrs,pi,qj,rk,sl->ijkl',
                                                  two_body_integrals,
                                                  torch.block_diag(initial_partial_unitary, initial_partial_unitary),
                                                  torch.block_diag(initial_partial_unitary, initial_partial_unitary),
                                                  torch.block_diag(initial_partial_unitary, initial_partial_unitary),
                                                  torch.block_diag(initial_partial_unitary, initial_partial_unitary)).detach().numpy()

cis_states = get_CIS_states(one_body_integrals=transformed_one_body_integrals,
                                            two_body_integrals=transformed_two_body_integrals,
                                            num_particles=q_molecule.num_particles,
                                            state_representation='dense')[:num_states]

initial_states = [QuantumCircuit(ansatz.num_qubits) for n in range(num_states)]
for n in range(num_states):
        initial_states[n].initialize(cis_states[n], initial_states[n].qubits)


HF_state = HartreeFock(qubit_mapper=mapper,
                       num_spatial_orbitals=int(num_reduced_qubits/2),
                       num_particles=num_particles)
        
excited_HF = QuantumCircuit(4)
excited_HF.x(1)
excited_HF.x(2)
print(excited_HF)

initial_states = [HF_state, excited_HF]
        
uccsd = UCCSD(qubit_mapper=mapper,
               num_spatial_orbitals=int(num_reduced_qubits/2),
               num_particles=num_particles)

outer_iteration = 0
vqe_start_time = perf_counter()
def ssvqe_callback(eval_count, parameters, mean, std):
        global vqe_start_time
        print(f'Outer loop iteration: {outer_iteration}, function evaluation: {eval_count}, energy: {mean}, time = {perf_counter() - vqe_start_time}')

        vqe_start_time = perf_counter()


orbital_rotation_start_time = perf_counter()
def orbital_rotation_callback(orbital_rotation_iteration, energy):
        global orbital_rotation_start_time
        print(f'Outer loop iteration: {outer_iteration}, Iteration: {orbital_rotation_iteration}, energy sum: {energy}, time: {perf_counter() - orbital_rotation_start_time}')
        orbital_rotation_start_time = perf_counter()
        

def outer_loop_callback(optorb_iteration, vqe_result, optorb_result):
        global outer_iteration
        outer_iteration += 1


partial_unitary_optimizer = PartialUnitaryProjectionOptimizer(initial_BBstepsize=10**-3,
                                                              stopping_tolerance=10**-5,
                                                              maxiter=10000,
                                                              gradient_method='autograd',
                                                              callback=orbital_rotation_callback)

ssvqe_instance = SSVQE(k=num_states,
                   initial_states=initial_states,
                   ansatz=ansatz,
                   initial_point=np.zeros(ansatz.num_parameters),
                   optimizer=l_bfgs_b,
                   estimator=estimator,
                   callback=ssvqe_callback,
                   weight_vector=[2,1])

optorbssvqe_instance = OptOrbSSVQE(problem=q_molecule,
                               integral_tensors=(one_body_integrals, two_body_integrals),
                               num_spin_orbitals=num_reduced_qubits,
                               excited_states_solver=ssvqe_instance,
                               mapper=mapper,
                               estimator=estimator,
                               partial_unitary_optimizer=partial_unitary_optimizer,
                               initial_partial_unitary=initial_partial_unitary,
                               maxiter=20,
                               wavefuntion_real=True,
                               spin_conserving=True,
                               stopping_tolerance=10**-5,
                               outer_loop_callback=outer_loop_callback,
                               partial_unitary_random_perturbation=0.0,
                               eigensolver_random_perturbation=0.0)

ground_state_energy_result = optorbssvqe_instance.compute_energies()
print(ground_state_energy_result)

