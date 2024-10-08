import sys
sys.path.append('..')

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms.optimizers import L_BFGS_B, COBYLA
from qiskit_aer.primitives import Estimator, Sampler
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.operators.tensor_ordering import to_physicist_ordering

from electronic_structure_algorithms.orbital_optimization import PartialUnitaryProjectionOptimizer, OptOrbVQD
from electronic_structure_algorithms.excited_states_eigensolvers import VQD
from electronic_structure_algorithms.initializations import get_HF_permutation_matrix, get_CIS_states

from time import perf_counter

estimator = Estimator(approximation=True)
sampler = Sampler(run_options={'shots': None})
fidelity = ComputeUncompute(sampler=sampler)
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

initial_states = [QuantumCircuit(num_reduced_qubits) for n in range(num_states)]
for n in range(num_states):
        initial_states[n].initialize(cis_states[n], initial_states[n].qubits)
        initial_states[n] = transpile(initial_states[n], optimization_level=3).decompose()

for state in initial_states:
        
        for n, instruction in enumerate(state.data):

                if instruction.operation.name == 'reset':
                        del state.data[n]

HF_state = HartreeFock(qubit_mapper=mapper,
                       num_spatial_orbitals=int(num_reduced_qubits/2),
                       num_particles=num_particles)
        
excited_HF = QuantumCircuit(4)
excited_HF.x(1)
excited_HF.x(2)

initial_states = [HF_state, excited_HF]

ansatz_list = [UCCSD(qubit_mapper=mapper,
               num_spatial_orbitals=int(num_reduced_qubits/2),
               num_particles=num_particles,
               reps=2,
               initial_state=initial_states[n]) for n in range(num_states)]

outer_iteration = 0
vqd_start_time = perf_counter()
def vqd_callback(eval_count, parameters, mean, std, step, costs):
        global vqd_start_time
        print(f'OptOrb iteration: {outer_iteration}, function evaluation: {eval_count}, energy level {step - 1}: {mean}, time = {perf_counter() - vqd_start_time}')
        vqd_start_time = perf_counter()

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

vqd_instance = VQD(k=num_states,
                   ansatz=ansatz_list,
                   initial_point=[np.zeros(ansatz_list[n].num_parameters) for n in range(num_states)],
                   optimizer=l_bfgs_b,
                   estimator=estimator,
                   callback=vqd_callback,
                   fidelity=fidelity,
                   betas=[2,2])

optorbssvqe_instance = OptOrbVQD(problem=q_molecule,
                               integral_tensors=(one_body_integrals, two_body_integrals),
                               num_spin_orbitals=num_reduced_qubits,
                               excited_states_solver=vqd_instance,
                               mapper=mapper,
                               estimator=estimator,
                               partial_unitary_optimizer=partial_unitary_optimizer,
                               maxiter=20,
                               wavefuntion_real=True,
                               spin_conserving=True,
                               stopping_tolerance=10**-5,
                               outer_loop_callback=outer_loop_callback,
                               partial_unitary_random_perturbation=0.01,
                               eigensolver_random_perturbation=0.0)

ground_state_energy_result = optorbssvqe_instance.compute_energies()
print(ground_state_energy_result)