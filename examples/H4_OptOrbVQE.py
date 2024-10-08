import sys
sys.path.append('..')

import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import L_BFGS_B, COBYLA
from qiskit_aer.primitives import Estimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

from electronic_structure_algorithms.orbital_optimization import PartialUnitaryProjectionOptimizer, OptOrbVQE

from time import perf_counter

estimator = Estimator(approximation=True)
mapper=JordanWignerMapper()

interatomic_distance = 1.23
driver = PySCFDriver(atom=f'H 0 0 0; H 0 {interatomic_distance} 0; H {interatomic_distance} 0 0; H {interatomic_distance} {interatomic_distance} 0',
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM, basis='cc-pVDZ')

q_molecule = driver.run()
num_particles = q_molecule.num_particles

l_bfgs_b = L_BFGS_B(maxfun=10**6, maxiter=10**6)
cobyla = COBYLA(maxiter=10**6)

num_reduced_qubits = 8

HF_state = HartreeFock(qubit_mapper=mapper,
                       num_spatial_orbitals=int(num_reduced_qubits/2),
                       num_particles=num_particles)

ansatz = UCCSD(qubit_mapper=mapper,
               num_spatial_orbitals=int(num_reduced_qubits/2),
               num_particles=num_particles,
               initial_state=HF_state,
               reps=2)


outer_iteration = 0
vqe_start_time = perf_counter()
def vqe_callback(eval_count, parameters, mean, std):
        global vqe_start_time
        print(f'Outer loop iteration: {outer_iteration}, function evaluation: {eval_count}, energy: {mean}, time = {perf_counter() - vqe_start_time}')

        vqe_start_time = perf_counter()


orbital_rotation_start_time = perf_counter()
def orbital_rotation_callback(orbital_rotation_iteration, energy):
        global orbital_rotation_start_time
        print(f'Outer loop iteration: {outer_iteration}, Iteration: {orbital_rotation_iteration}, energy: {energy}, time: {perf_counter() - orbital_rotation_start_time}')
        orbital_rotation_start_time = perf_counter()
        

def outer_loop_callback(optorb_iteration, vqe_result, optorb_result):
        global outer_iteration
        outer_iteration += 1


partial_unitary_optimizer = PartialUnitaryProjectionOptimizer(initial_BBstepsize=10**-3,
                                                              stopping_tolerance=10**-5,
                                                              maxiter=10000,
                                                              gradient_method='autograd',
                                                              callback=orbital_rotation_callback)

vqe_instance = VQE(ansatz=ansatz,
                   initial_point=np.zeros(ansatz.num_parameters),
                   optimizer=l_bfgs_b,
                   estimator=estimator,
                   callback=vqe_callback)

optorbvqe_instance = OptOrbVQE(problem=q_molecule,
                               integral_tensors = None,
                               num_spin_orbitals=num_reduced_qubits,
                               ground_state_solver=vqe_instance,
                               mapper=mapper,
                               estimator=estimator,
                               partial_unitary_optimizer=partial_unitary_optimizer,
                               maxiter=20,
                               wavefuntion_real=True,
                               spin_conserving=True,
                               stopping_tolerance=10**-5,
                               outer_loop_callback=outer_loop_callback,
                               partial_unitary_random_perturbation=0.01,
                               minimum_eigensolver_random_perturbation=0.0)

ground_state_energy_result = optorbvqe_instance.compute_minimum_energy()
print(ground_state_energy_result)

