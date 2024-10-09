import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms.optimizers import L_BFGS_B, COBYLA
from qiskit_aer.primitives import Estimator
from qiskit_nature.second_q.circuit.library import UCCSD

from electronic_structure_algorithms.orbital_optimization import PartialUnitaryProjectionOptimizer, OptOrbMCVQE
from electronic_structure_algorithms.excited_states_eigensolvers import MCVQE

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

l_bfgs_b = L_BFGS_B(maxfun=10**6, maxiter=10**6)

num_reduced_qubits = 4

ansatz = UCCSD(qubit_mapper=mapper,
               num_spatial_orbitals=int(num_reduced_qubits/2),
               num_particles=num_particles,
               reps=2)


outer_iteration = 0
vqe_start_time = perf_counter()
def mcvqe_callback(eval_count, parameters, mean, std):
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

mcvqe_instance = MCVQE(k=2,
                   excitations='s',
                   num_particles=q_molecule.num_particles,
                   ansatz=ansatz,
                   initial_point=np.zeros(ansatz.num_parameters),
                   optimizer=l_bfgs_b,
                   estimator=estimator,
                   callback=mcvqe_callback)

mcvqe_instance.weight_vector = mcvqe_instance._check_weight_vector()
optorbvqe_instance = OptOrbMCVQE(problem=q_molecule,
                               integral_tensors = None,
                               num_spin_orbitals=num_reduced_qubits,
                               excited_states_solver=mcvqe_instance,
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

ground_state_energy_result = optorbvqe_instance.compute_energies()
print(ground_state_energy_result)

