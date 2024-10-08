"""Test OptOrbVQD"""

import unittest

import numpy as np
import torch
from ddt import data, ddt

from qiskit import QuantumCircuit

from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_algorithms.state_fidelities import ComputeUncompute

import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_aer.primitives import Estimator, Sampler
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.operators.tensor_ordering import to_physicist_ordering

from electronic_structure_algorithms.orbital_optimization import PartialUnitaryProjectionOptimizer, OptOrbVQD
from electronic_structure_algorithms.excited_states_eigensolvers import VQD

estimator = Estimator(approximation=True)
sampler = Sampler(run_options={'shots': None})
fidelity = ComputeUncompute(sampler=sampler)
mapper=JordanWignerMapper()

driver = PySCFDriver(atom=f'H 0 0 0; H 0 0 {0.735}',
                     charge=0,
                     spin=0,
                     unit=DistanceUnit.ANGSTROM,
                     basis='6-31G')

es_problem = driver.run()
num_particles = es_problem.num_particles
num_reduced_qubits = 4

one_body_integrals = torch.from_numpy(np.asarray(es_problem.hamiltonian.electronic_integrals.second_q_coeffs()["+-"].to_dense()))
two_body_integrals = torch.from_numpy(np.asarray(-1*to_physicist_ordering(es_problem.hamiltonian.electronic_integrals.second_q_coeffs()["++--"].to_dense())))

integrals = one_body_integrals, two_body_integrals

@ddt
class TestOptOrbVQD(unittest.TestCase):
    """Test OptOrbVQD"""

    def setUp(self):
        super().setUp()

        self.k = 2
        HF_state = HartreeFock(qubit_mapper=mapper,
                       num_spatial_orbitals=int(num_reduced_qubits/2),
                       num_particles=num_particles)
        
        excited_HF = QuantumCircuit(4)
        excited_HF.x(1)
        excited_HF.x(2)

        self.initial_states = [HF_state, excited_HF]
        
        self.uccsd_list = [UCCSD(qubit_mapper=mapper,
               num_spatial_orbitals=int(num_reduced_qubits/2),
               num_particles=num_particles,
               reps=2,
               initial_state=self.initial_states[n]) for n in range(self.k)]

        self.h2_energies = [-1.8540352 , -1.37044389]
        self.estimator = Estimator(approximation=True)

    @data(integrals)
    def test_ground_state_integrals(self, integrals):

        partial_unitary_optimizer = PartialUnitaryProjectionOptimizer(initial_BBstepsize=10**-3,
                                                              stopping_tolerance=10**-5,
                                                              maxiter=10000,
                                                              gradient_method='autograd')
        
        vqd_instance = VQD(k=self.k,
                   ansatz=self.uccsd_list,
                   initial_point=[np.zeros(self.uccsd_list[n].num_parameters) for n in range(self.k)],
                   optimizer=L_BFGS_B(),
                   estimator=estimator,
                   fidelity=fidelity,
                   betas=[2,2])
        
        optorbvqd_instance = OptOrbVQD(problem=None,
                               integral_tensors=integrals,
                               num_spin_orbitals=num_reduced_qubits,
                               excited_states_solver=vqd_instance,
                               mapper=mapper,
                               estimator=estimator,
                               partial_unitary_optimizer=partial_unitary_optimizer,
                               maxiter=20,
                               wavefuntion_real=True,
                               spin_conserving=True)
        
        result = optorbvqd_instance.compute_energies()

        with self.subTest(msg="test eigenvalue, provided integrals"):
            np.testing.assert_array_almost_equal(
                [result.eigenvalues.real], [self.h2_energies], decimal=3
            )


    @data(es_problem)
    def test_ground_state_es_problem(self, problem):

        partial_unitary_optimizer = PartialUnitaryProjectionOptimizer(initial_BBstepsize=10**-3,
                                                              stopping_tolerance=10**-5,
                                                              maxiter=10000,
                                                              gradient_method='autograd')
        
        vqd_instance = VQD(k=self.k,
                   ansatz=self.uccsd_list,
                   initial_point=[np.zeros(self.uccsd_list[n].num_parameters) for n in range(self.k)],
                   optimizer=L_BFGS_B(),
                   estimator=estimator,
                   fidelity=fidelity,
                   betas=[2,2])
        
        optorbvqd_instance = OptOrbVQD(problem=problem,
                               integral_tensors = None,
                               num_spin_orbitals=num_reduced_qubits,
                               excited_states_solver=vqd_instance,
                               mapper=mapper,
                               estimator=estimator,
                               partial_unitary_optimizer=partial_unitary_optimizer,
                               maxiter=20,
                               wavefuntion_real=True,
                               spin_conserving=True)
        
        result = optorbvqd_instance.compute_energies()

        with self.subTest(msg="test eigenvalue, provided ElectronicStructureProblem"):
            np.testing.assert_array_almost_equal(
                [result.eigenvalues.real], [self.h2_energies], decimal=3
            )

    
    @data(integrals)
    def test_ground_state_integrals_2(self, integrals):

        partial_unitary_optimizer = PartialUnitaryProjectionOptimizer(initial_BBstepsize=10**-3,
                                                              stopping_tolerance=10**-5,
                                                              maxiter=10000,
                                                              gradient_method='autograd')
        
        vqd_instance = VQD(k=self.k,
                   ansatz=self.uccsd_list,
                   initial_point=[np.zeros(self.uccsd_list[n].num_parameters) for n in range(self.k)],
                   optimizer=L_BFGS_B(),
                   estimator=estimator,
                   fidelity=fidelity,
                   betas=[2,2])
        
        optorbvqd_instance = OptOrbVQD(problem=None,
                               integral_tensors=integrals,
                               num_spin_orbitals=num_reduced_qubits,
                               excited_states_solver=vqd_instance,
                               mapper=mapper,
                               estimator=estimator,
                               partial_unitary_optimizer=partial_unitary_optimizer,
                               maxiter=20,
                               wavefuntion_real=False,
                               spin_conserving=False)
        
        result = optorbvqd_instance.compute_energies()

        with self.subTest(msg="test eigenvalue, provided integrals, no RDM symmetry exploited"):
            np.testing.assert_array_almost_equal(
                [result.eigenvalues.real], [self.h2_energies], decimal=3
            )


    @data(es_problem)
    def test_ground_state_es_problem_2(self, problem):

        partial_unitary_optimizer = PartialUnitaryProjectionOptimizer(initial_BBstepsize=10**-3,
                                                              stopping_tolerance=10**-5,
                                                              maxiter=10000,
                                                              gradient_method='autograd')
        
        vqd_instance = VQD(k=self.k,
                   ansatz=self.uccsd_list,
                   initial_point=[np.zeros(self.uccsd_list[n].num_parameters) for n in range(self.k)],
                   optimizer=L_BFGS_B(),
                   estimator=estimator,
                   fidelity=fidelity,
                   betas=[2,2])
        
        optorbvqd_instance = OptOrbVQD(problem=problem,
                               integral_tensors = None,
                               num_spin_orbitals=num_reduced_qubits,
                               excited_states_solver=vqd_instance,
                               mapper=mapper,
                               estimator=estimator,
                               partial_unitary_optimizer=partial_unitary_optimizer,
                               maxiter=20,
                               wavefuntion_real=False,
                               spin_conserving=False)
        
        result = optorbvqd_instance.compute_energies()

        with self.subTest(msg="test eigenvalue, provided ElectronicStructureProblem, no RDM symmetry exploited"):
            np.testing.assert_array_almost_equal(
                [result.eigenvalues.real], [self.h2_energies], decimal=3
            )



if __name__ == "__main__":
    unittest.main()