"""Test OptOrbVQE"""
import sys
sys.path.append("..")

import unittest

import numpy as np
import torch
from ddt import data, ddt

from qiskit_algorithms.optimizers import L_BFGS_B

import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_aer.primitives import Estimator
#from qiskit_aer.primitives import EstimatorV2
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.operators.tensor_ordering import to_physicist_ordering

from electronic_structure_algorithms.orbital_optimization import PartialUnitaryProjectionOptimizer, OptOrbVQE

estimator = Estimator(approximation=True)
#estimator = EstimatorV2()
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

HF_state = HartreeFock(qubit_mapper=mapper,
                       num_spatial_orbitals=int(num_reduced_qubits/2),
                       num_particles=num_particles)

uccsd = UCCSD(qubit_mapper=mapper,
               num_spatial_orbitals=int(num_reduced_qubits/2),
               num_particles=num_particles,
               initial_state=HF_state)

@ddt
class TestOptOrbVQE(unittest.TestCase):
    """Test OptOrbVQE"""

    def setUp(self):
        super().setUp()

        self.HF_state = HartreeFock(qubit_mapper=mapper,
                       num_spatial_orbitals=int(num_reduced_qubits/2),
                       num_particles=num_particles)
        
        self.uccsd = UCCSD(qubit_mapper=mapper,
               num_spatial_orbitals=int(num_reduced_qubits/2),
               num_particles=num_particles,
               initial_state=HF_state)

        self.h2_energy = -1.8661038079694765
        self.estimator = Estimator(approximation=True)
        #self.estimator = EstimatorV2()

    @data(integrals)
    def test_ground_state_integrals(self, integrals):

        partial_unitary_optimizer = PartialUnitaryProjectionOptimizer(initial_BBstepsize=10**-3,
                                                              stopping_tolerance=10**-5,
                                                              maxiter=10000,
                                                              gradient_method='autograd')
        
        vqe_instance = VQE(ansatz=self.uccsd,
                initial_point=np.zeros(self.uccsd.num_parameters),
                optimizer=L_BFGS_B(),
                estimator=estimator)
        
        optorbvqe_instance = OptOrbVQE(problem=None,
                               integral_tensors=integrals,
                               num_spin_orbitals=num_reduced_qubits,
                               ground_state_solver=vqe_instance,
                               mapper=mapper,
                               estimator=estimator,
                               partial_unitary_optimizer=partial_unitary_optimizer,
                               maxiter=20,
                               wavefuntion_real=True,
                               spin_conserving=True)
        
        result = optorbvqe_instance.compute_minimum_energy()

        with self.subTest(msg="test eigenvalue, provided integrals"):
            np.testing.assert_array_almost_equal(
                [result.eigenvalue.real], [self.h2_energy], decimal=3
            )


    @data(es_problem)
    def test_ground_state_es_problem(self, problem):

        partial_unitary_optimizer = PartialUnitaryProjectionOptimizer(initial_BBstepsize=10**-3,
                                                              stopping_tolerance=10**-5,
                                                              maxiter=10000,
                                                              gradient_method='autograd')
        
        vqe_instance = VQE(ansatz=self.uccsd,
                initial_point=np.zeros(self.uccsd.num_parameters),
                optimizer=L_BFGS_B(),
                estimator=estimator)
        
        optorbvqe_instance = OptOrbVQE(problem=problem,
                               integral_tensors = None,
                               num_spin_orbitals=num_reduced_qubits,
                               ground_state_solver=vqe_instance,
                               mapper=mapper,
                               estimator=estimator,
                               partial_unitary_optimizer=partial_unitary_optimizer,
                               maxiter=20,
                               wavefuntion_real=True,
                               spin_conserving=True)
        
        result = optorbvqe_instance.compute_minimum_energy()

        with self.subTest(msg="test eigenvalue, provided ElectronicStructureProblem"):
            np.testing.assert_array_almost_equal(
                [result.eigenvalue.real], [self.h2_energy], decimal=3
            )

    
    @data(integrals)
    def test_ground_state_integrals_2(self, integrals):

        partial_unitary_optimizer = PartialUnitaryProjectionOptimizer(initial_BBstepsize=10**-3,
                                                              stopping_tolerance=10**-5,
                                                              maxiter=10000,
                                                              gradient_method='autograd')
        
        vqe_instance = VQE(ansatz=self.uccsd,
                initial_point=np.zeros(self.uccsd.num_parameters),
                optimizer=L_BFGS_B(),
                estimator=estimator)
        
        optorbvqe_instance = OptOrbVQE(problem=None,
                               integral_tensors=integrals,
                               num_spin_orbitals=num_reduced_qubits,
                               ground_state_solver=vqe_instance,
                               mapper=mapper,
                               estimator=estimator,
                               partial_unitary_optimizer=partial_unitary_optimizer,
                               maxiter=20,
                               wavefuntion_real=False,
                               spin_conserving=False)
        
        result = optorbvqe_instance.compute_minimum_energy()

        with self.subTest(msg="test eigenvalue, provided integrals, no RDM symmetry exploited"):
            np.testing.assert_array_almost_equal(
                [result.eigenvalue.real], [self.h2_energy], decimal=3
            )


    @data(es_problem)
    def test_ground_state_es_problem_2(self, problem):

        partial_unitary_optimizer = PartialUnitaryProjectionOptimizer(initial_BBstepsize=10**-3,
                                                              stopping_tolerance=10**-5,
                                                              maxiter=10000,
                                                              gradient_method='autograd')
        
        vqe_instance = VQE(ansatz=self.uccsd,
                initial_point=np.zeros(self.uccsd.num_parameters),
                optimizer=L_BFGS_B(),
                estimator=estimator)
        
        optorbvqe_instance = OptOrbVQE(problem=problem,
                               integral_tensors = None,
                               num_spin_orbitals=num_reduced_qubits,
                               ground_state_solver=vqe_instance,
                               mapper=mapper,
                               estimator=estimator,
                               partial_unitary_optimizer=partial_unitary_optimizer,
                               maxiter=20,
                               wavefuntion_real=False,
                               spin_conserving=False)
        
        result = optorbvqe_instance.compute_minimum_energy()

        with self.subTest(msg="test eigenvalue, provided ElectronicStructureProblem, no RDM symmetry exploited"):
            np.testing.assert_array_almost_equal(
                [result.eigenvalue.real], [self.h2_energy], decimal=3
            )



if __name__ == "__main__":
    unittest.main()