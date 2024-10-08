"""Test configuration interaction state initialization methods"""

import sys
sys.path.append("..")
sys.path.append(".")

import unittest

import numpy as np
import torch
from ddt import data, ddt

from qiskit import QuantumCircuit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_aer.primitives import Estimator
from qiskit_nature.second_q.operators.tensor_ordering import to_physicist_ordering
from qiskit_nature.second_q.mappers import JordanWignerMapper

from electronic_structure_algorithms.initializations.configuration_interaction_states import get_CIS_states, get_CISD_states

estimator = Estimator(approximation=True)
mapper = JordanWignerMapper()

driver = PySCFDriver(atom=f'H 0 0 0; H 0 0 {0.735}',
                     charge=0,
                     spin=0,
                     unit=DistanceUnit.ANGSTROM,
                     basis='sto-3g')

es_problem = driver.run()
hamiltonian = mapper.map(es_problem.hamiltonian.second_q_op())
num_particles = es_problem.num_particles

one_body_integrals = torch.from_numpy(np.asarray(es_problem.hamiltonian.electronic_integrals.second_q_coeffs()["+-"].to_dense()))
two_body_integrals = torch.from_numpy(np.asarray(-1*to_physicist_ordering(es_problem.hamiltonian.electronic_integrals.second_q_coeffs()["++--"].to_dense())))

integrals = one_body_integrals, two_body_integrals

@ddt
class TestCI(unittest.TestCase):
    """Test configuration interaction state initialization methods."""

    def setUp(self):
        super().setUp()

        self.k = 3
        self.cis_energies = [-1.83696799, -1.24458455, -0.88272215]
        self.cisd_energies = [-1.85727503, -1.24458455, -0.88272215, -0.22491125]

    @data(integrals)
    def test_CIS_states(self, integrals):

        cis_states = get_CIS_states(one_body_integrals=integrals[0],
                                            two_body_integrals=integrals[1],
                                            num_particles=num_particles,
                                            state_representation='dense')[:self.k]
        
        circs = [QuantumCircuit(4) for n in range(self.k)]
        for n in range(self.k):
            circs[n].initialize(cis_states[n], circs[n].qubits)

        energies = estimator.run(circuits=circs, observables=[hamiltonian, hamiltonian, hamiltonian]).result().values

        with self.subTest(msg="test CIS energies"):
            np.testing.assert_array_almost_equal(
                energies, self.cis_energies, decimal=3
            )

    
    @data(integrals)
    def test_CISD_states(self, integrals):

        cisd_states = get_CISD_states(one_body_integrals=integrals[0],
                                            two_body_integrals=integrals[1],
                                            num_particles=num_particles,
                                            state_representation='dense')[:self.k + 1]
        
        circs = [QuantumCircuit(4) for n in range(self.k + 1)]
        for n in range(self.k + 1):
            circs[n].initialize(cisd_states[n], circs[n].qubits)

        energies = estimator.run(circuits=circs, observables=[hamiltonian, hamiltonian, hamiltonian, hamiltonian]).result().values

        with self.subTest(msg="test CISD energies"):
            np.testing.assert_array_almost_equal(
                energies, self.cisd_energies, decimal=3
            )

if __name__ == "__main__":
    unittest.main()