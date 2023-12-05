![ecosystem](https://raw.githubusercontent.com/qiskit-community/ecosystem/main/badges/qiskit-ecosystem_template.svg)

# Electronic Structure Orbital Optimization

Orbital optimization refers to a class of methods in quantum chemistry where the single-particle wavefunctions (the orbitals) are mapped via some parameterized transformation onto a new set of orbitals. An eigensolver is then used to solve for the ground and/or excited states in this transformed basis. The solution of this eigensolver is then used to re-initialize the basis transformation to re-optimize the orbitals. This interleaving of orbital optimization and eigensolver subproblems is repeated until some outer loop stopping condition is reached. This repository contains implementations of the OptOrbVQE algorithm (![arxiv2208.14431](https://arxiv.org/abs/2208.14431v2)) and its excited states variants (![arxiv2310.09418](https://arxiv.org/abs/2310.09418)). In these methods, the orbital transformation takes the form of a partial unitary matrix and quantum eigensolvers such as VQE, SSVQE, MCVQE, and VQD are used in between orbital optimization runs. The starting basis is chosen to be one with a large number $M$ of orbitals (e.g. cc-pVTZ, cc-pVQZ, cc-pV5Z, ect...) and compressing the active space to one of size $N < M$. Thus, the orbital optimization searches for an optimal transformation in the space of $M \times N$ real partial unitaries. Currently, orbital optimization supports the VQE and AdaptVQE ground state solvers and the SSVQE, MCVQE, and VQD excited states solvers. Additional eigensolvers such as QPE, QITE, and VarQITE could be added in the future.

## Installation

To install this project, run:

```
git clone https://github.com/JoelHBierman/electronic-structure-orbital-optimization.git
cd electronic-structure-orbital-optimization-main
pip install .
```

## OptOrbVQE Code Example

This is an example of how a user might use this orbital optimization scheme in combination with VQE to find the orbital-optimized ground state energy.

```
import numpy as np
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import L_BFGS_B, COBYLA
from qiskit_aer.primitives import Estimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

from electronic_structure_algorithms.orbital_optimization import PartialUnitaryProjectionOptimizer, OptOrbVQE

estimator = Estimator(approximation=True)
mapper=JordanWignerMapper()

# Initialize an ElectronicStructureProblem with a large basis.
# Here we use cc-pVTZ.
driver = PySCFDriver(atom=f'H 0 0 0; H 0 0 {0.735}',
                     charge=0,
                     spin=0,
                     unit=DistanceUnit.ANGSTROM,
                     basis='cc-pVTZ')

q_molecule = driver.run()
num_particles = q_molecule.num_particles

# The size of the active space for OptOrbVQE to use.
num_reduced_spin_orbitals = 4

# Choose an initialization and an ansatz for VQE.
# Here we initalize the UCCSD ansatz in the Hartree-Fock state.
HF_state = HartreeFock(qubit_mapper=mapper,
                       num_spatial_orbitals=int(num_reduced_spin_orbitals/2),
                       num_particles=num_particles)

ansatz = UCCSD(qubit_mapper=mapper,
               num_spatial_orbitals=int(num_reduced_spin_orbitals/2),
               num_particles=num_particles,
               initial_state=HF_state)

# Initialize the optimizer for the orbital optimization subroutine.
partial_unitary_optimizer = PartialUnitaryProjectionOptimizer(initial_BBstepsize=10**-3,
                                                              stopping_tolerance=10**-5,
                                                              maxiter=10000,
                                                              gradient_method='autograd')

# Initialize the VQE solver.
vqe_instance = VQE(ansatz=ansatz,
                   initial_point=np.zeros(ansatz.num_parameters),
                   optimizer=L_BFGS_B(maxfun=10**6, maxiter=10**6),
                   estimator=estimator)

# Initialize the OptOrbVQE solver.
optorbvqe_instance = OptOrbVQE(num_spin_orbitals=num_reduced_spin_orbitals,
                               ground_state_solver=vqe_instance,
                               mapper=mapper,
                               estimator=estimator,
                               partial_unitary_optimizer=partial_unitary_optimizer,
                               problem=q_molecule,
                               maxiter=20,
                               wavefuntion_real=True,
                               spin_conserving=True)

ground_state_energy_result = optorbvqe_instance.compute_minimum_energy()
print(f'Orbital-optimized ground state energy: {ground_state_energy_result.eigenvalue} Ha (4 spin-orbitals)')
print(f'Ground state energy in STO-3G basis: {-1.85727503} Ha (4 spin-orbitals)')
print(f'Ground state energy in cc-pVTZ basis: {-1.89226657} Ha (56 spin-orbitals)')

```

We can see that OptOrbVQE greatly improves upon the VQE ground state energy, despite using the same number of qubits, however there is still room to improve, as evidenced by the fact that the energy in the full cc-pVTZ active space is still much more accurate. The accuracy of OptOrbVQE can be systematically improved
by increasing the value of `num_reduced_spin_orbitals` at the cost of more qubits being required.
