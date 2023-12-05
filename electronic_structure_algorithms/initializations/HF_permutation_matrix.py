import torch

def get_HF_permutation_matrix(num_original_spin_orbitals: int,
                           num_spin_orbitals: int):

    num_original_molecular_orbitals = int(num_original_spin_orbitals/2)
    num_molecular_orbitals = int(num_spin_orbitals/2)
    
    initial_partial_unitary_guess = torch.zeros(size=(num_original_molecular_orbitals, num_molecular_orbitals), dtype=torch.float64)
    for n in range(int(num_molecular_orbitals)):
        initial_partial_unitary_guess[n,n] = 1.0

    return initial_partial_unitary_guess

