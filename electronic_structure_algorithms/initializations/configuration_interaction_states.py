import numpy as np
from typing import Optional
from qiskit.quantum_info import Statevector

def count_mismatches(bitstring1: str,
                     bitstring2: str) -> int:
    
    """Determines the number of pairs of orbital occupation mismatches of two
        Slater determinants represented by bitstrings.
        
        Args:
            
            bitstring1, bitstring2: the two Slater determinant bitstrings.
            
        Returns:
        
            The number of pairs of orbital occupation mismatches between
            bitstring1 and bitstring2.
            
    """

    return int(sum(index1 != index2 for index1, index2 in zip(bitstring1, bitstring2))/2)

def get_occupation_indices(bitstring: str):

    """Determines the indices corresponding to occupied orbitals
        in a Slater determinant represented by a bitstring.

        Args:

            bitstring: A bitstring representing a Slater determinant.

        Returns:

            A list of the indices of the orbitals which are occupied
            in the Slater determinant.
    
    """

    indices = []
    for n in range(len(bitstring)):

        if bitstring[-(n+1)] == '1':

            indices.append(n)

    return indices

def get_one_mismatched_orbital_pair(bitstring1: str,
                                    bitstring2: str):
    
    """Given two bitstrings bitstring1 and bitstring2 which
        represent Slater determinants and are known to have two
        pairs of mismatched orbital occupancies, return a tuple 
        where each entry denotes the one orbital index for each
        bitstring involved in one of the mismatched pairs.

        Args:

            bitstring1, bitstring2: The two Slater determinant bitstrings.

        Returns:

            A tuple storing the occupied index involved in a mismatched pair
            for each bitstring.

    """

    for n in range(len(bitstring1)):
        
        if bitstring1[-(n+1)] != bitstring2[-(n+1)]:

            if bitstring1[-(n+1)] == '1':

                first_orbital_occupation = n

            else:

                second_orbital_occuptation = n
    
    return first_orbital_occupation, second_orbital_occuptation

def get_two_mismatched_orbital_pairs(bitstring1: str,
                                     bitstring2: str):
    
    """Given two bitstrings bitstring1 and bitstring2 which represent
        Slater determinants and are known to have two mismatched pairs
        of mismatched orbital occupancies, return a 4-tuple of the 
        occupied indices involved in each pair.

        Args:

            bitstring1, bitstring2: The two Slater determinant bitstrings.

        Returns:

            A 4-tuple of the occupied indices involved in each pair.
    
    """

    state1_occupied_mismatches = []
    state2_occupied_mismatches = []

    for n in range(len(bitstring1)):

        if bitstring1[-(n+1)] != bitstring2[-(n+1)]:

            if bitstring1[-(n+1)] == '1':

                state1_occupied_mismatches.append(n)

            else:

                state2_occupied_mismatches.append(n)

    return min(state1_occupied_mismatches), max(state1_occupied_mismatches), min(state2_occupied_mismatches), max(state2_occupied_mismatches)

def gamma(bitstring: str,
          index: int):
    
    """Given an index and a Slater determinant represented by a bitstring,
        calculated the gamma factor representing the parity of the occupation
        numbers of orbitals 0 through index - 1. For a parity p, the gamma
        factor is given by (-1)^p.

        Args:

            bitstring: The slater determinant bitstring.
            index: The orbital index.

        Returns:

            The gamma factor for the bitstring and orbital index.
    
    """

    if index == 0:

        bitstring = ''

    else:

        bitstring = bitstring[-index:]

    gamma_factor = 1

    for _ in bitstring:
    
        if int(_) == 1:
            gamma_factor *= -1
        else:
            pass

    return gamma_factor     

def get_CIS_states(one_body_integrals: np.ndarray,
                   two_body_integrals: np.ndarray,
                   num_particles: tuple[int, int],
                   state_representation: Optional[str] = 'sparse', # must be either 'sparse' or 'dense'
                   truncation_threshold: Optional[float] = 10**-10):
    
    """Calculates the configuration interaction singles states in either a sparse or dense representation.

        Args:

            one_body_integrals: The one body integrals in the spin-orbital representation.
            two_body_integrals: The two body integrals in the spin-orbital representation in
                the physicist's notation.
            num_particles: A tuple representation the number of alpha and beta electrons.
            state_representation: Either 'sparse' or 'dense'. Setting equal to 'sparse' 
                returns a representation mapping Slater determinant bitstrings to coefficients.
                Setting equal to 'dense' returns a dense statevector.
            truncation_threshold: The threshold for discarding CI coefficients.

        Returns:

            The CIS states.

    """
    num_spin_orbitals = one_body_integrals.shape[0]
    num_molecular_orbitals = int(num_spin_orbitals/2)
    num_alpha = num_particles[0]
    num_beta = num_particles[1]
    
    alpha_string = '0'*(num_molecular_orbitals - num_alpha) + '1'*num_alpha
    beta_string = '0'*(num_molecular_orbitals - num_beta) + '1'*num_beta

    HF_string = alpha_string + beta_string
    HF_alpha_occupied_list = []
    HF_alpha_unoccupied_list = []
    HF_beta_occupied_list = []
    HF_beta_unoccupied_list = []

    possible_alpha_excitations = []
    possible_beta_excitations = []
    excited_bitstrings = [HF_string]

    for n in range(num_molecular_orbitals):

        if alpha_string[-(n+1)] == '1':

            HF_alpha_occupied_list.append(n + num_molecular_orbitals)

        elif alpha_string[-(n+1)] == '0':

            HF_alpha_unoccupied_list.append(n + num_molecular_orbitals)


        if beta_string[-(n+1)] == '1':

            HF_beta_occupied_list.append(n)

        elif beta_string[-(n+1)] == '0':

            HF_beta_unoccupied_list.append(n)


    for occupied_index in HF_alpha_occupied_list:

        for unoccupied_index in HF_alpha_unoccupied_list:

            possible_alpha_excitations.append((unoccupied_index, occupied_index))

    for occupied_index in HF_beta_occupied_list:

        for unoccupied_index in HF_beta_unoccupied_list:

            possible_beta_excitations.append((unoccupied_index, occupied_index))


    for possible_excitation in possible_alpha_excitations + possible_beta_excitations:
        
        excited_bitstring = ''
        for n in range(len(HF_string)):

            if n == possible_excitation[0]:
                excited_bitstring = '1' + excited_bitstring
            elif n == possible_excitation[1]:
                excited_bitstring = '0' + excited_bitstring
            else:
                excited_bitstring = HF_string[-(n+1)] + excited_bitstring

        excited_bitstrings.append(excited_bitstring)

    def get_matrix_element(i: int,
                           j: int):
        
        """Given two indices i and j, compute the Hamiltonian matrix element <i|H|j>,
            where i and j index Slater determinants.
            
            Args:

                i, j: The slater determinant indices.

            Returns:

                The Hamiltonian matrix eleent <i|H|j>.
            """

        ith_state_occupations = get_occupation_indices(excited_bitstrings[i])
        jth_state_occupations = get_occupation_indices(excited_bitstrings[j])

        num_mismatches = count_mismatches(excited_bitstrings[i], excited_bitstrings[j])

        if num_mismatches == 0:
            
            matrix_element = 0

            for index in ith_state_occupations:

                matrix_element += one_body_integrals[index, index]

            for index1 in ith_state_occupations:
                for index2 in jth_state_occupations:

                    matrix_element += two_body_integrals[index1, index2, index1, index2]
                    matrix_element -= two_body_integrals[index1, index2, index2, index1]

        elif num_mismatches == 1:
            
            m, p = get_one_mismatched_orbital_pair(excited_bitstrings[i], excited_bitstrings[j])
            matrix_element = one_body_integrals[m,p]

            for n in ith_state_occupations:

                matrix_element += 2*two_body_integrals[m, n, p, n]
                matrix_element -= 2*two_body_integrals[m, n, n, p]

            matrix_element *= gamma(excited_bitstrings[i], m)*gamma(excited_bitstrings[j], p)

        elif num_mismatches == 2:
            
            m, n, p, q = get_two_mismatched_orbital_pairs(excited_bitstrings[i], excited_bitstrings[j])

            matrix_element = two_body_integrals[m,n,p,q] - two_body_integrals[m,n,q,p]
            matrix_element  *= 2*gamma(excited_bitstrings[i], m)*gamma(excited_bitstrings[i], n)*gamma(excited_bitstrings[j], p)*gamma(excited_bitstrings[j], q)

        elif num_mismatches > 2:
            
            matrix_element = 0
        
        
        return matrix_element

    
    CIS_matrix = np.empty(shape=(len(excited_bitstrings), len(excited_bitstrings)))
    for x in range(len(excited_bitstrings)):
        for y in range(len(excited_bitstrings)):

            CIS_matrix[x,y] = get_matrix_element(x,y)

    CIS_results = np.linalg.eigh(CIS_matrix)
    
    sparse_results = [dict(zip(excited_bitstrings, CIS_results[1].transpose()[n])) for n in range(len(excited_bitstrings))]
    for state in sparse_results:
        for key in state:
            if np.abs(state[key]) <= truncation_threshold:
                state[key] = 0.0
    
    if state_representation == 'sparse':

        return sparse_results

    if state_representation =='dense':

        dense_results = []

        for n, state in enumerate(sparse_results):

            key_counter = 0
            for key in state:

                if key_counter == 0:

                    dense_results.append(state[key]*Statevector.from_label(key))

                else:

                    dense_results[n] += state[key]*Statevector.from_label(key)

                key_counter += 1
            
        for n, state in enumerate(dense_results):

            if not state.is_valid():
            
                norm = np.sqrt(np.sum([elem**2 for elem in state]))
                dense_results[n] = Statevector(np.array([elem/norm for elem in state]))


        return dense_results


def get_CISD_states(one_body_integrals: np.ndarray,
                    two_body_integrals: np.ndarray,
                    num_particles: tuple[int, int],
                    num_spin_orbitals: int,
                    state_representation: Optional[str] = 'sparse',  # must be either 'sparse' or 'dense'
                    truncation_threshold: Optional[float] = 10 ** -10):
    
    """Calculates the configuration interaction singles and doubles (CISD) states in either a sparse
        or dense representation.

        Args:

            one_body_integrals: The one body integrals in the spin-orbital representation.
            two_body_integrals: The two body integrals in the spin-orbital representation in
                the physicist's notation.
            num_particles: A tuple representation the number of alpha and beta electrons.
            state_representation: Either 'sparse' or 'dense'. Setting equal to 'sparse' 
                returns a representation mapping Slater determinant bitstrings to coefficients.
                Setting equal to 'dense' returns a dense statevector.
            truncation_threshold: The threshold for discarding CI coefficients.

        Returns:

            The CIS states.

    """

    num_molecular_orbitals = int(num_spin_orbitals / 2)
    num_alpha = num_particles[0]
    num_beta = num_particles[1]

    alpha_string = '0' * (num_molecular_orbitals - num_alpha) + '1' * num_alpha
    beta_string = '0' * (num_molecular_orbitals - num_beta) + '1' * num_beta

    HF_string = alpha_string + beta_string
    HF_alpha_occupied_list = []
    HF_alpha_unoccupied_list = []
    HF_beta_occupied_list = []
    HF_beta_unoccupied_list = []

    excited_bitstrings = []

    for n in range(num_molecular_orbitals):

        if alpha_string[-(n + 1)] == '1':

            HF_alpha_occupied_list.append(n + num_molecular_orbitals)

        elif alpha_string[-(n + 1)] == '0':

            HF_alpha_unoccupied_list.append(n + num_molecular_orbitals)

        if beta_string[-(n + 1)] == '1':

            HF_beta_occupied_list.append(n)

        elif beta_string[-(n + 1)] == '0':

            HF_beta_unoccupied_list.append(n)

    for n in range(2**num_spin_orbitals):

        bitstr = '0' * (num_spin_orbitals - len(bin(n)[2:])) + bin(n)[2:]
        alpha_bitstr, beta_bitstr = bitstr[:num_molecular_orbitals], bitstr[num_molecular_orbitals:]

        if (alpha_bitstr.count('1') == num_alpha and beta_bitstr.count('1') == num_beta):

            num_alpha_excitations, num_beta_excitations = alpha_bitstr[:-num_alpha].count('1'), beta_bitstr[:-num_beta].count('1')
            if num_alpha_excitations + num_beta_excitations <= 2:

                excited_bitstrings.append(bitstr)


    def get_matrix_element(i: int,
                           j: int):
        
        """Given two indices i and j, compute the Hamiltonian matrix element <i|H|j>,
            where i and j index Slater determinants.
            
            Args:

                i, j: The slater determinant indices.

            Returns:

                The Hamiltonian matrix eleent <i|H|j>.
            """

        ith_state_occupations = get_occupation_indices(excited_bitstrings[i])
        jth_state_occupations = get_occupation_indices(excited_bitstrings[j])

        num_mismatches = count_mismatches(excited_bitstrings[i], excited_bitstrings[j])

        if num_mismatches == 0:
            
            matrix_element = 0

            for index in ith_state_occupations:
                matrix_element += one_body_integrals[index, index]

            for index1 in ith_state_occupations:
                for index2 in jth_state_occupations:
                    matrix_element += two_body_integrals[index1, index2, index1, index2]
                    matrix_element -= two_body_integrals[index1, index2, index2, index1]

        elif num_mismatches == 1:

            m, p = get_one_mismatched_orbital_pair(excited_bitstrings[i], excited_bitstrings[j])
            matrix_element = one_body_integrals[m, p]

            for n in ith_state_occupations:
                matrix_element += 2 * two_body_integrals[m, n, p, n]
                matrix_element -= 2 * two_body_integrals[m, n, n, p]

            matrix_element *= gamma(excited_bitstrings[i], m) * gamma(excited_bitstrings[j], p)

        elif num_mismatches == 2:

            m, n, p, q = get_two_mismatched_orbital_pairs(excited_bitstrings[i], excited_bitstrings[j])

            matrix_element = two_body_integrals[m, n, p, q] - two_body_integrals[m, n, q, p]
            matrix_element *= 2 * gamma(excited_bitstrings[i], m) * gamma(excited_bitstrings[i], n) * gamma(
                excited_bitstrings[j], p) * gamma(excited_bitstrings[j], q)

        elif num_mismatches > 2:

            matrix_element = 0

        return matrix_element

    CISD_matrix = np.empty(shape=(len(excited_bitstrings), len(excited_bitstrings)))
    for x in range(len(excited_bitstrings)):
        for y in range(len(excited_bitstrings)):
            CISD_matrix[x, y] = get_matrix_element(x, y)

    CISD_results = np.linalg.eigh(CISD_matrix)

    sparse_results = [dict(zip(excited_bitstrings, CISD_results[1].transpose()[n])) for n in
                      range(len(excited_bitstrings))]
    for state in sparse_results:
        for key in state:
            if np.abs(state[key]) <= truncation_threshold:
                state[key] = 0.0

    if state_representation == 'sparse':

        return sparse_results

    elif state_representation == 'dense':

        dense_results = []

        for n, state in enumerate(sparse_results):

            key_counter = 0
            for key in state:

                if key_counter == 0:

                    dense_results.append(state[key] * Statevector.from_label(key))

                else:

                    dense_results[n] += state[key] * Statevector.from_label(key)

                key_counter += 1

        for n, state in enumerate(dense_results):

            if not state.is_valid():

                norm = np.sqrt(np.sum([elem**2 for elem in dense_results[n]]))
                dense_results[n] = Statevector(np.array([elem/norm for elem in dense_results[n]]))

        return dense_results


