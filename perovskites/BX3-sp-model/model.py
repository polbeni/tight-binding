# Pol Benítez Colominas, September 2024
# Universitat Politècnica de Catalunya and University of Cambridge

# Tight binding model for a BX_3 perovskite system with s and p electrons for both B and X atoms
# this model is based in the following work: https://doi.org/10.1021/acs.jpclett.6b01749

import numpy as np
import matplotlib.pyplot as plt


# let's give value to the parameters, based in the empyrical values provided in the paper
E_s0 = -9.01          # s electron of B
E_s1 = -13.01         # s electron of X
E_p0 = 2.34           # p electron of B
E_p1 = -1.96          # p electron of X

V_ss = -1.10          # s(B) and s(X)
V_s0p1 = 1.19         # s(B) and p(X)
V_s1p0 = 0.70         # p(B) and s(X)
V_pp_sigma = -3.65    # pp bonding
V_pp_pi = 0.55        # pp anti-bonding

# create a list for orbitals s and p (x, y, z), and for atoms 0 (for B) and 1, 2, 3 (for X)
orbitals = ['s', 'x', 'y', 'z']
atoms = [0, 1, 2, 3]

# let's define a function that computes each hamiltionian element
def hamiltonian_element(orbital_1, atom_1, pos_1, orbital_2, atom_2, pos_2, pos_2_pbc, k_point):
    """
    Computes the hamiltonian elements for a given k-point, the diagonal terms reffer to the each atom hamiltonian (energy)
    while the non-diagonal terms account for the interaction between two orbitals

    Inputs:
        orbital_1 -> electron orbital of the first atom
        atom_1 -> atom corresponding to the first atom
        pos_1 -> position for the first atom
        orbital_2 -> electron orbital of the second atom
        atom_2 -> atom corresponding to the second atom
        pos_2 -> position for the second atom
        pos_2_pbc -> position of the second atom in periodic boundary conditions
        k_point -> point in the reciprocal space
    """
        
    # non-diagonal elements (interaction elements)
    distance = pos_1 - pos_2
    distance_pbc = pos_1 - pos_2_pbc

    # s-s orbitals
    if (orbital_1 == 's') and (orbital_2 == 's') and (atom_1 == 0) and (atom_2 != 0):
        element = V_ss * (np.exp(1j * np.dot(k_point, distance)) + np.exp(1j * np.dot(k_point, distance_pbc)))
    # s-p orbitals
    elif (orbital_1 == 's') and (orbital_2 != 's') and (atom_1 == 0) and (atom_2 != 0):
        if (orbital_2 == 'x' and atom_2 == 1) or (orbital_2 == 'y' and atom_2 == 2) or (orbital_2 == 'z' and atom_2 == 3):
            element = V_s0p1 * (np.exp(1j * np.dot(k_point, distance)) - np.exp(1j * np.dot(k_point, distance_pbc)))
        else:
            element = 0
    # p-s orbitals
    elif (orbital_1 != 's') and (orbital_2 == 's') and (atom_1 == 0) and (atom_2 != 0):
        if (orbital_1 == 'x' and atom_2 == 1) or (orbital_1 == 'y' and atom_2 == 2) or (orbital_1 == 'z' and atom_2 == 3):
            element = -V_s0p1 * (np.exp(1j * np.dot(k_point, distance)) - np.exp(1j * np.dot(k_point, distance_pbc)))
        else:
            element = 0
    # p-p orbitals
    elif (orbital_1 != 's') and (orbital_2 != 's') and (atom_1 == 0) and (atom_2 != 0):
        if (orbital_1 == 'x') and (orbital_2 == 'x'):
            if atom_2 == 1:
                element = V_pp_sigma * (np.exp(1j * np.dot(k_point, distance)) + np.exp(1j * np.dot(k_point, distance_pbc)))
            else:
                element = V_pp_pi * (np.exp(1j * np.dot(k_point, distance)) + np.exp(1j * np.dot(k_point, distance_pbc)))
        elif (orbital_1 == 'y') and (orbital_2 == 'y'):
            if atom_2 == 2:
                element = V_pp_sigma * (np.exp(1j * np.dot(k_point, distance)) + np.exp(1j * np.dot(k_point, distance_pbc)))
            else:
                element = V_pp_pi * (np.exp(1j * np.dot(k_point, distance)) + np.exp(1j * np.dot(k_point, distance_pbc)))
        elif (orbital_1 == 'z') and (orbital_2 == 'z'):
            if atom_2 == 3:
                element = V_pp_sigma * (np.exp(1j * np.dot(k_point, distance)) + np.exp(1j * np.dot(k_point, distance_pbc)))
            else:
                element = V_pp_pi * (np.exp(1j * np.dot(k_point, distance)) + np.exp(1j * np.dot(k_point, distance_pbc)))
        else:
            element = 0
    # otherwise
    else:
        element = 0

    # diagonal elements
    if (orbital_1 == orbital_2) and (atom_1 == atom_2):
        if (orbital_1 == 's') and (atom_1 == 0):
            element = E_s0
        elif (orbital_1 == 's') and (atom_1 != 0):
            element = E_s1
        elif (orbital_1 != 's') and (atom_1 == 0):
            element = E_p0
        elif (orbital_1 != 's') and (atom_1 != 0):
            element = E_p1

    return element

# define atom positions
lattice_parameter = 3.8
pos_B = np.array([0, 0, 0])
pos_X1 = np.array([0.5 * lattice_parameter, 0, 0])
pos_X2 = np.array([0, 0.5 * lattice_parameter, 0])
pos_X3 = np.array([0, 0, 0.5 * lattice_parameter])

# now we can compute determine the hamiltonian for a given k-point
num_elements = 16 # total number of electronic orbitals

def hamiltonian_k(k_point, num_elements):
    """
    Determines the hamiltonian for a given k-point in the reciprocal space

    Inputs:
        k_point -> k-point in the reciprocal space
        num_elements -> total number of electronic orbitals
    """
    hamiltonian = np.zeros((num_elements, num_elements), dtype=complex)

    n_index = 0
    m_index = 0
    for orb1 in orbitals:
        for atom1 in atoms:
            n_state = [orb1, atom1]
            for orb2 in orbitals:
                for atom2 in atoms:
                    m_state = [orb2, atom2]

                    if atom1 == 0:
                        pos1 = pos_B
                    elif atom1 == 1:
                        pos1 = pos_X1
                    elif atom1 == 2:
                        pos1 = pos_X2
                    elif atom1 == 3:
                        pos1 = pos_X3

                    if atom2 == 0:
                        pos2 = pos_B
                    elif atom2 == 1:
                        pos2 = pos_X1
                    elif atom2 == 2:
                        pos2 = pos_X2
                    elif atom2 == 3:
                        pos2 = pos_X3

                    if (atom1 == 0) and (atom2 == 1):
                        pos2_pbc =  np.array([pos_X1[0] - lattice_parameter, pos_X1[1], pos_X1[2]])
                    elif (atom1 == 0) and (atom2 == 2):
                        pos2_pbc =  np.array([pos_X2[0], pos_X2[1] - lattice_parameter, pos_X2[2]])
                    elif (atom1 == 0) and (atom2 == 3):
                        pos2_pbc =  np.array([pos_X3[0], pos_X3[1], pos_X3[2] - lattice_parameter])
                    elif (atom1 == 1) and (atom2 == 0):
                        pos2_pbc =  np.array([pos_B[0] + lattice_parameter, pos_B[1], pos_B[2]])
                    elif (atom1 == 2) and (atom2 == 0):
                        pos2_pbc =  np.array([pos_B[0], pos_B[1] + lattice_parameter, pos_B[2]])
                    elif (atom1 == 3) and (atom2 == 0):
                        pos2_pbc =  np.array([pos_B[0], pos_B[1], pos_B[2] + lattice_parameter])
                    else:
                        pos2_pbc = np.array([0, 0, 0])
                    
                    hamiltonian[n_index, m_index] = hamiltonian_element(n_state[0], n_state[1], pos1, m_state[0], m_state[1], pos2, pos2_pbc, k_point)

                    m_index = m_index + 1
            
            n_index = n_index + 1
            m_index = 0

    # down-diagonal elements (complex conjugate of )
    n_index = 0
    m_index = 0
    for orb1 in orbitals:
        for atom1 in atoms:
            n_state = [orb1, atom1]
            for orb2 in orbitals:
                for atom2 in atoms:
                    m_state = [orb2, atom2]

                    if n_index > m_index:
                        hamiltonian[n_index, m_index] = hamiltonian[m_index, n_index]

                    m_index = m_index + 1
            
            n_index = n_index + 1
            m_index = 0

    return hamiltonian

# and we can create another function that diagonalizes the hamiltonian and returns the eigenvalues sorted
def eigenvalues_hamiltonian(hamiltonian):
    """
    Determine the eigenvalues of a given hamiltonian

    Inputs:
        hamiltonian -> the hamiltonian we want to diagonalise
    """

    eigenvalues = np.linalg.eigvalsh(hamiltonian)

    return eigenvalues

# we can define a function to generate a path in the reciprocal space
def create_k_path(sym_points, step, lattice_parameter):
    """
    Creates a path in the reciprocal space crossing the desired symmetry points

    Inputs:
        sym_points: points of symmetry in the reciprocal space orderer in the path we want to follow (give the values without b)
        step: distance of each step in the reciprocal space
        lattice_parameter: lattice parameter in the real space to determine the lattice constants in the reciprocal space (cubic system assumed)
    Outputs:
        path_points: an array with all the k-points in the path
        position_of_sym_points: an array with the index of each symmetry point
    """

    reciprocal_lat_param = (2 * np.pi) / lattice_parameter

    path_points = []

    total_number_of_points = 0
    position_of_sym_points = []

    for point_it in range(len(sym_points) - 1):
        initial_point = sym_points[point_it] * reciprocal_lat_param
        final_point = sym_points[point_it + 1] * reciprocal_lat_param

        distance = np.linalg.norm(final_point - initial_point)

        num_steps = int(distance / step)

        if point_it == 0:
            path_points.append(initial_point)

            total_number_of_points = total_number_of_points + 1
            position_of_sym_points.append(total_number_of_points)

        num_dif_components = 0
        if initial_point[0] != final_point[0]:
            num_dif_components = num_dif_components + 1
        if initial_point[1] != final_point[1]:
            num_dif_components = num_dif_components + 1
        if initial_point[2] != final_point[2]:
            num_dif_components = num_dif_components + 1

        actual_point = initial_point
        for nstep in range(num_steps):
            if num_dif_components == 1:
                if initial_point[0] != final_point[0]:
                    if initial_point[0] > final_point[0]:
                        step_apply = -step
                    else:
                        step_apply = step
                    actual_point = np.array([actual_point[0] + step_apply, actual_point[1], actual_point[2]])

                elif initial_point[1] != final_point[1]:
                    if initial_point[1] > final_point[1]:
                        step_apply = -step
                    else:
                        step_apply = step
                    actual_point = np.array([actual_point[0], actual_point[1] + step_apply, actual_point[2]])

                elif initial_point[2] != final_point[2]:
                    if initial_point[2] > final_point[2]:
                        step_apply = -step
                    else:
                        step_apply = step
                    actual_point = np.array([actual_point[0], actual_point[1], actual_point[2] + step_apply])

            elif num_dif_components == 2:
                if initial_point[0] != final_point[0]:
                    if initial_point[0] > final_point[0]:
                        step_apply = -step * (1 / np.sqrt(2))
                    else:
                        step_apply = step * (1 / np.sqrt(2))
                    actual_point = np.array([actual_point[0] + step_apply, actual_point[1], actual_point[2]])

                if initial_point[1] != final_point[1]:
                    if initial_point[1] > final_point[1]:
                        step_apply = -step * (1 / np.sqrt(2))
                    else:
                        step_apply = step * (1 / np.sqrt(2))
                    actual_point = np.array([actual_point[0], actual_point[1] + step_apply, actual_point[2]])

                if initial_point[2] != final_point[2]:
                    if initial_point[2] > final_point[2]:
                        step_apply = -step * (1 / np.sqrt(2))
                    else:
                        step_apply = step * (1 / np.sqrt(2))
                    actual_point = np.array([actual_point[0], actual_point[1], actual_point[2] + step_apply])

            elif num_dif_components == 3:
                if initial_point[0] != final_point[0]:
                    if initial_point[0] > final_point[0]:
                        step_apply = -step * (1 / np.sqrt(3))
                    else:
                        step_apply = step * (1 / np.sqrt(3))
                    actual_point = np.array([actual_point[0] + step_apply, actual_point[1], actual_point[2]])

                if initial_point[1] != final_point[1]:
                    if initial_point[1] > final_point[1]:
                        step_apply = -step * (1 / np.sqrt(3))
                    else:
                        step_apply = step * (1 / np.sqrt(3))
                    actual_point = np.array([actual_point[0], actual_point[1] + step_apply, actual_point[2]])

                if initial_point[2] != final_point[2]:
                    if initial_point[2] > final_point[2]:
                        step_apply = -step * (1 / np.sqrt(3))
                    else:
                        step_apply = step * (1 / np.sqrt(3))
                    actual_point = np.array([actual_point[0], actual_point[1], actual_point[2] + step_apply])
            
            if nstep == (num_steps - 1):
                actual_point = final_point

                total_number_of_points = total_number_of_points + 1
                position_of_sym_points.append(total_number_of_points)
            else:
                total_number_of_points = total_number_of_points + 1
            
            path_points.append(actual_point)

    return path_points, position_of_sym_points

# now we can define the path we are interested
sym_points = [np.array([0, 0, 0]), np.array([0, 0.5, 0]), np.array([0.5, 0.5, 0]), np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5]), np.array([0, 0.5, 0])]
name_sym_points = ['$\\Gamma$', 'X', 'M', '$\\Gamma$', 'R', 'X']
k_path, position_sym_points = create_k_path(sym_points, 0.01, lattice_parameter)

# let's compute and save the bands for each k-point
bands = []

for k_point in k_path:
    hamiltonian = hamiltonian_k(k_point, num_elements)
    eigenvalues = eigenvalues_hamiltonian(hamiltonian)
    bands.append(eigenvalues)

correct_bands = []
for x in range(num_elements):
    band = []
    for y in range(position_sym_points[-1]):
        band.append(bands[y][x])
    correct_bands.append(band)

# finally let's plot the bands result
fig, ax = plt.subplots(figsize=(4, 3))
ax.set_title('Tight Binding for BX$_3$ with $sp$ orbitals')
ax.set_ylabel('Energy (eV)')

k_points = np.linspace(1, position_sym_points[-1], position_sym_points[-1])
for band in correct_bands:
    ax.plot(k_points, band, color='navy')

ax.set_xlim(0, position_sym_points[-1])

ax.set_xticks(ticks=position_sym_points, labels=name_sym_points)

plt.tight_layout()
plt.savefig('bands.pdf')