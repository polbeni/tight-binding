# Pol Benítez Colominas, September 2024
# Universitat Politècnica de Catalunya and University of Cambridge

# Tight binding model for a pervskite 3D system

import numpy as np
import matplotlib.pyplot as plt

def t_correction(t_nm, distance_original, distance_distorted):
    """
    Correction to the hopping therms with the distance between atoms (related with the original distance)

    Inputs:
        t_nm: original hopping value for the hamiltonian component nm
        distance_original: original distance (euclidean) between atoms (because symmetry of the problem will always be the same)
        distance_distorted: the new distance (euclidean) between the orbitals n and m (or n and m')
    """

    fraction_of_change = distance_distorted / distance_original
    change = 1.75 * np.exp(fraction_of_change - 1)

    t_nm_d = t_nm * change

    return t_nm_d

def h_nm_element(t_nm, k_point, R_n, R_m1, R_m2, is_TiO):
    """
    Computes the n, m element of the Hamiltonian of the system, and implements the correction with the distance

    Inputs:
        t_nm: hopping integral or self-energy
        k_point: k point in the reciprocal space (3D)
        R_n: cartesian position of the atom n (3D)
        R_m1: cartesian position of the atom m (3D), R_m2: for periodicity
        is_TiO: boolean, if n and m are orbital of Ti and O (True) or not (False)
                if True, then the distance correction will be applied
    """

    if is_TiO == True:
        original_distance = 0.5

        diff_R = R_n - R_m1
        escalar1 = np.dot(k_point, diff_R)
        dist1 = np.linalg.norm(diff_R)
        corrected_t1 = t_correction(t_nm, original_distance, dist1)

        diff_R = R_n - R_m2
        escalar2 = np.dot(k_point, diff_R)
        dist2 = np.linalg.norm(diff_R)
        corrected_t2 = t_correction(t_nm, original_distance, dist2)


        element = 0.5 * ((corrected_t1 * np.exp(1j * escalar1)) + (corrected_t2 * np.exp(1j * escalar2)))

    else:
        diff_R = R_n - R_m1
        escalar1 = np.dot(k_point, diff_R)

        diff_R = R_n - R_m2
        escalar2 = np.dot(k_point, diff_R)

        element = t_nm * 0.5 * (np.exp(1j * escalar1) + np.exp(1j * escalar2))

    return element

def t_nm_element(n, m):
    """
    Generates the t_nm element (empyrical (and qualitative) hopping integral element) for the states n and m.
    The states are px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, dxy4, dxz4, dyz4, dx2-y24, dz24
    If n == m, the function returns the energy (the same for all the electrons)
    """

    #energy = [-4.0, -4.0, -4.0, -3.5, -3.5, -3.5, -3.0, -3.0, -3.0, -2, -2, -2, -2, -2]
    energy = [-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -2, -2, -2, -2, -2]
    #energy = [0]*14
    t_pp_bonding = 4
    t_pp_antibonding = -1
    t_pd_bonding = -1.6
    t_pd_partially = -1.0
    t_pd_antibonding = 2
    
    if n == m:
        t_nm = energy[n]
    else:
        if (abs(n - m) == 3 or abs(n - m) == 6) and n <= 8 and m <= 8:
            t_nm = t_pp_bonding
        elif (abs(n - m) != 3 and abs(n - m) != 6) and n <= 8 and m <= 8:
            t_nm = t_pp_antibonding
        elif ((n == 0 or n == 3 or n == 6) and (m == 9 or m == 10 or m == 12)) or ((m == 0 or m == 3 or m == 6) and (n == 9 or n == 10 or n == 12)):
            t_nm = t_pd_partially
        elif ((n == 1 or n == 4 or n == 7) and (m == 9 or m == 11 or m == 12)) or ((m == 1 or m == 4 or m == 7) and (n == 9 or n == 11 or n == 12)):
            t_nm = t_pd_partially
        elif ((n == 2 or n == 5 or n == 8) and (m == 10 or m == 11)) or ((m == 2 or m == 5 or m == 8) and (n == 10 or n == 11)):
            t_nm = t_pd_partially
        elif ((n == 2 or n == 5 or n == 8) and (m == 13)) or ((m == 2 or m == 5 or m == 8) and (n == 13)):
            t_nm = t_pd_bonding
        else:
            t_nm = t_pd_antibonding

        if (abs(n - m) == 1 or abs(n - m) == 2) and (n <= 8 and m <= 8):
            t_nm = 0
        
        if (n != m) and (n > 8 and m >8):
            t_nm = 0

    return t_nm


# define the path in the reciprocal space
band_points = 25
k_intervals_up = np.linspace(0, np.pi, band_points)
k_intervals_down = np.linspace(np.pi, 0, band_points)
k_path = []
# gamma to X
for x in range(band_points):
    k_path.append(np.array([0, k_intervals_up[x], 0]))
# X to M
for x in range(band_points - 1):
    k_path.append(np.array([k_intervals_up[x + 1], np.pi, 0]))
# M to gamma
for x in range(band_points - 1):
    k_path.append(np.array([k_intervals_down[x + 1], k_intervals_down[x + 1], 0]))
# gamma to R
for x in range(band_points - 1):
    k_path.append(np.array([k_intervals_up[x + 1], k_intervals_up[x + 1], k_intervals_up[x + 1]]))
# R to X
for x in range(band_points - 1):
    k_path.append(np.array([k_intervals_down[x + 1], np.pi, k_intervals_down[x + 1]]))

num_k_points = 121
k_points = np.linspace(0, 1, num_k_points)

# create hopping integral matrix and evaluate it
num_elements = 14 # total number of electronic orbitals

hopping_matrix = np.zeros((num_elements, num_elements))
for n in range(num_elements):
    for m in range(num_elements):
        hopping_matrix[n, m] = t_nm_element(n, m)


#### ORIGINAL STRUCTURE ####

# define the positions of the atoms (let's assume the cubic unit cell has size 1 (arbitraty units))
R_O1 = np.array([0.5, 0.5, 0])
R_O2 = np.array([0.5, 0, 0.5])
R_O3 = np.array([0, 0.5, 0.5])
R_Ti = np.array([0.5, 0.5, 0.5])
R_O4 = np.array([0.5, 0.5, 1])
R_O5 = np.array([0.5, 1, 0.5])
R_O6 = np.array([1, 0.5, 0.5])

# create the hamiltionan matrix
hamiltonian = np.zeros((num_elements, num_elements), dtype=complex)

# define the tensor with the energy bands
bands = []

# generate all the hamiltonian elements
for k_point in k_path:
    for n in range(num_elements):
        if n in [0, 1, 2]:
            R_n = R_O1
        elif n in [3, 4, 5]:
            R_n = R_O2
        elif n in [6, 7, 8]:
            R_n = R_O3
        else:
            R_n = R_Ti

        for m in range(num_elements):
            if m in [0, 1, 2]:
                R_m1 = R_O1
                R_m2 = R_O4
            elif m in [3, 4, 5]:
                R_m1 = R_O2
                R_m2 = R_O5
            elif m in [6, 7, 8]:
                R_m1 = R_O3
                R_m2 = R_O6
            else:
                R_m1 = R_Ti
                if n in [0, 1, 2]:
                    R_m2 = np.array([0.5, 0.5, -0.5])
                elif n in [3, 4, 5]:
                    R_m2 = np.array([0.5, -0.5, 0.5])
                elif n in [6, 7, 8]:
                    R_m2 = np.array([-0.5, 0.5, 0.5])

            if n == m:
                hamiltonian[n, m] = hopping_matrix[n, m]
            else:
                if (n in [0, 1, 2, 3, 4, 5, 6, 7, 8] and m in [9, 10, 11, 12, 13, 14]) or (m in [0, 1, 2, 3, 4, 5, 6, 7, 8] and n in [9, 10, 11, 12, 13, 14]):
                    is_TiO = True
                else:
                    is_TiO = False

                hamiltonian[n, m] = h_nm_element(hopping_matrix[n, m], k_point, R_n, R_m1, R_m2, is_TiO)

            # compute the complex conjugate for the elements below the diagonal
            if m < n:
                hamiltonian[n, m] = np.conjugate(hamiltonian[n, m])

    bands.append(np.linalg.eigvalsh(hamiltonian))

correct_bands = []
for x in range(14):
    band = []
    for y in range(121):
        band.append(bands[y][x])
    correct_bands.append(band)


#### DISTORTED STRUCTURE ####

# define the positions of the atoms (let's assume the cubic unit cell has size 1 (arbitraty units))
R_O1 = np.array([0.5, 0.5, 0])
R_O2 = np.array([0.5, 0, 0.5])
R_O3 = np.array([0, 0.5, 0.5])
R_Ti = np.array([0.5, 0.5, 0.25])
R_O4 = np.array([0.5, 0.5, 1])
R_O5 = np.array([0.5, 1, 0.5])
R_O6 = np.array([1, 0.5, 0.5])

# create the hamiltionan matrix
hamiltonian = np.zeros((num_elements, num_elements), dtype=complex)

# define the tensor with the energy bands
bands_distorted = []

# generate all the hamiltonian elements
for k_point in k_path:
    for n in range(num_elements):
        if n in [0, 1, 2]:
            R_n = R_O1
        elif n in [3, 4, 5]:
            R_n = R_O2
        elif n in [6, 7, 8]:
            R_n = R_O3
        else:
            R_n = R_Ti

        for m in range(num_elements):
            if m in [0, 1, 2]:
                R_m1 = R_O1
                R_m2 = R_O4
            elif m in [3, 4, 5]:
                R_m1 = R_O2
                R_m2 = R_O5
            elif m in [6, 7, 8]:
                R_m1 = R_O3
                R_m2 = R_O6
            else:
                R_m1 = R_Ti
                if n in [0, 1, 2]:
                    R_m2 = np.array([0.5, 0.5, -0.75])
                elif n in [3, 4, 5]:
                    R_m2 = np.array([0.5, -0.5, 0.5])
                elif n in [6, 7, 8]:
                    R_m2 = np.array([-0.5, 0.5, 0.5])

            if n == m:
                hamiltonian[n, m] = hopping_matrix[n, m]
            else:
                if (n in [0, 1, 2, 3, 4, 5, 6, 7, 8] and m in [9, 10, 11, 12, 13, 14]) or (m in [0, 1, 2, 3, 4, 5, 6, 7, 8] and n in [9, 10, 11, 12, 13, 14]):
                    is_TiO = True
                else:
                    is_TiO = False

                hamiltonian[n, m] = h_nm_element(hopping_matrix[n, m], k_point, R_n, R_m1, R_m2, is_TiO)

            # compute the complex conjugate for the elements below the diagonal
            if m < n:
                hamiltonian[n, m] = np.conjugate(hamiltonian[n, m])

    bands_distorted.append(np.linalg.eigvalsh(hamiltonian))

correct_bands_distorted = []
for x in range(14):
    band = []
    for y in range(121):
        band.append(bands_distorted[y][x])
    correct_bands_distorted.append(band)



# plot the resulting bands
fig, ax = plt.subplots(figsize=(4, 3))

ax.set_title('Tight Binding Perovskite TiO$_3$')
ax.set_ylabel('Energy')

it = 0
for band in correct_bands:
    if it == 0:
        ax.plot(k_points, band, color='navy', alpha=0.8, label='Non-distorted')
    else:
        ax.plot(k_points, band, color='navy', alpha=0.8)

    it = it + 1

it = 0
for band in correct_bands_distorted:
    if it == 0:
        ax.plot(k_points, band, color='lightcoral', linestyle='dashed', alpha=0.8, label='Distorted')
    else:
        ax.plot(k_points, band, color='lightcoral', linestyle='dashed', alpha=0.8)

    it = it + 1

ax.set_xlim(0, 1)
ax.set_ylim(-17.5, 27.5)

x_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
x_labels = ['$\\Gamma$', 'X', 'M', '$\\Gamma$', 'R', 'X']
ax.set_xticks(ticks=x_ticks, labels=x_labels)

ax.legend(frameon=False)

plt.tight_layout()
plt.savefig('bands.pdf')

ax.set_ylim(-5, 25)
plt.savefig('bands-zoom.pdf')