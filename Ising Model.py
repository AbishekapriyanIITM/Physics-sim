import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'notebook', 'grid'])
import numba
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure

N = 50
init_random = np.random.random((N, N))
lattice_n = np.zeros((N, N))
lattice_n[init_random >= 0.75] = 1
lattice_n[init_random < 0.75] = -1

init_random = np.random.random((N, N))
lattice_p = np.zeros((N, N))
lattice_p[init_random >= 0.25] = 1
lattice_p[init_random < 0.25] = -1

def get_energy(lattice):
    kern = generate_binary_structure(2, 1)
    kern[1][1] = False
    arr = -lattice * convolve(lattice, kern, mode='constant')
    return float(arr.sum())  


@numba.njit('UniTuple(f8[:],2)(f8[:,:],i8,f8,f8)', nogil=True)
def metropolis(spin_arr, times, BJ, energy):
    spin_arr = spin_arr.copy()
    net_spins = np.zeros(times - 1)
    net_energy = np.zeros(times - 1)
    
    for t in range(0, times - 1):
        x = np.random.randint(0, N)
        y = np.random.randint(0, N)
        spin_i = spin_arr[x, y]
        spin_f = spin_i * -1

        E_i = 0
        E_f = 0
        if x > 0:
            E_i += -spin_i * spin_arr[x - 1, y]
            E_f += -spin_f * spin_arr[x - 1, y]
        if x < N - 1:
            E_i += -spin_i * spin_arr[x + 1, y]
            E_f += -spin_f * spin_arr[x + 1, y]
        if y > 0:
            E_i += -spin_i * spin_arr[x, y - 1]
            E_f += -spin_f * spin_arr[x, y - 1]
        if y < N - 1:
            E_i += -spin_i * spin_arr[x, y + 1]
            E_f += -spin_f * spin_arr[x, y + 1]
            
        dE = E_f - E_i
        
        
        if dE <= 0 or (np.random.random() < np.exp(-BJ * dE)):
            spin_arr[x, y] = spin_f
            energy += dE
            
        net_spins[t] = spin_arr.sum()
        net_energy[t] = energy
        
    return net_spins, net_energy  

# Run the simulation
spins, energies = metropolis(lattice_n, 1000000, 0.2 , get_energy(lattice_n))

# Plotting the results
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(spins / N**2)
ax.set_xlabel('Alg time steps')
ax.set_ylabel('avg spin')
ax.grid(True)

ax = axes[1]
ax.plot(energies)
ax.set_xlabel('Alg time steps')
ax.set_ylabel('energy')
ax.grid(True)

fig.tight_layout()  
fig.suptitle('spin, energy vs timesteps', y=1.05)
plt.show()
