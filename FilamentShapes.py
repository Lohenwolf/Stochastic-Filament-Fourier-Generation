import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

#parameters
N = 500 #number of filaments to generate
L = 1 #unitary length
lp = 1 #persistence length of the filament type
num_filaments = 500 #number of filaments to generate
s_points = 1000  #point numbers for each filament shape

def filament(L, lp, N):
    s = np.linspace(0, L, 1000)
    sig = np.sqrt(L / lp)
    a0 = np.random.randn() * sig
    k = np.ones_like(s) * a0

    for n in range(1, N):
        anr = np.random.randn() * sig
        ani = np.random.randn() * sig
        qn = 2 * np.pi * n / L
        k += 2 * anr * np.cos(qn * s) - 2 * ani * np.sin(qn * s)

    theta = cumulative_trapezoid(k, s, initial=0)
    x = cumulative_trapezoid(np.cos(theta), s, initial=0)
    y = cumulative_trapezoid(np.sin(theta), s, initial=0)

    return x, y

#array to accumulate filaments coordinates
all_x = np.zeros((num_filaments, s_points))
all_y = np.zeros((num_filaments, s_points))

#filament generation loop
for i in range(num_filaments):
    x, y = filament(L, lp, N)
    all_x[i] = x
    all_y[i] = y
    plt.plot(x, y, color='blue', alpha=0.04)

#mean output
mean_x = np.mean(all_x, axis=0)
mean_y = np.mean(all_y, axis=0)
plt.plot(mean_x, mean_y, color=(0.5, 1, 0.5), linestyle='--', linewidth=2, alpha=1.0, label='Mean')

#single filament highlight
x, y = filament(L, lp, N)
plt.plot(x, y, color='black', linewidth=2, label='Single Filament')

plt.title(fr'$\xi={lp}$')
plt.axis('off')
plt.axis('equal')
plt.savefig(f"lp{lp}.png", dpi=300, bbox_inches='tight')
plt.show()