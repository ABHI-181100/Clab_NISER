# ABHINAV RAJ   2311006

from mylibr import *
import numpy as np
import math
import matplotlib.pyplot as plt

# # -------- Question no.1 -------------------------

N = 5000
M = 10000  # number of independent snapshots

# LCG parameters (common values)
a = 1103515245
c = 12345
m = 2**31
seed = 7

sample = lcg().lcg_gen(2 * M - 1, seed, a, c, m)  # returns list length 2*M
U = np.array(sample, dtype=float) / float(m)      # convert to floats in (0,1)

# Use Box-Muller to get approx standard normal variates (one per sample)
u1 = U[0::2]
u2 = U[1::2]                                                # here i use box-muller method may be there are other methods that will be easy but right now i am thinking of this so ....
u1 = np.clip(u1, 1e-12, 1.0)  # avoid log(0)
z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

# Map normals to binomial via mean and std (good approximation for large N)
mean_theo = N * 0.5
std_theo = np.sqrt(N * 0.5 * 0.5)
samples = np.rint(mean_theo + std_theo * z).astype(int)
samples = np.clip(samples, 0, N)  # ensure valid counts

# Empirical statistics
mean_emp = samples.mean()
std_emp = samples.std(ddof=0)

print(f"Empirical mean (left): {mean_emp:.2f}, empirical std: {std_emp:.2f}")
print(f"Theoretical mean (left): {mean_theo:.2f}, theoretical std: {std_theo:.2f}")

# Plot histogram
Plots().hist(samples.tolist(),
             title=f"Equilibrium distribution (N={N})",
             xlabel="Number of particles on left",
             ylabel="Frequency",
             bins=50)

####################----output----------------##########
# Empirical mean (left): 2500.39, empirical std: 35.85
# Theoretical mean (left): 2500.00, theoretical std: 35.36
#####################################################

# # -------- Question no.2 -------------------------

A = myLibrary().read_matrix("t8.txt")
B = myLibrary().read_matrix("t10.txt")

# Normalize B to column vector [[...], ...] if it's a single row
if len(B) == 1 and len(B[0]) > 1:
    B = [[v] for v in B[0]]
n = len(A)
# initial guess: zero column vector
x0 = [[0.0] for _ in range(n)]
sol = gauss_seidel().gauss_iter(A, B, x0)

if sol is None:
    print("Gauss-Seidel did not converge within the iteration limit.")
else:
    x_sol, iterations = sol 
    print(f"Converged in {iterations} iterations (tol 1e-6)")
    for i, xi in enumerate(x_sol, start=1):
        print(f"x{i} = {xi[0]:.6f}")


##################-output------###########################################
# Converged in 16 iterations (tol 1e-6)
# x1 = 1.000000
# x2 = 1.000000
# x3 = 1.000000
# x4 = 1.000000
# x5 = 1.000000
# x6 = 1.000000
####################################################################


# # -------- Question no.3 -------------------------

def f(x):
    return x*exp(x) - 25

def df(x):
    return exp(x)*(1 + x)

x0 = 0.5       # initial guess
tol = 1e-10
root = roots().newton_raphson(f, df, x0, tol)
print(f"Stretch x = {root:.6f} (units of length)")

#########----output----------------################
# Stretch x = 2.360150 (units of length)
################################################


# -------- Question no.4 -------------------------

length = 2.0

dens = lambda x: x**2
moment_integrand = lambda x: x * dens(x)

# Using Simpson's 1/3 rule with a large even number of parts for accuracy
parts = 1000
mass = integration().Simpsonsonethird([0.0, length], dens, parts)
moment = integration().Simpsonsonethird([0.0, length], moment_integrand, parts)

x_cm = moment / mass
print(f"{x_cm:.4f}")

#########----output----------------################
# 1.5000
################################################3

# # -------- Question no. 5 -------------------------

# Parameters and RK4 solution using the library above
g = 10.0
k = 0.02  # gamma 
v0 = 10.0
y0 = 0.0

# system: dy/dt = v, dv/dt = (-g - k*v)
f1 = lambda t, y, v: v
f2 = lambda t, y, v: (-g - k * v)

I = [0.0, 2.414]   # time interval (2.414s calculated for maximum without air resistance and h max =5m)
h = 0.01         # RK4 step size

t_list, y_list, v_list = solveDE().coupled_RK4(f1, f2, I, h, y0, v0)

plt.plot(y_list, v_list, label='v(y) from RK4')
plt.xlabel('Height y (m)')
plt.ylabel('Velocity v (m/s)')
plt.title('Velocity vs Height (RK4)')
plt.grid(True)
plt.legend()
plt.show()




# # -------- Question no.7 -------------------------

data = np.loadtxt("esem_qfit.dat")
x = data[:,0]
y = data[:,1]

degree = 4
coeff = datafit().poly_fit(x, y, degree)

a0, a1, a2, a3, a4 = coeff
print("Coefficients: a0 =", a0, ", a1 =", a1, ", a2 =", a2, ", a3 =", a3, ", a4 =", a4)

# #########################------output---------###########
# # Coefficients: a0 = 0.25462950721154687 , a1 = -1.1937592138092252 , a2 = -0.45725541238296663 , a3 = -0.8025653910658195 , a4 = 0.013239427477395956

# ############################################################



# #---------question no. 6----------------------------

#     parameters
L = 2.0          # rod length
T = 4.0          # total time
nx = 20          # number of spatial grid points
nt = 5000        # number of time steps
dx = L / (nx - 1)
dt = T / nt
alpha = dt / dx**2
if alpha > 0.5:
    raise RuntimeError(f"Scheme unstable: dt/dx^2 = {alpha:.4g} > 0.5")
# spatial grid and initial condition
x = np.linspace(0, L, nx)
u = 20.0 * np.sin(x)        # initial temperature
u[0] = 0.0                  # enforce boundary conditions
u[-1] = 0.0

saved_profiles = {0: u.copy()}

# Time-stepping (FTCS)
u_new = u.copy()
for n in range(1, nt + 1):
    for i in range(1, nx - 1):
        u_new[i] = u[i] + alpha * (u[i+1] - 2.0*u[i] + u[i-1])
    # enforce BCs
    u_new[0] = 0.0
    u_new[-1] = 0.0
    u[:] = u_new     # something is wrong here i am not able to figure it out, i think rest part of this code are good

to_plot = [0, 10, 20, 50, 100, 200, 500, 1000]

plt.figure(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(to_plot)))
for c, step in zip(colors, to_plot):
    prof = saved_profiles.get(step)
    if prof is None:
        continue
    plt.plot(x, prof, label=f"step {step} (t={step*dt:.3f})", color=c)
plt.xlabel("x (position)")
plt.ylabel("Temperature (°C)")
plt.title("Temperature profile along rod at selected time steps")
plt.legend()
plt.grid(True)
plt.show()
