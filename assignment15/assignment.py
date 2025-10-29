import matplotlib.pyplot as plt
import numpy as np
        

def f1(x, T, z):
    return z  # dT/dx = z

def f2(x, T, z):
    return alpha * (T - Ta)  # dz/dx = alpha*(T - Ta)



def rk4_step(x, T, z, h):
    k1T = h * f1(x, T, z)
    k1z = h * f2(x, T, z)
    
    k2T = h * f1(x + h/2, T + k1T/2, z + k1z/2)
    k2z = h * f2(x + h/2, T + k1T/2, z + k1z/2)
    
    k3T = h * f1(x + h/2, T + k2T/2, z + k2z/2)
    k3z = h * f2(x + h/2, T + k2T/2, z + k2z/2)
    
    k4T = h * f1(x + h, T + k3T, z + k3z)
    k4z = h * f2(x + h, T + k3T, z + k3z)
    
    T_new = T + (k1T + 2*k2T + 2*k3T + k4T) / 6
    z_new = z + (k1z + 2*k2z + 2*k3z + k4z) / 6
    return T_new, z_new

# --- Function to integrate and return T(L) for a given slope ---

# --- Given parameters ---
alpha = 0.01      
Ta = 20          
L = 10            
T0 = 40           
TL = 200          
N = 100           
h = L / N 

def shoot(z0):
    x = 0.0
    T = T0
    z = z0
    for i in range(N):
        T, z = rk4_step(x, T, z, h)
        x += h
    return T

zl = 5.0
zh = 30.0
Tl_end = shoot(zl)
Th_end = shoot(zh)

# --- Shooting iteration ---
for _ in range(20):
    z = zl + (zh - zl) * (TL - Tl_end) / (Th_end - Tl_end)
    T_end = shoot(z)
    if abs(T_end - TL) < 1e-3:
        break
    if T_end < TL:
        zl, Tl_end = z, T_end
    else:
        zh, Th_end = z, T_end

print(f"Initial slope (T'(0)) = {z:.4f}")
print(f"T(L) ≈ {T_end:.4f} °C")


x_vals = [0.0]
T_vals = [T0]
x = 0.0
T = T0
z = z
for i in range(N):
    T, z = rk4_step(x, T, z, h)
    x += h
    x_vals.append(x)
    T_vals.append(T)

# --- Find where T = 100 °C ---
for i in range(len(T_vals) - 1):
    if (T_vals[i] - 100) * (T_vals[i+1] - 100) < 0:
        x100 = x_vals[i] + (100 - T_vals[i]) * (x_vals[i+1] - x_vals[i]) / (T_vals[i+1] - T_vals[i])
        break

print(f"Temperature T = 100°C occurs at x ≈ {x100:.3f} m")

plt.plot(x_vals, T_vals, 'b-', linewidth=2)
plt.axhline(100, color='r', linestyle='--', label='T = 100°C')
plt.axvline(x100, color='g', linestyle='--', label=f'x ≈ {x100:.2f} m')
plt.title('Temperature Distribution along the Rod')
plt.xlabel('x (m)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

##################   QUESTION 2 ###########################

L = 2.0           
nx = 41          
dx = L / (nx - 1) 
alpha = 0.4       
dt = alpha * dx**2
nt = 300          

x = np.linspace(0, L, nx)
u = np.zeros(nx)      
u_new = np.zeros(nx)

# Initial condition: 300°C at center
center = nx // 2
u[center] = 300

# Time evolution
for n in range(nt):
    for i in range(1, nx-1):
        u_new[i] = alpha * (u[i+1] + u[i-1]) + (1 - 2*alpha) * u[i]
    u_new[0] = 0       # boundary condition
    u_new[-1] = 0
    u[:] = u_new[:]    # update for next step

    # Plot every 50 steps
    if n % 50 == 0:
        plt.plot(x, u, label=f't={n*dt:.2f}')

plt.xlabel('Position x')
plt.ylabel('Temperature T (°C)')
plt.title('Heat Diffusion in a 1D Bar')
plt.legend()
plt.grid(True)
plt.show()

