# ---------  abhinav raj 2311006 -----------
#########################  question-1 ###########################


import numpy as np
import matplotlib.pyplot as plt
from mylib import *


def f2(x,y):
    return (x + y)**2

def f4(x):
    return math.tan(x + math.pi/4) - x


x_vals2 = np.linspace(0, math.pi/5, 100)
y_vals_f4 = [f4(x) for x in x_vals2]

x_rk1, y_rk1 = runge_kutta_solving().runge_kutta(f2, 0, math.pi/5, 1, 0.1)
x_rk2, y_rk2 = runge_kutta_solving().runge_kutta(f2, 0, math.pi/5, 1, 0.25)
x_rk3, y_rk3 = runge_kutta_solving().runge_kutta(f2, 0, math.pi/5, 1, 0.45)

plt.plot(x_vals2, y_vals_f4, label="Analytic f2")
plt.plot(x_rk1, y_rk1, linestyle='dashed' , label="Runge-Kutta f2 h=0.1")
plt.plot(x_rk2, y_rk2, linestyle='dotted' , label="Runge-Kutta f2 h=0.25")
plt.plot(x_rk3, y_rk3, linestyle='dashdot' , label="Runge-Kutta f2 h=0.45")
plt.xlabel("x")
plt.ylabel("y")
plt.title("dy/dx = (x + y)^2")
plt.legend()
plt.show()

#########################  question-2 ###########################


t_vals, x_vals, v_vals = Damped_Harmonic_Oscillator().runge_kutta_damped(Damped_Harmonic_Oscillator().f_damped_oscillation, 0, 40, 1.0, 0.0, 0.1)
plt.plot(t_vals, x_vals, label="Position")
plt.plot(t_vals, v_vals, label="Velocity")
plt.xlabel("Time")
plt.ylabel("Value ")
plt.title("Damped Harmonic Oscillator")
plt.legend()
plt.show()
