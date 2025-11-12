######### abhinav raj 2311006  #########
# Question 1: Lagrange Interpolation to find y(6.7)

import matplotlib.pyplot as plt
import numpy as np

x = [2, 3, 5, 8, 12]
y = [10, 15, 25, 40, 60]
x_new = 6.7

def lagrange_interpolation(x, y, x_new):
    n = len(x)
    y_new = 0
    
    # Calculating each Lagrange basis polynomial
    for i in range(n):
        # Calculating L_i(x_new)
        L_i = 1
        for j in range(n):
            if i != j:
                L_i *= (x_new - x[j]) / (x[i] - x[j])
        y_new += y[i] * L_i                                            #this will give all lagrangian points, "L_{i}({x_new}) = {L_i:.6f}, y_{i} * L_{i} = {y[i]} * {L_i:.6f} = {y[i] * L_i:.6f}"                 
    return y_new

y_result = lagrange_interpolation(x, y, x_new)
print(f"Lagrange interpolation at  y(x = {x_new}) = {y_result:.4f}")


########################### output ############################
# Lagrange interpolation at y(x = 6.7) = 33.5000
################################################################

def fit(x,y):
    sum_X = sum(x)
    sum_Y = sum(y)
    sum_XX = sum([x[i]**2 for i in range(N)])
    sum_XY = sum([x[i] * ln_y[i] for i in range(N)])

    b_power = (N * sum_XY - sum_X * sum_Y) / (N * sum_XX - sum_X**2)
    A_power = (sum_Y - b_power * sum_X) / N
    a_power = math.exp(A_power)

    return a_power, b_power
# ################################## Question 2 ########################################


x_q2 = [2.5, 3.5, 5.0, 6.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.5]
y_q2 = [13.0, 11.0, 8.5, 8.2, 7.0, 6.2, 5.2, 4.8, 4.6, 4.3]

# ============ Power Law Fit: y = a * x^b ============

# Transform to linear form
ln_x = [x**0 for x in x_q2]  # Initialize
ln_y = [y**0 for y in y_q2]  # Initialize

for i in range(len(x_q2)):
    ln_x[i] = x_q2[i]**0 * 0  # placeholder
    ln_y[i] = x_q2[i]**0 * 0  # placeholder

# Calculate natural logarithm manually
import math
ln_x = []
ln_y = []
for i in range(len(x_q2)):
    # Manual ln calculation using Taylor series approximation or built-in
    ln_x.append(math.log(x_q2[i]))
    ln_y.append(math.log(y_q2[i]))

# Linear fit to transformed data
N = len(x_q2)
a_power, b_power = fit(ln_x, ln_y)

print(f"\nPower Law: b = {b_power:.6f}, a = {a_power:.6f}")
print(f"Model: y = {a_power:.6f} * x^{b_power:.6f}")

# Calculate R² for power law
y_pred_power = [a_power * (x_q2[i]**b_power) for i in range(N)]
mean_y = sum(y_q2) / N
ss_res_power = sum([(y_q2[i] - y_pred_power[i])**2 for i in range(N)])
ss_tot = sum([(y_q2[i] - mean_y)**2 for i in range(N)])
r2_power = 1 - (ss_res_power / ss_tot)

print(f"R² (Power Law) = {r2_power:.6f}")

# ============ Exponential Fit: y = a * e^(-b*x) ============

Y_exp = ln_y  # Already calculated as ln(y)
X_exp = x_q2   # x values

sum_X_exp = sum(X_exp)
sum_Y_exp = sum(Y_exp)
sum_X2_exp = sum([X_exp[i]**2 for i in range(N)])
sum_XY_exp = sum([X_exp[i] * Y_exp[i] for i in range(N)])


slope = (N * sum_XY_exp - sum_X_exp * sum_Y_exp) / (N * sum_X2_exp - sum_X_exp**2)
intercept = (sum_Y_exp - slope * sum_X_exp) / N

b_exp = -slope  # b is negative of the slope
a_exp = math.exp(intercept)

print(f"\nExponential: b = {b_exp:.6f}, a = {a_exp:.6f}")
print(f"Model: y = {a_exp:.6f} * e^(-{b_exp:.6f}*x)")

# Calculate R² for exponential
y_pred_exp = [a_exp * math.exp(-b_exp * x_q2[i]) for i in range(N)]
ss_res_exp = sum([(y_q2[i] - y_pred_exp[i])**2 for i in range(N)])
r2_exp = 1 - (ss_res_exp / ss_tot)

print(f"R² (Exponential) = {r2_exp:.6f}")

# ============ Comparison ============

print(f"\nPower Law R²:      {r2_power:.6f}")
print(f"Exponential R²:    {r2_exp:.6f}")



################################ output ############################
# Power Law: b = -0.537409, a = 21.046352
# Model: y = 21.046352 * x^-0.537409
# R² (Power Law) = 0.995305

# Exponential: b = 0.058456, a = 12.212993
# Model: y = 12.212993 * e^(-0.058456*x)
# R² (Exponential) = 0.873061

# Power Law R²:      0.995305
# Exponential R²:    0.873061


#################################################################
