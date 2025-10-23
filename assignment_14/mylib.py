
class matrix_multiplication():
    def __init__(self):
        pass

    def matrix_multiply2(self,A, B):
        result=[]
        for i in range(len(A)):
            row=[]
            for j in range (len(B[0])):
                product_sum=0
                for k in range (len(B)):
                    product_sum += A[i][k]*B[k][j]

                row.append(product_sum)
            result.append(row)
    
        return result

    def matrix_multiply(self,A, B):
        result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):
            for j in range (len(B[0])):
                for k in range (len(B)):
                    result[i][j] += A[i][k]*B[k][j]
        return result
    
    def dot_product(self,C,D):
        
        if len(C) == len(D):
            q=0
            for i in range(len(C)-1):
                q+=C[i][0] * D[i][0]
            return q
        else:
            None
    

class read_matrix():

    def __init__(self):
        pass

    def read_matrix1(self, filepath):
        with open(filepath, 'r') as file:
            matrix = []
            for line in file:
                    line = line.strip()
                    if line:  # Only process non-empty lines
                        row = [float(num) for num in line.split()]
                        matrix.append(row)

        return matrix
    
# class read():
#     def __init__(self):
#         pass

#     def read_matrix1(file):
#         return np.loadtxt(file)

#     def read_vector(file):
#         return np.loadtxt(file)

    
class mycomplex():

    def __init__(self,R,I):
        self.real=R
        self.img= I
        

    def display_complex(self):
        return f"{self.real} + {self.img}j"

    # def sum(self,c1,c2):
    #     return f"{self.real+c1}+{self.img+c2 :.4f}j"
    
    # def sub(self,c1,c2):

    #     return f"{self.real-c1}+{self.img-c2 :.4f}j"

    # def multiply(self,c1,c2):
    #     return f"{self.real*c1 - self.img*c2}+({self.img*c1 + self.real*c2 :.4f})j"
    
    # def modulus(self):
    #     return f"{((self.real)**2 + (self.img)**2)**(0.5) :.4f}"              
    
    # -------f for making string and :.4f for turncating----
    
    def sum(self,s,c1,c2):
        if s == "+" :
            return f"{self.real+c1}+{self.img+c2 :.4f}j"
        elif s == "-":
            return f"{self.real-c1}+{self.img-c2 :.4f}j"
        elif s== "*":
            return f"{self.real*c1 - self.img*c2}+({self.img*c1 + self.real*c2 :.4f})j"
    
    def modulus(self):
        return f"{((self.real)**2 + (self.img)**2)**(0.5) :.4f}"              
    

class lcg():

    def __init__(self):
        pass

    def l_c_g(self,seed, a, c, m, n):
        random_numbers = []
        x = seed
        for _ in range(n):
            x = (a * x + c) % m

            random_numbers.append(x)
        return random_numbers
    
    
    def lcg_gen(self, limit, x, a, c, m):
        l = [x]
        for i in range(limit):
            l.append((a*l[-1] + c)%m)
        return l

class gauss_jordan:
    def __init__(self):
        pass
    def gauss_jordan1(self, A, B):
        for i in range(len(A)):
            A[i].append(B[i][0])

        for i in range(len(A)):
            diagonal = A[i][i] 
            # value of diagonal elements

            if diagonal == 0:
                for r in range(i + 1, len(A)):
                     # check for non-zero pivot
                    if A[r][i] != 0:
                        A[i], A[r] = A[r], A[i] 
                        break
                diagonal = A[i][i]
            
            for j in range(i, len(A[0])):
                A[i][j] /= diagonal
                
            for k in range(len(A)):
                if k != i:
                    f = A[k][i] # factor to eliminate
                    for l in range(i, len(A[0])):
                        A[k][l] -= f * A[i][l]
        return A
    

class L_matrix():
    def __init__(self):
        pass
    def gauss_jordan1(self, A):
        for i in range(len(A)):
            diagonal = A[i][i] 
            # value of diagonal elements
            if diagonal == 0:
                for r in range(i + 1, len(A)):
                    # check for non-zero pivot
                    if A[r][i] != 0:
                        A[i], A[r] = A[r], A[i] 
                        break
                diagonal = A[i][i]
            
            for j in range(i, len(A[0])):
                A[i][j] /= diagonal
            for k in range(len(A)):
                if k < i:
                    f = A[k][i] # factor to eliminate
                    for l in range(i, len(A[0])):
                        A[k][l] -= f * A[i][l]
        return A

class U_matrix():
    def __init__(self):
        pass
    def gauss_jordan2(self, B):

        for i in range(len(B)):
            diagonal = B[i][i] 
            # value of diagonal elements

            if diagonal == 0:
                for r in range(i + 1, len(B)):
                     # check for non-zero pivot
                    if B[r][i] != 0:
                        B[i], B[r] = B[r], B[i] 
                        
                        
                        break
                diagonal = B[i][i]

            # for j in range(i, len(B[0])):
            #     B[i][j] /= diagonal

            for k in range(len(B)):

                if k > i:
                    f = B[k][i] # factor to eliminate
                    for l in range(i, len(B[0])):
                        B[k][l] -= f * B[i][l]

        return B
    

class LU_matrix():
    def __init__(self):
        pass
    def dolittle(self, A):
        n = len(A)
        L = [[0 for _ in range(n)] for _ in range(n)]
        U = [[0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            L[i][i] = 1

        for j in range(n):
            for i in range(j + 1):
                U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
            for i in range(j + 1, n):
                L[i][j] = (A[i][j] - sum(L[i][k] * U[k][j] for k in range(j))) / U[j][j]

        return L, U
    
class LU_decomposition():
    def __init__(self):
        pass

    def LU_matrix(self,A):
        n=len(A)
        # for i in range(len(A)):
        #     A[0][i]=A[0][i]
        for i in range(1,n):
            A[i][0]= A[i][0]/A[0][0]


        for j in range (1,n):
            for i in range(1,n):
                if i<=j:
                    sum = 0
                    for k in range (0,i):
                        sum += A[i][k] * A[k][j]
                    A[i][j] = A[i][j] - sum

                else:
                    sum2 = 0 
                    for k in range(0,j):
                        sum2 += A[i][k]*A[k][j]
                    A[i][j] = (A[i][j] - sum2)/ (A[j][j])

        return A
    
class LU_decom():
    def __init__(self):
        pass

    def LU_matrix(self,A):
        n = len(A)
        L = [[0.0]*n for _ in range(n)]
        U = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == 0:
                    U[0][j] = A[0][j]
                    L[j][0] = A[j][0] / U[0][0] if U[0][0] != 0 else 0
                elif i >= j:
                    sum_LU = sum(L[i][k] * U[k][j] for k in range(j))
                    L[i][j] = A[i][j] - sum_LU if i != j else 1.0
                else:
                    sum_LU = sum(L[i][k] * U[k][j] for k in range(i))
                    U[i][j] = A[i][j] - sum_LU
            if i > 0:
                U[i][i] = A[i][i] - sum(L[i][k] * U[k][i] for k in range(i))
        return L, U

    def forward_substitution(self,L, b):
        n = len(L)
        y = [0] * n
        for i in range(n):
            y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
        return y
    def backward_substitution(self,U, y):
        n = len(U)
        x = [0] * n
        for i in reversed(range(n)):
            x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]
        return x
    def extract_LU(self,A):
        n = len(A)
        L = [[0.0]*n for _ in range(n)]
        U = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i > j:
                    L[i][j] = A[i][j]
                elif i == j:
                    L[i][j] = 1.0
                    U[i][j] = A[i][j]
                else:
                    U[i][j] = A[i][j]
        return L, U

    def inverse_from_LU(self,L, U):
        n = len(L)
        inv = []
        for col in range(n):
            e = [0.0]*n
            e[col] = 1.0
            y = self.forward_substitution(L, e)
            x = self.backward_substitution(U, y)
            inv.append(x)
        # Transpose result to get columns as rows
        inv_matrix = [[round(inv[j][i], 3) for j in range(n)] for i in range(n)]
        return inv_matrix


class solve():
    def __init__(self):
        pass

    def solve2(self,A,B):
        y=[[0] for _ in range(len(A))]
        x=[[0] for _ in range(len(A))]

        n=len(A)

        y[0][0] = B[0][0]
        

        for i in range(1,n):
            sum = 0
            for j in range (i):
                sum += A[i][j] * y[j][0]
                

            y[i][0] = B[i][0] - sum

        x[n-1][0] = y[n-1][0] / A[n-1][n-1]

        for i in range (n-2,-1,-1):
            sum2=0
            for j in range (i+1,n):
                sum2 += A[i][j] * x[j][0]

            x[i][0] = (y[i][0]-sum2)/ A[i][i]

        return x
    
class cd():

    def __init__(self):
        pass
    def cholesky_decomposition(self,A):
        n = len(A)
        L = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1):
                sum_val = sum(L[i][k]*L[j][k] for k in range(j))
                if i == j:
                    L[i][j] = (A[i][i] - sum_val) ** 0.5
                else:
                    L[i][j] = (A[i][j] - sum_val) / L[j][j]
        return L

    def transpose(self,A):
        n = len(A)
        return [[A[j][i] for j in range(n)] for i in range(n)]


    def print_matrix(self, mat, name):
        print(f"{name}:")
        for row in mat:
            print(" ".join(f"{x:.4f}" for x in row))
        print()

    def mix(self, L, L_dagger):
        n = len(L)
        c = []
        for i in range(n):
            row = []
            for j in range(n):
                if L[i][j] != 0:
                    row.append(L[i][j])
                else:
                    row.append(L_dagger[i][j])
            c.append(row)
        return c

    def matrix_multiply(self, A, B):
        n = len(A)
        result = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    result[i][j] += A[i][k] * B[k][j]
        return result




    def solve2(self,A,B):
            y=[[0] for _ in range(len(A))]
            x=[[0] for _ in range(len(A))]

            n=len(A)

            y[0][0] = (B[0][0]) / A[0][0]

            for i in range(1,n):
                sum = 0
                for j in range (i):
                    sum += A[i][j] * y[j][0]
                    

                y[i][0] = (B[i][0] - sum)/A[i][i]

            x[n-1][0] = y[n-1][0] / A[n-1][n-1]
            return x
    
class Jacobi1():
    def __init__(self):
        pass
    
    def jacobi_iter(self,A,B):
        n = len(A)
        D = [[A[i][j] if i==j else 0 for j in range(n)] for i in range(n)]
        T = [[A[i][j] if i != j else 0 for j in range(n)] for i in range(n)]
        seed = [[0] for _ in range(n)]
        for _ in range(100): # iterative steps
            new_seed = [[0] for _ in range(n)]
            for i in range(n):
                sum1 = 0
                for j in range(n):
                    if i != j:
                        sum1 += T[i][j] * seed[j][0]
                new_seed[i][0] = (B[i][0] - sum1) / D[i][i]

            if all(abs(new_seed[i][0] - seed[i][0]) < 10**(-6) for i in range(n)):
                # check if the difference is difference is very less for more precision!
                return new_seed
            seed = new_seed # change the seed

    
class jacobi_():
    def __init__(self):
        pass
        
    def jacobi(self,A,B):
        n =len(A)
        initial = [[0] for i in range(n)]
        new_initial = [[0] for i in range(n)]

        for i in range(n):
                sum = 0
                for j in range (n):
                    if j != i:
                        sum += A[i][j] * initial[j][0]

                new_initial[i][0] = (B[i][0] - sum)/ A[i][i]

        if all(abs(new_initial[i][0] - initial[i][0]) < 10**(-6) for i in range (n)):
            return new_initial
        initial = new_initial


class cholesky_decom():
    def __init__(self):
        pass
    def cholesky(self, A):
        n = len(A)
        for i in range(n):
            sum1 = 0
            for j in range(i):
                sum1 += A[i][j]**2
            A[i][i] = (A[i][i] - sum1)**0.5
            for j in range(n):
                if i < j :
                    sum2 = 0
                    for k in range(i):
                        sum2 += A[i][k]*A[k][j]
                    A[i][j] = (A[i][j] - sum2) / A[i][i]
                    A[j][i] = A[i][j]
        return A

    def cholesky_solve( self,A, B):
        n = len(A)
        L = self.cholesky(A)
        
        # Forward substitution 
        Y = [[0] for _ in range(n)]
        for i in range(n):
            sum1 = 0
            for j in range(i):
                sum1 += L[i][j] * Y[j][0]
            Y[i][0] = (B[i][0] - sum1) / L[i][i]
        
        # Back substitution 
        X = [[0] for _ in range(n)]
        for i in range(n-1, -1, -1):
            sum2 = 0
            for j in range(i+1, n):
                sum2 += L[j][i] * X[j][0]
            X[i][0] = (Y[i][0] - sum2) / L[i][i]
        
        return X
    
class GaussSeidel():
    def __init__(self):
        pass

    def gauss_seidel_iter(self, A, B):
        n = len(A)
        x = [0.0 for _ in range(n)]
        if isinstance(B[0], list):
            B = [row[0] for row in B]
        for _ in range(100):
            x_old = x.copy()
            for i in range(n):
                sum1 = sum(A[i][j] * x[j] for j in range(i))
                sum2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
                x[i] = (B[i] - sum1 - sum2) / A[i][i]
            if all(abs(x[i] - x_old[i]) < 1e-6 for i in range(n)):
                return [[xi] for xi in x]
        return [[xi] for xi in x]
    

class diadominant():
    def __init__(self):
        pass
    
    
    def dia_dominant(self,A):
        n = len(A)
        used = [False] * n
        new_A = [[0]*n for _ in range(n)]
        for i in range(n):
            found = False
            for row in range(n):
                if not used[row]:
                    diag = abs(A[row][i])
                    off_diag_sum = sum(abs(A[row][j]) for j in range(n) if j != i)
                    if diag >= off_diag_sum:
                        new_A[i] = A[row][:]
                        used[row] = True
                        found = True
                        break
            if not found:
                # If no suitable row, just pick the next unused row
                for row in range(n):
                    if not used[row]:
                        new_A[i] = A[row][:]
                        used[row] = True
                        break
        return new_A
    
class root():
    def __init__(self):
        pass

    def rfalsi(self,f,a,b,iter=100,acc=10e-6):          
        if f(a)*f(b) < 0: 
            for i in range(iter):
                c = b - (f(b)*(b-a)) / (f(b)-f(a))
                if abs(f(c)) < acc or abs(b - a) < acc:
                    y=i+1
                    return [c,y]
                if f(a) * f(c) < 0:
                    b = c
                else:
                    a = c
                    max_i = i + 1 
            print("iteration steps:", max_i)
            return c
        else:
            return "No root bracketed in [a, b]"
        
    def bracketting(self,f,a,b,iter=100,acc=10e-6):

     if f(a)*f(b) > 0 :
          if abs(f(a)) < abs(f(b)):
               a = a - 1.5*(b-a)
          elif abs(f(a)) > abs(f(b)):
               b = b + 1.5*(b-a)
        
          return [a,b]
    
    def bisection(self,f,a,b,iter=100,acc=10e-6):          
        if f(a)*f(b) < 0: 
            for i in range(iter):
                c = (a + b) / 2
                if abs(f(c)) < acc or abs(b - a) < acc:
                    y=i+1
                    return [c,y]
                if f(a) * f(c) < 0:
                    b = c
                else:
                    a = c
                    max_i = i + 1 
            print("iteration steps:", max_i)
            return (a + b) / 2
        else:
            return "No root bracketed in [a, b]"
        
    def n_raphson(self,f,f1,a,b,x0,iter=100,acc=10e-6):          
       if f(a)*f(b) < 0: 
           for i in range(iter):
               x0 = x0 - f(x0)/ f1(x0) 
               if abs(f(x0)) < acc or abs(x0) < acc:
                   y=i+1
                   return [x0,y]

           return [x0,y]
       else:
           return "No root bracketed in [a, b]"
       
    def fixed_point(self,g, x0, iter=100, acc=1e-6):
        y = 0
        for i in range(iter):
            x1 = g(x0)
            if abs(x1 - x0) < acc:
                y = i + 1
                return [x1, y]
            x0 = x1

        return [x0, i+1]
    def fix1(self,x,f):
        x1 = f(x)
        if abs(x1 - x) < 1e-6:
            return x1
        else:
            return self.fix1(x1,f)
        
class MultivariableFunction():
    def __init__(self):
        pass



    def fixed_point(self,g, x0, iter=100, acc=1e-6):
        for i in range(iter):
            x1 = g(x0)
            if max(abs(x1[j] - x0[j]) for j in range(3)) < acc:
                return x1, i+1
            x0 = x1
        return x0, iter

    def det3(self, m):
        det = 0
        for i in range(3):
            det += m[0][i] * (m[1][(i+1)%3] * m[2][(i+2)%3] - m[1][(i+2)%3] * m[2][(i+1)%3])
        return det


    def solve3_gauss_jordan(self,A, b):
        # Prepare augmented matrix
        aug = [A[i][:] + [b[i]] for i in range(3)]
        # Forward elimination
        for i in range(3):
            # Find pivot
            if abs(aug[i][i]) < 1e-12:
                for r in range(i+1, 3):
                    if abs(aug[r][i]) > 1e-12:
                        aug[i], aug[r] = aug[r], aug[i]
                        break
            # Normalizing pivot row
            pivot = aug[i][i]
            for j in range(i, 4):
                aug[i][j] /= pivot
            # Eliminating other rows
            for k in range(3):
                if k != i:
                    factor = aug[k][i]
                    for j in range(i, 4):
                        aug[k][j] -= factor * aug[i][j]
        # Extracting solution
        return [aug[i][3] for i in range(3)]

    def newton_raphson_gj(self,F, J, x0, iter=100, acc=1e-6):
        for i in range(iter):
            fval = F(x0)
            if max(abs(fval[j]) for j in range(3)) < acc:
                return x0, i+1
            Jval = J(x0)
            dx = MultivariableFunction().solve3_gauss_jordan(Jval, [-f for f in fval])
            x0 = [x0[j] + dx[j] for j in range(3)]
        return x0, iter

import math

#     Args:
#         P (function): The polynomial function.
#         P_prime (function): The first derivative of the polynomial.
#         P_double_prime (function): The second derivative of the polynomial.
#         x0 (float): The initial guess for the root.
#         n (int): The degree of the polynomial.
#         tol (float): The desired tolerance for the root.
#         max_iter (int): The maximum number of iterations.


class laguerre():

    def __init__(self):
        pass

    def laguerre(self, P, P_prime, P_double_prime, x0, n, tol=1e-6, max_iter=100):

        x = x0
        for i in range(max_iter):
            P_val = P(x)
            P_prime_val = P_prime(x)
            P_double_prime_val = P_double_prime(x)

            if abs(P_val) < tol:
                return x

            G = P_prime_val / P_val
            H = G*G - (P_double_prime_val / P_val)
            sqrt_term = math.sqrt((n - 1) * (n*H - G*G))    # Calculating the square root term

            if abs(G + sqrt_term) > abs(G - sqrt_term):     # Choosing sign to maximize the denominator's magnitude
                denominator = G + sqrt_term
            else:
                denominator = G - sqrt_term

            if abs(denominator) < 1e-10:
                print("Denominator is too close to zero. Cannot proceed.")
                return None

            a = n / denominator # Apply the Laguerre's formula
            x_new = x - a

            if abs(x_new - x) < tol:
                return x_new

            x = x_new

        print("Maximum iterations reached without finding a root within tolerance.")
        return None



class deflation():

    def __init__(self):
        pass

    def synthetic_division(self,coeffs, root):
        """Performs synthetic division of a polynomial by (x - root).
        Args:
            coeffs (list): Coefficients of the polynomial (highest degree first).
            root (float): The root to deflate.
        Returns:
            list: Coefficients of the deflated polynomial.
        """
        n = len(coeffs)
        new_coeffs = [coeffs[0]]
        for i in range(1, n):
            new_coeffs.append(coeffs[i] + new_coeffs[-1] * root)   # The last element is the remainder, discard it

        return new_coeffs[:-1]

    def poly_eval(self, coeffs, x):
        result = 0              # Evaluates a polynomial at x given its coefficients.
        for c in coeffs:
            result = result * x + c
        return result

    def poly_derivative(self, coeffs):
        n = len(coeffs)
        return [coeffs[i] * (n - i - 1) for i in range(n - 1)]

    def poly_double_derivative(self, coeffs):
        first = self.poly_derivative(coeffs)
        return self.poly_derivative(first)

    def find_all_roots(self, coeffs, tol=1e-6, max_iter=100):
        roots = []
        current_coeffs = coeffs[:]
        n = len(current_coeffs) - 1

        while n > 0:

            P = lambda x: self.poly_eval(current_coeffs, x)              # Define polynomial and its derivatives for current coefficients
            P_prime = lambda x: self.poly_eval(self.poly_derivative(current_coeffs), x)
            P_double_prime = lambda x: self.poly_eval(self.poly_double_derivative(current_coeffs), x)

            x0 = 0.0         # Initial guess (can be improved)
            root = laguerre().laguerre(P, P_prime, P_double_prime, x0, n, tol, max_iter)
            if root is None:
                print("Failed to find root.")
                break
            
            if abs(P(root)) < tol:    # Refine root
                roots.append(root)
                current_coeffs = self.synthetic_division(current_coeffs, root)
                n -= 1
            else:
                print("Root not accurate enough, stopping.")
                break
        return roots
    
class integral_solving():
    
    def __init__(self):
        pass

    def midpoint_integral(self,a,b,n,f):
        h = (b - a) / n  #width
    
        total_sum = 0
        for i in range(n):
            total_sum += f(a + (i + 0.5) * h)
        return total_sum * h
    
    def trapezoidal_integral(self,a,b,n,f):

        h = (b - a) / n  #width
        x=[a]
        for j in range(n):
            x.append(x[j] + h)
        total_sum = 0
        for i in range(1,n):
            total_sum += (h*(f(x[i-1])+f(x[i])))*(0.5)
        return total_sum

    def trap_integral(self, Interval, parts, function):
        sum = 0
        h = (Interval[1] - Interval[0])/parts
        for i in range(1,parts+1):
            x = Interval[0] + (i-1)*h
            y = Interval[0] + i*h
            sum += h*(function(x)+function(y))/2
        return sum
    

    def Simpsons(self, Interval, function, parts):
        if parts % 2 == 1: # if parts is odd then make it even
            parts += 1
        h = (Interval[1] - Interval[0])/parts
        sum = function(Interval[0]) + function(Interval[1])
        for i in range(1, parts, 2):
            x = Interval[0] + i*h
            sum += 4*function(x) # Adding based on formula
        for i in range(2, parts-1, 2):
            x = Interval[0] + i*h
            sum += 2*function(x) # Adding based on formula
        sum *= h/3
        return sum
    
    def monte_carlo(self, iter, Interval, function, parts):
        L = lcg().lcg_gen(iter, 10, 1103515245, 12345, 32768) # upto 32768 (m)
        for i in range(len(L)):
            L[i] /= 32768 # Getting all random numeber less than 1
        sum = 0
        for i in range(parts):
            x = Interval[0] + (Interval[1] - Interval[0]) * L[i] # Scaling numbers in range
            sum += function(x)
        return sum*(Interval[1] - Interval[0])/parts
    
import numpy as np
from numpy.polynomial import legendre as P

class integral_solve():
    def __init__(self):
        pass

# n is N mention in pdf
    def gauss_quadrature(self,f,i,n):   

        root,weight = P.leggauss(n)
        root = 0.5*(i[1]-i[0])*root + 0.5*(i[1]+i[0])
        weight = 0.5*(i[1]-i[0])*weight

        s=0
        for i in range(n):
            s += weight[i]*f(root[i])
        return s


    def Simpsons( self,Interval, function, parts):
        if parts % 2 == 1: # if parts is odd then make it even
            parts += 1
        h = (Interval[1] - Interval[0])/parts
        sum = function(Interval[0]) + function(Interval[1])
        for i in range(1, parts, 2):
            x = Interval[0] + i*h
            sum += 4*function(x) # Adding based on formula
        for i in range(2, parts-1, 2):
            x = Interval[0] + i*h
            sum += 2*function(x) # Adding based on formula
        sum *= h/3
        return sum
    

class runge_kutta_solving():
    def __init__(self):
        pass

    def runge_kutta(self,f, a, b, y0, h):
        L_x = [a]
        L_y = [y0]

        for _ in range(int((b - a) / h)):
            k1 = h * f(L_x[-1], L_y[-1])
            k2 = h * f(L_x[-1] + h / 2, L_y[-1] + k1 / 2)
            k3 = h * f(L_x[-1] + h / 2, L_y[-1] + k2 / 2)
            k4 = h * f(L_x[-1] + h, L_y[-1] + k3)

            y_new = L_y[-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            x_new = L_x[-1] + h

            L_x.append(x_new)
            L_y.append(y_new)

        return L_x, L_y


class Damped_Harmonic_Oscillator():
    def __init__(self):
        pass

    def f_damped_oscillation(self,t, x,v):
        u = 0.15
        w = 1.0
        dxdt = v
        dvdt = -u * v - w**2 * x
        return dxdt, dvdt

    def runge_kutta_damped(self,f, a, b, x0, v0, h):
        L_t = [a]
        L_x = [x0]
        L_v = [v0]
        L_e = [(x0**2 + v0**2)/2]

        for _ in range(int((b - a) / h)):
            t = L_t[-1]
            x = L_x[-1]
            v = L_v[-1]
           

            k1_x, k1_v = f(t, x, v)
            k2_x, k2_v = f(t + h / 2, x + h * k1_x / 2, v + h * k1_v / 2)
            k3_x, k3_v = f(t + h / 2, x + h * k2_x / 2, v + h * k2_v / 2)
            k4_x, k4_v = f(t + h, x + h * k3_x, v + h * k3_v)

            x_new = x + (h / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
            v_new = v + (h / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
            t_new = t + h

            L_t.append(t_new)
            L_x.append(x_new)
            L_v.append(v_new)
            L_e.append((x_new**2 + v_new**2)/2)

        return L_t, L_x, L_v, L_e

