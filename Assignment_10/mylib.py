
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

# class myLibrary:
#     def __init__(self):
#         pass
#     def read_matrix(self, filename) :
#         with open(filename, 'r' ) as f :
#             matrix = []
#             for line in f :
#                 row = [float (num) for num in line.strip( ).split( )]
#                 matrix.append(row)
#         return matrix

#     def matrix_multi(self, a, b):
#         nrowa= len(a)
#         ncolumna = len(a[0])
#         nrowb = len(b)
#         ncolumnb = len(b[0])
#         answer = []
#         for i in range(nrowa):
#             row = []
#             for j in range(ncolumnb):
#                 val = 0
#                 for k in range(ncolumna):
#                     val += a[i][k] * b[k][j]
#                 row.append(val)
#             answer.append(row)
#         return answer

#     def transpose(self, mat):
#         nrow = len(mat)
#         ncolumn = len(mat[0])
#         answer = []
#         for i in range(ncolumn):
#             row = []
#             for j in range(nrow):
#                 row.append(mat[j][i])
#             answer.append(row)
#         return answer
    
    

# class myComplex:
#     def __init__(self, a, b):
#         self.real = a
#         self.img = b
#     def show(self):
#         return f"{self.real} + {self.img}j"
    
#     def sum(self, s, a, b):
#         if s == "+" :
#             return f"{self.real + a : .4f} + {self.img + b : .4f}j"
#         elif s == "-" :
#             return f"{self.real - a : .4f} + {self.img - b : .4f}j"
#         elif s == "*" :
#             return f"{self.real * a - self.img * b : .4f} + {self.real * b + self.img * a : .4f}j"
#         else:
#             return "enter valid operation!"
#     def modC(self):
#         return f"{((self.real)**2 + (self.img)**2)**0.5 : .4f}"

# class lcg:
#     def __init__(self):
#         pass
#     def lcg_gen(self, limit, x, a, c, m):
#         l = [x]
#         for i in range(limit):
#             l.append((a*l[-1] + c)%m)
#         return l
# class gauss_jordan:
#     def __init__(self):
#         pass
#     def jordan(self, A, B):
#         T = [A[i][:] + B[i] for i in range(len(A))]
#         z = len(T)
#         for i in range(z):
#             dig = T[i][i] # value of diagonal elements

#             if dig == 0:
#                 for r in range(i + 1, z): # check for non-zero pivot
#                     if T[r][i] != 0:
#                         T[i], T[r] = T[r], T[i] # swap
#                         break
#                 dig = T[i][i]
#                 if dig == 0: # again if diagonal is zero then lines are parallel and no solution
#                     return "No Solution"

#             for j in range(i, len(T[0])):
#                 T[i][j] /=  dig
#             for k in range(z):
#                 if k != i:
#                     f = T[k][i] # factor to eliminate
#                     for l in range(i, len(T[0])):
#                         T[k][l] -= f * T[i][l]
#         # return A               
#         C = []
#         for w in range (z):
#             C.append(T[w][-len(B[0]):])
#         return C
    
#     def identity(self, mat):
#         l = len(mat)
#         return [[1 if i == j else 0 for j in range(l)] for i in range(l)]
# class LU_decomp:
#     def __init__(self):
#         pass
#     def storeLU(self,A): # Gives the elements of L and U in single Matrix
#         val = len(A)
#         # ============== Using DOOLITTLE Composition ==================== #
#         for i in range(val):
#             A[0][i] = A[0][i]
#         for j in range(1,val):
#             A[j][0] = A[j][0] / A[0][0]
#         for i in range(1,val):
#             for j in range(1,val):
#                 if j<=i:
#                     sum = 0
#                     for k in range(0, j):
#                         sum+= A[j][k]*A[k][i]
#                     A[j][i] -= sum
#                 else:
#                     sum = 0
#                     for k in range(0, i):
#                         sum += A[j][k]*A[k][i]
#                     A[j][i] = (A[j][i] - sum) / A[i][i]
#         return A
#     def forback(self,A,B):
#         val = len(A)
#         result = self.storeLU(A)

#         # Forward substitution: solve L * Y = B
#         Y = [[0] for _ in range(val)]
#         Y[0][0] = B[0][0] 

#         for i in range(1, val):
#             sum1 = 0
#             for j in range(i):
#                 sum1 += result[i][j] * Y[j][0]
#             Y[i][0] = B[i][0] - sum1            

#         # Back substitution: solve U * X = Y
#         X = [[0] for _ in range(val)]
#         X[-1][0] = Y[-1][0] / result[-1][-1]

#         for i in range(val-2, -1, -1):
#             sum2 = 0
#             for j in range(i+1, val):
#                 sum2 += result[i][j] * X[j][0]
#             X[i][0] = (Y[i][0] - sum2) / result[i][i]
        
#         return X
    
#     def det(self, A):
#         D = self.storeLU(A)
#         p = 1
#         for i in range(len(D)):
#             p *= D[i][i]
#         return p
# class hermi_cholesky:
#     def __init__(self):
#         pass
#     def cholesky(self, A):
#         n = len(A)
#         for i in range(n):
#             sum1 = 0
#             for j in range(i):
#                 sum1 += A[i][j]**2
#             A[i][i] = (A[i][i] - sum1)**0.5
#             for j in range(n):
#                 if i < j :
#                     sum2 = 0
#                     for k in range(i):
#                         sum2 += A[i][k]*A[k][j]
#                     A[i][j] = (A[i][j] - sum2) / A[i][i]
#                     A[j][i] = A[i][j]
#         return A
#     def cholesky_solve(self, A, B):
#         n = len(A)
#         L = self.cholesky(A)
        
#         # Forward substitution to solve L * Y = B
#         Y = [[0] for _ in range(n)]
#         for i in range(n):
#             sum1 = 0
#             for j in range(i):
#                 sum1 += L[i][j] * Y[j][0]
#             Y[i][0] = (B[i][0] - sum1) / L[i][i]
        
#         # Back substitution to solve L^T * X = Y
#         X = [[0] for _ in range(n)]
#         for i in range(n-1, -1, -1):
#             sum2 = 0
#             for j in range(i+1, n):
#                 sum2 += L[j][i] * X[j][0]
#             X[i][0] = (Y[i][0] - sum2) / L[i][i]
        
#         return X
# class jacobi:
#     def __init__(self):
#         pass
    
#     def jacobi_iter(self,A,B,seed):
#         n = len(A)
#         D = [[A[i][j] if i==j else 0 for j in range(n)] for i in range(n)]
#         T = [[A[i][j] if i != j else 0 for j in range(n)] for i in range(n)]

#         for t in range(1,100): # iterative steps
#             new_seed = [[0] for _ in range(n)]
#             for i in range(n):
#                 sum1 = 0
#                 for j in range(n):
#                     if i != j:
#                         sum1 += T[i][j] * seed[j][0]
#                 new_seed[i][0] = (B[i][0] - sum1) / D[i][i]

#             if all(abs(new_seed[i][0] - seed[i][0]) < 10**(-6) for i in range(n)):
#                 # check if the difference is difference is very less for more precision!
#                 return new_seed, t
#             seed = new_seed # change the seed

# class gauss_seidel:
#     def __init__(self):
#         pass
    
#     def gauss_iter(self, A, B, x):
#         n = len(A)
        
        
#         for k in range(100):
#             x_new = [[0] for _ in range(n)]
            
#             for i in range(n):
#                 sum1 = 0
#                 sum2 = 0
#                 for j in range(n):
#                     if j < i:
#                         sum1 += A[i][j] * x_new[j][0]
#                     if j > i:
#                         sum2 += A[i][j] * x[j][0]

#                 x_new[i][0] = (B[i][0] - sum1 - sum2) / A[i][i]
            
#             if all(abs(x_new[i][0] - x[i][0]) < 10**(-6) for i in range(n)):
#                 return x_new, k + 1
            
#             x = x_new

# class roots:
#     def __init__(self):
#         pass
#     def bracketing(self,f, a, b):
#         beta = 0.1
#         while f(a) * f(b) >= 0: # ensure that the root is bracketed
#             if abs(f(a)) < abs(f(b)):
#                 a-= beta*(b-a)
#             else:
#                 b+= beta*(b-a)
#         return a,b # return the bracketing interval
#     def bisecting(self,f,a,b,tol):
#         while abs((b - a) / 2) > tol: # while the interval is larger than tolerance
#             mid = (a + b) / 2
#             if f(mid) == 0:
#                 return mid
#             elif f(a) * f(mid) < 0:
#                 b = mid
#             else:
#                 a = mid
#         return (a + b) / 2 # return midpoint
    
#     def regula_falsi(self,f, a, b, tol):
#         while abs(b - a) > tol:
#             c = (a * f(b) - b * f(a)) / (f(b) - f(a))  # Regula Falsi formula
#             if f(c) == 0:
#                 return c  # Found exact root
#             elif f(a) * f(c) < 0:
#                 b = c  # Root is in left subinterval
#             else:
#                 a = c  # Root is in right subinterval
        

#         return (a + b) / 2  # Return midpoint as the root approximation
    
#     def fixed_point(self, g, x0, tol):
#         max_iter = 100
#         for i in range(max_iter):
#             x1 = g(x0)
#             if abs(x1 - x0) < tol:
#                 return x1 # write "return x1, i" if you want the number of iterations
#             x0 = x1
#         return x0  # Return the last approximation if not converged

#     def newton_raphson(self, f, df, x0, tol):
#         max_iter = 100
#         for i in range(max_iter):
#             fx0 = f(x0)
#             dfx0 = df(x0)

#             x1 = x0 - fx0 / dfx0 # Iterative formula
#             if abs(x1 - x0) < tol:
#                 return x1 # write "return x1, i" if you want the number of iterations
#             x0 = x1
#         return x0  # Return the last approximation if not converged
    
#     def multivar_fixed(self, g, x0, tol):
#         # g and x0 are lists of functions and initial guess
#         max_iter = 100
#         n = len(x0)
#         for i in range(max_iter):
#             x1 = [0] * n
#             for j in range(n):
#                 x1[j] = g[j](*x0)  # Evaluate g_j at the current guess x0

#             if all(abs(x1[j] - x0[j]) < tol for j in range(n)):
#                 return x1, i
#             x0 = x1
#         return x0  # Return the last approximation if not converged
    
#     # multivariable newton raphson for n variables
#     def multivar_newton(self, f, J, x0, tol):
#         # f is a list of functions, J is a function that returns the Jacobian matrix
#         max_iter = 100
#         n = len(x0)
#         for i in range(max_iter):
#             F = [f[j](*x0) for j in range(n)]  # Evaluate f at the current guess x0
#             J_matrix = J(x0)  # Get the Jacobian matrix at x0

#             # Solve J * delta = -F using Gaussian elimination or any other method
#             delta = gauss_jordan().jordan(J_matrix, [[-F[j]] for j in range(n)])

#             x1 = [x0[j] + delta[j][0] for j in range(n)]  # Update the guess

#             if all(abs(x1[j] - x0[j]) < tol for j in range(n)):
#                 return x1, i  
#             x0 = x1
#         return x0  # Return the last approximation if not converged

# class poly_roots:
#     def __init__(self):
#         pass
#     def pol(self,F,x): # F is list of coefficients of polynomial
#         l = len(F)
#         sum = 0
#         for i in range(l):
#             sum += F[i]*(x**(l-i-1))
#         return sum
     
#     def defltn(self,P,r): # divide P by divisor r
#         L = []
#         L.append(P[0])
#         for i in range(1,len(P)):
#             L.append(P[i] + r*L[i-1])
        
#         return L[:-1]

#     def D(self,P): # derivative of a polynomial function
#         l = len(P)
#         L = []
#         for i in range(l-1):
#             L.append(P[i]*(l-1-i))
#         return L

#     def laguerre(self,P, r, tol, iter=100):
#         n = len(P) - 1
#         for _ in range(iter):
#             if abs(self.pol(P, r)) < tol:
#                 return r
            
#             G = self.pol(self.D(P), r) / self.pol(P, r)
#             H = G * G - self.pol(self.D(self.D(P)), r) / self.pol(P, r)
#             denom1 = G + ((n - 1) * (n * H - G * G))**0.5
#             denom2 = G - ((n - 1) * (n * H - G * G))**0.5
#             if abs(denom1) > abs(denom2):
#                 a = n / denom1
#             else:
#                 a = n / denom2
#             r -= a
#         return r 

#     def laguerre_all(self,P, tol):
#         roots = []
#         poly = P[:]
#         while len(poly) > 2:
#             root = self.laguerre(poly, 1.0, tol)  # start guess = 1.0
#             roots.append(root)
#             poly = self.defltn(poly, root)
#         if len(poly) == 2:
#             roots.append(-poly[1] / poly[0])
#         return roots