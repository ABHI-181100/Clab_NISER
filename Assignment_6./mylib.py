
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
            for _ in range(iter):
                c = b - (f(b)*(b-a)) / (f(b)-f(a))
                if abs(f(c)) < acc or abs(b - a) < acc:
                    return c
                if f(a) * f(c) < 0:
                    b = c
                else:
                    a = c
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
            for _ in range(iter):
                c = (a + b) / 2
                if abs(f(c)) < acc or abs(b - a) < acc:
                    return c
                if f(a) * f(c) < 0:
                    b = c
                else:
                    a = c
            return (a + b) / 2
        else:
            return "No root bracketed in [a, b]"