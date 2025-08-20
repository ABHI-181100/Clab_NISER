import numpy as np

class matrix_multiplication():
    def __init__(self):
        pass
    # def matrix_multiply(self,A, B):
    #     result=[]
    #     for i in range(len(A)):
    #         row=[]
    #         for j in range (len(B[0])):
    #             product_sum=0
    #             for k in range (len(B)):
    #                 product_sum += A[i][k]*B[k][j]

    #             row.append(product_sum)
    #         result.append(row)
    
    #     return result

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
                row = [float(num) for num in line.strip().split()]
                matrix.append(row)

        return matrix
    
class read():
    def __init__(self):
        pass

    def read_matrix1(file):
        return np.loadtxt(file)

    def read_vector(file):
        return np.loadtxt(file)

    
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
    