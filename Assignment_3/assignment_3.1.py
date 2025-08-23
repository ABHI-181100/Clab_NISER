# -----------Abhinav Raj 2311006----------
# ------------Question : 1 ---------------
# Find LU decomposition of the matrix and verify. The LU decomposition routine
# must be kept in the library for later use. Mention your choice of Doolittle or Crout clearly -----
# ---------------using Dolittle method------------------------------------


from mylib import *

A = read_matrix().read_matrix1("matrixA.txt")

LU_decomposition().LU_matrix(A)

L=[[0 for _ in range(len(A))] for _ in range(len(A))]
U=[[0 for _ in range(len(A))] for _ in range(len(A))]

for i in range (len(A)):
    for j in range (len(A)):
        if i > j :
            L[i][j] = A[i][j]

        elif i <= j :
            U[i][j] = A[i][j]
            if i==j:
                L[i][i] = 1
print("Original matrix : ", read_matrix().read_matrix1("matrixA.txt"))
print(f"Lower is : {L} \n Upper is : {U}")
print("Multiplication of L and U is : ",matrix_multiplication().matrix_multiply(L,U))

###########################################################################################################
# Original matrix :  [[1.0, 2.0, 4.0], [3.0, 8.0, 14.0], [2.0, 6.0, 13.0]]
# Lower is : [[1, 0, 0], [3.0, 1, 0], [2.0, 1.0, 1]] 
#  Upper is : [[1.0, 2.0, 4.0], [0, 2.0, 2.0], [0, 0, 3.0]]
# Multiplication of L and U is :  [[1.0, 2.0, 4.0], [3.0, 8.0, 14.0], [2.0, 6.0, 13.0]]
###########################################################################################################
