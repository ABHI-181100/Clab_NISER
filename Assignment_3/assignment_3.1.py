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


from mylib import *

A = read_matrix().read_matrix1("t3.txt")
B = read_matrix().read_matrix1("t4.txt")
LU_decomposition().LU_matrix(A)
n = len(A)

for i in range(n):
    print(f"value of a_{i+1} is : ", solve().solve2(A,B)[i][0])

##################################################
# ----------output----------------------------

# value of a_1 is :  -1.761817043997862
# value of a_2 is :  0.8962280338740133
# value of a_3 is :  4.051931404116158
# value of a_4 is :  -1.6171308025395421
# value of a_5 is :  2.041913538501913
# value of a_6 is :  0.15183248715593525   


################################################

