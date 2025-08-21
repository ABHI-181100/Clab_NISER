# ---------abhinav raj 2311006-------------------------------
# --------question no. 1 -------using dolittle method---------

from mylib import *

read1 = read_matrix()
A = read1.read_matrix1("matrixA.txt")
print("given matrix : ",A)


d= LU_matrix().dolittle(A)
print("L_matrix, U_matrix are: ",d)

y=matrix_multiplication().matrix_multiply(d[0], d[1])
print("matrix verified : ",y)

# ##############################################
# ---------output -----------------------------

# given matrix : [[1.0, 2.0, 4.0], [3.0, 8.0, 14.0], [2.0, 6.0, 13.0]]
# L_matrix, U_matrix are:  ([[1, 0, 0], [3.0, 1, 0], [2.0, 1.0, 1]], [[1.0, 2.0, 4.0], [0, 2.0, 2.0], [0, 0, 3.0]])
# matrix verified : [[1.0, 2.0, 4.0], [3.0, 8.0, 14.0], [2.0, 6.0, 13.0]]

#######################################################3


