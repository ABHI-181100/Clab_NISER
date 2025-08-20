
# ----------abhinav raj 2311006-------------------------------
# ------------question 1 solving linear equation for 3 variable --------


from mylib import read_matrix , gauss_jordan

matrix1 = read_matrix().read_matrix1("t1.txt")
matrix2 = read_matrix().read_matrix1("t2.txt")
value = gauss_jordan().gauss_jordan1(matrix1,matrix2)       


print("The value of x,y,z are: ", value[0][-1],value[1][-1],value[2][-1])

# --------------2nd question solving linear equation  for 5 variable-------

matrix3= read_matrix().read_matrix1("t3.txt")
matrix4 = read_matrix().read_matrix1("t4.txt")
value2 = gauss_jordan().gauss_jordan1(matrix3,matrix4)

for i in range(len(matrix3)):
    print(f"a_{i+1} is", value2[i][-1])