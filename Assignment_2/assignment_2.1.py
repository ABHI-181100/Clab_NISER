
# ----------abhinav raj 2311006-------------------------------
# ------------question 1 solving linear equation for 3 variable --------


from mylib import read_matrix , gauss_jordan

matrix1 = read_matrix().read_matrix1("t1.txt")
matrix2 = read_matrix().read_matrix1("t2.txt")
value = gauss_jordan().gauss_jordan1(matrix1,matrix2)   


print("The value of x,y,z are: ", value[0][-1],value[1][-1],value[2][-1])


# ##################---------output----###########################

# The value of x,y,z are:  -2.0 -2.0 1.0

########################################################################



# --------------2nd question solving linear equation  for 5 variable-------

matrix3 = read_matrix().read_matrix1("t3.txt")
matrix4 = read_matrix().read_matrix1("t4.txt")
value2 = gauss_jordan().gauss_jordan1(matrix3, matrix4)
print(value2)
for i in range(len(matrix3)):
    print(f"value of a_{i+1} is ", value2[i][-1])

################-------output---#############################

# value of a_1 is  -1.7618170439978567
# value of a_2 is  0.8962280338740136
# value of a_3 is  4.051931404116157
# value of a_4 is  -1.6171308025395428
# value of a_5 is  2.041913538501914
# value of a_6 is  0.15183248715593495

#############################################################
