# -----question no. 1 ----code to find the sum of first 20 odd numbers------
# -------------------Abhinav raj-----roll no: 2311006----------



s=0
k=0
i=1
for i in range(50):
        if i % 2 != 0:
            while k < 20:
                k += 1
                s += i
                break
print("sum of first 20 odd numbers is:", s)


# ------question no. 1)b)-----code to find the sum of factorial of n=8 starting from 0------


n = 8
factorial = 1
for i in range(1, n + 1):
    factorial *= i
print("sum of factorial of n=8 starting from 0 is:", factorial + 1)  # Adding 1 for the factorial of 0




# ------question no. 2)a)-------code for sum of terms of GP having common ratio is 0.5  and first term is 1.25----------

n=15
a=1.25
r=0.5
sum_gp = 0
for i in range(n):
    terms = a * r** i
    sum_gp += terms
print("sum of first 15 terms of GP is:", sum_gp)


# -----------question no. 2)b)---code for sum of HP terms having common difference is 1.5  and first term is 1.25----------

n=15
a=1.25
d=1.5
sum_hp = 0
for i in range(15):
    terms = 1/(a + i*d)
    sum_hp += terms
print("sum of first 15 terms of HP is:", sum_hp)

# ---------question no. 3)---------code for matrix multiplication----------------

from mylibrary import *

mylib = read_matrix()

A = mylib.read_matrix1('asgn0_matA')
B = mylib.read_matrix1('asgn0_matB')
C = mylib.read_matrix1('asgn0_vecC')
D = mylib.read_matrix1('asgn0_vecD')

mylib2 = matrix_multiplication()

print("Matrix A:", A)
print("Matrix B:", B)
print("Vector C:", C)
print("Vector D:", D)



X = mylib2.matrix_multiply(A, B)
print("matrix AB is :", X)
Y = mylib2.matrix_multiply(B, C)
print("matrix BC is :", Y)

Z= mylib2.dot_product(C,D)
print("dotpro:",Z)


# --------question no. 4)------for algebric operation b/w complex nos.----------


from mylibrary import *

# C1= 1.3 - 2.2j
# C2 = -0.8 +1.7j

mc=mycomplex(1.3,-2.2)
mc2=mycomplex(-0.8,1.7)


print("Complex number 1:", mc.display_complex())
print("Complex number 2:", mc2.display_complex())


# res= mc.sum(mc2.real,mc2.img)
# print("sum of given complex no. is : ", res)

# res1 = mc.sub(mc2.real,mc2.img)
# print("sub of given complex no. is : ", res1)

# res2 = mc.multiply(mc2.real,mc2.img)
# print("mul of given complex no. is : ", res2)

# print("modulus of given complex no. is : ", mc.modulus())

# ------here mc representing self so it take their real and img. as argument in init one....----------

res= mc.sum("+", mc2.real,mc2.img)
print("sum of given complex no. is : ", res)

res1 = mc.sum("-",mc2.real,mc2.img)
print("sub of given complex no. is : ", res1)

res2 = mc.sum("*",mc2.real,mc2.img)
print("mul of given complex no. is : ", res2)

print("modulus of given complex no. is : ", mc.modulus())