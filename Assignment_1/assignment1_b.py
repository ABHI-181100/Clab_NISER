
#--------------- determining the value of pi-----------
#  ----------------abhinav raj 2311006---------------


import matplotlib.pyplot as plt
import numpy as np      
from mylib import *

mylib1 = lcg()
pie = []
num = []
n = 10000
for j in range(20,n):
    num.append(j)
    x_rand = mylib1.l_c_g(567, 1103, 12345, 32768, j)
    y_rand = mylib1.l_c_g(5679, 117679, 12345, 32768, j)
    p=[]
    q=[]
    for i in x_rand:
        p.append(i/32768)
    for i in y_rand:
        q.append(i/32768)
    
    count=0
    for i in range(len(p)):
        if p[i]**2 + q[i]**2 <= 1:
            count += 1
      
    pi_estimate = (count / j) * 4
    pie.append(pi_estimate)
   
print("mean value is : ", np.mean(pie))
plt.plot(num,pie, color='blue')
plt.title('Estimate of Pi')
plt.show()


#####################################################################

# mean value is :  3.116475167343179

######################################################################


# ------------------random no. in exponential form ------


from mylib import *
import numpy as np
import matplotlib.pyplot as plt

mylib1 = lcg()
x_rand = mylib1.l_c_g(567, 1103, 12345, 32768, 10000)
L=[]
for i in x_rand:
    L.append(i/32768)
# print(L)

p=[]
for i in L:
    p.append(-np.log(i))

plt.hist(p, bins=30, alpha=0.5, color='blue', edgecolor='black')
plt.show()

