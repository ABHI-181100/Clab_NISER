# # --------------------code for generating random no. plot---------
# # ------------------abhinav raj   2311006------------

def random(c,n):
    X= [0.1]  
    for i in range(n):
        p = c * X[i]*(1-X[i])
        X.append(p)
        
    return X
        

import matplotlib.pyplot as plt

def plot_graph(x, y):
    plt.scatter(x, y, marker='o', color='b', linestyle='-')
    plt.title('Random Number Generation')
    plt.xlabel('')
    plt.ylabel('Random Number')
    plt.grid(True)
    plt.show()


def newlist(Y,k):
    a=[]
    b=[]
    for i in range (len(Y)):
        a.append(Y[i])
        if (i+k)<len(Y):
            b.append(Y[i+k])

    return b



Y=random(3.98,1000)     
y=Y[:-5]
x=newlist(Y,5)
plot_graph(y,x)


Y=random(3.7,1000)
y=Y[:-5]
x=newlist(Y,5)
plot_graph(y,x)

Y=random(3.987654321,1000)
y=Y[:-5]
x=newlist(Y,5)
plot_graph(y,x)

Y=random(3.98,1000)
y=Y[:-10]
x=newlist(Y,10)
plot_graph(y,x)


# #------------------------ linear congruential generator-------------

class lcg():
    def __init__(self):
        pass

def linear_congruential_generator(seed, a, c, m, n):
    random_numbers = []
    x = seed
    for _ in range(n):
        x = (a * x + c) % m
        random_numbers.append(x)
    return random_numbers

random_numbers = linear_congruential_generator(0.1,1103515245,12345,32768,1000)
print(random_numbers) 

Y=random_numbers         #ploting for k=5
y=Y[:-5]
x=newlist(Y,5)
plot_graph(y,x)
