import numpy as np

n = 10 ** 6
data = 10 ** 9 + np.random.uniform(0, 1, n)

somme = 0

for i in data:
    somme += i
moyenne = somme/(len(data))

# print moyenne
# print np.mean(data)
# print np.var(data)

def variance(moyenne, data):
    s = 0
    for i in data:
        s += (moyenne - i) * (moyenne - i)
    return s / (len(data))

# print variance(moyenne, data)

def Welford(p):
    m = p[0]
    temp = 0
    s = 0
    for k in range(len(p) - 1):
        temp = p[k] - m
        m = m + temp / (k+1)
        s = s + temp * (p[k] - m)
    return s/len(p)

# print Welford(data)