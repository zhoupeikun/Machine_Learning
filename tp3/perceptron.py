import numpy as np
import random


def data_reader(filename):
    to_binary = {"?": 3, "y": 2, "n": 1}
    labels = {"democrat": 1, "republican": -1}

    data = []
    for line in open(filename, "r"):
        line = line.strip()

        label = int(labels[line.split(",")[0]])
        observation = np.array([to_binary[obs] for obs in line.split(",")[1:]] + [1])
        data.append((label, observation))

    return data
#print data_reader("data.txt");

def spam_reader(filename):
    to_binary = {1: 1, 0: -1}
    data = []
    for line in open(filename, "r"):
        line = line.strip()
        label = to_binary[int(line.split(",")[-1])]
        observation = [float(obs) for obs in line.split(",")[:-1] + [1.0]]

        data.append((label, np.array(observation)))
        
    return data

def classify(array, vecteur):
    result = np.inner(array.tolist(), vecteur);
    if result >= 0 :
        return 1
    else: return -1

data = data_reader("data.txt")

def test(data, vecteur):
    nombre = 0
    for e, a in data:
        if classify(a, vecteur) != e:
            nombre += 1
    return nombre/float(len(data))

vecteur = [25, -12, 67, -104, -43, 46, -18, -10, 45, -33, 54, -39, 43, -19, 5, -2, 55];
#print test(data, vecteur)

def learn(data, nbPass, vecteur):
    for i in range(0, nbPass):
        for j in range(0, len(vecteur)):
            if classify(data[i][1], vecteur) != data[i][0]:
                vecteur[j] = vecteur[j] + data[i][1][j] * data[i][0]

def aret1(data, nbPass, vecteur):
    learn(data[nbPass:], nbPass, vecteur)
    while test(data[:nbPass], vecteur) != 0:
        random.shuffle(data)
        learn(data[nbPass:], nbPass, vecteur)
    return vecteur
#print aret1(data, 100, vecteur)
#print aret1(data, 50, vecteur)


def aret2(data, nbPass, vecteur, iter):
    for x in range(0, iter):
        random.shuffle(data)
        learn(data[nbPass:], nbPass, vecteur)
    return test(data[:nbPass], vecteur)

print aret2(data, 100, vecteur, 20)

def aret3(data, nbPass, vecteur, nbIter):
    list_taux_erreur = []
    list_iteration = []


