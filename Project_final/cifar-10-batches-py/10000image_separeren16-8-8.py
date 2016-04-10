#coding=utf-8
from sklearn import svm
import numpy as np
from numpy import *
import random
import operator
import time
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
w = 8
k = 30


#这个函数是网站上的读取图片的函数
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
#我将图片都读进来，放进了一个链表
images = [[] for i in range(5)]
images[0] = unpickle('./data_batch_1')
images[1] = unpickle('./data_batch_2')
images[2] = unpickle('./data_batch_3')
images[3] = unpickle('./data_batch_4')
images[4] = unpickle('./data_batch_5')
#print len(images[0]['data'])
#print images[0]['data']

#fonction definir**************************************************

#n,w are prameters in paper 219 page. In this example, n = 32,  w=10,  
#l2 is a new list or matrix: 1000*1024*3(3 =r g b)
# another list stores 40 thousands of images in 10*10(rgb)
#new_list for return
#reshape the liste in 32*32 to simplify for separe 
#return N*10*10*4
def separe( data, n, w):
	l = []
	for image in data:
		r = image[0:1024]
		g = image[1024:2048]
		b = image[2048:3072]
		
		r = np.dstack((r,g))
		r = np.dstack((r,b))
		r_new = r[0]
		r_new = np.reshape(r_new,(32,32,3))
		l.append(r_new)
		
	new_list = []	
	for i in l: 
            #1
            new_list.append(i[:w,:w])
            new_list.append(i[:w, w:2*w]) 
            new_list.append(i[:w, 2*w:3*w])
            new_list.append(i[:w, 32-w:])
            #2
            new_list.append(i[w:32-2*w, :w])
            new_list.append(i[w:32-2*w, w:2*w])
            new_list.append(i[w:32-2*w, 2*w:3*w])
            new_list.append(i[w:32-2*w, 32-w:])
            #3
            new_list.append(i[2*w:32-w, :w])
            new_list.append(i[2*w:32-w, w:2*w])
            new_list.append(i[2*w:32-w, 2*w:3*w])
            new_list.append(i[2*w:32-w, 32-w:])
            #4
            new_list.append(i[32-w:, :w])
            new_list.append(i[32-w:, w:2*w])
            new_list.append(i[32-w:, 2*w:3*w])            
            new_list.append(i[32-w:, 32-w:])
	#print len(new_list)
	#print len(new_list[399][9])
	return new_list



#Eu distance between two vector
def Dis_eu(vector1, vector2):
    return np.linalg.norm(np.array(vector1) - np.array(vector2))

#calcul k_means
'''
1.[r, g, b] -> get the means for both center and data
2.cycle whiel: if there are still some change of the center, we continue the cycle. And I note the nearest cneter[j] for data[i]  in the label_data[](label_data and data_value have the same indice).
3.we use the label_data to calcule the new center.
center_num is *4 time than the real number
'''
print "Calcul the Centroids"
def returnCneter(dataSet,k):
    n = shape(dataSet)[1]
    
    center = mat(zeros((k,n)))
    for i in range(n):
        minJ = min(dataSet[:,i])
        rangeJ = float(max(dataSet[:,i]) - minJ)
        center[:,i] = minJ + rangeJ*(np.random.rand(k,1))
    print "succsses calcul the centroids"
    return center
  
def k_means(dataSet, k, distMeas=Dis_eu, createCent=returnCneter):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
    print"................................."
    print len(centroids)
    print len(centroids[0])
    print len(centroids[0][0])
    for cent in range(k):#recalculate centroids
        ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
        centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment



#用的包的数量有限。。。
n1 = []	
n2 = []
n1 = separe(images[0]['data'][0:5000],32,w)
print len(n1)
#print (n1[1])
n2 = separe(images[0]['data'][5000:10000],32,w)
n1.extend(n2)
print (len(n1))
print len(n1[1])
print len(n1[0][0])

n1 = np.reshape(n1,(40000,768))
print "n1 changdu"
print "n1 1 shisha"


n7 = []
n8 = []
n7 = separe(images[3]['data'][0:5000],32,w)
n8 = separe(images[3]['data'][5000:10000],32,w)
n7.extend(n8)
n7 = np.reshape(n7,(40000,768))


print "data_separe no problem"
#make all the 2000000*2 together (one batch)  

n = shape(n1)[1]
print "lenth of n1[1] is k*k*3"
#Eu distance between two vector
def Dis_eu(vector1, vector2):
    return np.linalg.norm(np.array(vector1) - np.array(vector2))


#calcul k_means
'''
1.[r, g, b] -> get the means for both center and data
2.cycle whiel: if there are still some change of the center, we continue the cycle. And I note the nearest cneter[j] for data[i]  in the label_data[](label_data and data_value have the same indice).
3.we use the label_data to calcule the new center.
center_num is *4 time than the real number
'''
#这里是随机获得质心的函数。就是聚类的中心
print "Calcul the Centroids"
def returnCneter(dataSet,k):
    n = shape(dataSet)[1]
    #随机生成数组的函数，初始值为零。维度为k×n
    center = mat(zeros((k,n)))
    for i in range(n):
        minJ = min(dataSet[:,i])
        rangeJ = float(max(dataSet[:,i]) - minJ)
        center[:,i] = minJ + rangeJ*(np.random.rand(k,1))
    print "succsses calcul the centroids"
    return center
#Kmeans，这个应该不用说了  
def k_means(dataSet, k, distMeas=Dis_eu, createCent=returnCneter):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
    print"................................."
    print len(centroids)
    print len(centroids[0])
    print len(centroids[0][0])
    for cent in range(k):#recalculate centroids
        ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
        centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

center, clust_Label = k_means(n1,k)	
print len(center)
print len(center[0][0])
#calcul the vector for test_batch
#获得特征向量的函数。比如，我有1w张图片，切成4块，每块是16×16 = 256,获得的就是1w×4×256×3,下面要把它的形状变成40000×768，方便计算
def caractaire(data,center):
    m = np.shape(data)[0]
    print len(center)
    
    label = []
    for i in range(len(data)):
        for j in range(len(center)):
            label.append(0)
    label = np.reshape(label,(len(data),len(center)))
    
    print len(label)
    print len(label[0])
    

    flag = True
    while flag:
        flag =False
        for i in range(len(data)):
            minDist = inf
            minIndex = -1 
            for j in range(len(center)):
                if minDist>Dis_eu(data[i],center[j]):
                    minDist = Dis_eu(data[i],center[j])
                    minIndex = j
            label[i][minIndex] = 1
    print label[0]
    print label[1],label[2], label[3]
 
    return label 
    
#先处理一下要检测的目标，算出他的特征向量
test_batch = unpickle('./test_batch')
test_data = separe(test_batch['data'][:], 32,w)
test_data = reshape(test_data,(40000,768))
#lenth of test_data is 4000000
print "calcul caractaires vector for test_batch"
test_car = caractaire(test_data, center)
#print len(test_car)
#print len(test_car[0])


#！！！此处需注意 ： 计算n1包的图片特征向量，把特征向量和对应的初始label一一对应起来，通过SVM进行机器学习，然后把要检测的目标函数传进去，通过apprentissag函数的clf进行标签预测处理，再将预测标签和原始标签对比，得到正确率。
n1_car = caractaire(n1,center)


test = []
test = np.reshape(test_car,(len(test_car)/4,k*4))
 
n1_car_nor = []

n1_car_nor = np.reshape(n1_car,(len(n1_car)/4,k*4))   
print len(n1_car_nor)
print len(n1_car_nor[0])
#use the svm fonction
# change the caractaire vector from 40000*30 to 10000*120 
#apprendissage and give the vector labels.
def apprentissage(learn, test, labels):
            clf = svm.SVC()
	    
            clf.fit(learn, labels)
            
            return clf.predict(test)
            #print"teaux de correct", clf.score(test, c)
#apprentissage(n1_car_nor, test, images[0]["labels"])            
test_relabels = apprentissage(n1_car_nor,test , images[0]["labels"][:])



print "llongth of test_car"
a = []
for i in test_relabels :
    a.append(i)
b = []
print len(test_batch["labels"])
for i in test_batch["labels"] :
    b.append(i)

#test if the labels are right or not
def Final_test(a, b) : 
    temp = []
    for i in range(10):
        temp.append(0)
    counter = 0.000
    for i in range(len(a)) : 
        if a[i] == b[i] : 
            temp[a[i]] = temp[a[i]]+1
            counter = counter +1.00
    for i in range(10):
        print "classe i",i,"correct number: ",temp[i]
    print counter
    print ("taux correct :  is %.2f "%((counter / len(a))*100.0)) + "%"
Final_test(a, b)    
        
        
        