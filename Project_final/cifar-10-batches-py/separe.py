import numpy as np
import random
import math
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
images = [[] for i in range(5)]
images[0] = unpickle('./data_batch_1')
images[1] = unpickle('./data_batch_2')
images[2] = unpickle('./data_batch_3')
images[3] = unpickle('./data_batch_4')
images[4] = unpickle('./data_batch_5')


#print len(images[0]['data'])
#print images[0]['data']

#n,w are prameters in paper 219 page. In this example, n = 32,  w=10,  
#l2 is a new list or matrix: 1000*1024*3(3 =r g b)
# another list stores 40 thousands of images in 10*10(rgb)
#new_list for return
#reshape the liste in 32*32 to simplify for separe 
#return 10*10*4*5000 = 2000000
def separe( data, n, w):
	m = np.reshape(data,(len(data),3072))
	l = []
	for image in m:
		r = image[0:1024]
		g = image[1024:2048]
		b = image[2048:3072]
		
		r = np.dstack((r,g))
		r = np.dstack((r,b))
		r_new = r[0] 
		#print r_new
		np.reshape(r_new,(32,32,3))
		l.append(r_new)
	new_list = []	
	
	for i in l:
		for j in range(n):
			for k in range(n):
				if (k < w) and (j < w):
					new_list.append(i[j+n*k]) 
					#print i[j+n*k]
				if (j < w) and (k >= n - w):
					 new_list.append(i[j+k*n])	
				if (j >= n-w) and (k<w):
					 new_list.append(i[j+k*n])
				if (j >= n-w) and (k >= n - w):
					 new_list.append(i[j+k*n])
	#print len(new_list)
	return new_list
'''
n1 = []	
n2 = []
n1 = separe(images[0]['data'][0:5000],32,10)
n2 = separe(images[0]['data'][5000:10000],32,10)
n1.extend(n2)

n3 = []
n4 = []
n3 = separe(images[1]['data'][0:5000],32,10)
n4 = separe(images[1]['data'][5000:10000],32,10)
n3.extend(n2)

n5 = []
n6 = []
n5 = separe(images[2]['data'][0:5000],32,10)
n6 = separe(images[2]['data'][5000:10000],32,10)
n5.extend(n6)

n7 = []
n8 = []
n7 = separe(images[3]['data'][0:5000],32,10)
n8 = separe(images[3]['data'][5000:10000],32,10)
n7.extend(n8)
'''
n9 = []
n10 = []

n9 = separe(images[4]['data'][0:5000],32,10)
n10 = separe(images[4]['data'][5000:10000],32,10)			
n9.extend(n10)
print "len of n9"
print len(n9)
#make all the 2000000*2 together (one batch)  
'''
n7.extend(n9)
n5.extend(n7)
n3.extend(n5)
n1.extend(n3)'''

#Eu distance between two vector
def Dis_eu(vector1, vector2):
    sqDiffVector = vector1-vector2
    sqDiffVector=sqDiffVector**2
    sqDistances = sqDiffVector.sum()
    distance = sqDistances**0.5
    return distance


#calcul k_means
'''
1.[r, g, b] -> get the means for both center and data
2.cycle whiel: if there are still some change of the center, we continue the cycle. And I note the nearest cneter[j] for data[i]  in the label_data[](label_data and data_value have the same indice).
3.we use the label_data to calcule the new center.
'''
def k_means(data,center_num):
        data_value = []
        new_center = []
        center_value = []
	#print data[1]	 
	# choose center, center = center_num * 100	
	for i in range(center_num):
		a = random.sample(data,100)
		center_value.append(a)
	# compute the average value of center(num = center_num) 
	for i in range(center_num):		
		center_value[i] = np.mean(center_value[i],axis = 0)	
       # print center_value
	#comoute the average value of data(5000*4 = 20 milles)
	for j in range(10000):
		for k in range(4):
			data_value.append(np.mean(data[j*k:j*k+100],axis = 0))
        print "len of data_value"
        print len(data_value)
	label_data = []
	clusterChanged = True
	minDis = [[1000]*len(data_value)]
        t = []
        #initialisition of vector_car
        vector_car = np.zeros((len (data_value), center_num))
	while (clusterChanged):
		clusterChanged = False
		for i in range(len(data_value)):
			for j in range(center_num):
				distemp = Dis_eu(center_value[j], data_value[i])
				if distemp < minDis[0][i]:
					minDis[0][i] = distemp; minIndex = j
			ancien_label = label_data			 
			label_data.append(minIndex)
			if ancien_label != label_data:
				clusterChanged = True
		for i in range(center_num):
                        #sys.exit(0)
			for j in range(len(label_data)):
				if label_data[j] == i:
					t.append(data_value[j])
                                
                        #sys.exit(1)
                        x = np.mean(t[:], axis = 0)
                        if  x == None:
                            del t[:]
                        else:    
                            center_value[i] = x;  del t[:]
			#print center_value
	#get the list of vector_car		
	for i in range(len(data_value)):
                for j in range(center_num):
                        if label_data[i] == j :
                                vector_car[i][j] = 1

	return center_value
    
#get the center for vector of test_data
center = k_means(n9,30)	

#calcul the vector for test_batch
def caractaire(test_data,center_value):
        data_value = []
        
        center_num = len(center_value);
        
        for j in range(10000):
		for k in range(4):
			data_value.append(np.mean(test_data[j*k:j*k+100],axis = 0))
        
        minDis = [[10000]*len(data_value)]
        print len (data_value )
        print len(data_value[0])
        label_data = [];
        clusterChanged = True
        while (clusterChanged):
		clusterChanged = False
		for i in range(len(data_value)):
			for j in range(center_num):
				distemp = Dis_eu(center_value[j], data_value[i])
				if distemp < minDis[0][i]:
					minDis[0][i] = distemp; minIndex = j
			ancien_label = label_data			 
			label_data.append(minIndex)
			if ancien_label != label_data:
				clusterChanged = True
				
	vector_car = np.zeros((len (data_value), center_num))			
        for i in range(len(data_value)):
            for j in range(center_num):
                if label_data[i] == j :
                    vector_car[i][j] = 1
        
        return vector_car 
    
    
    
        
test_batch = unpickle('./test_batch')
test_data = separe(test_batch['data'][:], 32,10)
#lenth of test_data is 40000
test_car = caractaire(test_data, center)

from sklearn import svm, grid_search, datasets
#use the svm fonction

print "llongth of test_car"
print len(test_car[1])


# change the caractaire vector from 40000*30 to 10000*120 
test = []
for i in range(40000):
    for j in range(30):
        test.append(test_car[i][j])
test = np.reshape(test,(10000,120))
print "lenth of test"
print len(test)

def apprentissage(test):
            clf = svm.SVC()
            clf.fit(test)
            clf.predict(test)

apprentissage(test)	
	
