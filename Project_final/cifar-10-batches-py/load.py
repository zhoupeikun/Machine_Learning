import numpy as np

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
center = [[] for i in range(10)]
def naive(images , center):
	for k in images:
		for i in range(10000):
			#下标作为Label
			center[k['labels'][i]].append(k['data'][i])
			#center[images['labels'][i]].append(images['labels'][i])
	for i in range(10):		
		center[i] = np.mean(center[i],axis = 0)		
	return center
center = naive(images, center)


test = unpickle('./test_batch')
def calcul(OB,center):
	k = 0.0	
	distance = [0 for i in range(10)]
	for i in range(10000):
		distance = center - OB['data'][i]
		distance = np.linalg.norm(distance,axis = 1)
		#返回label
		labels = np.argmin(distance)
		#label与原来数据做比较
		if labels == OB['labels'][i]:
			k = k + 1
	k = k/100	
	return k

taux = calcul(test, center)
print taux
