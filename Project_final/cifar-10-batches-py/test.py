import numpy as np
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

#data, labels,batch_label, filenames

images = unpickle('./data_batch_1')#.values()[0]
data = images['data']
#print data
for key in images:
    print key, images[key];
# print images.values()[0]
# values = images.values()

'''
def kmeans(values , k=10):
    center = []
    for j in range(10):
        for i in values:
            if i[1] = j-1:
                center[j].append(i[0])
    center_1 = np.zeros(3072)
    center_2 = np.zeros(3072)+255
    center_3 = np.zeros(3072)+510
    center_4 = np.zeros(3072)+765
    center_5 = np.zeros(3072)+1020
    center_6 = np.zeros(3072)+1275
    center_7 = np.zeros(3072)+1530
    center_8 = np.zeros(3072)+1785
    center_9 = np.zeros(3072)+2040
    center_10 = np.zeros(3072)+2295
    for i in range(10):
        distance_1 = np.linalg.norm(images - center_1, axis=1)
        distance_2 = np.linalg.norm(images - center_2, axis=1)
        classes = np.array([1 if x <= 0 else 2 for x in distance_1 - distance_2])
        center_1 = np.mean(images[classes==1], axis=0)
        center_2 = np.mean(images[classes==2], axis=0)
    return classes
print kmeans(values,10)'''