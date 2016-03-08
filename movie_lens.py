#le nombre de jugement et le nombre d'utilisateur
fo = open("movie_lens.csv", 'rU');
count = len (fo.readlines());
print count;

counters = {}
with open('movie_lens.csv') as fp:
	for line in fp:
		out= line.split('|');
		counters[out(0)] +=1
		print len(counters)




	