import numpy as np
import matplotlib.pyplot as plt
import csv,math
def loadData():
	ll=[]
	with open('iris.data','r') as csvreader:
		rows=csv.reader(csvreader);
		for row in rows:
			if len(row) > 1:
				ll.append([float(x) for x in row[:-1]]);
		return ll;

if __name__ == "__main__":
	data=loadData()
	data=np.array((data));
	norm = np.linalg.norm(data,axis=1);
	maxThreshold = int(math.ceil((max(norm))/2));
	n_samples = len(data);
	n_attributes = len(data[1]);
	matrix = np.zeros((n_samples,n_samples));

	for i in xrange(n_samples):
		sample1 = np.array(data[i]);

		for j in xrange(n_samples):
			sample2=np.array(data[j]);
			matrix[i,j] = np.linalg.norm(sample1-sample2);

	stepsize=0.1
	maxThreshold=matrix.max()/2

	print matrix
	print maxThreshold
	
	thRange=np.arange(0.1,maxThreshold,stepsize)
	print thRange

	k_count = [];
	for Threshold in thRange:
		activities = np.zeros(n_samples)
		w=np.zeros((n_samples,n_samples))
		for i in xrange(n_samples):
			for j in xrange(n_samples):
				w[i,j] = float(Threshold**2)/(matrix[i,j]**2 + Threshold**2)

				#print w[i,j]
				if w[i,j] >= 0.5:
					pass
				else:
					w[i,j]=0;
			for j in range(n_samples):
				activities[i]=activities[i]+w[i,j]
			if activities[i] < 1:
				activities[i] = 0;

			
		t=0;
		alpha=0.1
		counter=0;
		while(True):
			for i in xrange(n_samples):
				total = 0;
				if activities[i] >0:
					for j in xrange(n_samples):
						if activities[j] > 0:
							total = total+w[i,j]*(activities[i]-activities[j])
					activities[i]=activities[i] + alpha*total;
					if activities[i] <=0:
						activities[i]=0;
			#print np.count_nonzero(activities)
			counter+=1;
			if counter == 100:
				k_count.append( np.count_nonzero(activities));
	
				break;
		#print Threshold

	print k_count
	print len(thRange)
	print Threshold
	#print activities
	non_zero_index = np.nonzero(activities)
	#print non_zero_index
	classes = np.zeros(n_samples,dtype=np.int)

	for i in xrange(n_samples):
		sample1 = data[i,:]
		distance = np.linalg.norm(sample1-data[non_zero_index[0][0]]);
		Class = 0;
		for j in xrange(len(non_zero_index[0])):
			dist = np.linalg.norm(sample1-data[non_zero_index[0][j]]);
			if dist < distance:
				distance = dist;
				Class = j;
		classes[i] = Class
	
	plt.xlabel('Threshold')
	plt.ylabel('Number Of Clusters ')
	plt.plot(thRange,k_count,'-o')
	plt.show()	


