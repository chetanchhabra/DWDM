import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
def loadData():
	ll=[]
	with open('data.csv','r') as csvreader:
		rows=csv.reader(csvreader);
		for row in rows:
			ll.append([int(x) for x in row]);
		return ll;

if __name__ == "__main__":
	data=loadData()
	data=np.array((data));
	np.random.shuffle(data);
	norm = np.linalg.norm(data,axis=1);
	maxThreshold = int(max(norm))/2;
	n_samples = len(data);
	n_attributes = len(data[1]);
	matrix = np.zeros((n_samples,n_samples));

	for i in xrange(n_samples):
		sample1 = np.array(data[i]);

		for j in xrange(n_samples):
			sample2=np.array(data[j]);
			matrix[i,j] = np.linalg.norm(sample1-sample2);

	print matrix
	k_count = np.zeros(maxThreshold);
	for Threshold in xrange(maxThreshold,maxThreshold+1):
		activities = np.zeros(n_samples)
		w=np.zeros((n_samples,n_samples))
		for i in xrange(n_samples):
			for j in xrange(n_samples):
				w[i,j] = float(Threshold**2)/(matrix[i,j]**2 + Threshold**2)
				print w[i,j]
				if w[i,j] >= 0.5:
					pass
				else:
					w[i,j]=0;
			for j in range(n_samples):
				activities[i]=activities[i]+w[i,j]
			if activities[i] < 1:
				activities[i] = 0;

			
		t=0;
		alpha=0.5
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
				k_count[Threshold-1] = np.count_nonzero(activities);
				break;
	#print k_count
	print activities
	non_zero_index = np.nonzero(activities)
	print non_zero_index
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
	print classes
	
	fig=plt.figure()
	ax=fig.add_subplot(111,projection='3d')

	symbol = ['o','^','+']
	centre_symbol = ['*','D','H']
	for i in range(n_samples):
		if i not in non_zero_index[0]:
			ax.scatter(data[i,0],data[i,1],data[i,2],c=data[i]/255.0,marker=symbol[classes[i]])
		else:
			ax.scatter(data[i,0],data[i,1],data[i,2],c=data[i]/255.0,marker=centre_symbol[classes[i]])
	plt.show()


	





