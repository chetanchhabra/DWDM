import random
import csv

ll=[]
for i in xrange(50):
	r=random.randint(200,255);
	g=random.randint(0,125);
	b=random.randint(0,125);
	ll.append([r,g,b])

for i in xrange(50):
	r=random.randint(0,125);
	g=random.randint(200,255);
	b=random.randint(0,125);
	ll.append([r,g,b])

for i in xrange(50):
	r=random.randint(0,125);
	g=random.randint(0,125);
	b=random.randint(200,255);
	ll.append([r,g,b])

with open("data.csv","w") as f:
	writer=csv.writer(f);
	writer.writerows(ll);