import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from scipy.spatial import distance
import random

class KMeansClustering:
	def __init__(self):
		self.centiroid=np.zeros([3,2])
		self.iteration=300
		self.clustered_data={}
		self.clustersize=3
	
	def find_centiroid(self,a,b):#function to find the centiroid 
		centi=np.zeros([1,2])
		for k in range(2):
			centi[0][k]=(a[k]+b[k])/2
		return centi

	def getmincent(self,data,show_data=False):#function to predict which class the point may belong to
		min=1000000000
		for i,cent in enumerate(self.centiroid):
			if(show_data):
				print("Centiroid:  {}   Distance:  {}".format(cent,self.find_euc(data,cent)))
			if(self.find_euc(data,cent)<min):
				ret=i
				min=self.find_euc(data,cent)
		return ret

	def find_euc(self,data,centiroid):
		return distance.euclidean(data,centiroid)
	
	def fit(self,data,clustersize=3,numiterations=100):#fitting the data
	    self.clustersize=clustersize
	    for i in range(clustersize):
	    	self.centiroid[i]=np.array([random.randint(0,max(data[:,0])),random.randint(0,max(data[:,1]))])
	    for _ in range(numiterations):
	    	for i,each in enumerate(data):
	    		cls=self.getmincent(each)
	    		self.clustered_data[i]=[each,cls]
	    		self.centiroid[cls]=self.find_centiroid(each,self.centiroid[cls])#to update the centiroid value
		
	def predict(self,sample):
		cls=self.getmincent(sample,True)
		print("predicted class: ",cls)
	
	def showplot(self):#to display the plot
		datapoints=[]
		colors=['g','b','r','y']
		for i in range(self.clustersize):
			datapoints.append(np.array([val[0] for key,val in self.clustered_data.items() if val[1]==i]))
		for l,each in enumerate(datapoints):
			plt.scatter(each[:,0],each[:,1],color=colors[l])
		plt.scatter(self.centiroid[:,0],self.centiroid[:,1],color=colors[3])
		plt.show()

	def showData(self):
		print("---------------------CENTIROID------------------------\n")
		for i in range(self.clustersize):
			print("centiroid {}: {}\n".format(i,self.centiroid[i]))
		print("---------------------DATA POINTS----------------------\nS.NO      data        class\n")
		for key,val in self.clustered_data.items():
			print("{}         {}       {}\n".format(key,val[0],val[1]))


Kmeans=KMeansClustering()
data=np.array([[1,1],[0,1],[2,2],[1,10],[1,11],[1,12],[2,10],[6,6],[7,7],[6,7],[8,8],[7,8],[4,6],[8,2]])
Kmeans.fit(data)
Kmeans.showData()
Kmeans.showplot()
Kmeans.predict([2,8])