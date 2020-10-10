
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
X = data['data']
Y = data['target']
print (X)
print(Y)
# Number of training data
ni = X.shape[0]
# Number of features in the data
ci = X.shape[1]
print(ni)
print(ci)

clrs = 10*["g","r","c","b","k","y","w"]


class kMean:
    def __init__(self, k=3, tolerance=0.0001, max_iteration=100):
        self.k = k
        self.tolerance = tolerance
        self.max_iteration = max_iteration

    def clusters(self,X):
        self.cntrds = {}    #empty dictionary for centroid values
        for i in range(self.k):
            self.cntrds[i] = X[i]

        for i in range(self.max_iteration):      #optimization begins
            self.classes = {}

            for i in range(self.k):
                self.classes[i] = []

    # Calculating distances
            for featureset in X:
                dstncs = [np.linalg.norm(featureset-self.cntrds[centroid]) for centroid in self.cntrds]
                classification = dstncs.index(min(dstncs))
                self.classes[classification].append(featureset)

            prev_centroids = dict(self.cntrds)

        # Finding the mean of the features
            for classification in self.classes:
                self.cntrds[classification] = np.average(self.classes[classification],axis=0)

            optimized = True
            # calculating the centroids till optimization
            for c in self.cntrds:
                original_centroid = prev_centroids[c]
                current_centroid = self.cntrds[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tolerance:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break   


classifier = kMean()   
classifier.clusters(X)

# Plotting the graph 
for centroid in classifier.cntrds:
    plt.scatter(classifier.cntrds[centroid][0], classifier.cntrds[centroid][1],
                marker="o", color="k", s=100, linewidths=4)

for classification in classifier.classes:
    color = clrs[classification]
    for featureset in classifier.classes[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=100, linewidths=4)

plt.show()
