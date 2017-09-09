# k nearest neighbors
# author: Eli Pandolfo
#
# implementation of the K nearest neighbors machine learning algorithm in python
# using numpy
# K nearest neighbors: think of an n dimensional graph, where n is the number
# of different categories of inputs (or attributes). If you are using 10
# different measures to predict an output (or label/classification), imagine a 10D
# matrix/graph. Find how close a sample datapoint (with 10 different coordinates/features,
# (x,y,z,a,b,c,d,e,f,g)) is to the rest of the datapoints, whose outputs/labels/
# classes are known, and thus predict the class of the sample datapoint
#
# distance is measured with euclidian distance: sqrt (sum from 1 to n (n = dimesnions) of
# (q_i - p_i)^2), where q_i and p_i are the coordinates in dimension i of 2 data
# points

import numpy as np
import pandas as pd
from math import sqrt

class K_nearest():

	def __init__(self):
		pass
	# simple python algorithm to compute distance between 2 n-dimensional points
	# p1 and p2 can be lists
	def euclidian_dist1(self, p1, p2):
		if len(p1) != len(p2):
			print('p1 has', len(p1), 'dimensions and p2 has', len(p2), 'dimensions; please input points with the same number of dimensions')
			return -1
		dist = 0
		for i in range(len(p1)):
			dist += (p1[i] - p2[i])**2
		return sqrt(dist)

	# But, linear algebra and numpy can compute euclidian distance way faster than that algorithm:
	def euclidian_dist(self, p1, p2):
		if len(p1) != len(p2):
			print('p1 has', len(p1), 'dimensions and p2 has', len(p2), 'dimensions; please input points with the same number of dimensions')
			return -1
		return np.linalg.norm(np.array(p1) - np.array(p2))

	# computes the mode and confidence of a list
	# example: [a,b,b,c,a,a,] has mode a and confidence 3/6 = .5
	# important to note that confidence is not the same as accuracy;
	# the model's accuracy is unrelated to the confidence of each predicted label
	def mode(self, l):
		keys = set(l)
		mode_dict = dict.fromkeys(keys, 0)
		for label in l:
			mode_dict[label] += 1;
		v = list(mode_dict.values())
		k = list(mode_dict.keys())
		conf = max(v) / len(l)
		return {'label': k[v.index(max(v))], 'confidence': conf}

	# classify a data point based on a pandas dataframe of features, a pandas
	# series of labels of the same length, a sample datapoint with known features
	# and an unknown label, and a value for k.
	# if no k value is provided, the program will use 1 more than the number of features,
	# to ensure no vote is split
	def k_nearest_neighbors(self, features, labels, predict_features, k=0):
		if k <= len(labels.unique()):
			k = len(labels.unique()) + 1
			print('setting k to', k)
		if features.shape[0] != labels.shape[0]:
			print('features and labels must have corresponding (the same number of) rows')
			return 'Error'
		if features.shape[1] != predict_features.shape[1]:
			print('features and predict_features must have the same number of columns (features must correspond)')
			return 'Error'

		predicted_labels = pd.DataFrame(columns=['label', 'confidence'])
		for i in range(predict_features.shape[0]):
			distances = []
			for j in range(features.shape[0]):
				distances.append([self.euclidian_dist(predict_features.loc[i], features.loc[j]), labels.loc[j]])
			votes = [l[1] for l in sorted(distances)[:k]]
			result_dict = self.mode(votes)
			predicted_labels = predicted_labels.append(result_dict, ignore_index=True)
		return predicted_labels


# def main():
# 	features = {'a': pd.Series([1,2,7,3,8]), 'b': pd.Series([2,3,7,3,9]), 'c': pd.Series([1,1,9,1,7]), 'd': pd.Series([1,2,6,1,9])}
# 	features = pd.DataFrame(features)
#
# 	labels = pd.Series(['nurp', 'nurp', 'durp', 'nurp', 'durp'], name='labels')
#
# 	predicted_features = {'a': pd.Series([1,2,6]), 'b': pd.Series([1,3,1]), 'c': pd.Series([1,1,7]), 'd': pd.Series([1,2,9])}
# 	predicted_features = pd.DataFrame(predicted_features)
#
# 	predicted_labels = k_nearest_neighbors(features, labels, predicted_features)
#
# 	features = features.join(labels)
# 	print(features)
# 	print()
# 	predicted_features = predicted_features.join(predicted_labels)
# 	print(predicted_features)
