# support vector machine
# author: Eli Pandolfo
#
# a machine learning classifier that uses vectors to classify groups of data
# attempts to create an n-D hyperplane (more than 2d plane) between 2 different
# sets of n-dimensional data
#
# This version, made from scratch, is horribly inefficient and should never
# be used on real datasets; since it solves the convex optimization problem
# with brute forcing different combinations of w and b, it takes a ridiculously
# long time to fit even with tiny data sets. I wrote this to understand how
# SVMs work, not to use for classifying data.

import numpy as np
import pandas as pd

class Support_Vector_Machine:
	def __init__(self):
		pass

	# find support vector w and offset b such that,
	# given y = labels (scalars), and x = features (vectors):
	# y(x.w + b) >= 1, where ||w|| is minimized and b is maximized
	#
	def train(self, features, labels):
		max_featL, min_featL = (max(features.values.tolist()),
								(min(features.values.tolist())))
		max_feat, min_feat = max(max_featL), min(min_featL)
		transform_vects = [[1, 1], [-1, 1], [1, -1], [-1, -1]]
		steps = [max_feat * .1, max_feat * .01, max_feat * .001]
		b_step = 5
		b_step_multiplier = 5
		curr_w = max_feat * 10
		optimized = {}

		for step in steps:
			w = np.array([curr_w, curr_w])
			min_found = False
			while not min_found:
				for b in np.arange(-1 * max_feat * b_step_multiplier,
									max_feat * b_step_multiplier,
									max_feat * step):
					for vec in transform_vects:
						tr_w = w * vec
						found = True
						for i in features.index:
							yi = labels[i]
							if not (yi * (np.dot(tr_w, features.loc[i].values.tolist()) + b) >= 1):
								found = False
								break
						if found:
							optimized[np.linalg.norm(tr_w)] = [tr_w, b]
				if w[0] < 0:
					min_found = True
				else:
					w = w - step
			mags = sorted([n for n in optimized])
			self.w = optimized[mags[0]][0]
			self.b = optimized[mags[0]][1]
			latest = self.w[0] + step * 2


	# given a set of known features (in the form of datapoints),
	# the trained SVM uses the w vector and b offset to predict the labels
	# associated with each datapoint.
	# the sign of the dot product of k_f and w, plus the offset b, is a value
	# eith er -1, 0, or 1. if -1, it belongs to one feature set; if 1, it
	# belongs to the other, and if 0, this datapoint is directly on the dividing
	# line between the two
	#
	# known features will be a pandas dataframe or a 2D ndarray
	def predict(self, known_features):
		labels = np.sign(np.dot(known_features, self.w) + self.b)
		return labels


def main():
	features = {'a': pd.Series([1,2,1,6,8,7]), 'b': pd.Series([0,2,1,9,7,8])}
	features = pd.DataFrame(features)

	labels = pd.Series([1,1,1,-1,-1,-1], name='label')

	predict_features = {'a': pd.Series([0,0,-1,4,5,6]), 'b': pd.Series([0,-2,1,6,7,9])}
	predict_features = pd.DataFrame(predict_features)

	svm = Support_Vector_Machine()
	svm.train(features, labels)

	predict_labels = svm.predict(predict_features)
	print(predict_labels)

main()
