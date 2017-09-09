# Masyu Strip Data
# author: Eli Pandolfo
#
# Strips data from an input masyu board and puts it into a dataframe
# dataframe has a row for every square on the board, and strips data for:
#	board corner: 0, 1 (yes or no)
#	board edge (not corner): 0, 1
#	top left: 0, 1, 2, 3 (off board, black, white, neither)
#	top mid: 0, 1, 2, 3
#	top right: 0, 1, 2, 3
#	left: 0, 1, 2, 3
#	right: 0, 1, 2, 3
#	bottom left: 0, 1, 2, 3
#	bottom mid: 0, 1, 2, 3
#	bottom right: 0, 1, 2, 3
#	type: N/A, C, (nothing, corner, line)
#
# Uses that dataframe to run a k nearest neighbors machine learning classifier
# and predict the type of square (vertical line, horizontal line, corner, or
# unknown), and print a board with those values filled in for each square, with
# the black and white squares remaining as they were and with high confidence
# labels capitalized.
# Right now, I am using as features the type (b, w, 0) of the 8 adjacent squares
# for each square on the board (excluding white and black squares since those
# are known)

import numpy as np
import pandas as pd
import k_nearest_neighbors

class Masyu_solver(k_nearest_neighbors.K_nearest):

	#right now, the board is set at 10x10 but the real puzzle can have nxn
	def __init__(self):
		self.rows = 0
		self.cols = 0

	# given an array representing a board and a row, col location,
	# classifies the object at that square as b, w, or neither
	def get_feature(self, array, r, c):
		if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
			return -99999
		elif array[r][c] == 'b':
			return 1
		elif array[r][c] == 'w':
			return 2
		else:
			return 3

	# reads a txt file of boards separated by commas, with rows separated by
	# line breaks, into a dataframe whose index is each location on each board
	# and whose columns are the attributes I am testing for: each adjacent
	# square, and whether the current square is a corner or board edge
	def strip_known(self, filename, allow_bw=False):
		with open(filename, 'r') as b_file:
			b_string = b_file.read()
			b_list = b_string.split(',')

		self.rows = 10
		self.cols = 10
		b_list = [b_str.split('\n') for b_str in b_list]
		b_list = [[list(row) for row in board] for board in b_list]

		df = pd.DataFrame(columns=['board_corner', 'board_edge', 'top_left',
							'top_mid', 'top_right', 'left', 'right',
							'bottom_left', 'bottom_mid', 'bottom_right',
							'type_'])
		for array in b_list:
			for r, row in enumerate(array):
				for c, obj in enumerate(row):
					if (allow_bw or (not allow_bw and str(obj) != 'b' and str(obj) != 'w')):
						index = (r, c)
						board_corner = 1 if (index == (0, 0) or index == (0, self.cols - 1)
							or index == (self.rows - 1, 0)
							or index == (self.rows - 1, self.cols - 1)) else 0
						board_edge = 1 if (index[0] == 0 ^ index[0] == self.rows - 1
							^ index[1] == 0 ^ index[1] == self.cols - 1) else 0
						top_left = self.get_feature(array, index[0] - 1, index[1] - 1)
						top_mid = self.get_feature(array, index[0] - 1, index[1])
						top_right = self.get_feature(array, index[0] - 1, index[1] + 1)
						left = self.get_feature(array, index[0], index[1] - 1)
						right = self.get_feature(array, index[0], index[1] + 1)
						bottom_left = self.get_feature(array, index[0] + 1, index[1] - 1)
						bottom_mid = self.get_feature(array, index[0] + 1, index[1])
						bottom_right = self.get_feature(array, index[0] + 1, index[1] + 1)
						type_ = obj

						append_dict = {'board_corner': board_corner,
										'board_edge': board_edge,
										'top_left': top_left,
										'top_mid': top_mid,
										'top_right': top_right,
										'left': left,
										'right': right,
										'bottom_left': bottom_left,
										'bottom_mid': bottom_mid,
										'bottom_right': bottom_right,
										'type_': type_}
						name = '(' + str(r) + ', ' + str(c) + ')'
						df = df.append(pd.Series(append_dict, name=name),
												 ignore_index=True)
		for col in df.columns:
			if col != 'type_':
				df[col] = df[col].astype(int)
		return df

	# applies the k nearest neighbors classifier on the data stripped from
	# the board map
	def apply_k_nearest(self, train_file, test_file, drop_cols):
		k = k_nearest_neighbors.K_nearest()
		features = self.strip_known(train_file)
		labels = features['type_']
		features.drop(drop_cols, inplace=True, axis=1)
		predict_features = self.strip_known(test_file)
		predict_features.drop(drop_cols, inplace=True, axis=1)
		predicted_labels = k.k_nearest_neighbors(features, labels,
													predict_features)
		avg_conf = np.mean(predicted_labels.loc[:]['confidence'])
		for row in predicted_labels.index:
			if predicted_labels.loc[row]['confidence'] < .5:
				predicted_labels.set_value(row, 'label', '0')
			elif predicted_labels.loc[row]['confidence'] == 1:
				predicted_labels.set_value(row, 'label',
					predicted_labels.loc[row]['label'].upper())
		return predicted_labels

	# displays the board with b and w in their correct location, with empty
	# squares filled in with the predicted labels
	def print_board(self, predicted_labels, test_file):
		predict_features = self.strip_known(test_file, allow_bw=True)
		print_list = []
		lnum = 0
		for fnum in range(predict_features.shape[0]):
			if fnum != 0 and fnum % self.cols == 0:
				print_list.append('\n')
			if str(predict_features['type_'][fnum]) == 'b':
				print_list.append('b')
			elif str(predict_features['type_'][fnum]) == 'w':
				print_list.append('w')
			else:
				print_list.append(predicted_labels.loc[lnum]['label'])
				lnum += 1
		print(''.join(print_list))

def main():
	solver = Masyu_solver()
	predicted = solver.apply_k_nearest('masyu_train.txt', 'masyu_test.txt',
		drop_cols=['type_'])
	solver.print_board(predicted, 'masyu_test.txt')

main()
#add shit to do classical analysis first, then use k nearest
