# linear regression
# author: Eli Pandolfo
#
# Linear regression: create a function that can be used to extrapolate y values
# based on given x values. Uses a data set containing (x,y) datapoints to create
# a line of best fit; plug in x values outside the dataset to extrapolate the
# corresponding y values (or within the dataset to interpolate).

import numpy as np
import matplotlib.pyplot as plt

# creates a linear regression model (a line of best fit in the form y = mx + b)
# from a dataset
# every x value should have a corresponding y value
def best_fit(xdata, ydata):
	# turn lists into nparrays if they are not already
	if len(xdata) != len(ydata):
		print(len(xdata), " x data points and", len(ydata), "y data points; please ensure your xdata and ydata correspond.\n Your results will be skewed.")
	m = (np.mean(xdata) * np.mean(ydata) - np.mean(xdata * ydata)) / (np.mean(xdata)**2 - np.mean(xdata**2))
	b = np.mean(ydata) - m * np.mean(xdata)
	return m, b

# computes the squared error (distance between each y coordinate and the line
# of best fit, squared, summed)
def squared_error(ydata, yline):
	return sum((yline - ydata)**2)

# computes the coefficient of determination (R squared) for the regression model
# takes in the original x and y coordinates, and the coefficients of the
# regression equation
def r_squared(xdata, ydata, m, b):
	y_predicted = [(m * x + b) for x in xdata]
	y_predicted_SE = squared_error(ydata, y_predicted)

	y_mean = [np.mean(ydata) for y in ydata]
	y_mean_SE = squared_error(ydata, y_mean)

	return 1 - y_predicted_SE / y_mean_SE

# takes in a list of x values to extrapolate upon, and the coefficients of a
# line of best fit
def predict(x_new, m, b):
	return [(m * x + b) for x in x_new]

def reg_model(xdata, ydata, x_new):
	m, b = best_fit(xdata, ydata)
	return predict(x_new, m , b), m, b

# def main():
# 	xdata = np.array([1,2,3,4,5])
# 	ydata = np.array([100,91,77,81,54])
# 	x_new = [6,7,8,9,10]
# 	plt.scatter(xdata, ydata, color='b', label='known data')
#
# 	y_new, m, b = reg_model(xdata, ydata, x_new)
# 	y_predicted = [(m * x + b) for x in xdata]
#
# 	regression_label = 'Regression line: y = ' + str(m) + 'x + ' + str(b) + '\nR-squared: ' + str(r_squared(xdata, ydata, m, b))
#
# 	plt.plot(xdata, y_predicted, color='k', label=regression_label)
# 	plt.scatter(x_new, y_new, color='g', label='extrapolated data')
#
# 	plt.legend(loc='best')
# 	plt.xlabel('x')
# 	plt.ylabel('y')
# 	plt.title('Regression Analysis Example')
# 	plt.savefig('lin_reg.pdf', bbox_inches='tight')
# 	plt.show()
#
# main()
