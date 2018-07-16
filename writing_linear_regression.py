#writing linear regression from scratch

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,7,6], dtype=np.float64)

def best_fit_slope(xs,ys):
    m =( ((mean(xs) * mean(ys)) - mean(xs*ys)) / 
        ((mean(xs)* mean(xs)) - mean(xs**2)))
    b = mean(ys) - (m * mean(xs))
    return m, b

def squared_error(ys_orig, ys_line): #squared error is the difference btwn orig y to the linear y.
    return sum((ys_line - ys_orig)**2)

def coefficient_of_determination(ys_orig,ys_line):
    ys_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, ys_mean_line)
    return 1- (squared_error_regr / squared_error_y_mean)

m, b = best_fit_slope(xs,ys)

regression_line = [(m*x)+b for x in xs]

predict_x = 8
predict_y = (m*predict_x)+b

r_squared = coefficient_of_determination(ys, regression_line)
print (r_squared)

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()