"""
    Linear regression with one variable using gradient descent
    ----------------------------------------------------------
    Source code based on Andrew Ng's excellent machine learning course and
    assignments (https://www.coursera.org/course/ml). Most of the code is 
    converted from Octave/Matlab code to python/numpy. 
"""

import sys
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


class gradientdescent(object):
    """
    Single feature gradient descent routine.
    
    Parameters
    ----------
    X : numpy array [ndata,2] dimension
        X-axis data points.
        
    y : numpy vector
        Y-axis data point vector
        
    theta : numpy vector
        Initial guess value of parameters
        
    alpha : float
        Learning rate
        
    niters : int
        Number of iterations
    """
    def __init__(self, X, y, theta, alpha = 0.01, niters = 500):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.theta = np.asarray(theta)
        self.alpha = float(alpha)
        self.niters = int(niters)
        
        self.ndata = len(self.y)


    def computecost(self):
        """
        Compute cost function for linear regression with multiple
        variables. 
        
        Parameters
        ----------        
        theta : numpy array
            Parameters of linear regression
            
        Returns
        -------
        J : float
            Cost function value
        """
        J = 0.0
    
        J = (1/(2.*self.ndata)) * np.sum((np.dot(self.X, self.theta) - self.y)**2)
    
        return J
    
    
    def gradientdescent(self):
        """
        Function to perform gradient descent to learn theta.
    
        Parameters
        ----------
        X : numpy array
            X-axis data points
        
        y : numpy array
            Y-axis data points
        
        theta : numpy array
            Linear regression parameters
        
        alpha : float
            Learning rate
        
        niters : int
            number of iterations
    
        Returns
        -------
        theta : numpy array
            Linear regression parameters
        
        J_history : numpy array
            Cost function values
        """        
        J_history = np.zeros(niters)
    
        for i in range(self.niters):
            tmp = np.zeros(self.X.shape[1])
        
            for j in range(self.X.shape[1]):
                tmp[j] = self.theta[j] - (self.alpha/self.ndata) * np.sum((np.dot(self.X, self.theta) - self.y) * self.X[:,j])
            
            self.theta = tmp
            J_history[i] = self.computecost()
    
        return self.theta, J_history
        

def computecost(X, y, theta):
    """
    Compute cost function for linear regression with multiple
    variables. 
        
    Parameters
    ----------
    X : numpy array
        X-axis data point array
    
    y : numpy array
        Y-axis data point vector
                
    theta : numpy array
        Parameters of linear regression
            
    Returns
    -------
    J : float
        Cost function value
    """
    ndata = len(y)
    J = 0.0
        
    J = (1/(2.*ndata)) * np.sum((np.dot(X, theta) - y)**2)
    
    return J


if __name__ == "__main__":
    # Exception if input file is not provided
    # Example - python gradientdescent.py ex1data1.txt
    if len(sys.argv) < 2:
        print "Usage: python gradientdescent.py data_file"
        sys.exit(-1)
        
    # Read input data (only two column separated by a comma
    X, y = np.loadtxt(sys.argv[1], delimiter = ",", unpack = True)

    # Number of iterations and learning rate
    niters = 1500
    alpha = 0.01

    # Construct X vector with first column as 1's and starting guess
    # parameter values (theta)
    X_new = np.vstack((np.ones_like(X), X)).T
    theta = np.zeros(X_new.shape[1])

    # gradientdescent object and perform gradient descent
    gd = gradientdescent(X_new, y, theta, alpha, niters)
    theta, j_hist = gd.gradientdescent()

    print "Linear Regression Equation: ", theta[0], " + ", theta[1], " * X"

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.dot(np.array([1, 3.5]),theta)
    print "For population = 35,000, we predict a profit of %5.6f" %(predict1*10000) 

    predict2 = np.dot(np.array([1, 7.0]), theta)
    print "For population = 70,000, we predict a profit of %5.6f" %(predict2*10000) 

    # Plot data points and overlay linear regression line        
    fig = plt.figure()
    plt.title("Linear Regression using Gradient Descent")
    plt.xlabel("Population (x10,000)")
    plt.ylabel("Profit (x$10,000)")
    plt.plot(X, y, "rx")
    plt.plot(X, theta[0] + X * theta[1], 'k-')
    plt.show()
    
    # Visualizing cost function on surface plot and as a contour
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    theta0, theta1 = np.meshgrid(theta0_vals,theta1_vals)
    J_vals = np.zeros([len(theta0_vals),len(theta1_vals)])
    
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array([theta0_vals[i], theta1_vals[j]])
            J_vals[i,j] = computecost(X_new, y, t);
                
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.xlabel(r"${\theta}_0$")
    plt.ylabel(r"${\theta}_1$")
    ax.plot_surface(theta0, theta1, J_vals.T, rstride=1, cstride=1, cmap = plt.cm.rainbow)
    plt.show()

    fig = plt.figure()
    plt.xlabel(r"${\theta}_0$")
    plt.ylabel(r"${\theta}_1$")
    plt.contour(theta0, theta1, np.log10(J_vals))
    plt.show()
