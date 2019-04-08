#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 20:25:45 2019

@author: YuFang
"""

import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct


# Define "true" and "noisy" fxn as follows:

def func(x1,x2):
    return 0.5*np.exp(-0.5*((x1+1.25)**2+(x2+1.75)**2))+np.exp(-0.5*((x1-2.25)**2+(x2-2.65)**2))

def noisy_func(x1,x2):
    output = func(x1,x2)
    noise = np.random.normal(0,0.1,np.shape(output))
    return output + noise

# 5a)
    
# Write code for probability of improvement
    
def probability_of_improvement(mu_x,sigma_x,opt_val):
    gamma = (mu_x - opt_val)/sigma_x
    return norm.cdf(gamma)

#Write code for Expected improvement
    
def expected_improvement(mu_x,sigma_x,opt_val):
    gamma = (mu_x - opt_val)/sigma_x
    return sigma_x * gamma * norm.cdf(gamma) + sigma_x * norm.pdf(gamma)


# Write code for upper confidence bound
    
def upper_confidence_bound(mu_x,sigma_x,k):
    return mu_x + k * sigma_x

#%%

# 5b)
#define a query fxn using "scipy.optmize.minimize" 
#and the acquisition fxn of your choice

def my_acquisition_function(mu_x, sigma_x, k):
    return upper_confidence_bound(mu_x,sigma_x,k=1)


def query(opt_val,gp):
    def obj(x):
        #do gaussian process prediction
        mu_x,sigma_x = gp.predict(x.reshape(1,-1),return_std=True)
        
        return -my_acquisition_function(mu_x,sigma_x,opt_val)
    
    x0 = np.random.uniform(-5,5,2)
    res = minimize(obj, x0=x0.reshape(1,-1), bounds=([-5, 5], [-5, 5]))
    return res.x

res = 50
lin = np.linspace(-5, 5, res)
meshX, meshY = np.meshgrid(lin, lin)
meshpts = np.vstack((meshX.flatten(), meshY.flatten())).T

def add_subplot(gp, subplt):
    mu = gp.predict(meshpts, return_std=False)
    ax = fig.add_subplot(2, 5, subplt, projection = '3d')
    ax.plot_surface(meshX, meshY, np. reshape (mu, (50 , 50)) , rstride =1, cstride =1, cmap=cm. jet , linewidth=0, antialiased=False) 

if __name__ == '__main__':
   true_y = func(meshX, meshY)

   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   ax.plot_surface(meshX, meshY, true_y, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
   plt.title('True function')
   plt.show()

   fig=plt.figure(figsize=plt.figaspect(0.5))

#%%
   
# 5c)
#Complete here
   
xi=np.random.uniform(-5,5,(4,2))
yi = noisy_func(xi[:,0], xi[:,1])

#%%
   
# 5d)

#complete here
gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0))

for i in range(10):
    gp.fit(xi,yi)
    
    #find the current optimal value and its location
    opt_val = max(yi)
    opt_x = xi[np.where(yi == opt_val)]
    
    print('Best_value:_', opt_val)
    print('at_', opt_x)
    
    next_x = query(opt_val, gp)
    
    #add next_x to the list of data points   
    xi = np.append(xi, [next_x],axis=0)

    next_y = noisy_func(xi[-1][0], xi[-1][1]).reshape(1)
    
    #add next_y to the list of observations
    yi = np.append(yi,next_y)

    add_subplot(gp, i+1)

plt.show()                 




