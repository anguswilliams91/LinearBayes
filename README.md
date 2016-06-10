# LinearBayes

A python implementation of the Bayesian approach to fitting a straight 
line to two-dimensional data with covariant errors on the two coordinates. 
Depends on `numpy`, `scipy`, `matplotlib` and `emcee`.

Clone the repo, and then run

python bayes_line.py

to generate some mock data and fit it.

Example code:

```python
import matplotlib.pyplot as plt
import numpy as numpy
import linear_bayes

#generate some mock data and fit it
data,true_parameters = linear_bayes.mock_data()
results = linear_bayes.fit_data(data,outfile="mcmc_output",make_cornerplot=False) 

#make a plot of the inference vs. data
x = np.linspace(np.min(data[:,0]),np.max(data[:,0]))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(data[:,0],data[:,1],xerr=data[:,2],yerr=data[:,3],ecolor='k',fmt='none',alpha=0.5)
ax.plot(x,true_parameters[0]*x+true_parameters[1],label="truth")
ax.plot(x,results[:,0][0]*x+results[:,0][1],label="inferred")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.legend(loc='lower right')
```

![Alt text](example.png?raw=true)