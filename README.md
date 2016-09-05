# LinearBayes

A simple `python 2` implementation of the Bayesian approach to fitting a straight 
line to two-dimensional data with covariant errors on the two coordinates and some outliers 
(see e.g. [Hogg, Bovy & Lang (2010)](http://arxiv.org/abs/1008.4686) for discussion). Depends 
on `numpy`, `scipy`, `matplotlib` and [`emcee`](https://github.com/dfm/emcee).

<<<<<<< HEAD
Download the tarball and then run

`pip install LinearBayes-master.tar.gz`
=======
To install using `pip`, download the zipped repo and run

`pip install LinearBayes-master.zip`
>>>>>>> new_version

to install the module (and any dependicies which aren't already installed). The function that 
does all of the work is called `fit_data` and has an explanatory docstring.

Example snippet, which runs a test on some mock data and produces two plots (shown below):

```python
import linear_bayes as lb

lb.mock_test(nproc=8) #run the test case and parallelise over 8 threads

```

![Alt text](mock_fit.png?raw=true)
![Alt text](mock_triangle.png?raw=true)