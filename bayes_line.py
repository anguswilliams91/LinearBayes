from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import os
import emcee
from bayes_triangle import triangle_plot

"""This code will fit a straight line with intrinsic dispersion to data with (optionally) covariant 
errors on both the independent and dependent variables, and takes care of outliers using a mixture 
model approach. Requires packages numpy,scipy,emcee and matplotlib. 

The main function is fit_data, which does the fitting and returns the results (and optionally makes a 
corner-plot.

There is a main() function so type "python bayes_line.py" to generate some mock data and then fit it.
"""

def likelihood_line(params,data):
    """the likelihood for the linear function. 
    Args:
        (1) np.ndarray or list, params = [slope,intercept,intrinsic standard deviation]
        (2) np.ndarray, data. Should have shape (N,4) (if no covariances on \
            errors) or (N,5) (if covariant errors).
    Returns:
        (1) float, the likelihood of the data given this set of model parameters.
    """
    m,b,sigma_intrinsic = params
    theta = np.arctan(m)
    #unpack the data
    if data.shape[1]==4:
        #no correlations on the errors
        x,y,dx,dy = data.T 
        dxy = np.zeros_like(x)
    elif data.shape[1]==5:
        #correlations
        x,y,dx,dy,dxy = data.T
    else:
        raise ValueError("data must have 4 or 5 columns, not {}. \
                Try transposing your data.".format(data.shape[1]))
    sint,cost = np.sin(theta),np.cos(theta)
    delta = -sint*x+cost*y-cost*b #perpendicular distance to the line
    Sigma_dd = sint**2.*dx**2. - np.sin(2.*theta)*dxy + cost**2.*dy**2. #projection of covariance matrix along line
    return (2.*np.pi*(Sigma_dd+sigma_intrinsic**2.))**-.5 * np.exp(-delta**2. / (2.*(Sigma_dd+sigma_intrinsic**2.)))


def outlier_distribution(params,data):
    """The likelihood for the outlier distribution, which is modelled as a uniform \
    distribution in x and a Gaussian distribution in y with some mean y0 and standard \
    deviation sigma0. 
    Args:
        (1) np.ndarray or list, params = [outlier y mean, outlier standard deviation]
        (2) np.ndarray, data. Should have shape (N,4) (if no covariances on \
            errors) or (N,5) (if covariant errors). Should be in the order \
            (x,y,dx,dy) or (x,y,dx,dy,dxy).
    Returns:
        (1) float, the likelihood of the data given this set of model parameters.
    """
    if data.shape[1]==4:
        #no correlations on the errors
        x,y,dx,dy = data.T 
        dxy = np.zeros_like(x)
    elif data.shape[1]==5:
        #correlations
        x,y,dx,dy,dxy = data.T
    else:
        raise ValueError("data must have 4 or 5 columns, not {}. \
                Try transposing your data.".format(data.shape[1]))
    y_outlier,sigma_outlier = params
    sigma_total2 = sigma_outlier**2.+dy**2.
    return (2.*np.pi*sigma_total2)**-0.5 * np.exp(-.5*(y-y_outlier)**2./sigma_total2)

def full_log_likelihood(params,data):
    """The log-likelihood of the data given the full mixture model of the linear function & \
    the outlier distribution. 
    Args:
        (1) np.ndarray or list, params = [slope,intercept,intrinsic scatter,outlier mean,\
            outlier standard deviation, outlier fraction]
        (2) np.ndarray, data. Should have shape (N,4) (if no covariances on \
            errors) or (N,5) (if covariant errors). Should be in the order \
            (x,y,dx,dy) or (x,y,dx,dy,dxy).
    Returns:
        (1) float, the likelihood of the data given this set of model parameters.
    """
    m,b,sigma_intrinsic,y_outlier,sigma_outlier,outlier_fraction = params
    return np.sum(np.log((1.-outlier_fraction)*likelihood_line([m,b,sigma_intrinsic],data)+\
            outlier_fraction*outlier_distribution([y_outlier,sigma_outlier],data)))

def log_priors(params,priorlimits):
    """Prior probabilities on the parameters, given upper and lower limits on each parameter. \
    Jeffreys priors are used for the intrinsic and outlier standard deviations, and a prior \
    that is flat in Arctan(slope) is used for the slope. For everything else, priors are uniform \ 
    within the given limits. 
    Args:
        (1) np.ndarray or list, params = [slope,intercept,intrinsic scatter,outlier mean,\
            outlier standard deviation, outlier fraction]
        (2) np.ndarray or list, priorlimits. Upper and lower values for each of the model parameters \
            (except the outlier fraction which has a flat prior between 0 and 1). 
    Returns:
        (1) float, the prior density of these parameters.
    """
    m,b,sigma_intrinsic,y_outlier,sigma_outlier,outlier_fraction = params
    mlo,mhi,blo,bhi,silo,sihi,yolo,yohi,solo,sohi = priorlimits
    if m<mlo or m>mhi or b<blo or b>bhi or sigma_intrinsic<silo or sigma_intrinsic>sihi or \
       sigma_outlier<solo or sigma_outlier>sohi or y_outlier<yolo or y_outlier>yohi or\
       outlier_fraction<0. or outlier_fraction>1.:
        return -np.inf
    else: 
        return -np.log(1.+m*m) - np.log(sigma_intrinsic) - np.log(sigma_outlier)

def full_posterior(params,data,priorlimits):
    """The log-posterior of the data given the full mixture model of the linear function & \
    the outlier distribution. 
    Args:
        (1) np.ndarray or list, params = [slope,intercept,intrinsic scatter,outlier mean,\
            outlier standard deviation, outlier fraction]
        (2) np.ndarray, data. Should have shape (N,4) (if no covariances on \
            errors) or (N,5) (if covariant errors). Should be in the order \
            (x,y,dx,dy) or (x,y,dx,dy,dxy).
        (3) np.ndarray, priorlimits. Upper and lower values for each of the model parameters \
            (except the outlier fraction which has a flat prior between 0 and 1). 
    Returns:
        (1) float, the posterior of the parameters given the data.
    """
    if log_priors(params,priorlimits)==-np.inf:
        return -np.inf
    else:
        return log_priors(params,priorlimits)+full_log_likelihood(params,data)

def mock_data():
    """Generate a mock sample of 100 + a few data points with errors to test the \
    fitting machinery.
    Args:
        This function takes no arguments.
    Returns:
        (1) np.ndarray data. A mock data-set with (x,y,dx,dy,dxy) as the columns.
        (2) np.ndarray params. The parameters of the model.
    """
    #generate random slope and intercept
    slope = np.float(np.random.uniform(1.,4.,1))
    intercept = np.float(np.random.uniform(-3.,3.,1))
    sigma_intrinsic = np.float(np.random.uniform(0.1,.3,1))
    theta = np.arctan(slope)
    #generate coordinates along the line with random intrinsic spread
    gamma = np.random.uniform(1.,20.,100)
    delta = np.random.normal(loc=0.,scale=sigma_intrinsic,size=100)
    #now transform to x and y
    sint,cost = np.sin(theta),np.cos(theta)
    xp = cost*gamma - sint*delta 
    yp = sint*gamma + cost*delta + intercept
    #now generate x and y errors
    dx = np.abs(np.random.normal(loc=0.,scale=0.3,size=100))
    dy = np.abs(np.random.normal(loc=0.,scale=0.3,size=100))
    rho = np.random.uniform(-1.,1.,size=100) #correlation parameters
    dxy = rho*dx*dy #off-diagonal terms
    #now scatter xp and yp by these errors
    x,y = np.zeros_like(xp),np.zeros_like(xp)
    for i in np.arange(100):
        cov = [[dx[i]**2.,rho[i]*dx[i]*dy[i]],[rho[i]*dx[i]*dy[i],dy[i]**2.]]
        mean = [xp[i],yp[i]]
        xi,yi = np.random.multivariate_normal(mean,cov,1).T
        x[i],y[i] = np.float(xi),np.float(yi)
    #now generate a few outliers
    sigma_outlier = np.float(np.random.uniform(1.,2.,1))
    y_mean = np.float(np.random.uniform(-10.,10.,1))
    noutlier=np.int(np.random.randint(2,high=5,size=1))
    y_outlier = np.random.normal(loc=y_mean,scale=sigma_outlier,size=noutlier)
    x_outlier = np.random.uniform(low=1.3*np.min(x),high=1.3*np.max(x),size=noutlier)
    dx_outlier = np.abs(np.random.normal(loc=0.,scale=0.3,size=noutlier))
    dy_outlier = np.abs(np.random.normal(loc=0.,scale=0.3,size=noutlier)) #don't bother scattering through these (lazy)
    rho_outlier = np.random.uniform(-1.,1.,noutlier)
    x =np.append(x,x_outlier)
    y=np.append(y,y_outlier)
    dx=np.append(dx,dx_outlier)
    dy=np.append(dy,dy_outlier)
    dxy = np.append(dxy,rho_outlier*dx_outlier*dy_outlier)
    return np.vstack((x,y,dx,dy,dxy)).T,np.array([slope,intercept,sigma_intrinsic,y_mean,sigma_outlier,noutlier/(noutlier+100)])

def write_to_file(sampler,outfile,p0,Nsteps=10):
    """Write an MCMC chain from emcee to a file.
    Args:
        (1) emcee.EnsembleSampler, sampler. The sampler to run.
        (2) string, outfile. File to save the results to.
        (3) np.npdarray, p0. Starting points for the walkers.
        (4) Nsteps (= 10) the number of steps to run the chain for.
    """
    f = open(outfile,"a")
    f.close()
    for result in sampler.sample(p0,iterations=Nsteps,storechain=False):
        position = result[0]
        f = open(outfile,"a")
        for k in range(position.shape[0]):
            f.write("{0:4d} {1:s}\n".format(k, " ".join(map(str,position[k]))))
        f.close()
    return None

def reshape_chain(chain):
    """Reshape the output from write_to_file so that it has shape . 
    Args: 
        (1) np.ndarray, chain. should have shape (n_samples*n_walkers,nparameters+1) (extra column \
            is because the 'walker ID' is stored as well.
    Returns:
        (1) np.ndarray, c. has shape (nwalkers,nsteps,nparameters)
    """
    nwalkers = len(np.unique(chain[:,0]))
    nsteps = len(chain[:,0])/nwalkers
    ndim = np.shape(chain)[1]-1
    c = np.zeros((nwalkers,nsteps,ndim))
    #put the chains into the right shape
    for i in np.arange(nwalkers):
        idx = chain[:,0]==i
        for j in np.arange(1,ndim+1):
            c[i,:,j-1] = chain[idx,j]
    return c

def load_chain(outfile):
    """Load an MCMC chain from file.
    Args:
        (1) string, outfile. The path to the text file containing the MCMC chain (assumed to be of the form 
            outputted by write_to_file).
    Returns:
        (1) np.ndarray, chain. numpy array of shape (nwalkers,nsteps,nparameters)
    """
    chain = np.genfromtxt(outfile)
    return reshape_chain(chain)

def chain_results(chain,burnin=None):
    """Get the results from a chain using the 16th, 50th and 84th percentiles of an MCMC chain. 
    Args: 
        (1) np.ndarray, chain. should have shape (n_samples*n_walkers,nparameters+1) (extra column \
            is because the 'walker ID' is stored as well.
        (2) int, burnin (= None), the number of steps to disregard from each walker before evaluating \
            the results of the chain.
    Returns:
        (1) For each parameter a tuple is returned (best_fit, +err, -err)
    """
    if burnin:
        nwalkers,nsteps,ndim = np.shape(reshape_chain(chain))
        chain = chain[nwalkers*burnin:,:]
    return np.array(map(lambda v: [v[1],v[2]-v[1],v[1]-v[0]],\
                zip(*np.percentile(chain[:,1:],[16,50,84],axis=0))))

def fit_data(data,priorlimits=[-10.,10.,-10.,10.,0.001,100.,-10.,10.,0.001,1000.],nwalkers=50,nsteps=5000\
    ,nproc=8,outfile="line_fit_chain",make_cornerplot=True,truths=None,delete_output=True):
    """Fit the model to data using emcee. An optimization routine from scipy is first used to 
    Args:
        (1) np.ndarray, data. Should have shape (N,4) (if no covariances on \
            errors) or (N,5) (if covariant errors). Should be in the order \
            (x,y,dx,dy) or (x,y,dx,dy,dxy).
        (2) np.ndarray, priorlimits. Upper and lower values for each of the model parameters \
            (except the outlier fraction which has a flat prior between 0 and 1). 
        (3) nwalkers (= 50), the number of emcee walkers to use in the fit.
        (4) nsteps (= 5000), the number of steps each walker should take in the MCMC.
        (5) nproc (= 8), the number of threads to use for parallelisation.
        (6) outfile (= "line_fit.dat"), the file to which the points in the MCMC chain will be written.
        (7) make_cornerplot (= True), make a corner plot and save it if True.
        (8) truths (= None), if you know the right answer, plot it on the corner plot.
        (9) delete_output (= True), if True, delete the text file that stores the chain. If not, the results \
            will be saved for you to analyse (e.g. with a corner plot).
    Returns:
        (1) results, for each parameter a tuple is returned (best_fit, +err, -err) (assuming 20 per cent burn-in time), \
            in the order (slope, intercept, intrinsic scatter, outlier mean, outlier deviation, outlier.
    """
    #first make some guesses at the values for things, this could be massively improved and will probably break sometimes.
    x,y = data[:,:2].T
    mguess = 2.
    bguess = 0.
    sig_guess = 2.
    y_outlier_guess = 0.
    sig_outlier_guess = 30.
    pguess = [mguess,bguess,sig_guess,y_outlier_guess,sig_outlier_guess,0.01]
    #now use a minimization routine to find the max-posterior point
    def minfun(params):
        return -full_posterior(params,data,priorlimits)
    print "Running optimization..."
    pstart = minimize(minfun,pguess,method="Nelder-Mead",options={'maxfev':1e6}).x
    print "found point {}".format(pstart)
    #make a blob at the max posterior point
    p0 = emcee.utils.sample_ball(pstart,0.01*np.ones_like(pstart),size=nwalkers)
    p0[:,-1] = np.abs(p0[:,-1]) #make sure no negative outlier fractions
    #sampler
    sampler = emcee.EnsembleSampler(nwalkers,6,full_posterior,args=[data,priorlimits],threads=nproc) #the sampler
    print "running fits..."
    write_to_file(sampler,outfile+".temp",p0,Nsteps=nsteps)
    print "done!"
    chain = np.genfromtxt(outfile+".temp")
    if make_cornerplot: 
        labels = ["$m$","$b$","$\\sigma_\\mathrm{in}$","$y_\\mathrm{out}$","$\\log_{10}\\sigma_\\mathrm{out}$","$f_\\mathrm{outlier}$"]
        chainplot = np.copy(chain)
        chainplot[:,5] = np.log10(chainplot[:,5])
        if truths is not None: 
            truths[4] = np.log10(truths[4])
            triangle_plot(chainplot,burnin=np.int(0.2*nsteps),axis_labels=labels,fname=outfile+"_corner.pdf",truths=truths)
        else:
            triangle_plot(chainplot,burnin=np.int(0.2*nsteps),axis_labels=labels,fname=outfile+"_corner.pdf",truths=truths)
        del chainplot
    results = chain_results(chain,burnin=np.int(0.2*nsteps))
    mcmc_chain = reshape_chain(chain)
    if not delete_output:
        np.save(outfile,mcmc_chain)
    os.remove(outfile+".temp")
    return results


import warnings
def main():
    """Run a quick test on mock data"""
    warnings.filterwarnings("ignore")
    data,ptrue = mock_data()
    os.makedirs("mock_fits")
    np.save("mock_fits/mock_data",data)
    np.save("mock_fits/correct_answer",ptrue)
    print "Correct answer: {}".format(ptrue)
    results = fit_data(data,outfile="mock_fits/line_fit_mock",delete_output=True,make_cornerplot=True,truths=ptrue)
    print "Inferred answer: {}".format(results[:,0])
    np.save("mock_fits/inferred_answer",results)
    #make a plot of the inference vs. data
    x = np.linspace(np.min(data[:,0]),np.max(data[:,0]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(data[:,0],data[:,1],xerr=data[:,2],yerr=data[:,3],ecolor='k',fmt='none',alpha=0.5)
    ax.plot(x,ptrue[0]*x+ptrue[1],label="truth")
    ax.plot(x,results[:,0][0]*x+results[:,0][1],label="inferred")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc='lower right')
    fig.savefig("mock_fits/data_vs_model.pdf")
    plt.close()


if __name__ == "__main__":
    main()










