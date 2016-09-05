from __future__ import division, print_function

import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import os
import emcee
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter, MaxNLocator, FuncFormatter
from matplotlib.colors import LogNorm
from matplotlib.pyplot import rc, axes
from sys import stderr
import matplotlib.cm as cm
import matplotlib.colorbar as cbar
import pylab
import matplotlib.colors as colors

warnings.filterwarnings("ignore") #sorry

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

def fit_data(data,guess=None,priorlimits=[-10.,10.,-10.,10.,0.001,100.,-10.,10.,0.001,1000.],nwalkers=50,nsteps=5000\
    ,nproc=8,outfile="line_fit_chain",make_cornerplot=True,truths=None,delete_output=True):
    """
    Args:
        (1) np.ndarray, data. Should have shape (N,4) (if no covariances on 
            errors) or (N,5) (if covariant errors). Should be in the order 
            (x,y,dx,dy) or (x,y,dx,dy,dxy).
        (2) np.ndarray or list, guess. A guess of the 6 model parameters. If not included 
            then a fixed starting guess will be used (which may well be inappropriate and 
            break everything!). Ordered as [slope,intercept,intrinsic_scatter,outlier_mean,
            outlier_scatter,outlier_fraction].
        (3) np.ndarray, priorlimits. Upper and lower values for each of the model parameters 
            (except the outlier fraction which has a flat prior between 0 and 1). The limits should 
            be provided in the order [slope,intercept,intrinsic_scatter,outlier_mean,
            outlier_scatter] (so that the array has 10 elements).
        (4) nwalkers (= 50), the number of emcee walkers to use in the fit.
        (5) nsteps (= 5000), the number of steps each walker should take in the MCMC.
        (6) nproc (= 8), the number of threads to use for parallelisation.
        (7) outfile (= "line_fit.dat"), the file to which the points in the MCMC chain will be written.
        (8) make_cornerplot (= True), make a corner plot and save it if True.
        (9) truths (= None), if you know the right answer, plot it on the corner plot.
        (10) delete_output (= True), if True, delete the text file that stores the chain. If not, the results 
            will be saved for you to analyse (e.g. with a corner plot).
    Returns:
        (1) results, for each parameter a tuple is returned (best_fit, +err, -err) (assuming 10 per cent burn-in time), 
            in the order (slope, intercept, intrinsic scatter, outlier mean, outlier deviation, outlier fraction).
    """
    #first make some guesses at the values for things, this could be massively improved and will probably break sometimes.
    x,y = data[:,:2].T
    if guess is None:
        #not recommended as will probably break!
        mguess = 2.
        bguess = 0.
        sig_guess = 2.
        y_outlier_guess = 0.
        sig_outlier_guess = 30.
        guess = [mguess,bguess,sig_guess,y_outlier_guess,sig_outlier_guess,0.01]
    #now use a minimization routine to find the max-posterior point
    def minfun(params):
        return -full_posterior(params,data,priorlimits)
    print("Running optimization...")
    pstart = minimize(minfun,guess,method="Nelder-Mead",options={'maxfev':1e6}).x
    print("found point {}".format(pstart))
    #make a blob at the max posterior point
    p0 = emcee.utils.sample_ball(pstart,0.01*np.ones_like(pstart),size=nwalkers)
    p0[:,-1] = np.abs(p0[:,-1]) #make sure no negative outlier fractions
    #sampler
    sampler = emcee.EnsembleSampler(nwalkers,6,full_posterior,args=[data,priorlimits],threads=nproc) #the sampler
    print("running fits...")
    write_to_file(sampler,outfile+".temp",p0,Nsteps=nsteps)
    print("done!")
    chain = np.genfromtxt(outfile+".temp")
    if make_cornerplot: 
        chainplot = np.copy(chain)
        labels = ["$m$","$b$","$\\sigma_\\mathrm{in}$"]
        if truths is not None: 
            triangle_plot(chainplot[:,:4],burnin=np.int(0.1*nsteps),axis_labels=labels,fname=outfile+"_corner.pdf",truths=truths)
        else:
            triangle_plot(chainplot[:,:4],burnin=np.int(0.1*nsteps),axis_labels=labels,fname=outfile+"_corner.pdf",truths=truths)
        del chainplot
    results = chain_results(chain,burnin=np.int(0.1*nsteps))
    mcmc_chain = reshape_chain(chain)
    if not delete_output:
        np.save(outfile,mcmc_chain)
    os.remove(outfile+".temp")
    return results

def mock_test(nproc=8):
    """Fit mock data to check everything is working"""
    fake_data = np.load(os.path.join(os.path.dirname(__file__),"data","mock_data.npy"))
    res = fit_data(fake_data,guess=None,priorlimits=[-10.,10.,-10.,10.,0.001,100.,-10.,10.,0.001,1000.],nwalkers=50,nsteps=7000\
    ,nproc=nproc,outfile="line_fit_chain",make_cornerplot=True,truths=None,delete_output=True)
    fig,ax = plt.subplots()
    x = np.linspace(np.min(fake_data[:,0]),np.max(fake_data[:,0]))
    ax.plot(x,res[0,0]*x+res[1,0],c='k',lw=3,label="fit")
    ax.errorbar(fake_data[:,0],fake_data[:,1],xerr=fake_data[:,2],yerr=fake_data[:,3],fmt="none",ecolor='0.5',label="mock data",zorder=0)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc='lower right',numpoints=1)
    return None

def my_formatter(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '${:g}$'.format(x)
    if np.abs(x) > 0 and np.abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str

def confidence_2d(xsamples,ysamples,ax=None,intervals=None,nbins=20,linecolor='k',histunder=False,cmap="Blues",filled=False,linewidth=1.):
    """Draw confidence intervals at the levels asked from a 2d sample of points (e.g. 
        output of MCMC)"""
    if intervals is None:
        intervals  = 1.0 - np.exp(-0.5 * np.array([0., 1., 2.]) ** 2)
    H,yedges,xedges = np.histogram2d(ysamples,xsamples,bins=nbins)


    #get the contour levels
    h = H.flatten()
    h = h[np.argsort(h)[::-1]]
    sm = np.cumsum(h)
    sm/=sm[-1]
    v = np.empty(len(intervals))
    for i,v0 in enumerate(intervals):
        try:
            v[i] = h[sm <= v0][-1]
        except:
            v[i] = h[0]
    v=v[::-1]

    xc = np.array([.5*(xedges[i]+xedges[i+1]) for i in np.arange(nbins)]) #bin centres
    yc = np.array([.5*(yedges[i]+yedges[i+1]) for i in np.arange(nbins)])

    xx,yy = np.meshgrid(xc,yc)

    if ax is None:
        if histunder:
            plt.hist2d(xsamples,ysamples,bins=nbins,cmap=cmap)
            plt.contour(xx,yy,H,levels=v,colors=linecolor,extend='max',linewidths=linewidth)
        elif filled:
            plt.contourf(xx,yy,H,levels=v,cmap=cmap)
        else:
            plt.contour(xx,yy,H,levels=v,colors=linecolor,linewidths=linewidth)
    else:
        if histunder:
            ax.hist2d(xsamples,ysamples,bins=nbins,cmap=cmap)
            ax.contour(xx,yy,H,levels=v,colors=linecolor,extend='max',linewidths=linewidth)
        elif filled:
            ax.contourf(xx,yy,H,levels=v,cmap=cmap)
            ax.contour(xx,yy,H,levels=v,colors=linecolor,extend='max',linewidths=linewidth)
        else:
            ax.contour(xx,yy,H,levels=v,colors=linecolor,linewidths=linewidth)        

    return None


def triangle_plot( chain, axis_labels=None, fname = None, nbins=40, filled=True, cmap="Blues", norm = None, truths = None,\
                         burnin=None, fontsize=20 , tickfontsize=15, nticks=4, linewidth=1.,wspace=0.5,hspace=0.5):

    """Plot a corner plot from an MCMC chain. the shape of the chain array should be (nwalkers*nsamples, ndim + 1). The extra column is for the walker ID 
    number (i.e. if you have 20 walkers the id numbers are np.arange(20)). Note the walker ID's are never used, theyre only assumed to be there because 
    of the way I write MCMC chains to file."""

    major_formatter = FuncFormatter(my_formatter)
    nwalkers = len(np.unique(chain[:,0]))

    if burnin is not None:
        traces = chain[nwalkers*burnin:,1:].T
    else:  
        traces = chain[:,1:].T

    if axis_labels is None:
        axis_labels = ['']*len(traces)

    #Defines the widths of the plots in inches
    plot_width = 15.
    plot_height = 15.
    axis_space = 0.05

    if truths != None and ( len(truths) != len(traces) ):
        print >> stderr, "ERROR: There must be the same number of true values as traces"

    num_samples = min([ len(trace) for trace in traces])
    n_traces = len(traces)


    #Set up the figure
    fig = plt.figure( num = None, figsize = (plot_height,plot_width))

    dim = 2*n_traces - 1

    gs = gridspec.GridSpec(dim+1,dim+1)
    gs.update(wspace=wspace,hspace=hspace)

    hist_2d_axes = {}

    #Create axes for 2D histograms
    for x_pos in xrange( n_traces - 1 ):
        for y_pos in range( n_traces - 1 - x_pos  ):
            x_var = x_pos
            y_var = n_traces - 1 - y_pos

            hist_2d_axes[(x_var, y_var)] = fig.add_subplot( \
                                           gs[ -1-(2*y_pos):-1-(2*y_pos), \
                                               2*x_pos:(2*x_pos+2) ] )
            hist_2d_axes[(x_var, y_var)].xaxis.set_major_formatter(major_formatter)
            hist_2d_axes[(x_var, y_var)].yaxis.set_major_formatter(major_formatter)

    #Create axes for 1D histograms
    hist_1d_axes = {}
    for var in xrange( n_traces -1 ):
        hist_1d_axes[var] = fig.add_subplot( gs[ (2*var):(2*var+2), 2*var:(2*var+2) ] )
        hist_1d_axes[var].xaxis.set_major_formatter(major_formatter)
        hist_1d_axes[var].yaxis.set_major_formatter(major_formatter)
    hist_1d_axes[n_traces-1] = fig.add_subplot( gs[-2:, -2:] )
    hist_1d_axes[n_traces-1].xaxis.set_major_formatter(major_formatter)
    hist_1d_axes[n_traces-1].yaxis.set_major_formatter(major_formatter)



    #Remove the ticks from the axes which don't need them
    for x_var in xrange( n_traces -1 ):
        for y_var in xrange( 1, n_traces - 1):
            try:
                hist_2d_axes[(x_var,y_var)].xaxis.set_visible(False)
            except KeyError:
                continue
    for var in xrange( n_traces - 1 ):
        hist_1d_axes[var].set_xticklabels([])
        hist_1d_axes[var].xaxis.set_major_locator(MaxNLocator(nticks))
        hist_1d_axes[var].yaxis.set_visible(False)

    for y_var in xrange( 1, n_traces ):
        for x_var in xrange( 1, n_traces - 1):
            try:
                hist_2d_axes[(x_var,y_var)].yaxis.set_visible(False)
            except KeyError:
                continue

    #Do the plotting
    #Firstly make the 1D histograms
    vals, walls = np.histogram(traces[-1][:num_samples], bins=nbins, normed = True)

    xplot = np.zeros( nbins*2 + 2 )
    yplot = np.zeros( nbins*2 + 2 )
    for i in xrange(1, nbins * 2 + 1 ):
        xplot[i] = walls[(i-1)/2]
        yplot[i] = vals[ (i-2)/2 ]

    xplot[0] = walls[0]
    xplot[-1] = walls[-1]
    yplot[0] = yplot[1]
    yplot[-1] = yplot[-2]

    Cmap = colors.Colormap(cmap)
    cNorm = colors.Normalize(vmin=0.,vmax=1.)
    scalarMap = cm.ScalarMappable(norm=cNorm,cmap=cmap)
    cVal = scalarMap.to_rgba(0.65)

    #this one's special, so do it on it's own
    hist_1d_axes[n_traces - 1].plot(xplot, yplot, color = 'k', lw=linewidth)
    if filled: hist_1d_axes[n_traces - 1].fill_between(xplot,yplot,color=cVal)
    hist_1d_axes[n_traces - 1].set_xlim( walls[0], walls[-1] )
    hist_1d_axes[n_traces - 1].set_xlabel(axis_labels[-1],fontsize=fontsize)
    hist_1d_axes[n_traces - 1].tick_params(labelsize=tickfontsize)
    hist_1d_axes[n_traces - 1].xaxis.set_major_locator(MaxNLocator(nticks))
    hist_1d_axes[n_traces - 1].yaxis.set_visible(False)
    plt.setp(hist_1d_axes[n_traces - 1].xaxis.get_majorticklabels(), rotation=45)
    if truths is not None:
        xlo,xhi = hist_1d_axes[n_traces-1].get_xlim()
        if truths[n_traces-1]<xlo:
            dx = xlo-truths[n_traces-1]
            hist_1d_axes[n_traces-1].set_xlim((xlo-dx-0.05*(xhi-xlo),xhi))
        elif truths[n_traces-1]>xhi:
            dx = truths[n_traces-1]-xhi
            hist_1d_axes[n_traces-1].set_xlim((xlo,xhi+dx+0.05*(xhi-xlo)))
        hist_1d_axes[n_traces - 1].axvline(truths[n_traces - 1],ls='--',c='k')


    #Now Make the 2D histograms
    for x_var in xrange( n_traces ):
        for y_var in xrange( n_traces):
            try:
                if norm == 'log':
                    H, y_edges, x_edges = np.histogram2d( traces[y_var][:num_samples], traces[x_var][:num_samples],\
                                                           bins = nbins, norm = LogNorm() )
                else:
                    H, y_edges, x_edges = np.histogram2d( traces[y_var][:num_samples], traces[x_var][:num_samples],\
                                                           bins = nbins )
                confidence_2d(traces[x_var][:num_samples],traces[y_var][:num_samples],ax=hist_2d_axes[(x_var,y_var)],\
                    nbins=nbins,intervals=None,linecolor='0.5',filled=filled,cmap=cmap,linewidth=linewidth)
                if truths is not None:
                    xlo,xhi = hist_2d_axes[(x_var,y_var)].get_xlim()
                    ylo,yhi = hist_2d_axes[(x_var,y_var)].get_ylim()
                    if truths[x_var]<xlo:
                        dx = xlo-truths[x_var]
                        hist_2d_axes[(x_var,y_var)].set_xlim((xlo-dx-0.05*(xhi-xlo),xhi))
                    elif truths[x_var]>xhi:
                        dx = truths[x_var]-xhi
                        hist_2d_axes[(x_var,y_var)].set_xlim((xlo,xhi+dx+0.05*(xhi-xlo)))
                    if truths[y_var]<ylo:
                        dy = ylo - truths[y_var]
                        hist_2d_axes[(x_var,y_var)].set_ylim((ylo-dy-0.05*(yhi-ylo),yhi))
                    elif truths[y_var]<ylo:
                        dy = truths[y_var] - yhi
                        hist_2d_axes[(x_var,y_var)].set_ylim((ylo,yhi+dy+0.05*(yhi-ylo)))
                    #TODO: deal with the pesky case of a prior edge
                    hist_2d_axes[(x_var,y_var)].plot( truths[x_var], truths[y_var], '*', color = 'k', markersize = 10 )
            except KeyError:
                pass
        if x_var < n_traces - 1:
            vals, walls = np.histogram( traces[x_var][:num_samples], bins=nbins, normed = True )

            xplot = np.zeros( nbins*2 + 2 )
            yplot = np.zeros( nbins*2 + 2 )
            for i in xrange(1, nbins * 2 + 1 ):
                xplot[i] = walls[(i-1)/2]
                yplot[i] = vals[ (i-2)/2 ]

            xplot[0] = walls[0]
            xplot[-1] = walls[-1]
            yplot[0] = yplot[1]
            yplot[-1] = yplot[-2]

            hist_1d_axes[x_var].plot(xplot, yplot, color = 'k' , lw=linewidth)
            if filled: hist_1d_axes[x_var].fill_between(xplot,yplot,color=cVal)
            hist_1d_axes[x_var].set_xlim( x_edges[0], x_edges[-1] )
            if truths is not None:
                xlo,xhi = hist_1d_axes[x_var].get_xlim()
                if truths[x_var]<xlo:
                    dx = xlo-truths[x_var]
                    hist_1d_axes[x_var].set_xlim((xlo-dx-0.05*(xhi-xlo),xhi))
                elif truths[x_var]>xhi:
                    dx = truths[x_var]-xhi
                    hist_1d_axes[x_var].set_xlim((xlo,xhi+dx+0.05*(xhi-xlo)))
                hist_1d_axes[x_var].axvline(truths[x_var],ls='--',c='k')

    #Finally Add the Axis Labels
    for x_var in xrange(n_traces - 1):
        hist_2d_axes[(x_var, n_traces-1)].set_xlabel(axis_labels[x_var],fontsize=fontsize)
        hist_2d_axes[(x_var, n_traces-1)].tick_params(labelsize=tickfontsize)
        hist_2d_axes[(x_var, n_traces-1)].xaxis.set_major_locator(MaxNLocator(nticks))
        plt.setp(hist_2d_axes[(x_var, n_traces-1)].xaxis.get_majorticklabels(), rotation=45)
    for y_var in xrange(1, n_traces ):
        hist_2d_axes[(0,y_var)].set_ylabel(axis_labels[y_var],fontsize=fontsize)
        hist_2d_axes[(0,y_var)].tick_params(labelsize=tickfontsize)
        plt.setp(hist_2d_axes[(0,y_var)].yaxis.get_majorticklabels(), rotation=45)
        hist_2d_axes[(0,y_var)].yaxis.set_major_locator(MaxNLocator(nticks))

    if fname != None:
        if len(fname.split('.')) == 1:
            fname += '.pdf'
        plt.savefig(fname, transparent=True, bbox_inches = "tight")

    return None