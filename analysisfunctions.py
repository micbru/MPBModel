# All numerical functions needed for the paper plots are included here.
import numpy as np
from scipy.special import betainc,beta
import scipy.integrate as integrate
import scipy.optimize as opt

############
# Functions
############
def ffull(m,k,p):
    return betainc(k,p+1,k/(k+m))

def fhill(m,n,g):
    return g**n/(m**n+g**n)

##############
# Derivatives
##############
def dffull(m,k,p):
    f1 = k/(k+m)
    f2 = m/(k+m)
    return -f1**k * f2**p / ((k+m)*beta(k,p+1))

def dfhill(m,n,g):
    return -n*g**n*m**(n-1) / (m**n+g**n)**2

###############################################
# Function to solve for roots of full function
###############################################
def fdf_eq(m,c,k,p):
    return [m/c - 1 + ffull(m,k,p), 1/c + dffull(m,k,p)]

##########################
# Functions to get Hill n
##########################
def dist(n,g,k,p,c):
    # Take integral distance between the two arrays
    var = lambda m: (fhill(m,n,g)-ffull(m,k,p))**2
    return integrate.quad(var,0,c)[0]

def get_hill_n(k,p,c,nmax=50):
    # Define a function to get the best hill n given all other parameters
    # This sets g = p, which may not be the best possible, but we'll do it for now.
    mindist = opt.minimize_scalar(dist,args=(p,k,p,c),bounds=(1,nmax))
    if not (mindist.success):
        print("Something is wrong!")
    return mindist.x

##################################
# Functions for stability analysis
##################################
def coef(m,s,c,f,df,*args):
    # Construct coefficient array
    # Args must be k,p
    a0 = s**50*(1-f(m,*args)+m*df(m,*args))
    a1 = s**49*(s*(1-f(m,*args))+(s*(m+c)-c)*df(m,*args))
    a2_50 = [(s*(s-f(m,*args))+(s*(m+c)-c)*df(m,*args))*s**(50-i) for i in np.arange(2,51)]
    a51 = s - f(m,*args) + (m+c)*df(m,*args)
    a52 = 1
    a = np.concatenate(([a0],[a1],a2_50,[a51],[a52]))
    return a

def specrad(m,s,c,f,df,*args):
    # Spectral radius
    coefs = coef(m,s,c,f,df,*args)
    return np.max(np.absolute(np.roots(coefs[::-1])))

####################################
# Functions for numerical simulation
####################################
def timestep(xt,s,N,agg,thresh,f=fhill,b_imm=0,c=1):
    '''Take in array of values at x_t and return values at x_{t+1}
    x_0-N-1 are juveniles
    x_N are susceptible
    x_N+1 are beetles
    agg, thresh are the aggregation and threshold parameter for the f function
    f is the function to simulate with, assumed to be the hill function
    b_imm are immigrating beetles, assumed zero.
    Finally c = 1 is assumed by default, which is the nondimensionalized version in the hill function case.'''
    # Set up return array
    xt1 = np.zeros(len(xt))
    
    # Juveniles
    xt1[0] = 1 - s*np.sum(xt[0:N]) - xt[N] - xt[N+1]/c
    for i in np.arange(N-1):
        xt1[i+1] = s*xt[i]

    # Get mean number of beetles from last year
    if xt[N] != 0:
        mt = (xt[N+1]+b_imm)/xt[N] # Where we add the number that invade + the number from the infested trees
    else:
        mt = 0
    
    # Susceptible trees
    xt1[N] = f(mt,agg,thresh)*xt[N]+s*xt[N-1]
    # Beetles
    xt1[N+1] = c*(1-f(mt,agg,thresh))*xt[N]
    
    return xt1