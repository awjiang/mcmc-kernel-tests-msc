import numpy as onp
# import jax.numpy as jnp
import scipy
import multiprocessing
from functools import reduce
from itertools import repeat
import pdb

#######################################################################
######################### Helper Functions ############################
#######################################################################

def XTX(X):
    return X.T @ X

def XTWX(X, W): 
    return X.T @ W @ X

def diagMatrix(z):
    '''
    Construct a diagonal matrix with diagonal `z`
    '''
    diag_z = onp.eye(len(z))
    onp.fill_diagonal(diag_z,z)
    return(diag_z)

def geweke_functions(samples):
    '''
    Returns a matrix with column means corresponding to the first and second empirical moments of `samples`.
    '''
    f1 = samples.copy()
    n, p = f1.shape
    f2 = onp.empty([n, int(p*(p+1)/2)])
    counter = 0
    for i in range(p):
        for j in range(i+1):
            f2[:, counter] = f1[:, i] * f1[:, j]
            counter += 1
    return onp.hstack([f1, f2])

def GaussianProductMV(mu_0, Sigma_0, lst_mu, lst_Sigma):
    '''
    Calculate mean and variance of the product of multivariate Gaussians
    '''
    assert(len(lst_mu) == len(lst_Sigma))
    mu_pr, Sigma_pr = mu_0, Sigma_0
    Sigma_pr_shape = onp.array(Sigma_pr).shape
    if Sigma_pr_shape == ():
        d = 1
    else:
        d = Sigma_pr_shape[0]
    for i in range(len(lst_mu)):
        if d == 1:
            mu_pr = (mu_pr*lst_Sigma[i] + lst_mu[i]*Sigma_pr)/(lst_Sigma[i] + Sigma_pr)
            Sigma_pr = (Sigma_pr*lst_Sigma[i])/(Sigma_pr + lst_Sigma[i])
        else:
            # Cholesky
            Sigma_sum = Sigma_pr + lst_Sigma[i]
            L = onp.linalg.cholesky(Sigma_sum)
            Sigma_1 = onp.linalg.solve(L, Sigma_pr)
            Sigma_2 = onp.linalg.solve(L, lst_Sigma[i])
            mu_1= onp.linalg.solve(L, mu_pr.reshape(-1, 1))
            mu_2 = onp.linalg.solve(L, lst_mu[i].reshape(-1, 1))
            mu_pr = Sigma_2.T @ mu_1 + Sigma_1.T @ mu_2
            Sigma_pr = Sigma_1.T @ Sigma_2
    return mu_pr, Sigma_pr

def splitIter(num_iter, nproc):
    ''' 
    Split `num_iter` iterations into `nproc` chunks for multithreading
    '''
    arr_iter = (onp.zeros(nproc) + num_iter // nproc).astype('int')
    for i in range(num_iter % nproc):
        arr_iter[i] += 1
    return arr_iter

class model_sampler(object):
    def __init__(self, **kwargs):
        '''
        Generic sampler class
        '''
        self._nproc = 1
        for key, value in kwargs.items():
            setattr(self, '_' + key, value)
        if hasattr(self, '_seed'):
            self._seed_sequence = onp.random.SeedSequence(self._seed)
        else:
            self._seed_sequence = onp.random.SeedSequence()
        self.set_nproc(self._nproc)
        pass
        
    @property
    def sample_dim(self):
        '''
        Dimension of the sample (Y, Theta)
        '''
        pass

    @property
    def theta_indices(self):
        '''
        Indices in sample (Y, Theta) corresponding to Theta
        '''
        pass

    def drawPrior(self):
        '''
        Draw from P(Theta)
        '''
        pass

    def drawLikelihood(self):
        '''
        Draw from P(Y|Theta)
        '''
        pass

    def drawPosterior(self):
        '''
        Draw from P(Theta|Y) via MCMC
        '''
        pass
    
    def set_seed(self, seed=None):
        '''
        Set seed
        '''
        if seed is None:
            self._seed_sequence = onp.random.SeedSequence()
        else:
            self._seed = seed
            self._seed_sequence = onp.random.SeedSequence(seed)
        self.init_rng()
        pass

    def set_nproc(self, nproc):
        '''
        Set number of parallel processes used for sampling
        '''
        self._nproc = nproc
        self.init_rng()
        pass
    
    def init_rng(self):
        '''
        Initialize random number generators used for single and multi-process functions
        '''
        child_seed_seq = self._seed_sequence.spawn(self._nproc + 1)
        # Multi-process
        child_seed_seq_m = child_seed_seq[:-1]
        self._bitgen_m = [onp.random.MT19937(s) for s in child_seed_seq_m]
        self._rng_m = [onp.random.Generator(bg) for bg in self._bitgen_m]
        # Single-process
        child_seed_seq_s = child_seed_seq[-1]
        self._bitgen_s = onp.random.MT19937(child_seed_seq_s)
        self._rng_s = onp.random.Generator(self._bitgen_s)        
        pass
    
    def jump_rng(self, type_rng):
        '''
        Advance random number generator of type `type_rng`
        '''
        if type_rng == 'm': # Multi-process
            self._bitgen_m = [bg.jumped() for bg in self._bitgen_m]
            self._rng_m = [onp.random.Generator(bg) for bg in self._bitgen_m]
        elif type_rng == 's': # Single-process
            self._bitgen_s = self._bitgen_s.jumped()
            self._rng_s = onp.random.Generator(self._bitgen_s)
        else:
            raise ValueError
        pass
    
    def forward(self, num_samples, rng):
        '''
        Algorithm 1: marginal-conditional simulator
        '''
        samples = onp.empty([num_samples, self.sample_dim])
        for i in range(num_samples):
            sample_prior = self.drawPrior(rng)
            sample_likelihood = self.drawLikelihood(rng)
            samples[i, :] = onp.hstack([sample_likelihood, sample_prior])
        return samples

    def successive(self, num_samples, rng):
        '''
        Algorithm 2: successive-conditional simulator
        '''
        samples = onp.empty([int(num_samples), self.sample_dim])
        self.drawPrior(rng)
        for i in range(int(num_samples)):
            sample_likelihood = self.drawLikelihood(rng)
            sample_posterior = self.drawPosterior(rng)
            samples[i, :] = onp.hstack([sample_likelihood, sample_posterior])
        return samples  
    
    def backward(self, num_samples, burn_in_samples, rng):
        '''
        Algorithm 3: backward-conditional simulator
        '''
        samples = onp.empty([num_samples, self.sample_dim])
        for i in range(int(num_samples)):
            self.drawPrior(rng)
            sample_likelihood = self.drawLikelihood(rng)
            for _ in range(int(burn_in_samples+1)):           
                sample_posterior = self.drawPosterior(rng)
            samples[i, :] = onp.hstack([sample_likelihood, sample_posterior])
        return samples
    
    def sample_mc(self, num_samples):
        '''
        Parallelized wrapper for Algorithm 1: marginal-conditional simulator
        '''
        if self._nproc == 1:
            samples = self.forward(int(num_samples), self._rng_s)
        else:
            lst_num_samples = splitIter(int(num_samples), self._nproc)
            pool = multiprocessing.Pool(processes=self._nproc)
            out = pool.starmap(self.forward, zip(lst_num_samples, self._rng_m))
            pool.close()
            samples = onp.vstack(out)
            self.jump_rng('m')
        return samples

    def sample_sc(self, num_samples):
        '''
        Wrapper for Algorithm 2: successive-conditional simulator
        '''      
        samples = self.successive(int(num_samples), self._rng_s)
        return samples

    def sample_bc(self, num_samples, burn_in_samples):
        '''
        Parallelized wrapper for Algorithm 3: backward-conditional simulator
        '''
        if self._nproc == 1:
            samples = self.backward(int(num_samples), int(burn_in_samples), self._rng_s)
        else:    
            lst_num_samples = splitIter(int(num_samples), self._nproc)
            pool = multiprocessing.Pool(processes=self._nproc)
            out = pool.starmap(self.backward, zip(lst_num_samples, repeat(burn_in_samples), self._rng_m))
            pool.close()
            samples = onp.vstack(out)
            self.jump_rng('m')
        return samples

    def test_functions(self, samples):
        '''
        Test functions computed on (Y, Theta)
        '''
        return geweke_functions(samples)