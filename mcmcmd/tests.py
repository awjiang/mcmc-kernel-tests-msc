import multiprocessing
import numpy as onp
from matplotlib import pyplot as plt
import scipy
import arch.covariance.kernel
import os
import pickle
from time import perf_counter

def splitIter(num_iter, nproc):
    ''' 
    Split `num_iter` iterations into `nproc` chunks for multithreading
    '''
    arr_iter = (onp.zeros(nproc) + num_iter // nproc).astype('int')
    for i in range(num_iter % nproc):
        arr_iter[i] += 1
    return arr_iter

def normalizeTwoSamples(X, Y):
    '''
    Normalize feature scales of two samples `X` and `Y` by dividing by the pooled standard deviations of the features
    '''
    assert len(X.shape) == 2 and len(Y.shape) == 2
    assert X.shape[1] == Y.shape[1]
    XY = onp.vstack([X, Y])
    std = onp.std(XY, axis=0)
    if (std==0).sum() > 0:
        std[std==0]=1
    X_tilde = X/std.reshape(1, X.shape[1])
    Y_tilde = Y/std.reshape(1, Y.shape[1])
    return X_tilde, Y_tilde

#######################################################################
############################# Kernels #################################
#######################################################################

class kernel(object):
    def __init__(self, X, Y):
        '''
        Generic kernel class
        `X`, `Y` are (n_x x p) and (n_y x p) samples
        '''
        self._X = X
        self._Y = Y
        pass

    @property
    def params(self):
        pass

    def set_params(self, params):
        pass
    
    def learn(self):
        '''
        Learn kernel parameters
        '''
        pass
    
    def eval(self):
        '''
        Generate kernel matrix K from samples X and Y with K_(i, j) = k(x_i, y_j)
        '''
        pass

    def f_kernel(self):
        '''
        Kernel function k(x, y)
        '''
        pass

class rbf_kernel(kernel):
    def __init__(self, X, Y, tau=None, **kwargs):
        ''' 
        RBF kernel class; k(x, y) = exp(-(||x-y||^2)/tau)
        `X`, `Y` are (n_x x p) and (n_y x p) samples
        If bandwidth `tau` is None, uses the median heuristic
        '''
        assert X.shape[1] == Y.shape[1]
        assert len(X.shape) == 2 and len(X.shape) == len(Y.shape)
        if tau is not None:
            assert isinstance(tau, int) or isinstance(tau, float)
        self._X = X
        self._Y = Y
        self._tau = tau
        pass

    @property
    def params(self):
        return self._tau

    def set_params(self, params):
        self._tau = params
        pass

    def learn(self, method='median_heuristic', eval=False):
        '''
        Learn kernel parameters
        '''
        assert method in ['median_heuristic']
        n_X, p = self._X.shape
        n_Y = self._Y.shape[0]
        if method == 'median_heuristic':
            # Pool the samples if not already pooled
            if onp.allclose(self._X, self._Y):
                X = self._X
                Y = self._Y
            else:
                X = Y = onp.vstack([self._X, self._Y])
            
            n = X.shape[0]
            norm2 = ((X.reshape(n, 1, p) - Y.reshape(1, n, p))**2).sum(2)
            norm2_sorted = onp.sort(norm2.reshape(n**2, 1).flatten())
            if (n**2) % 2 == 0:
                tau = (norm2_sorted[int(n**2/2)-1] + norm2_sorted[int(n**2/2)])/2
            else:
                tau = norm2_sorted[int(n**2/2)]
        else:
            raise ValueError
        self._tau = tau

        if eval == True:
            if n_X != n: 
                # If we pooled the samples, unpool here
                return onp.exp(-norm2[:, :n_X][n_X:, :]/self._tau)
            else:
                return onp.exp(-norm2/self._tau)
        else:
            pass

    def eval(self):
        '''
        Generate kernel matrix K from samples X and Y with K_(i, j) = exp(-(||x_i-y_j||^2)/tau)
        '''
        if self._tau is None:
            K = self.learn(eval=True)
        else:
            n_X, p = self._X.shape
            n_Y = self._Y.shape[0]
            K = onp.exp(-(((self._X.reshape(n_X, 1, p) - self._Y.reshape(1, n_Y, p))**2).sum(2))/self._tau)
        return K

    def f_kernel(self, x, y, tau=None):
        '''
        Kernel function k(x, y) = exp(-(||x-y||^2)/tau)
        '''
        if tau is None:
            tau = self._tau
        assert len(x.shape) == len(y.shape) and len(x.shape) == 1
        return onp.exp(-((x-y)**2).sum()/tau)
      
class linear_kernel(kernel):
    '''
    Linear kernel class; k(x, y) = <x, y>
    `X`, `Y` are (n_x x p) and (n_y x p) samples
    '''
    def __init__(self, X, Y, **kwargs):
        assert X.shape[1] == Y.shape[1]
        assert len(X.shape) == 2 and len(X.shape) == len(Y.shape)
        self._X = X
        self._Y = Y
        pass

    @property
    def params(self):
        pass
      
    def set_params(self, params):
        pass

    def learn(self, eval=False):
        '''
        Learn kernel parameters
        '''
        if eval == True:
            return self.eval()
        else:
            pass

    def eval(self):
        ''' 
        Generate kernel matrix K from samples X and Y with K_(i, j) = <x_i, y_j>
        '''
        n_X, p = self._X.shape
        n_Y = self._Y.shape[0]
        K = (self._X.reshape(n_X, 1, p) * self._Y.reshape(1, n_Y, p)).sum(2)
        return K

    def f_kernel(self, x, y):
        '''
        Kernel function k(x, y) = <x, y>
        '''
        assert len(x.shape) == len(y.shape) and len(x.shape) == 1
        return onp.dot(x, y)
      
class sum_kernel(kernel):
    def __init__(self, X, Y, lst_classes, lst_groups, lst_weights=None, lst_params=None, lst_kwargs=None, **kwargs):
        '''
        Sum of kernels
        `X`, `Y` are (n_x x p) and (n_y x p) samples
        `lst_classes` = list of classes used for each kernel
        `lst_groups` = list of sample indices used for each kernel
        `lst_weights` = list of sum weights used for each kernel
        `lst_params` = list of parameters used for each kernel and set via the `set_params` class method
        `lst_kwargs` = list of **kwargs to pass to each kernel
        '''
        assert X.shape[1] == Y.shape[1]
        assert len(X.shape) == 2 and len(X.shape) == len(Y.shape)
        if lst_weights is None:
            lst_weights = [1] * len(lst_classes)
        if lst_kwargs is None:
            lst_kwargs = [{}] * len(lst_classes)
        
        lst_lst = [lst_classes, lst_groups, lst_weights, lst_kwargs]
        assert all([len(x) == len(lst_lst[0]) for x in lst_lst])
        
        if lst_params is not None:
            assert len(lst_params) == len(lst_classes)
        
        self._X = X
        self._Y = Y
        self._lst_kernels = [None] * len(lst_classes)
        self._num_kernels = len(lst_classes)
        
        self._lst_classes = lst_classes
        self._lst_params = lst_params
        self._lst_groups = lst_groups
        self._lst_weights = onp.array(lst_weights)
        self._lst_weights = self._lst_weights/self._lst_weights.sum() # normalize weights
        self._lst_kwargs = lst_kwargs
        
        pass
    
    def learn(self, eval=False):
        n_X = self._X.shape[0]
        n_Y = self._Y.shape[0]
        if eval==True:
            K = onp.zeros(shape=[n_X, n_Y])
        
        for i in range(self._num_kernels):
            self._lst_kernels[i] = self._lst_classes[i](self._X[:, self._lst_groups[i]].reshape(-1, len(self._lst_groups[i])), self._Y[:, self._lst_groups[i]].reshape(-1, len(self._lst_groups[i])), **self._lst_kwargs[i])
            if eval==True:
                K_i = self._lst_kernels[i].learn(eval=True)
                K += self._lst_weights[i] * K_i
            else:
                self._lst_kernels[i].learn(eval=False)
                
        if eval == True:
            return K
        else:
            pass        

    def eval(self):
        if self._lst_params is None:
            K = self.learn(eval=True)
        else:
            n_X = self._X.shape[0]
            n_Y = self._Y.shape[0]
            K = onp.zeros(shape=[n_X, n_Y])    
            for i in range(self._num_kernels):
                self._lst_kernels[i] = self._lst_classes[i](self._X[:, self._lst_groups[i]].reshape(-1, len(self._lst_groups[i])), self._Y[:, self._lst_groups[i]].reshape(-1, len(self._lst_groups[i])))
                self._lst_kernels[i].set_params(self._lst_params[i])
                K_i = self._lst_kernels[i].eval()
                K += self._lst_weights[i] * K_i
        return K

    def f_kernel(self, x, y, **kwargs):
        assert len(x.shape) == len(y.shape) and len(x.shape) == 1
        out = 0.
        for i, group in enumerate(self._lst_groups):
            x_group = x[group]
            y_group = y[group]
            out += lst_weights[i] * lst_kernels[i].f_eval(x_group, y_group)
        return out

class prod_kernel(kernel):
    def __init__(self, X, Y, lst_classes, lst_groups, lst_params=None, lst_kwargs=None, **kwargs):
        '''
        Product of kernels
        `X`, `Y` are (n_x x p) and (n_y x p) samples
        `lst_classes` = list of classes used for each kernel
        `lst_groups` = list of sample indices used for each kernel
        `lst_weights` = list of sum weights used for each kernel
        `lst_params` = list of parameters used for each kernel and set via the `set_params` class method
        `lst_kwargs` = list of **kwargs to pass to each kernel
        '''
        assert X.shape[1] == Y.shape[1]
        assert len(X.shape) == 2 and len(X.shape) == len(Y.shape)
        if lst_kwargs is None:
            lst_kwargs = [{}] * len(lst_classes)
        
        lst_lst = [lst_classes, lst_groups, lst_kwargs]
        assert all([len(x) == len(lst_lst[0]) for x in lst_lst])
        
        if lst_params is not None:
            assert len(lst_params) == len(lst_classes)
        
        self._X = X
        self._Y = Y
        self._lst_kernels = [None] * len(lst_classes)
        self._num_kernels = len(lst_classes)
        
        self._lst_classes = lst_classes
        self._lst_params = lst_params
        self._lst_groups = lst_groups
        self._lst_kwargs = lst_kwargs
        
        pass
    
    def learn(self, eval=False):
        n_X = self._X.shape[0]
        n_Y = self._Y.shape[0]
        if eval==True:
            K = onp.ones(shape=[n_X, n_Y])
        
        for i in range(self._num_kernels):
            self._lst_kernels[i] = self._lst_classes[i](
              self._X[:, self._lst_groups[i]].reshape(-1, len(self._lst_groups[i])), 
              self._Y[:, self._lst_groups[i]].reshape(-1, len(self._lst_groups[i])), 
              **self._lst_kwargs[i]
            )
            if eval==True:
                K_i = self._lst_kernels[i].learn(eval=True)
                K *= K_i
            else:
                self._lst_kernels[i].learn(eval=False)
                
        if eval == True:
            return K
        else:
            pass        

    def eval(self):
        if self._lst_params is None:
            K = self.learn(eval=True)
        else:
            n_X = self._X.shape[0]
            n_Y = self._Y.shape[0]
            K = onp.ones(shape=[n_X, n_Y])    
            for i in range(self._num_kernels):
                self._lst_kernels[i] = self._lst_classes[i](self._X[:, self._lst_groups[i]], self._Y[:, self._lst_groups[i]])
                self._lst_kernels[i].set_params(self._lst_params[i])
                K_i = self._lst_kernels[i].eval()
                K *= K_i
        return K

    def f_kernel(self, x, y, **kwargs):
        assert len(x.shape) == len(y.shape) and len(x.shape) == 1
        out = 1.
        for i, group in enumerate(self._lst_groups):
            x_group = x[group]
            y_group = y[group]
            out *= lst_kernels[i].f_eval(x_group, y_group)
        return out   
      
#######################################################################
############################# Geweke test #############################
#######################################################################
# def geweke_se2(g, L=0, center=True):
#     '''
#     Calculate the squared standard error of the estimate of E[`g`] (Geweke 1999, 3.7-8).
#     '''
#     L = int(L)
#     M = g.shape[0]
#     if center==True:
#         g -= g.mean(axis=0)
#     v = geweke_c(g=g, s=0)
#     v_L = 0.
#     if L != 0:
#         w = (L-onp.arange(0, L))/L  # weights
#         v *= w[0]
#         assert L > 0 and L < M
#         for s in range(1, L):
#             v_L += w[s] * geweke_c(g=g, s=s)
#     v = (v + 2*v_L)/M
#     return v

# def geweke_c(g, s):
#     '''
#     Calculate the biased `s`-lag autocovariance of *centered* samples `g`
#     '''
#     if s == 0:
#         out = (g ** 2).mean(axis=0)
#     else:
#         M = g.shape[0]
#         out = ((g[s:, :]) * (g[:(M-s), :])).sum(0)/float(M)  # biased
#     return out

def geweke_se2(g, L=None, force_int_L=False):
    '''
    Calculate the squared standard error of the estimate of E[`g`] (Geweke 1999, 3.7-8); depends on arch.
    If `L`=None, automatically selects bandwidth for the lag window based on an asymptotic MSE criterion (Andrews 1991). This assumes that `g` is fourth-moment stationary and the autocovariances are L1-summable
    '''
    if len(g.shape) == 1:
        g = g.reshape(-1, 1)
    M = g.shape[0]
    if L is not None:
        bw = max(L-1, 0)
    else:
        bw = None
    v = onp.array([float(arch.covariance.kernel.Bartlett(g[:, j], bandwidth=bw,
                                  force_int=force_int_L).cov.long_run) for j in range(g.shape[1])])
    v /= M
    return v

def geweke_test(g_mc, g_sc, alpha=0.05, l=None, test_correction='bh'):
    '''
    Run Geweke test (Geweke 2004) on marginal-conditional and successive-conditional test function arrays `g_mc` and `g_sc`, each row corresponding to a sample
    Uses a maximum window size of `l`*M to estimate of the squared standard error of E[g_sc], where M is the number of successive-conditional samples
    Example values of `l` are 0.04, 0.08, 0.15. `l`=None for automatic lag window bandwidth selection (Andrews 1991)
    `test_correction` corrects for multiple testing if set to 'b' (for Bonferroni) or 'bh' (for Benjamini-Hochberg)
    '''
    
    assert test_correction in ['b', 'bh']
    num_tests = g_mc.shape[1]
    
    if len(g_mc.shape) == 1 or len(g_sc.shape) == 1:
        g_mc = g_mc.reshape(-1, 1)
        g_sc = g_sc.reshape(-1, 1)
    assert len(g_mc.shape) == 2 and len(g_sc.shape) == 2
    assert g_mc.shape[1] == g_sc.shape[1]
    
    mean_mc = g_mc.mean(axis=0)
    se2_mc = geweke_se2(g_mc, L=0)

    M_sc = float(g_sc.shape[0])
    mean_sc = g_sc.mean(axis=0)
    if l is not None:
        L_sc = l*M_sc
    else:
        L_sc = None
    se2_sc = geweke_se2(g_sc, L=L_sc)

    test_statistic = (mean_mc - mean_sc)/onp.sqrt(se2_mc + se2_sc)
    p_value = 2.*(1-scipy.stats.norm.cdf(abs(test_statistic)))
        
    if test_correction == 'b':
        threshold = scipy.stats.norm.ppf(1.-alpha/(2.)) # asymptotic
        alpha /= num_tests
        result = p_value <= alpha
    elif test_correction == 'bh':
        threshold = None
        rank = onp.empty_like(p_value)
        rank[onp.argsort(p_value)] = onp.arange(1, len(p_value)+1)
        under = p_value <= rank/num_tests * alpha
        if under.sum() > 0:
          rank_max = rank[under].max()
        else:
          rank_max = 0
        result = rank <= rank_max
    
    return {'result': result, 'p_value': p_value, 'test_statistic': test_statistic, 'critical_value': threshold, 'test_correction': test_correction}

def prob_plot(x, y, plot_type='PP', step = 0.005):
    '''
    Generate Geweke P-P plot (Grosse and Duvenaud 2014) for sample vectors x, y. Can also generate Q-Q plots.
    '''
    assert plot_type in ['PP', 'QQ']
    x = onp.sort(x)
    y = onp.sort(y)
    z_min = min(onp.min(x), onp.min(y))
    z_max = max(onp.max(x), onp.max(y))

    ecdf = lambda z, x: (x <= z).sum()/float(len(x))
    if plot_type == 'PP':
        pp_x = [ecdf(z, x) for z in onp.arange(z_min, z_max, step * (z_max-z_min))]
        pp_y = [ecdf(z, y) for z in onp.arange(z_min, z_max, step * (z_max-z_min))]
        plt.plot(pp_x, pp_y, marker='o', color='black', fillstyle='none', linestyle='none')
        plt.plot(pp_x, pp_x, color='black')
    elif plot_type == 'QQ':
        q = onp.arange(0., 1.+step, step)
        qq_x = onp.quantile(x, q)
        qq_y = onp.quantile(y, q)
        plt.plot(qq_x, qq_y, marker='o', color='black', fillstyle='none', linestyle='none')
        plt.plot(qq_x, qq_x, color='black')
    pass

#######################################################################
############################## MMD test ###############################
#######################################################################

def mmd_test(X, Y, kernel=rbf_kernel, alpha=0.05, null_samples=100, kernel_learn_method=None, mmd_type='unbiased', rng=None, X_train=None, Y_train=None, **kwargs):
    '''
    Quadratic/Linear time MMD test
    '''
    assert X.shape[1] == Y.shape[1] and len(X.shape) == 2 and len(Y.shape) == 2
    assert mmd_type in ['biased', 'unbiased', 'linear']
    assert kernel_learn_method is None or kernel_learn_method in ['median_heuristic']
    if kernel_learn_method is None and (kernel == rbf_kernel):
        kernel_learn_method = 'median_heuristic'
    
    if rng is None:
        rng = onp.random.default_rng()
    
    XY = onp.vstack([X,Y])
    K = kernel(XY, XY, **kwargs)
    
    if kernel in [rbf_kernel]:
        K.learn(method=kernel_learn_method)
    elif kernel == linear_kernel:
        pass
    elif kernel == sum_kernel:
        K.learn()

    if mmd_type == 'linear':
        assert X.shape == Y.shape
        n, p = X.shape
        f_kernel = K.f_kernel

        # Calculate test statistic
        test_statistic, var = mmd_l(X, Y, f_kernel, return_2nd_moment=True)
        var -= test_statistic**2
        scale = onp.sqrt(2.*var/n)
        p_value = scipy.stats.norm.sf(test_statistic, scale=scale)
        result = p_value <= alpha
        threshold = scipy.stats.norm.ppf(1.-alpha, scale=scale)
    else:
        
        # Calculate null distribution       
        K_XYXY = K.eval()
        n_X, p = X.shape
        n_Y = Y.shape[0]
        null_distr = onp.zeros(null_samples)
        for i in range(null_samples):
            ind = rng.permutation(int(n_X+n_Y))
            ind_X = ind[:n_X]
            ind_Y = ind[n_Y:]

            K_XX = K_XYXY[ind_X, :][:, ind_X]
            K_YY = K_XYXY[ind_Y, :][:, ind_Y]
            K_XY = K_XYXY[ind_X, :][:, ind_Y]
            
            if mmd_type == 'unbiased':
                null_distr[i] = mmd_u(K_XX=K_XX, K_YY=K_YY, K_XY=K_XY)
            elif mmd_type == 'biased':
                null_distr[i] = mmd_v(K_XX=K_XX, K_YY=K_YY, K_XY=K_XY)

        # Calculate test statistic
        K_XX = K_XYXY[:n_X, :][:, :n_X]
        K_YY = K_XYXY[n_X:, :][:, n_X:]
        K_XY = K_XYXY[:n_X, :][:, n_X:]

        if mmd_type == 'unbiased':
            test_statistic = mmd_u(K_XX=K_XX, K_YY=K_YY, K_XY=K_XY)
        elif mmd_type == 'biased':
            test_statistic = mmd_v(K_XX=K_XX, K_YY=K_YY, K_XY=K_XY)    

        threshold = onp.quantile(null_distr, 1.-alpha)
        result = test_statistic >= threshold
        p_value = (null_distr >= test_statistic).mean()
    
    return {'result':result, 'p_value':p_value, 'test_statistic':test_statistic, 'critical_value':threshold, 'kernel_param':K.params}

def mmd_u(K_XX, K_YY, K_XY, normalize=True):
    '''
    Generate (squared) Quadratic Time MMD u-statistic from kernel matrices
    '''
    assert K_XY.shape[0] == K_XY.shape[1]
    m, n = K_XY.shape
    if normalize == True:
        if m == n:
            z = n
        else:
            z = m * n / (m + n)
    else:
        z = 1.
    return z*(1./(m*(m-1)) * (K_XX.sum() - onp.diag(K_XX).sum()) + 1./(n*(n-1)) * (K_YY.sum() - onp.diag(K_YY).sum()) - 2.*K_XY.mean())

def mmd_l(X, Y, f_kernel, return_2nd_moment=False):
    '''
    Generate (squared) Linear Time MMD u-statistic from samples `X` and `Y` given kernel function `f_kernel`
    '''
    assert X.shape == Y.shape
    n, p = X.shape

    h = lambda x_i, y_i, x_j, y_j: f_kernel(x_i, x_j) + f_kernel(y_i, y_j) - f_kernel(x_i, y_j) - f_kernel(x_j, y_i)

    n_2 = int(n/2)
    stat = 0
    second = 0
    for i in range(n_2):
        h_i = h(x_i=X[2*i, :], y_i=Y[2*i, :], x_j=X[2*i+1, :], y_j=Y[2*i+1, :])
        stat += h_i
        second += h_i**2
    stat /= n_2
    second /= n_2

    if return_2nd_moment == True:
        return stat, second
    else:
        return stat

#######################################################################
############################ Wild MMD test ############################
#######################################################################

def wb_process(n, k=1, l_n=20, center=False, rng=None):
    '''
    Generate `k` wild bootstrap processes of length `n` for the Wild MMD test. Returns an (n x k) matrix
    '''
    if rng is None:
        rng = onp.random.default_rng()
    epsilon = rng.normal(size=(n, k))
    W = onp.sqrt(1-onp.exp(-2/l_n)) * epsilon
    
    for i in range(1, n):
        W[i, :] += W[i-1, :] * onp.exp(-1/l_n)

    if center==True:
        W -= W.mean(0).reshape(1, k)
    return W

def mmd_wb(K_XX, K_YY, K_XY, normalize=True, wb_l_n=20, wb_center=False, rng=None):
    '''
    Generate wild bootstrapped MMD v-statistic for the Wild MMD test using kernel matrices
    `normalize`=True will return the normalized bootstrapped statistics
    '''
    if rng is None:
        rng = onp.random.default_rng()
    n_X, n_Y = K_XY.shape
    z = 1.
    if n_X == n_Y:
        if normalize == True:
            z = n_X
        W = wb_process(n_X, l_n=wb_l_n, center=wb_center, rng=rng).reshape(-1, 1)
        return z*(W.T @ (K_XX + K_YY - 2*K_XY) @ W)/(n_X**2)
    else:
        w_X = wb_process(n_X, l_n=wb_l_n, center=wb_center, rng=rng).reshape(-1, 1)
        w_Y = wb_process(n_Y, l_n=wb_l_n, center=wb_center, rng=rng).reshape(-1, 1)
        if normalize == True:
            z = n_X * n_Y / (n_X + n_Y)
        return z*(1./n_X**2 * w_X.T @ K_XX @ w_X + 1./n_Y**2 * w_Y.T @ K_YY @ w_Y - 2./(n_X*n_Y) * w_X.T @ K_XY @ w_Y) 

def mmd_v(K_XX, K_YY, K_XY, normalize=True):
    '''
    Generate (squared) MMD v-statistic for the Wild MMD test
    `normalize`=True will return the normalized statistic
    '''
    n_X, n_Y = K_XY.shape
    z = 1.
    if n_X == n_Y:
        if normalize == True:
            z = n_X
    else:
        if normalize == True:
            z = n_X * n_Y / (n_X + n_Y)
    return z*(K_XX.mean() + K_YY.mean() - 2.*K_XY.mean())

def mmd_wb_test(X, Y, kernel=rbf_kernel, alpha=0.05, null_samples=100, kernel_learn_method=None, wb_l_n=20, wb_center=False, rng=None, **kwargs):
    '''
    Run Wild MMD test on samples with shape (n x p)
    '''
    if len(X.shape) == 1:
        X = X.reshape(X.shape[0], 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(Y.shape[0], 1)
    if rng is None:
        rng = onp.random.default_rng()
    assert kernel_learn_method is None or kernel_learn_method in ['median_heuristic']
    if kernel_learn_method is None and (kernel == rbf_kernel):
        kernel_learn_method = 'median_heuristic'    
    
    XY = onp.vstack([X, Y])
    K = kernel(XY, XY, **kwargs)
    if kernel in [rbf_kernel]:
        K.learn(method=kernel_learn_method)
    elif kernel == linear_kernel:
        pass
    elif kernel == sum_kernel:
        K.learn() 
        
    # Calculate test statistic
    n_X, p = X.shape
    n_Y = Y.shape[0]
    
    K_XYXY = K.eval()
    K_XX = K_XYXY[:n_X, :][:, :n_X]
    K_YY = K_XYXY[n_X:, :][:, n_X:]
    K_XY = K_XYXY[:n_X, :][:, n_X:]

    B = onp.empty(null_samples)
    for i in range(null_samples):
        B[i] = mmd_wb(K_XX, K_YY, K_XY, normalize=True, wb_l_n=wb_l_n, wb_center=wb_center, rng=rng)

    threshold = onp.quantile(B, 1.-alpha)
    test_statistic = mmd_v(K_XX, K_YY, K_XY, normalize=True)
    result = test_statistic >= threshold
    p_value = (B >= test_statistic).mean() # one-sided
    
    return {'result':result, 'p_value':p_value, 'test_statistic':test_statistic, 'critical_value':threshold, 'kernel_param':K.params}

def mmd_var(K_XX, K_XY, K_YY):
    '''
    Estimate MMD variance. From Sutherland et al. 2016
    '''
    m = K_XX.shape[0]

    diag_X = onp.diag(K_XX)
    diag_Y = onp.diag(K_YY)

    sum_diag_X = diag_X.sum()
    sum_diag_Y = diag_Y.sum()

    sum_diag2_X = diag_X.dot(diag_X)
    sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y
    K_XY_2_sum  = (K_XY ** 2).sum()

    var_est = (
          2 / (m**2 * (m-1)**2) * (
              2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum
            + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
        - (4*m-6) / (m**3 * (m-1)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4*(m-2) / (m**3 * (m-1)**2) * (
              K_XY_sums_1.dot(K_XY_sums_1)
            + K_XY_sums_0.dot(K_XY_sums_0))
        - 4 * (m-3) / (m**3 * (m-1)**2) * K_XY_2_sum
        - (8*m - 12) / (m**5 * (m-1)) * K_XY_sum**2
        + 8 / (m**3 * (m-1)) * (
              1/m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0))
    )

    return var_est

#######################################################################
#################### Tests from Gandy & Scott 2020 ####################
#######################################################################

def ks_test(X, Y, alpha=0.05, test_correction='bh'):
    '''
    Exact two-sample Kolmogorov-Smirnov test from Gandy and Scott 2020
    `test_correction` corrects for multiple testing if set to 'b' (for Bonferroni) or 'bh' (for Benjamini-Hochberg)
    '''
    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[1] == Y.shape[1]
    
    assert test_correction in ['b', 'bh']
    num_tests = X.shape[1]
    
    p_value = onp.array([scipy.stats.ks_2samp(X[:, j], Y[:, j]).pvalue for j in range(X.shape[1])])
    
    if test_correction == 'b':
        threshold = scipy.stats.norm.ppf(1.-alpha/(2.)) # asymptotic
        alpha /= num_tests
        result = p_value <= alpha
    elif test_correction == 'bh':
        threshold = None
        rank = onp.empty_like(p_value)
        rank[onp.argsort(p_value)] = onp.arange(1, len(p_value)+1)
        under = p_value <= rank/num_tests * alpha
        if under.sum() > 0:
          rank_max = rank[under].max()
        else:
          rank_max = 0
        result = rank <= rank_max
    
    return {'result': result, 'p_value': p_value}

def rank_stat(model, L, test_functions=None, rng=None):
    '''
    Generate Rank statistic from Gandy and Scott 2020 using a chain of length `L` sampled from `model`
    '''
    if test_functions is None:
        test_functions = model.test_functions
    if rng is None:
        rng = onp.random.default_rng()
    
    M = rng.choice(L)

    chain = onp.zeros(shape=(L, len(model.theta_indices)))
    chain[M, :] = model.drawPrior()

    y = model.drawLikelihood()
    
    stateDict = model.__dict__.copy()

    # Backward
    for i in range(M-1, -1, -1):
        chain[i, :] = model.drawPosterior()

    # Forward
    model.__dict__ = stateDict.copy()
    for j in range(M+1, L):
        chain[j, :] = model.drawPosterior()
    
    # Apply test functions
    chain = onp.hstack([onp.repeat(y.reshape(1, model._N*model._D), repeats=L, axis=0), chain])
    chain = test_functions(chain)
    return scipy.stats.rankdata(chain, 'ordinal', axis = 0)[M, :]

def rank_test(model, N, L, alpha=0.05, test_functions=None, test_correction='bh', rng=None):
    '''
    Rank test from Gandy and Scott 2020
    `N` = number of Rank statistics to compute
    `L` = length of each Rank statistic's chain
    `test_correction` corrects for multiple testing if set to 'b' (for Bonferroni) or 'bh' (for Benjamini-Hochberg)
    ''' 
    if rng is None:
        rng = model._rng_s
    if test_functions is None:
        test_functions = model.test_functions
    if test_correction is not None:
        assert test_correction in ['b', 'bh']
        
    ranks = onp.vstack([rank_stat(model=model, L=L, test_functions=test_functions) for _ in range(N)])
    f_obs = onp.apply_along_axis(lambda x: onp.bincount(x, minlength=L), axis=0, arr=ranks-1)
    p_value = onp.array([scipy.stats.chisquare(f_obs[:, j]).pvalue for j in range(ranks.shape[1])])

    num_tests = ranks.shape[1]
    if test_correction == 'b':
        result = (p_value <= alpha/num_tests)
    elif test_correction == 'bh':
        rank_bh = onp.empty_like(p_value)
        rank_bh[onp.argsort(p_value)] = onp.arange(1, len(p_value)+1)
        under = p_value <= rank_bh/num_tests * alpha
        if under.sum() > 0:
            rank_bh_max = rank_bh[under].max()
        else:
            rank_bh_max = 0
        result = rank_bh <= rank_bh_max        
      
    return {'result': result, 'p_value': p_value}  

def f_test_sequential(sample_size, model, test_type, **kwargs):
    '''
    Helper function for the sequential test from Gandy and Scott 2020
    Returns p-values from a `test_type` test using samples generated from `model`
    '''
    assert test_type in ['rank', 'ks', 'mmd', 'mmd-wb', 'geweke']
        
    if test_type == 'rank':
        p_values = rank_test(model, N=500, L=5)['p_value']
    elif test_type in ['ks', 'mmd']:
        X = model.test_functions(model.sample_mc(sample_size))
        Y = model.test_functions(model.sample_bc(sample_size, burn_in_samples=5))
        if test_type == 'ks':
            p_values = ks_test(X, Y)['p_value']
        elif test_type == 'mmd':
            p_values = mmd_test(X, Y, kernel=rbf_kernel, mmd_type='unbiased')['p_value']
    elif test_type in ['mmd-wb', 'geweke']:
        if test_type == 'mmd-wb':
            mmd_test_size = int(sample_size)
            mmd_thinning = onp.arange(0, int(sample_size), 5)
            X = model.test_functions(model.sample_mc(mmd_test_size))
            Y = model.test_functions(model.sample_sc(sample_size))
            p_values = mmd_wb_test(X, Y[mmd_thinning, :])['p_value']
        elif test_type == 'geweke':
            geweke_thinning = onp.arange(0, int(sample_size), 5)
            X = model.test_functions(model.sample_mc(sample_size))
            Y = model.test_functions(model.sample_sc(sample_size))
            p_values = geweke_test(X, Y[geweke_thinning, :], l=0.08, test_correction='b')['p_value']        
    return p_values
    
def sequential_test(f_test, n, alpha, k, Delta):
    '''
    Sequential wrapper from Gandy and Scott 2020
    '''  
    beta = alpha/k
    gamma = beta**(1/k)

    for i in range(k):
        p = f_test(n)
        if type(p).__name__ == 'ndarray':
            d = p.flatten().shape[0]
            q = onp.min(p)*d
        else:
            q = p
        
        if onp.isnan(q):
            return True
        if q <= beta:
            return True
        if q > gamma + beta:
            break
        beta = beta/gamma

        if i == 0:
            n = Delta * n
    return False