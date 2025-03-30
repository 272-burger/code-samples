import numpy as np
import pandas as pd
import scipy.stats as sps
from sdm import nw_func, hac_var_func


def Beta_TPreg(f, R, w_intercept = 0, idx='Shanken'):
    
    [T,K] = f.shape
    N = R.shape[1]
    F = np.hstack([np.ones((T,1)), f])
    
    Lambda0 = np.zeros((K+1)) if w_intercept else np.zeros((K)) 
    
    
    ### Time Series Regression ###
    # Rt = alpha + beta @ Ft + e
    B = np.linalg.inv(F.T @ F) @ F.T @ R # (F'F)^-1 @ F'R 
    alpha = B[0,:].T  # alpha
    beta = B[1:,:].T # betas

    e = R - F @ B  # residuals
    SSR = e.T @ e  # sum of squared residuals
    # R_bar = np.mean(R, axis=0).T   # average returns  
    # SST = (R - np.tile(R_bar.T, (T,1))).T @ (R - np.tile(R_bar.T, (T,1)))  # sum of squared demeaned returns  

    S = (1/T) * SSR # Cov-var matrix of residuals under i.i.d.-assumption
    Sf = np.cov(f.T, ddof = 0) 
    SSf = np.vstack([np.hstack([np.zeros((1,K+1))]),
                    np.hstack([np.zeros((K,1)), Sf])]) if w_intercept else Sf # Cov-var matrix of factors
    Sr = np.cov(R.T, ddof = 0) # Cov-var matrix of test asset returns
    
    ### Cross Sectional Regression ###
    R_bar = np.mean(R, axis=0).T 
    X = np.hstack([np.ones((N, 1)), beta]) if w_intercept else beta
    A = np.linalg.inv(X.T @ X) @ X.T
    inv_Sr = np.linalg.inv(Sr)

    Theta_OLS = A @ R_bar
    Theta_GLS = np.linalg.inv(X.T @ inv_Sr @ X) @ X.T @ inv_Sr @ R_bar

    if w_intercept == 0:
        Lambda_OLS = Theta_OLS
        # Lambda_GLS = Theta_GLS
        # const_OLS = np.nan
        # const_GLS = np.nan
    else:
        Lambda_OLS = Theta_OLS[1:]
        # Lambda_GLS = Theta_GLS[1:]
        # const_OLS = Theta_OLS[0]
        # const_GLS = Theta_GLS[0] 
        
    if idx == 'OLS': # OLS standard error
        
        AV_Lambda = A @ S @ A.T + SSf 
        SE_Lambda = np.sqrt(np.diag(AV_Lambda / T))
        T_Lambda = ((Theta_OLS - Lambda0) / SE_Lambda)
        pval_Lambda = 1 - sps.t.cdf(np.abs(T_Lambda), T-K) 
    
    else: # Shanken
        Shanken = 1 + Lambda_OLS.T @ np.linalg.inv(Sf) @ Lambda_OLS

        AV_Lambda = Shanken * (A @ S @ A.T) + SSf 
        SE_Lambda = np.sqrt(np.diag(AV_Lambda / T))
        T_Lambda = ((Theta_OLS - Lambda0) / SE_Lambda)
        pval_Lambda = 1 - sps.t.cdf(np.abs(T_Lambda), T-K) 
        
    return  SE_Lambda, T_Lambda, pval_Lambda, Theta_OLS #,  Theta_OLS, e, X


def Beta_GMM(f, R, w_intercept=0, idx='NW', lags = 3):
    [T,K] = f.shape
    N = R.shape[1]
    F = np.hstack([np.ones((T,1)), f])
    
    Lambda0 = np.zeros((K+1)) if w_intercept else np.zeros((K)) 
    
    B = np.linalg.inv(F.T @ F) @ F.T @ R # (F'F)^-1 @ F'R 
    beta = B[1:,:].T # betas
    e = R - F @ B  # residuals
    R_bar = np.mean(R, axis=0).T 
    X = np.hstack([np.ones((N, 1)), beta]) if w_intercept else beta
    A = np.linalg.inv(X.T @ X) @ X.T
    Theta_OLS = A @ R_bar
    Lambda_OLS = Theta_OLS[1:] if w_intercept else Theta_OLS

    # Moment Conditions
    g_ts = np.hstack([e[:, [i]] * F for i in range(N)]) # T * N(K+1)
    g_cs = R - np.tile((X @ Theta_OLS), (T,1)) # T * N
    g_GMM = np.hstack([g_ts, g_cs])
    
    # aT & dT
    aT = np.vstack([np.hstack([np.eye((N*(K+1))), np.zeros((N*(K+1), X.shape[0]))]),
                np.hstack([np.zeros((X.shape[1], (N*(K+1)))), X.T])])
    Mf = (F.T @ F)/T

    dT = np.vstack([np.hstack([np.kron(-np.eye(N), Mf), np.zeros((N*(K+1), X.shape[1]))]),
                    np.hstack([np.kron(-np.eye(N), np.insert(Lambda_OLS,0,0)), -X])])
    
    if idx == 'NW': # Newey & West
        SDM = nw_func(g_GMM, lags)
        
    if idx == 'HAC': 
        SDM = hac_var_func(g_GMM, 1, 0)
        
    AV = np.linalg.solve((aT @ dT), aT) @ SDM @ np.linalg.solve((aT @ dT), aT).T
    AV_Lambda = AV[N*(K+1):, N*(K+1):] # Extract elements corresponding to lambda 
    SE_Lambda = np.sqrt(np.diag(AV_Lambda / T))
    T_Lambda = ((Theta_OLS - Lambda0) / SE_Lambda)
    pval_Lambda = 1 - sps.t.cdf(np.abs(T_Lambda), T-K) # p-value
    
    return SE_Lambda, T_Lambda, pval_Lambda, Theta_OLS

def SDF_GMM(f, R, w_intercept=0, idx='I', lags=3):
    [T,K] = f.shape
    N = R.shape[1]
    F = np.hstack([np.ones((T,1)), f])
    
    Lambda0 = np.zeros((K+1)) if w_intercept else np.zeros((K)) 
    
    R_bar = np.mean(R, axis=0).reshape(-1,1)
    f_bar = np.mean(f, axis=0).reshape(-1,1) 
    DT = (R.T @ f) / T

    ## First-stage estimates W = I
    dT = np.hstack([np.ones((N,1)), DT - R_bar @ f_bar.T])  if w_intercept else DT - R_bar @ f_bar.T
    mu0 = f_bar # mu = E[f] estimates
    b0 = np.linalg.inv(dT.T @ dT) @ dT.T @ R_bar # b estimates


    ## Second-stage estimates W 
    # Moment conditions
    m = 1 - (f - np.tile(mu0.T, (T, 1))) @ b0[1:, :]  if w_intercept else 1 - (f - np.tile(mu0.T, (T, 1))) @ b0 # SDF; 1 - b'(f-E[f])
    u1 = np.multiply(R, m @ np.ones((1, N)))  - b0[0, 0]  if w_intercept else  np.multiply(R, m @ np.ones((1, N))) # pricing error
    u2 = f - np.tile(mu0.T, (T, 1)) # mean of factors; f - E[f]

    u3 = np.zeros((T, int(K * (K + 1) / 2)))  # covariance of factors; Sigma_f
    Sf = np.cov(f.T, ddof=0)  
    a = 0 
    for i in range(K):  
        for j in range(i, K):  
            mean_f_i = np.mean(f[:, i])  
            mean_f_j = np.mean(f[:, j])
                    
            u3[:, a] = (f[:, i] - mean_f_i) * (f[:, j] - mean_f_j) - Sf[i, j] 
            a += 1  
    u = np.concatenate((u1,u2,u3), axis = 1)
    g1T = np.mean(u1, axis=0).reshape(-1,1)

    # Spectral Density Matrix
    SDM = nw_func(u, lags)
    # SDM = hac_var_func(u, 1, 0)
    if idx=='I':
        W = np.eye(N)
    else:
        W = np.linalg.inv(SDM[0:(N),0:(N)])

    # aT & delT
    b = np.linalg.inv(dT.T @ W @ dT) @ dT.T @ W @ R_bar # b estimates
    aT = np.vstack([np.hstack([W @ dT, np.zeros((N, K))]),
        np.hstack([np.zeros((K,b0.shape[0])), np.eye(K)])])

    delT = np.vstack([np.hstack([dT, -(R_bar @ b[1:,:].T)]),
                    np.hstack([np.zeros((K,b0.shape[0])), np.eye(K)])]) if w_intercept else np.vstack([np.hstack([dT, -(R_bar @ b.T)]),
                                                                                                    np.hstack([np.zeros((K,b.shape[0])), np.eye(K)])]) 


    # Covariance matrix of estimated b
    B = np.linalg.inv(aT.T @ delT) @ aT.T
    Vb = (B @ SDM[0:(N+K),0:(N+K)] @ B.T) / T
    SE_b = np.sqrt(np.diag(Vb[0:b.shape[0],0:b.shape[0]]))[:,None] # Extract elements corresponding to lambda 
    T_b = (b/SE_b).flatten()
    pval_b = 1 - sps.t.cdf(np.abs((b/SE_b).flatten()),T-K)


    ## J-test
    Md = np.eye(N) - np.linalg.solve((dT.T @ dT).T, dT.T).T @ dT.T
    Vu = Md @ SDM[0:N,0:N] @ Md.T
    JT = (T * g1T.T @ np.linalg.pinv(Vu) @ g1T)[0][0]
    pval_JT = 1 - sps.chi2.cdf((T * g1T.T @ np.linalg.pinv(Vu) @ g1T)[0][0],N-K)
    
    return SE_b, T_b, pval_b,  b.flatten(), pval_JT

def block_bootstrap_func(data, B, w): #,  seed):
    
    # Input Checking
    [t,k] = data.shape # Get length of data
    
    if k>1:
        raise ValueError("DATA must be a column vector")
    
    if t<2:
        raise ValueError("DATA must have at least 2 observations")
    
    if not np.isscalar(w) or w<1 or np.floor(w) != w or w>t:
        raise ValueError("W must be a positive scalar integer smaller than T")
    
    if not np.isscalar(B) or B<1 or np.floor(B) != B:
        raise ValueError("B must be a positive scalar integer")
        
    # Compute the number of blocks needed
    s = int(np.ceil(t / w))
    
    # Generate the starting points (Bs)
    Bs = np.floor(np.random.rand(s, B) * t).astype(int) 
    
    # Initialize indices matrix
    indices = np.zeros((s * w, B), dtype=int)
    adder = np.tile(np.arange(w).reshape(-1, 1), (1, B)) 
    
    index = 0
    for i in range(0, t, w):  
        indices[i:(i+w), :] = np.tile(Bs[index, :], (w, 1)) + adder
        index += 1

    # Trim indices to ensure it does not exceed t
    indices = indices[:t, :]
    
    # Circular indexing: Wrap around if indices exceed t
    indices[indices >= t] -= t  

    # Generate bootstrap samples
    bsdata = data[indices]

    return bsdata, indices


def FMB_coefficients_func(R,f,with_intercept):
    [T,K] = f.shape
    N = np.size(R, axis=1)
    
    # Time Series Regression to determine Beta (Cochrane, 2005, p.230, ff)
    F = np.array(np.concatenate((np.ones((T,1),float),f), axis=1))
    FFi = np.linalg.solve(F.T @ F, np.identity(K+1))             
    B = FFi @ F.T @ R
    alpha = np.array(B[0]).T                                   # Alphas
    beta = np.array(B[1:K+1]).T                                # Betas
    R_bar = np.mean(R, axis=0).T                               # Average returns
    
    # Cross-sectional regression to determine Lambda
    
    if with_intercept == 1:
        
        X = np.concatenate((np.ones((N,1)), beta), axis = 1)
        A = np.linalg.solve(X.T @ X,X.T)
        Theta = A @ R_bar
        Lambda = np.array(Theta[1:K+1])                    
        const = np.array(Theta[0])
        Lambda = np.hstack([const, Lambda])
    
    else:
        A = np.linalg.solve((beta.T @ beta),beta.T)
        Lambda = A @ R_bar                                     
        # const = np.nan
    return Lambda


def Burnside_CI(f, R, w_intercept, level, B, W):
    """
        B: Bootstrapping Sample Size
        W: Bootstrapping Block Size
    """
    [_,K] = f.shape
    Lambda0 = np.zeros((K+1)) if w_intercept else np.zeros((K)) 
    Lambda_dist = np.full((B, K+1), np.nan) if w_intercept else np.full((B, K), np.nan)
    idx = block_bootstrap_func(f[:, :1], B, W)[1]

    for tau in range(0, B):
        # Sampling by using the block bootstrapped indices
        f_BOOT = f[idx[:, tau], :]
        R_BOOT = R[idx[:, tau], :]
        # Run the 2-Pass Regression and generate the lambda estimates from each bootstrapped sample
        Lambda_dist[tau, :] = FMB_coefficients_func(R_BOOT, f_BOOT, w_intercept)
        
    ci = np.quantile(Lambda_dist, (level/2, 1-level/2), axis=0)
    av_sign = -1 + 2*(np.mean(Lambda_dist > 0, axis=0) > 0.5)
    p_val = (np.mean((np.ones((B,1)) @ av_sign[:,None].T * (Lambda_dist - np.ones((B,1)) @ Lambda0.reshape(-1, 1).T)) < 0, axis=0))[:,None].T
    
    CI = []
    lamb = []
    ci_num = K+1 if w_intercept else K
    for k in range(0,ci_num):
        Lambda_CIlow = round(ci[0,k], 3)
        Lambda_CIhigh = round(ci[1,k], 3)
        CI.append("(" + str(Lambda_CIlow) + "," + str(Lambda_CIhigh) + ")")
        lamb.append((Lambda_CIlow+Lambda_CIhigh)/2)
    return CI, p_val[0], lamb