import numpy as np
import numpy.matlib
from sklearn import preprocessing


def emp_covar_mat(dataset, with_std=False):
    '''
    calculate empirical covariance matrix S,
    dataset is given as 2-dimension numpy array (Sample_Num, Node_Num)
    '''
    SN = dataset.shape[0]

    # NN = dataset.shape[1]
    # S = np.matlib.zeros((NN, NN), dtype=dataset.dtype)
    # x_aver = np.sum(dataset, axis=0) / SN
    # x_aver_mat = np.matrix(x_aver)
    # for x in dataset:
    #     x_mat = np.matrix(x)
    #     S += (x_mat - x_aver_mat).T * (x_mat - x_aver_mat)
    # S = S / SN

    scaler = preprocessing.StandardScaler(with_std=with_std).fit(dataset)
    dataset_scaled = scaler.transform(dataset)
    dataset_scaled = np.matrix(dataset_scaled)
    S = dataset_scaled.T * dataset_scaled / SN

    return S


def ggmlasso(S, lam):
    S = np.matrix(S)
    
    p = S.shape[0]
    noff = p * (p-1) / 2

    Smag = np.sum(np.abs(np.triu(S))) / noff
    dW = np.inf

    theta = np.matlib.zeros((p, p), dtype=S.dtype)

    W = S + lam * np.matrix(np.eye(p))
    MaxOuter = 20
    MaxInner = 20
    Tol = 1e-4

    nmax = 0
    m = 1
    while (m <= MaxOuter) and (dW > Tol):
        W0 = W

        for i in range(p):
            noti = list(range(p))
            noti.pop(i)
            W11 = W[noti][:, noti]
            W12 = W[noti][:, i]
            S22 = S[i, i]
            S12 = S[noti][:, i]
            W22 = W[i, i]

            V = W11
            beta = np.matlib.zeros((p-1, 1), dtype=S.dtype)
            dbeta = np.inf
            n = 1
            while (n <= MaxInner) and (dbeta > Tol):
                beta0 = beta.copy()
                for j in range(p-1):
                    notj = list(range(p-1))
                    notj.pop(j)
                    res = S12[j] - V[notj][:, j].T * beta[notj]
                    beta[j] = np.sign(res) * np.max([np.abs(res)-lam, 0]) / V[j, j]
                dbeta = np.mean(np.abs(beta-beta0)) / (np.mean(np.abs(beta0)) + 1e-16)
                n = n + 1
            if n > nmax:
                nmax = n
            W12 = W11 * beta
            W[noti, i] = W12.T
            W[i, noti] = W12.T

            # update precision matrix theta
            theta22 = np.max([0, 1 / (W22 - W12.T * beta)])
            theta12 = - beta * theta22
            theta[noti, i] = theta12.T
            theta[i, noti] = theta12.T
            theta[i, i] = theta22

        m = m + 1
        dW = (np.sum(np.abs(np.triu(W) - np.triu(W0))) / noff) / Smag

    return W, theta, (m, dW)