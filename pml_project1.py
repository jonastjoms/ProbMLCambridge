#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')  # ignore DeprecationWarnings from tensorflow

import matplotlib.pyplot as plt

import gpflow

from gpflow.utilities import print_summary, set_trainable
from gpflow.ci_utils import ci_niter

from multiclass_classification import plot_posterior_predictions, colors

np.random.seed(0)  # reproducibility


# In[2]:


# Number of functions and number of data points
C = 3
N = 100

# RBF kernel lengthscale
lengthscale = 0.1

# Jitter
jitter_eye = np.eye(N) * 1e-6

# Input
X = np.random.rand(N, 1)


# In[3]:


# SquaredExponential kernel matrix
kernel_se = gpflow.kernels.SquaredExponential(lengthscale=lengthscale)
K = kernel_se(X) + jitter_eye

# Latents prior sample
f = np.random.multivariate_normal(mean=np.zeros(N), cov=K, size=(C)).T

# Hard max observation
Y = np.argmax(f, 1).reshape(-1,).astype(int)

# One-hot encoding
Y_hot = np.zeros((N, C), dtype=bool)
Y_hot[np.arange(N), Y] = 1

data = X, Y


# In[4]:


plt.figure(figsize=(12, 6))
order = np.argsort(X.reshape(-1,))

for c in range(C):
    plt.plot(X[order], f[order, c], '.', color=colors[c], label=str(c))
    plt.plot(X[order], Y_hot[order, c], '-', color=colors[c])


plt.legend()
plt.xlabel('$X$')
plt.ylabel('Latent (dots) and one-hot labels (lines)')
plt.title('Sample from the joint $p(Y, \mathbf{f})$')
plt.grid()
plt.show()


# In[5]:


# sum kernel: Matern32 + White
kernel = gpflow.kernels.Matern32() + gpflow.kernels.White(variance=0.01)

# Robustmax Multiclass Likelihood
invlink = gpflow.likelihoods.RobustMax(C)  # Robustmax inverse link function
likelihood = gpflow.likelihoods.MultiClass(3, invlink=invlink)  # Multiclass likelihood
Z = X[::5].copy()  # inducing inputs

m = gpflow.models.SVGP(kernel=kernel, likelihood=likelihood,
    inducing_variable=Z, num_latent=C, whiten=True, q_diag=True)

# Only train the variational parameters
set_trainable(m.kernel.kernels[1].variance, False)
set_trainable(m.inducing_variable, False)
print_summary(m, fmt='notebook')


# In[ ]:


opt = gpflow.optimizers.Scipy()

@tf.function(autograph=False)
def objective_closure():
    return - m.log_likelihood(X,Y)

opt_logs = opt.minimize(objective_closure,
                        m.trainable_variables,
                        options=dict(maxiter=ci_niter(1000)))
print_summary(m, fmt='notebook')


# In[ ]:
