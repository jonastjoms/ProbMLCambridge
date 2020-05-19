import gpflow
import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


#make a one dimensional classification problem
np.random.seed(1)
X = np.random.rand(100,1)
K = np.exp(-0.5*np.square(X - X.T)/0.01) + np.eye(100)*1e-6
f = np.dot(np.linalg.cholesky(K), np.random.randn(100,3))

plt.figure(figsize=(12,6))
plt.plot(X, f, '.')
#plt.show()

Y = np.array(np.argmax(f, 1).reshape(-1,1), dtype=float)


m = gpflow.models.SVGP(X, Y, kern=gpflow.kernels.Matern32(1) + gpflow.kernels.White(1, variance=0.01), likelihood=gpflow.likelihoods.MultiClass(3), Z=X[::5].copy(), num_latent=3, whiten=True, q_diag=True)

m.kern.white.variance.trainable = False
m.feature.trainable = False
m.as_pandas_table()
