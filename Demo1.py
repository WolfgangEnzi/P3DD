from P3DD import *

# number of used samples
N = 10000

# create a coil
mean_samps1 = np.array([0.0, 0.0, 0.0])
cov_samps1 = np.array([[0.01, 0.0, 0.0],
                       [0.0, 0.01, 0.0],
                       [0.0, 0.0, 0.01]])
samps1_aux = np.random.multivariate_normal(mean_samps1,
                                           cov_samps1, N).T
phi1 = np.linspace(0, 6*np.pi, N)
r1 = np.random.uniform(2, 3, N)
samps1 = samps1_aux + np.array([phi1 - np.mean(phi1),
                                r1 * np.cos(phi1), r1 * np.sin(phi1)])
wsamps1 = np.ones((N))

# create a Mobius loop
mean_samps2 = np.array([0.0, 0.0, 0.0])
cov_samps2 = np.array([[0.01, 0.0, 0.0],
                       [0.0, 0.01, 0.0],
                       [0.0, 0.0, 0.01]])
samps2_aux = np.random.multivariate_normal(mean_samps2,
                                           cov_samps2, N).T
wsamps2 = np.ones((N))
b2 = np.linspace(0, 2 * np.pi, N)
a2 = np.random.uniform(-2.0, 2.0, N)
samps2 = 5 * np.array([(2 + a2 / 2.0 * np.cos(b2 / 2.0)) * np.cos(b2),
                       a2 / 2.0 * np.sin(b2 / 2.0),
                       (2 + a2 / 2.0 * np.cos(b2 / 2.0)) * np.sin(b2)])
samps2 = samps2 + samps2_aux

# combine the samples
samples = np.array([[samps1, wsamps1], [samps2, wsamps2]])

# Plot the densities

fig = plt.figure(figsize=(6, 6))
ax3d = plt.gca()
Density3d(samples, ax3d)
plt.tight_layout()
plt.subplots_adjust(hspace=0.0, wspace=0.0)
plt.show()
