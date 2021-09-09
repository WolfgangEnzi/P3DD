from P3DD import *

# generating Samples
N = 40000

# create gaussian noise
mean_samps1 = np.array([0.0, 0.0, 0.0])
cov_samps1 = np.array([[0.1, 0.0, 0.0],
                       [0.0, 0.1, 0.0],
                       [0.0, 0.0, 0.1]])
samps1_aux = np.random.multivariate_normal(mean_samps1,
                                           cov_samps1, N).T
x = np.random.uniform(-5, 5, N)
y = np.random.uniform(-5, 5, N)
z = np.random.uniform(-0.1, 0.1, N)


# pseudo galaxy shape for the generated samples
def pseudo_galdens(r, p, z, a, b, c):
    aux = np.exp(-0.5 * r * r * (1 + 0.8 * np.sin(a * p + b * r) + z * z * c))
    return aux


r = np.sqrt(x * x + y * y)
p = np.arctan2(y, x)

samps1 = roti(-1.6, 2).dot(roti(2.3, 0).dot(np.array([x, y, z])))
wsamps1 = pseudo_galdens(r, p, z, 5, 6, 4)

samps2 = 0.5 * roti(0.2, 0).dot(roti(0.2, 2).dot(np.array([x, y, z])))
samps2 = samps2 + np.array([-4.5, 2.5, 2.5])[:, np.newaxis]
wsamps2 = pseudo_galdens(r, p, z, 2, 3, 4)

samples = np.array([[samps1, wsamps1], [samps2, wsamps2]])

# plot the densities
fig = plt.figure(figsize=(6, 6))
ax3d = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)

ax_a_2d = plt.subplot2grid((3, 3), (0, 0), sharex=ax3d, colspan=2)
ax_b_2d = plt.subplot2grid((3, 3), (1, 2), sharey=ax3d, rowspan=2)

ax_a_2d.spines['right'].set_visible(False)
ax_a_2d.spines['top'].set_visible(False)
ax_a_2d.spines['left'].set_visible(False)
ax_a_2d.yaxis.set_visible(False)

ax_b_2d.spines['right'].set_visible(False)
ax_b_2d.spines['top'].set_visible(False)
ax_b_2d.spines['bottom'].set_visible(False)
ax_b_2d.xaxis.set_visible(False)


def draw_sample1d(s, w, P, env1, env2, cc, Nbins=50, sms=2, nsig=6):
    Ps = project(s, P)
    min = np.min(Ps, axis=1)
    max = np.max(Ps, axis=1)
    amin = min[0] - 0.1 * (max[0] - min[0])
    amax = max[0] + 0.1 * (max[0] - min[0])
    arange = np.linspace(amin, amax, Nbins)
    bmin = min[1] - 0.1 * (max[1] - min[1])
    bmax = max[1] + 0.1 * (max[1] - min[1])
    brange = np.linspace(bmin, bmax, Nbins)
    hista, ha = np.histogram(Ps[0], weights=w,
                             bins=arange, density=True)
    histb, hb = np.histogram(Ps[1], weights=w,
                             bins=brange, density=True)
    cov = get_cov(Ps[:2], w)
    smooth_hista = gauf(hista, np.sqrt(cov[0, 0]) / (arange[1] - arange[0]))
    smooth_histb = gauf(histb, np.sqrt(cov[1, 1]) / (brange[1] - brange[0]))
    samps_plota = env1.fill_between(ha[1:], 0, smooth_hista,
                                    color=cc, alpha=0.25)
    samps_plotb = env2.fill_betweenx(hb[1:], 0, smooth_histb,
                                     color=cc, alpha=0.25)

    return Ps


def backend(D3d):
    ax_a_2d.clear()
    ax_b_2d.clear()
    N = len(samples)
    for i in range(N):
        draw_sample1d(D3d.samples[i][0], D3d.samples[i][1], D3d.P,
                      ax_a_2d, ax_b_2d, ccycle[i], Nbins=D3d.Nbins,
                      sms=D3d.sms, nsig=D3d.nsig)


Density3d(samples, ax3d, backend, Nbins=200, sms=1.8)
plt.tight_layout()

plt.show()
