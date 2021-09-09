"""
================================================================================

P3DD v1.0 - Wolfgang Enzi 2021

A short script that creates a projection of 3D densities (P3DD) of samples
and is designed to visualize 3D samples.

Requires only: scipy, matplotlib, and numpy

Notes on this version:
- It would be nice to add to Demo2 the option to choose 3 dimensions of general
  nD data.

================================================================================
"""

import matplotlib as mpl
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as gauf
import numpy as np
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap

ccycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
P0 = np.array([[1.0, 0.0, 0.0],
               [0.0, 0.0, 1.0],
               [0.0, 1.0, 0.0]])


# function to derive colormaps that increase in alpha for a given color
def get_alpha_colormap(auxc=ccycle[0]):
    u = colors.hex2color(auxc)
    cdict = {'red': [[0.0, u[0], u[0]],
                     [1.0, u[0], u[0]]],
             'green': [[0.0, u[1], u[1]],
                       [1.0, u[1], u[1]]],
             'blue': [[0.0, u[2], u[2]],
                      [1.0, u[2], u[2]]],
             'alpha': [[0.0, 0.0, 0.0],
                       [1.0, 0.8, 0.8]]}
    newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict)
    return newcmp


# function that creates a rotation matrix with angle p around the axis i
def roti(p, i):
    u = (i + 1) % 3
    v = (i + 2) % 3
    R = np.identity(3)
    R[u, u] = np.cos(p)
    R[v, v] = np.cos(p)
    R[u, v] = np.sin(p)
    R[v, u] = -np.sin(p)
    return R


# function that projects a set of vectors onto a 2d plane
def project(v, P=P0):
    p_v = P.dot(v)
    return p_v


# function that determines colors of axes depending on their distance
# in the projected direction
def cauxf(x, Dmin, Dmax):
    return np.ones((3)) * np.clip(0.6 - 0.6 * (x - Dmin) / (Dmax - Dmin), 0, 1)


# function to draw an axis in to the projected image
def draw_axis(env, i, Pl_all, L, label="x", shift=0.0):
    K = L.shape[1]
    Dmax = np.max(Pl_all[:, 2])
    Dmin = np.min(Pl_all[:, 2])
    Pl = Pl_all[i]

    if i == 0:
        zero_label = "(%.2f, %.2f, %.2f)" % (L[0, 0], L[1, 0], L[2, 0])
        a_zero = shift * 0.05 + Pl[0, 0]
        b_zero = shift * 0.05 + Pl[1, 0]
        c_zero = cauxf(Pl[2, 0], Dmin, Dmax)
        env.text(a_zero, b_zero, zero_label, fontsize="x-small", color=c_zero)

    # add labels to axes currently in x and y direction
    if np.fabs(Pl[1, -1] - Pl[1, 0]) < np.fabs(Pl[0, -1] - Pl[0, 0]):
        axm = 3
        a_mid = np.mean(Pl[0])
        b_mid = np.mean(Pl[1]) - shift * 0.3
        c_mid = cauxf(np.mean(Pl[2]), Dmin, Dmax)
        env.text(a_mid, b_mid, label, color=c_mid)
        a_end = Pl[0, -1]
        b_end = Pl[1, -1] - shift * 0.2
        c_end = cauxf(Pl[2, -1], Dmin, Dmax)
        l_end = "%.2f" % (L[i, -1])
        env.text(a_end, b_end, l_end, fontsize="x-small", color=c_end)

    # add labels to axes currently in z direction
    else:
        axm = 0
        a_mid = np.mean(Pl[0]) - shift * 0.2
        b_mid = np.mean(Pl[1])
        c_mid = cauxf(np.mean(Pl[2]), Dmin, Dmax)
        env.text(a_mid, b_mid, label, color=c_mid)
        a_end = Pl[0, -1] - shift * 0.4
        b_end = Pl[1, -1]
        c_end = cauxf(Pl[2, -1], Dmin, Dmax)
        l_end = "%.2f" % (L[i, -1])
        env.text(a_end, b_end, l_end, fontsize="x-small", color=c_end)

    # draw the line with ticks
    for u in range(K - 1):
        caux = cauxf(Pl[2, u], Dmin, Dmax)
        a_ax = Pl[0, u:u + 2]
        b_ax = Pl[1, u:u + 2]
        env.plot(a_ax, b_ax, c=caux, marker=axm, lw=1,
                 markersize=4, zorder=Pl[2, u])


# function that determines the
def draw_axes(lx, ly, lz, P, off, env):
    Plx = project(lx, P)
    Ply = project(ly, P)
    Plz = project(lz, P)
    draw_axis(env, 0, np.array([Plx, Ply, Plz]), lx, "x", off)
    draw_axis(env, 1, np.array([Plx, Ply, Plz]), ly, "y", off)
    draw_axis(env, 2, np.array([Plx, Ply, Plz]), lz, "z", off)
    return Plx, Ply, Plz


# covariance estimation following silverman,
# Monographs on Statistics and Applied Probability,
# Chapman and Hall, London, 1986
def get_cov(y, w):
    neff = np.sum(w) * np.sum(w) / np.sum(w * w)
    covf = (np.power(neff * (y.shape[0] + 2) / 4.0, -1.0 / (y.shape[0] + 4)))
    covy = np.cov(y, aweights=w)
    return covy * np.power(covf, 2)


# determine the porbability mass of 2D levels of size ind*Sigma
# use 2d levels because the density is drawn in projection
def contour_perc(ind):
    return (1 - np.exp(- (ind * 1.0) ** 2.0 / 2.0))


# function that determines the sigma levels of the sampled distribution
def get_sigmalevels(y, nsig):
    y_sort = np.sort(y.flatten())[::-1]
    y_cumsum = np.cumsum(y_sort)
    y_cumsum /= y_cumsum[-1]
    lvla_si = []

    for i in range(nsig):
        ind_lvl = np.argmin(np.power(y_cumsum - contour_perc(i + 1), 2.0))
        lvla_si += [(y_sort)[ind_lvl]]

    return np.append(np.sort(lvla_si), np.max(y))


# function that draws the density of
def draw_density(s, w, P, cmap, env, Nbins=50, sms=2, nsig=4):
    Ps = project(s, P)
    min = np.min(Ps, axis=1)
    max = np.max(Ps, axis=1)
    amin = min[0] - 0.1 * (max[0] - min[0])
    amax = max[0] + 0.1 * (max[0] - min[0])
    arange = np.linspace(amin, amax, Nbins)
    bmin = min[1] - 0.1 * (max[1] - min[1])
    bmax = max[1] + 0.1 * (max[1] - min[1])
    brange = np.linspace(bmin, bmax, Nbins)
    hist2d, ha, hb = np.histogram2d(Ps[0], Ps[1], weights=w,
                                    bins=[arange, brange], density=True)
    cov = get_cov(Ps[:2], w)
    invdab = np.array([[1 / (arange[1] - arange[0]), 0],
                       [0, 1 / (brange[1] - brange[0])]])
    smooth_hist = gauf(hist2d.T, np.sqrt(invdab.dot(np.diag(cov))))
    lvls = get_sigmalevels(smooth_hist, nsig)
    samps_plot = env.contourf(ha[1:], hb[1:],
                              smooth_hist, cmap=cmap,
                              levels=lvls, zorder=np.mean(Ps[2]))
    return Ps


# function for updating plot when rotated
def update(p, t, off, env, lx, ly, lz, samples, Nbins=50, sms=2, nsig=4):
    P = np.copy(P0).dot(roti(-t, 0).dot(roti(-p, 2)))
    env.clear()
    env.axis('off')
    Plx, Ply, Plz = draw_axes(lx, ly, lz, P, off, env)
    N = len(samples)
    Psamples = []

    for i in range(N):
        cmap = get_alpha_colormap(auxc=ccycle[i])
        Psamp = draw_density(samples[i][0], samples[i][1], P, cmap, env,
                             Nbins, sms, nsig)
        Psamples += [Psamp]

    max = np.max(np.concatenate([Plx, Ply, Plz]+Psamples,
                 axis=1), axis=1)
    min = np.min(np.concatenate([Plx, Ply, Plz]+Psamples,
                 axis=1), axis=1)

    env.set_xlim(min[0] - 0.1 * (max[0] - min[0]),
                 max[0] + 0.1 * (max[0] - min[0]))
    env.set_ylim(min[1] - 0.1 * (max[1] - min[1]),
                 max[1] + 0.1 * (max[1] - min[1]))

    return Psamples, P


# function that determines how long the drawn axes have to be
def get_L(samples):
    max = np.max(np.concatenate(samples[:, 0], axis=1), axis=1)
    min = np.min(np.concatenate(samples[:, 0], axis=1), axis=1)
    off = np.max((max - min) / 3.0)
    K = 5
    lx = np.ones((3, K)) * min[:, np.newaxis]
    lx[0] = np.linspace(min[0], max[0], K)
    ly = np.ones((3, K)) * min[:, np.newaxis]
    ly[1] = np.linspace(min[1], max[1], K)
    lz = np.ones((3, K)) * min[:, np.newaxis]
    lz[2] = np.linspace(min[2], max[2], K)
    return lx, ly, lz, max, min, off


# dummy function, with the idea that one can add any function for
# special purposes that is called once the plot is updated
# in place of this dummy
def dummy(x):
    return 0


# Create a class that determines the plot object
class Density3d:

    # initialize the plot with all the necessary input
    def __init__(self, samples, ax3d, backend=dummy, Nbins=60, sms=1, nsig=4):
        self.p0 = 0.0
        self.t0 = 0.0
        self.pr = 0.0
        self.tr = 0.0
        self.p = 0.362
        self.t = -0.316
        self.Nbins = Nbins
        self.sms = sms
        self.nsig = nsig
        self.backend = backend
        self.P = np.copy(P0).dot(roti(-self.t, 0).dot(roti(-self.p, 2)))
        self.rot_flag = 0
        self.ax3d = ax3d
        self.canvas3d = ax3d.figure.canvas
        self.samples = samples
        self.ax3d.set_aspect('auto')
        aux = get_L(samples)
        self.lx, self.ly, self.lz, self.max, self.min, self.off = aux
        self.backend(self)
        self.Psamples, self.P = update(self.p, self.t, self.off, self.ax3d,
                                       self.lx, self.ly, self.lz, self.samples,
                                       self.Nbins, self.sms, self.nsig)
        plt.sca(self.ax3d)

        # on movement update the plot according to the previous functions
        def on_move(event):

            if self.rot_flag == 1:
                x, y = event.x, event.y
                self.p = self.p0 - (x - self.pr) * 2e-3
                self.t = self.t0 + (y - self.tr) * 2e-3
                self.backend(self)
                self.Psamples, self.P = update(self.p, self.t, self.off,
                                               self.ax3d, self.lx,
                                               self.ly, self.lz, self.samples,
                                               self.Nbins, self.sms, self.nsig)

                self.canvas3d.draw_idle()

        # on click determine the 0 point around which one rotates
        def on_click(event):

            if event.inaxes == self.ax3d and event.button == 1:
                x, y = event.x, event.y
                self.p0 = self.p
                self.t0 = self.t
                self.pr = x
                self.tr = y
                self.rot_flag = 1

        # when the cursor is no longer held down, stop rotation
        def on_up(event):

            if self.rot_flag == 1 and event.button == 1:
                self.rot_flag = 0

        binding_id = plt.connect('motion_notify_event', on_move)
        plt.connect('button_press_event', on_click)
        plt.connect('button_release_event', on_up)
