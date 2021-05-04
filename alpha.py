"""
Programmer: Chris Tralie
Purpose: To animate an alpha filtration
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from scipy import spatial
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from persim import plot_diagrams
import gudhi


def gudhi2persim(pers):
    """
    Convert a persistence diagram from GUDHI's format into
    a format that persim can plot
    """
    Is = [[], []]
    for i in range(len(pers)):
        (dim, (b, d)) = pers[i]
        Is[dim].append([b, d])
    #Put onto diameter scale so it matches rips more closely
    return [np.sqrt(np.array(I)) for I in Is]


def draw_alpha(X, filtration, alpha, draw_balls=True, draw_voronoi_edges=True):
    """
    Draw the delaunay triangulation in dotted lines, with the alpha faces at
    a particular scale

    Parameters
    ----------
    X: ndarray(N, 2)
        A 2D point cloud
    filtration: list of [(idxs, d)]
        List of simplices in the filtration, listed by idxs, which indexes into
        X, and with an associated scale d at which the simplex enters the filtration
    alpha: float
        The radius/scale up to which to plot balls/simplices
    draw_balls: boolean
        Whether to draw the balls (discs intersected with voronoi regions)
    draw_voronoi_edges: boolean
        Whether to draw the voronoi edges showing the boundaries of the alpha balls
    """
    
    # Determine limits of plot
    pad = 0.3
    xlims = [np.min(X[:, 0]), np.max(X[:, 0])]
    xr = xlims[1]-xlims[0]
    ylims = [np.min(X[:, 1]), np.max(X[:, 1])]
    yr = ylims[1]-ylims[0]
    xlims[0] -= xr*pad
    xlims[1] += xr*pad
    ylims[0] -= yr*pad
    ylims[1] += yr*pad

    if draw_balls:
        resol = 2000
        xr = np.linspace(xlims[0], xlims[1], resol)
        yr = np.linspace(ylims[0], ylims[1], resol)
        xpix, ypix = np.meshgrid(xr, yr)
        P = np.ones((xpix.shape[0], xpix.shape[1], 4))
        PComponent = np.ones_like(xpix)
        PBound = np.zeros_like(PComponent)
        # First make balls
        tree = spatial.KDTree(X)
        XPix = np.array([xpix.flatten(), ypix.flatten()]).T
        neighbs = tree.query(XPix, 1)[1].flatten()
        neighbs = np.reshape(neighbs, xpix.shape)
        if draw_voronoi_edges:
            PBound = filters.sobel(neighbs) > 0
        else:
            PBound = np.zeros_like(neighbs)
        for i in range(X.shape[0]):
            # First make the ball part
            ballPart = (xpix-X[i, 0])**2 + (ypix-X[i, 1])**2 <= alpha**2
            # Now make the Voronoi part
            voronoiPart = np.reshape(neighbs == i, ballPart.shape)
            Pi = ballPart*voronoiPart
            PComponent[Pi == 1] = 0
        # Now make Voronoi regions
        P[:, :, 0] = PComponent
        P[:, :, 1] = PComponent
        P[:, :, 3] = 0.2 + 0.8*PBound
        plt.imshow(np.flipud(P), cmap='magma', extent=(xlims[0], xlims[1], ylims[0], ylims[1]))

    # Plot simplices
    patches = []
    for (idxs, d) in filtration:
        if len(idxs) == 2:
            if d < alpha:
                plt.plot(X[idxs, 0], X[idxs, 1], 'k', 2)
            else:
                plt.plot(X[idxs, 0], X[idxs, 1], 'gray', linestyle='--', linewidth=1)
        elif len(idxs) == 3 and d < alpha:
            patches.append(Polygon(X[idxs, :]))
    ax = plt.gca()
    p = PatchCollection(patches, alpha=0.2, facecolors='C1')
    ax.add_collection(p)
    plt.scatter(X[:, 0], X[:, 1], zorder=0)
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    #plt.axis('equal')


def alpha_animation(X, scales=np.array([])):
    """
    Create an animation of an alpha filtration of a 2D point cloud
    with the point cloud on the left and a plot on the right

    Parameters
    ----------
    X: ndarray(N, 2)
        A point cloud
    scales: ndarray(T)
        Scales to use in each frame of the animation.  If left blank,
        the program will choose 100 uniformly spaced scales between
        the minimum and maximum events in birth/death
    """
    alpha_complex = gudhi.AlphaComplex(points=X)
    simplex_tree = alpha_complex.create_simplex_tree()
    filtration = []
    for i, f in enumerate(simplex_tree.get_filtration()):
        filtration.append((f[0], np.sqrt(f[1])))
    diag = simplex_tree.persistence()
    dgmsalpha = gudhi2persim(diag)[0:2]
    if scales.size == 0:
        # Choose some default scales based on persistence
        smin = min(np.min(dgmsalpha[0]), np.min(dgmsalpha[1]))
        smax = max(np.max(dgmsalpha[0][np.isfinite(dgmsalpha[0])]), np.max(dgmsalpha[1]))
        rg = smax-smin
        smin = max(0, smin-0.05*rg)
        smax += 0.05*rg
        scales = np.linspace(smin, smax, 100)

    plt.figure(figsize=(12, 6))
    for frame, alpha in enumerate(scales):
        plt.clf()
        plt.subplot(121)
        draw_alpha(X, filtration, alpha, True)
        plt.title("$\\alpha = {:.3f}$".format(alpha))
        plt.subplot(122)
        plot_diagrams(dgmsalpha)
        plt.plot([-0.01, alpha], [alpha, alpha], 'gray', linestyle='--', linewidth=1, zorder=0)
        plt.plot([alpha, alpha], [alpha, 1.0], 'gray', linestyle='--', linewidth=1, zorder=0)
        plt.text(alpha+0.01, alpha-0.01, "{:.3f}".format(alpha))
        plt.title("Persistence Diagram")
        plt.savefig("{}.png".format(frame), bbox_inches='tight')

def load_pointcloud(path):
    """
    Load a point cloud from an image.  Every black pixel
    is considered to represent a point

    Parameters
    ----------
    path: string
        Path to point cloud
    
    Returns
    -------
    ndarray(N, 2)
        A point cloud with N points
    """
    from skimage.io import imread
    I = imread(path)
    I = np.mean(I, axis=2)
    X, Y = np.meshgrid(np.arange(I.shape[1]), np.arange(I.shape[1]))
    x = X[I == 0]
    y = I.shape[0]-Y[I == 0]
    return np.array([x, y]).T

def get_noisy_circle():
    """
    An example of a simple noisy circle with 20 points
    """
    np.random.seed(0)
    X = np.random.randn(20, 2)
    X /= np.sqrt(np.sum(X**2, 1))[:, None]
    X += 0.2*np.random.randn(X.shape[0], 2)
    return X

if __name__ == '__main__':
    X = load_pointcloud("snowman.png")
    # If scales left empty, will choose 100 automatically
    # based on persistence diagrams
    scales = np.array([]) 
    #scales = np.linspace(3, 18, 500) # Choose custom scales
    alpha_animation(X, scales)