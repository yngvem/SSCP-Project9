import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_vcg(vcg, figure=None, axes=None, set_lims=True, color='b'):
    """Creates a 3D curve plot of a VCG signal.

    Parameters:
    -----------
    vcg : Array like
        An array of shape (Nx3) where the first index is
        the time steps and the second index is the
        denotes the position in the x, y and z direction
        respecitvely.
    figure : matplotlib.figure.Figure
        Figure object to plot in. New figure is created
        if this is None.
    axes : matplotlib.axes._subplots.Axes3DSubplot
        Axes object to plot the VCG in. New axes object
        is created if this is None. 
    set_lims : Bool
        Whether or not to set new limits in the axes object
    color : str
        The color of the wireframe.
    
    Returns:
    --------
    figure : matplotlib.Figure
        Figure object that was drawn in.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        Axes the figure was drawn in.
    wire : mpl_toolkits.mplot3d.art3d.Line3DCollection
        The wireframe (plot) of the vcg.

    Raises:
    -------
    ValueError
        If axes object is passed but no figure
    """
    if ax is not None and figure is None:
        raise ValueError("Can't use axes object without its figure.")

    # Create figure and axes
    fig = plt.figure() if figure is None else figure
    ax = fig.add_subplot(111, projection='3d') if axes is None else axes

    # Set limits
    if set_lims:
        ax.set_xlim(np.min(vcg[:, 0]),np.max(vcg[:, 0]))
        ax.set_ylim(np.min(vcg[:, 1]),np.max(vcg[:, 1]))
        ax.set_zlim(np.min(vcg[:, 2]),np.max(vcg[:, 2]))

    # Plot VCG
    wire = ax.plot_wireframe(vcg[:, 0], vcg[:, 1], vcg[:, 2], color=color)

    return fig, ax, wire


def plot_vcg_axes(vcg, figure=None, axes=None, set_lims=True, 
                  color=('b', 'b', 'b')):
    """Plots the x, y and z position of the heart vector in different subplots.

    Parameters:
    -----------
    vcg : Array like
        An array of shape (Nx3) where the first index is
        the time steps and the second index is the
        denotes the position in the x, y and z direction
        respecitvely.
    figure : matplotlib.figure.Figure
        Figure object to plot in. New figure is created
        if this is None.
    axes : Array like
        List of the three different axes used for the plot.
    set_lims : Bool
        Wether or not to automatically set the limit of the plots
    color : Array like
        List of the different colors used for the plots.

    Returns:
    --------
    figure : matplotlib.Figure
        Figure object that was drawn in.
    ax : List
        List with the subplots the figure was drawn in.
    plots : List
        List with the three line collections used for the plots.

    Raises:
    -------
    ValueError
        If axes object is passed but no figure
    """
    if ax is not None and figure is None:
        raise ValueError("Can't use axes object without its figure.")

    # Create figure and axes
    fig = plt.figure() if figure is not None else figure
    ax = [fig.add_subplot(1, 3, i) for i in range(3)] if axes is None else axes

    # Set limits
    if set_lims:
        ax[0].set_xlim(np.min(vcg[:, 0]),np.max(vcg[:, 0]))
        ax[1].set_ylim(np.min(vcg[:, 1]),np.max(vcg[:, 1]))
        ax[2].set_zlim(np.min(vcg[:, 2]),np.max(vcg[:, 2]))

    # Plot VCG axes
    plots = [ax[i].plot(vcg[:, i], color=color[i]) for i in range(n)]

    return fig, ax, plots
