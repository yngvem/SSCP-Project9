import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_vcg(vcg, figure=None, axes=None, wire=None, set_lims=True, color=None):
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
    wire : mpl_toolkits.mplot3d.art3d.Line3DCollection
        Wireframe to remove from the axes. Ignored if None.
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
    if axes is not None and figure is None:
        raise ValueError("Can't use axes object without its figure.")

    # Create figure and axes
    fig = plt.figure() if figure is None else figure
    ax = fig.add_subplot(111, projection='3d') if axes is None else axes
    if wire is not None:
        ax.collections.remove(wire)

    # Set limits
    if set_lims:
        ax.set_xlim(np.min(vcg[:, 0]),np.max(vcg[:, 0]))
        ax.set_ylim(np.min(vcg[:, 1]),np.max(vcg[:, 1]))
        ax.set_zlim(np.min(vcg[:, 2]),np.max(vcg[:, 2]))

    # Plot VCG
    if color is not None:
        wire = ax.plot_wireframe(vcg[:, 0], vcg[:, 1], vcg[:, 2], color=color)
    else:
        wire = ax.plot_wireframe(vcg[:, 0], vcg[:, 1], vcg[:, 2])

    return fig, ax, wire


def plot_vcg_axes(vcg, figure=None, axes=None, plots=None,
                  set_lims=True, color=None, titles=None,
                  transforms=None):
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
    plots : Array like
        A list of length 3 containing the three plots to update.
        If this is none new plots are created.
    axes : Array like
        List of the three different axes used for the plot.
    set_lims : Bool
        Wether or not to automatically set the limit of the plots
    color : Array like
        List of the different colors used for the plots.
    titles : Array like
        List of the titles for the three different plots. If None this is 
        set to x, y and z position of heart vector.
    transforms : Array like
        Array of length 3 containing the transformations that are supposed
        to be used for the three different plots.

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
    if axes is not None and figure is None:
        raise ValueError("Can't use axes object without its figure.")

    if transforms is not None:
        transforms = [
        trans if trans is not None else lambda x: x for trans in transforms
    ]
    else:
        transforms = [lambda x: x for _ in range(3)]

    # Set titles:
    titles = (
        'X-position\nof VCG', 
        'Y-position\nof VCG', 
        'Z-position\nof VCG'
        ) if titles is None else titles

    # Create figure and axes
    fig = plt.figure() if figure is None else figure
    ax = [fig.add_subplot(1, 3, i) for i in range(1, 4)] if axes is None else axes

    # Set limits
    if set_lims:
        for i, trans in enumerate(transforms):
            ax[i].set_ylim(np.min(trans(vcg[:, i])),np.max(trans(vcg[:, i])))

    # Set titles
    for i in range(3):
        ax[i].set_title(titles[i])

    # Plot VCG axes
    if plots is None:
        if color is not None:
            plots = [ax[i].plot(trans(vcg[:, i]), color=color[i])[0] for i, trans in enumerate(transforms)]
        else:
            plots = [ax[i].plot(trans(vcg[:, i]))[0] for i, trans in enumerate(transforms)]
    else:
        for i, (plot, trans) in enumerate(zip(plots, transforms)):
            plot.set_data(range(len(vcg[:, i])), trans(vcg[:, i]))
            plot.axes.set_title(titles[i])

    return fig, ax, plots
