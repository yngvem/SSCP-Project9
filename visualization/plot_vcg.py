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
	fig = plt.figure() if figure is not None else figure
	ax = fig.add_subplot(111, projection='3d') if axes is not None else axes

	if set_lims:
		ax.set_xlim(np.min(vcg[:, 0]),np.max(vcg[:, 0]))
		ax.set_ylim(np.min(vcg[:, 1]),np.max(vcg[:, 1]))
		ax.set_zlim(np.min(vcg[:, 2]),np.max(vcg[:, 2]))

	wire = ax.plot_wireframe(vcg[:, 0], vcg[:, 1], vcg[:, 2], color=color)