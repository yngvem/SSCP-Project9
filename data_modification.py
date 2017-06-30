import numpy as np


def cart_to_spherical(vcg):
	"""Transform VCG signal from a cartesian to a spherical coordinate system.

	Returns:
	--------
	vcg_spher : np.ndarray
		Numpy array of samze size as the vcg signal, with first row being
		the radius, the second row being the first spherical angle and the 
		third row being the second spherical angle.
	"""
	assert(vcg.shape[1] == 3)

	vcg_spher = np.zeros(np.shape(vcg))
	vcg_spher[:, 0] = np.sqrt(np.sum(vcg**2, axis=1))
	vcg_spher[:, 1] = np.arctan2(vcg[:, 0], vcg[:, 1])
	vcg_spher[:, 2] = np.arccos(vcg[:, 2]/vcg_spher[:, 0])
	return vcg_spher

def cart_to_cylindrical(vcg):
	"""Transform the VCG signal from cartesian to cylindrical coordinate system.

	Returns:
	--------
	vcg_spher : np.ndarray
		Numpy array of samze size as the vcg signal, with first row being
		the radius, the second row being the cylindrical angle and the third
		row being the z value.
	"""
	assert(vcg.shape[1] == 3)

	vcg_cyl = np.zeros(np.shape(vcg))
	vcg_cyl[:, 0] = np.sqrt(vcg[:, 0]**2+vcg[:, 1]**2)
	vcg_cyl[:, 1] = np.arctan2(vcg[:, 1], vcg[:, 0])
	vcg_cyl[:, 2] = np.copy(vcg[:, 2])
	return vcg_cyl


def center(vcg):
	"""Center the curve so that it's average is 0. 

	Warning: This function changes its arguments, nothing is retuned. 
	"""
	vcg -= np.mean(vcg, axis=0)

