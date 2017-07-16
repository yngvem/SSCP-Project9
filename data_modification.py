import numpy as np
import pandas as pd
from copy import deepcopy


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


def normalise_patient(patient):
    """Normalises the VCG by dividing by the maximum heart vector.
    """
    patient = deepcopy(patient)

    # Standardise VCG signal:
    vcg_real_sq = patient['vcg_real']**2
    vcg_real_len = np.sqrt(
        vcg_real_sq['px']**2 + vcg_real_sq['py']**2 + vcg_real_sq['pz']**2
    ).max()
    patient['vcg_real'] /= vcg_real_len

    for i, vcg_model in enumerate(patient['vcg_model']):

        # Standardise VCG signal:
        vcg_model_len = np.sqrt(
            vcg_model['px']**2 + vcg_model['py']**2 + vcg_model['pz']**2
        ).max()
        patient['vcg_model'][i] /= vcg_model_len

    return patient


def cylindrical_patient(patient):
    """Converts the patients VCG signal to cylindrical coordinates.
    """
    patient = deepcopy(patient)

    cyl_df = cart_to_cylindrical(patient['vcg_real'].values)
    patient['vcg_real'] = pd.DataFrame(cyl_df, columns=['pr', 'pphi', 'pz'])

    for i, vcg_model in enumerate(patient['vcg_model']):
        cyl_df = cart_to_cylindrical(vcg_model.values)
        patient['vcg_model'][i] = pd.DataFrame(cyl_df, columns=['pr', 'pphi', 'pz'])

    return patient


def center_patient(patient):
    """Centers the heart vector for each patient such that its average is 0.

    This requires the heart vector to be in cartesian coordinates!
    """

    patient = deepcopy(patient)

    patient['vcg_real'] -= patient['vcg_real'].mean()

    for i, vcg_model in enumerate(patient['vcg_model']):
        patient['vcg_model'][i] -= vcg_model.mean()

    return patient


def create_data_matrix(patient, transforms=None):
    """Returns a datamatrix for the patients where each column is a simulation.

    Arguments
    ---------
    patient : pandas.DataFrame
        The dataframe containing all information about the patient
    transforms : Array like
        List containing three transformations to be applied to the different
        axis of the VCG signal.
    
    Returns
    -------
    data_matrix : np.ndarray
        A matrix where each column is a VCG signal for each simulation (the
        three dimensions are just concatenated to one vector).
    """
    data_matrix = np.zeros((len(patient['vcg_model']), np.prod(patient['vcg_model'][0].shape)))
    for i, simulation in enumerate(patient['vcg_model']):
        sim = simulation.values.copy()

        if transforms is not None:
            for j in range(3):
                if transforms[j] is not None:
                    sim[j, :] = transforms[j](sim[j, :])

        data_matrix[i] = sim.reshape(np.prod(sim.shape))

    return data_matrix