import numpy as np
import pandas as pd
from copy import deepcopy

from numpy.linalg import eig, inv
from numpy import mat
from sklearn.decomposition import PCA
from scipy import interpolate, ndimage


def get_velocity(vcg, mode=3):
#https://stackoverflow.com/questions/18991408/python-finite-difference-functions
    if mode==0:
        dx_dt = ndimage.gaussian_filter1d(vcg[:, 0], sigma=1, order=1, mode='wrap')
        dy_dt = ndimage.gaussian_filter1d(vcg[:, 1], sigma=1, order=1, mode='wrap')
        dz_dt = ndimage.gaussian_filter1d(vcg[:, 2], sigma=1, order=1, mode='wrap')
        return np.column_stack((dx_dt, dy_dt, dz_dt))
        
    elif mode==1:
        dx_dt = np.gradient(vcg[:, 0])
        dy_dt = np.gradient(vcg[:, 1])
        dz_dt = np.gradient(vcg[:, 2])
        return np.column_stack((dx_dt, dy_dt, dz_dt))
    
    else:
        #np.diff(vcg, axis=0)
        return np.append([[0,0,0]],np.diff(vcg, axis=0), axis=0)

    
def get_curvature(vcg, mode=3, given_vel=False):
    
    if not given_vel:
        vel = get_velocity(vcg, mode)
    else:
        vel = vcg
        
    acc = get_velocity(vel, mode)
    norm_accXvel = np.linalg.norm(np.cross(acc, vel), axis=1)
    v3 = np.power(np.linalg.norm(vel, axis=1),3)
    
    return norm_accXvel/v3


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


def resample_by_velocity(patient_vcg, mode=3, length=None, velosi=False, curvsi=False, plotting=False):
    """Resample the VCG signal uniformly in space (instead of in time).

    Returns:
    --------
    x(s), y(s), z(s) : np.ndarray, np.ndarray, np.ndarray
        Numpy arrays of same sizes as the vcg signal. x, y, z coordinates
        for the resampled points.
    """    
    vcg = patient_vcg.as_matrix()
    velocitat = get_velocity(vcg, mode)
    norma = np.linalg.norm(velocitat, axis=1)
    param = np.cumsum(norma)
    
    if length==None: length = len(param)
    x = interpolate.interp1d(param, vcg[:,0])
    y = interpolate.interp1d(param, vcg[:,1])
    z = interpolate.interp1d(param, vcg[:,2])
    s = np.linspace(0, param.max(), length)

    if plotting:
        ax = plt.axes(projection='3d')
        ax.plot(vcg[:,0], vcg[:,1], vcg[:,2], color ='r')
        ax.scatter(x(s), y(s), z(s), color ='b')
        plt.show()

    
    if velosi:
        vx = interpolate.interp1d(param, velocitat[:,0])
        vy = interpolate.interp1d(param, velocitat[:,1])
        vz = interpolate.interp1d(param, velocitat[:,2])
        norm = interpolate.interp1d(param, norma)

        if curvsi:
            curvature = get_curvature(velocitat, mode, given_vel=True)
            k = interpolate.interp1d(param, curvature)
            return x(s), y(s), z(s), vx(s), vy(s), vz(s), k(s)
    
        return x(s), y(s), z(s), vx(s), vy(s), vz(s)
    
    return x(s), y(s), z(s)


def project_PCA(patient_vcg, by_coords=False, plotting=False):
    """Transform the VCG signal  with PCA. The new x, y, z axis are ordered
    by decreasing variance.

    Returns:
    --------
    by_coords = False
     vcg_projected : 
        Numpy arrays of same sizes as the vcg signal. With transformed coordinates.
    
    by_coords = True
    x(s), y(s), z(s) : np.ndarray, np.ndarray, np.ndarray
        Numpy arrays of same sizes as the vcg signal. x, y, z coordinates
        for the transformed points. In this way it is easier to project on the 2D plane
        by grabbing the first two components.
    """    
    #http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_3d.html
    
    vcg = patient_vcg.as_matrix()
    center(vcg)
    pca = PCA(n_components=3)
    pca.fit(vcg)
    vcg_projected = pca.transform(vcg)
    
    if plotting:
        plt.scatter(vcg_projected[:,0], vcg_projected[:,1])
        plt.show()

    if by_coords: return vcg_projected[:,0], vcg_projected[:,1], vcg_projected[:,2]
    return vcg_projected


def center(vcg):
    vcg -= vcg.mean(0, keepdims=True) 


def normalise_patient(patient):
    """Normalises the VCG by dividing by the maximum heart vector.
    """
    patient = deepcopy(patient)
    patient['vcg_model'] = deepcopy(patient['vcg_model'])

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
    patient['vcg_model'] = deepcopy(patient['vcg_model'])

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
    patient['vcg_model'] = deepcopy(patient['vcg_model'])

    patient['vcg_real'] -= patient['vcg_real'].mean()

    for i, vcg_model in enumerate(patient['vcg_model']):
        patient['vcg_model'][i] -= vcg_model.mean()

    return patient


def resample_patient(patient, length=None):
    """Resamples the heart vector for each patient by velocity so that it is uniformly sampled in space (instead of in time).

    This requires the heart vector to be in cartesian coordinates!
    """
    
    patient = deepcopy(patient)
    patient['vcg_model'] = deepcopy(patient['vcg_model'])

    for i, vcg in enumerate(patient['vcg_model']):
        resampled_vcg = np.array(resample_by_velocity(vcg, length=length)).T
        patient['vcg_model'][i] = pd.DataFrame(resampled_vcg, columns=vcg.columns)
        #patient['vcg_model'][i]['px'], patient['vcg_model'][i]['py'], patient['vcg_model'][i]['pz'] = resample_by_velocity(patient['vcg_model'][i], length=length)

    resampled_vcg = np.array(resample_by_velocity(vcg, length=length)).T
    patient['vcg_real'] = pd.DataFrame(resampled_vcg, columns = patient['vcg_real'].columns)
    #patient['vcg_real']['px'], patient['vcg_real']['py'], patient['vcg_real']['pz'] = resample_by_velocity(patient['vcg_real'], length=length)

    return patient


def project_patient(patient):
    """Uses PCA on the heart vector and transforms it so that the z axis contains the least variance.

    This requires the heart vector to be in cartesian coordinates!
    """
    
    patient = deepcopy(patient)
    patient['vcg_model'] = deepcopy(patient['vcg_model'])
    
    for i in range(len(patient['vcg_model'])):
        patient['vcg_model'][i]['px'], patient['vcg_model'][i]['py'], patient['vcg_model'][i]['pz'] = project_PCA(patient['vcg_model'][i], by_coords=True)   

    patient['vcg_real']['px'], patient['vcg_real']['py'], patient['vcg_real']['pz'] = project_PCA(patient['vcg_real'], by_coords=True)

    return patient


def add_velocity_patient(patient, curvsi=False, mode=3):
    """Adds the velocity (and the curvature) to the dataframe
    """
    
    patient = deepcopy(patient)
    patient['vcg_model'] = deepcopy(patient['vcg_model'])

    long = len(patient['vcg_model'])
    vcg_vel = [None]*long
    if curvsi: vcg_curv = [None]*long
    
    for i in range(long):
            velocitat = get_velocity(patient['vcg_model'][i].as_matrix(), mode=mode)
            vcg_vel[i] = pd.DataFrame(velocitat,  columns=['vx','vy','vz'])
            if curvsi: vcg_curv[i] = pd.DataFrame(get_curvature(velocitat, mode=mode, given_vel=True),  columns=['k'])

    patient['velocity'] = vcg_vel
    if curvsi: patient['curvature'] = vcg_curv

    return patient


def resample_add_velocity_patient(patient, mode=3, curvsi=False):
    """Resamples the heart vector for each patient by velocity so that it is uniformly sampled in space (instead of in time).
    Also adds velocity (and curvature) to the dataframe
    
    This requires the heart vector to be in cartesian coordinates!
    """
    
    patient = deepcopy(patient)
    patient['vcg_model'] = deepcopy(patient['vcg_model'])
    long = len(patient['vcg_model'])
    vcg_vel = [None]*long
    
    if not curvsi: 
        for i in range(long):
            patient['vcg_model'][i]['px'], patient['vcg_model'][i]['py'], patient['vcg_model'][i]['pz'], vx, vy, vz = resample_by_velocity(patient['vcg_model'][i], mode=mode, velosi=True)
            vcg_vel[i] = pd.DataFrame(np.column_stack((vx, vy, vz)),  columns=['vx','vy','vz'])
        patient['velocity'] = vcg_vel

    else:
        vcg_curv = [None]*long
        for i in range(long):
            patient['vcg_model'][i]['px'], patient['vcg_model'][i]['py'], patient['vcg_model'][i]['pz'], vx, vy, vz, k = resample_by_velocity(patient['vcg_model'][i], mode=mode, velosi=True, curvsi=True)
            vcg_vel[i] = pd.DataFrame(np.column_stack((vx, vy, vz)),  columns=['vx','vy','vz'])
            vcg_curv[i] = pd.DataFrame(k,  columns=['k'])
        patient['velocity'] = vcg_vel
        if curvsi: patient['curvature'] = vcg_curv

    patient['vcg_real']['px'], patient['vcg_real']['py'], patient['vcg_real']['pz'] = resample_by_velocity(patient['vcg_real'])

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

#ellipse stuff

def add_ellipse_patient(patient):
    """Adds ellipse measures to the dataframe
    """
    
    patient = deepcopy(patient)
    patient['vcg_model'] = deepcopy(patient['vcg_model'])

    long = len(patient['vcg_model'])
    vcg_ellipse = [None]*long
    
    for i in range(long):
            fitted_ellipse = fitEllipse(patient['vcg_model'][i].values[:,0], patient['vcg_model'][i].values[:,1])
            axis = ellipse_axis_length(fitted_ellipse)
            vcg_ellipse[i] = pd.DataFrame(np.column_stack(axis),  columns=['a','b'])

    patient['ellipse'] = vcg_ellipse

    return patient


def fitEllipse(x,y):
    #http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html#the-approach
    #https://stackoverflow.com/questions/29051168/data-fitting-an-ellipse-in-3d-space
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def ellipse_angle_of_rotation2( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

