import os
import glob as gb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

from visualization.plot_vcg import plot_3d_vcg, plot_vcg_axes
from data_modification import cart_to_cylindrical, center

from sklearn.decomposition import PCA

from numpy.linalg import eig, inv

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

def project_PCA(patient, plotting=False):
    #http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_3d.html
    vcg = patient['vcg_real'].as_matrix()
    center(vcg)
    pca = PCA(n_components=3)
    pca.fit(vcg)
    vcg_projected = pca.transform(vcg)
    
    if plotting:
        plt.scatter(vcg_projected[:,0], vcg_projected[:,1])
        plt.show()
    
    return vcg_projected

def ellipse_fit_projection(patient, plotting=False):
    vcg_projected = project_PCA(patient)
    a = fitEllipse(vcg_projected[:,0], vcg_projected[:,1])
    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation2(a)
    axes = ellipse_axis_length(a)
    if plotting:
        print("center = ",  center)
        print("angle of rotation = ",  phi)
        print("axes = ", axes)
        aa, b = axes
        arc = 2*np.pi
        R = np.arange(0,arc*np.pi, 0.01)
        xx = center[0] + aa*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
        yy = center[1] + aa*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
        plt.scatter(vcg_projected[:,0], vcg_projected[:,1])
        plt.plot(xx, yy, color='r')
        plt.show()
    return a

def mean_plot(patient):

    vcg_real = patient['vcg_real']   
    vcg_model = patient['vcg_model']     

    fig = plt.figure()
    ax = plt.axes(projection='3d')    

    real_mean = [vcg_real['px'].mean()], [vcg_real['py'].mean()], [vcg_real['pz'].mean()]
    model_mean = np.array([[s['px'].mean(), s['py'].mean(), s['pz'].mean()] for s in vcg_model])
    #vcg_model[1].mean().as_matrix()

    ax.scatter([vcg_real['px'].mean()], [vcg_real['py'].mean()], [vcg_real['pz'].mean()], color ='r')
    ax.scatter(np.array(model_mean)[:,0], np.array(model_mean)[:,1], np.array(model_mean)[:,2], color='b')
    plt.show()


def parse(initial_path = "/Users/aleix/Downloads/LBBB_LVdyssync", vcg_which = True, vcg_select = '*', patient_number=0):

    """
    Get all data from patient number 'patient_number' in folder LBBB_LVdyssync stored into the 'patient'variable
	Returns:
	--------
	patient: 
        pandas.Series containing all data of the patient. See comments in __main__ for indexing.
        
        
    Warning: python3 is __next__, python2 is next
    Warning: in folder patient 1 you need to copy the sync file out of the vcg folder, otherwise it will be missing
    """

    vcg_type = 'dVCG' if vcg_which else 'kVCG'

    patients_folders = [os.path.join(initial_path, sub_path) for sub_path in os.walk(initial_path).__next__()[1]]
    patients = [None]*len(patients_folders)

    # useless loop, leaving it here in case
    # for patient_folder in patients_folders[patient_number:]:
    for patient_folder in [patients_folders[patient_number]]:

        # load the optimal desynchrony
        opt_desync = pd.read_csv(gb.glob(patient_folder + '/*sync_opt*')[0], header = None)

        # load the desynchrony times of each simulation
        desync = pd.read_csv(gb.glob(patient_folder + '/*sync.*')[0], header = None)

        # load the parameters of each simulation
        eval_values = pd.read_csv(gb.glob(patient_folder + '/*eval*')[0],
                                  header = None, names=['x','y','z','cond','L','R'], sep='\t', index_col=False)

        # load the experimental VCG
        vcg_real = pd.read_csv(gb.glob(patient_folder + '/*measured*/*' + vcg_type + '*')[0],
                               header = None, names=['px','py','pz'], sep=' ', index_col=False)    

        # load the VCG of each simulation
        vcg_reading = gb.glob(patient_folder + '/*model*/' + vcg_select + '.txt')
        vcg_model = [None]*len(vcg_reading)
        for i in range(len(vcg_reading)):
            vcg_model[i] = pd.read_csv(vcg_reading[i], header = None, names=['px','py','pz'], sep='\t', index_col=False)
    

        # store all into a panda series
        patient = pd.Series([opt_desync[0], desync[0], eval_values, vcg_real,
                             vcg_model], index=['opt_desync', 'desync', 'eval_values', 'vcg_real', 'vcg_model'])

    return patient

if __name__ == "__main__":

    patient = parse(initial_path = "/Users/aleix/Desktop/SURPH/SSCP-Project9-master/LBBB_LVdyssync", vcg_which = False, patient_number=7)

    # plot comparison betnween the x,y,z means of the simulated vcgs compared to the real vcg
    #mean_plot(patient)
    
    # vcg plots
    #fig, ax, wire = plot_3d_vcg(patient['vcg_real'].as_matrix())
    #fig, ax, plots = plot_vcg_axes(patient['vcg_real'].as_matrix())
    #plt.show(wire)
    
    
    # vcg polar plots
    #vcg = patient['vcg_real'].as_matrix()
    #center(vcg)
    #vcg_polar = cart_to_cylindrical(vcg)
    #fig, ax, plots = plot_vcg_axes(vcg_polar)
    #fig, ax, wire = plot_3d_vcg(vcg)
    #plt.show(plots)
    #plt.show(wire)

    # PCA projection
    #project_PCA(patient, plotting=True)

    # Ellipse fit of the PCA projection
    ellipse_fit_projection(patient, plotting=True)
    # not really ellipses? maybe because of the irregular sampling?
    # you just need to resample and use the same function

    # example calls
    #print patient['opt_desync'][0]
    #print patient['desync']
    #print patient['eval_values']
    #print(patient['vcg_real'].as_matrix()) # ['px'],['py'],['pz']
    #print patient['vcg_model'] # [i] # ['px'],['py'],['pz']

	




