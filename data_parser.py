import os
import glob as gb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

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
    Get all data from the folder LBBB_LVdyssync.
	Returns:
	--------
	patient: description not yet done
		Numpy array of samze size as the vcg signal, with first row being
		the radius, the second row being the first spherical angle and the 
		third row being the second spherical angle.
    """

    vcg_type = 'dVCG' if vcg_which else 'kVCG'

    patients_folders = [os.path.join(initial_path, sub_path) for sub_path in os.walk(initial_path).next()[1]] 
    patients = [None]*len(patients_folders)

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

    patient = parse(initial_path = "/Users/aleix/Downloads/LBBB_LVdyssync", vcg_which = False, patient_number=7)

    #mean_plot(patient)
            
    # example calls
    #print patient['opt_desync'][0]
    #print patient['desync']
    #print patient['eval_values']
    #print patient['vcg_real'] # ['px'],['py'],['pz']
    #print patient['vcg_model'] # [i] # ['px'],['py'],['pz']

    #http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html#the-approach
    #https://stackoverflow.com/questions/29051168/data-fitting-an-ellipse-in-3d-space


	




