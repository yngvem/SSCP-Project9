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

	
def parse_all(initial_path = "/Users/aleix/Desktop/LBBB_LVdyssync", vcg_which = True, with_geometry = False, vcg_select = '*', patient_number = None):
    
    """
        Get all data from the patients number in the list 'patient_number' (e.g. patient_number = [0,1,2,4,5])  in folder LBBB_LVdyssync stored into the 'patient' variable
        with_geometry takes the geometry data from geometric_chambers_data.txt
        Returns:
        --------
        patient:
        pandas.Series containing all data of the patient. See comments in __main__ for indexing.
        
        
        Warning: python3 is __next__, python2 is next
        Warning: in folder patient 1 you need to copy the sync file out of the vcg folder, otherwise it will be missing
        """
    
    vcg_type = 'dVCG' if vcg_which else 'kVCG'
    
    patients_folders = [os.path.join(initial_path, sub_path) for sub_path in os.walk(initial_path).next()[1]]
    patients = [None]*len(patients_folders)
    
    # for patient_folder in patients_folders[patient_number:]:
    dys = np.array([])
    params = np.array([]).reshape(0,7)
    vcgs = np.array([]).reshape(0,3)
    vcg_model = []
    
    if with_geometry: count = []
    
    for number in patient_number:
        
        patient_folder = patients_folders[number]
        # load the desynchrony times of each simulation
        desync = pd.read_csv(gb.glob(patient_folder + '/*sync.*')[0], header = None)
        dys = np.concatenate((dys, desync[0]))
        
        # load the parameters of each simulation
        eval_values = pd.read_csv(gb.glob(patient_folder + '/*eval*')[0],
                                  header = None, names=['x','y','z','cond','L','R'], sep='\t', index_col=False)
        pt_num = np.array([number]*len(desync[0].values))
        params = np.concatenate((params, np.column_stack((eval_values.values,pt_num))))
        
        # load the optimal desynchrony, keeping it so the other functions work
        opt_desync = pd.read_csv(gb.glob(patient_folder + '/*sync_opt*')[0], header = None)
        
        # load the experimental VCG, keeping it so the other functions work
        vcg_real = pd.read_csv(gb.glob(patient_folder + '/*measured*/*' + vcg_type + '*')[0],
                               header = None, names=['px','py','pz'], sep=' ', index_col=False)
        
                                  
        # load the VCG of each simulation
        vcg_reading = gb.glob(patient_folder + '/*model*/' + vcg_select + '.txt')
        llargada = len(vcg_model)
        vcg_model = vcg_model + [None]*len(vcg_reading)
        
        if with_geometry: count += [len(vcg_reading)]
        
        for i in range(len(vcg_reading)):
            vcg_model[i+llargada] = pd.read_csv(vcg_reading[i], header = None, names=['px','py','pz'], sep='\t', index_col=False)
                                  
                                  
        # store all into a panda series

    if with_geometry:
        geo_values = pd.read_csv(gb.glob(initial_path + '/*geo*')[0], sep=' ', header = None)
        #geo_matrix = np.repeat(geo_values.values[np.newaxis,:], len(desync[0].values), axis=0)
        geo_values = np.delete(geo_values.values, [4,5,6], axis=1) #. values. Deleting unknowns, 'AoR_Diam', 'LA_Dimen', 'LA_Vol_Index'
        geo_matrix = np.repeat(geo_values[patient_number, :], count, axis=0)
        params = np.column_stack((params, geo_matrix))
        print params[0]
        patient = pd.Series([opt_desync[0], dys, pd.DataFrame(params.astype(float), columns=['x','y','z','cond','L','R', 'patient_number', 'IVSd', 'LVIDd', 'LVIDs', 'LVPWd', 'EDV', 'ESV', 'EF']), vcg_real,
                             vcg_model], index=['opt_desync', 'desync', 'eval_values', 'vcg_real', 'vcg_model'])
        print patient['desync']
        return patient

    patient = pd.Series([opt_desync[0], dys, pd.DataFrame(params, columns=['x','y','z','cond','L','R', 'patient_number']), vcg_real,
                     vcg_model], index=['opt_desync', 'desync', 'eval_values', 'vcg_real', 'vcg_model'])

    return patient

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


	




