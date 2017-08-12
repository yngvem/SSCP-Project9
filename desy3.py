import os
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from data_parser import *
from data_modification import *
from visualization.plot_vcg import *

from scipy.stats import binned_statistic
import numpy_indexed as npi

#0, 1, 2, 4, 5
if __name__ == "__main__":
    
    patient = parse_all(with_geometry=True, patient_number = [0,1,2,4,5])
    patient = center_patient(patient)
    patient = normalise_patient(patient)
    patient = resample_add_velocity_patient(patient)
    patient = project_patient(patient)
    patient = add_ellipse_patient(patient)
    patient_cyl = cylindrical_patient(patient)
    patient_cyl = add_features_cyl_patient(patient_cyl)
    patient_cyl = add_features_cartcyl_patient(patient, patient_cyl)
    
    integral_radis = patient_cyl['cyl_features']['integral_radi_vel']
    #integral_radis = patient_cyl['cartcyl_features']['winding_num']
    #integral_radis = patient_cyl['cyl_features']['turning_num']

    
    #integral_radis = patient['ellipse']['a']
    dyssyncs = patient_cyl['desync'][np.isfinite(integral_radis)]
    #Rs = patient['eval_values']['R'].values[np.isfinite(integral_radis)]
    integral_radis = integral_radis[np.isfinite(integral_radis)]
    #plt.scatter(integral_radis, Ls, color ='r')
    #plt.show()
    
    num=25
    
    integral_radis=integral_radis[dyssyncs!=0]
    dyssyncs=dyssyncs[dyssyncs!=0]
    print np.cov([integral_radis, dyssyncs])
    plt.scatter(integral_radis, dyssyncs, color ='r')
    plt.show()
    hh = [ dyssyncs[np.logical_and(i*max(integral_radis)/num < integral_radis, integral_radis < ((1+i)*max(integral_radis)/num))].mean() for i in range(num)]
    hhh = np.linspace(0,max(integral_radis),num)
    plt.scatter(hhh, hh, color ='r')
    plt.ylim(0.7, 0.9)
    plt.show()

