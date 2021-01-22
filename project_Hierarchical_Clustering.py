# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:26:08 2020

@author: andreea
"""

#Importing necessary libraries =================================================

import nilearn
from nilearn import datasets
from nilearn.input_data import NiftiMasker

from nilearn.image import index_img

import pandas as pd
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt

from termcolor import colored   #for formated output in prompt


''' Import the fMRI data from the Haxby(2001) study --------------------'''
data=nilearn.datasets.fetch_haxby(subjects=(1, 2, 3, 4, 5, 6), fetch_stimuli=False)



'''
 Define a function which extract all data that belongs to a single stimulus and 
 calculate its average '''
 
def stimulus_mean (stimulus, data_matrix, all_labels):
    label_of_stim=(all_labels==stimulus)
    stim=data_matrix[label_of_stim]
    return stim.mean(axis=0)

'''
 Process data into a form that can be easily rendered into a dendrogram.
 For this step only a subpart of the data should be selected, since plotting the 
 entire data makes the dendrogram impossible to interpret. For this project, 
 the data will be averaged according to the shown stimulus.
'''

def process_data(subject):
    n=subject-1
    file=data.func[n] #location of functional data of "subject"

    
    ''' Manage imported data ------------------------------------------------'''
    
    labels_subject = pd.read_csv(data.session_target[n], sep=" ") 
    L = labels_subject['labels']
    #L contains the names of the stimulus seen during each fMRI run.
    #Total: 1452 runs
    
    #   For easier use, the stimuli will be coded into numbers 
    #   from 0 to 9 in a vector names labels, as follows:
    labels=np.zeros(1452)
    for i in range (1452):
        if (L[i]=='face'):                 ####Legend: 
            labels[i]=0                     ### 0 = face
        elif (L[i]=='cat'):                 ### 1 = cat
            labels[i]=1                     ### 2 = bottle
        elif (L[i]=='bottle'):              ### 3 = shoe
            labels[i]=2                     ### 4 = house
        elif (L[i]=='shoe'):                ### 5 = scissors
            labels[i]=3                     ### 6 = chair
        elif (L[i]=='house'):               ### 7 = scrambled picture
            labels[i]=4                     ### 8 = rest (no visual stimulus)
        elif (L[i]=='scissors'): 
            labels[i]=5
        elif (L[i]=='chair'): 
            labels[i]=6
        elif (L[i]=='scrambledpix'): 
            labels[i]=7
        elif (L[i]=='rest'): 
            labels[i]=8
        else:
           print('error')
    
    '''
    Extract the relevant stimuli----------------------------------------------------------
    
    Selecting all labels that are not rest or scrambled pictures'''
    labels_test = ~labels_subject['labels'].isin(['rest', 'scrambledpix'])
    
    
    '''Selecting the fmri runs corresponding to the stimuli '''
    
    X_test = index_img(file, labels_test)
    
    '''
         Selecting the VT voxels only + preprocessing
         Note: the Haxby dataset comes with its own VT mask
    Preprocessing:
          Spatial smoothing (fwhm), filter = 3mm
          Time course normalization: baseline (z) standardization
    '''
    mask_filename = data.mask_vt[0]
    
    maskvt=NiftiMasker(mask_img=mask_filename, standardize=True, smoothing_fwhm=3)
    
    Xtest=maskvt.fit_transform(X_test) # processed (final) fMRI data (preprocessed, VT mask, only relevant stimuli)
    
    ytest=labels[labels_test]          # associated labels of fMRI data (belonging to only relevant stimuli)
    
    ###################################################################################################
    #     In order to plot some of the dendograms in the slides:   ###################
    ###################################################################################################
    
    ### Plotting a dendrogram of all the runs of a subject:
    #======================================================================    
    ##  Plotting out entire data in a dendrogram 
    ##  renders it impossible to descipher due to its large amoung
    
    #plot_dendrogram (Xtest, 'median', ytest)
    
    
    ### Plotting subparts of the data:
    #=====================================================================
    ## Plot the first 65 runs, distance method = median
    #plot_dendrogram (Xtest[0:65,:], 'median', ytest[0:65])
    
    ##Plot 100 runs, distance method = median
    #plot_dendrogram (Xtest[100:200,:], 'median', ytest[100:200])


    '''Makinf means of all runs in regard to stimuli
    ######################################################################'''
    
    means_matrix=np.zeros(shape=(7, Xtest.shape[1]), dtype=float)
    for i in range(7):
        means_matrix[i,:]=stimulus_mean(i, Xtest, ytest)
    return means_matrix
    



''' Hierarchical agglutination using the linkage function and drawing of the dendrogram'''

def plot_dendrogram (X, method, y):
    linkage_matrix = linkage(X, method)
    
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix,
                orientation='top',
                labels=y,
                distance_sort='descending',
                color_threshold = 11,
                show_leaf_counts=True)
    plt.show()
    a=squareform(cophenet(linkage_matrix))
    plt.matshow(a)
    plt.colorbar()
    plt.show()


def dendr(means_matrix, distance_method):
    plot_dendrogram(means_matrix, distance_method, ['face', 'cat', 'bottle', 'shoe', 'house', 'scissors', 'chair'])

'''
The reason why the two functions above were split rather than written together is 
so that the option of using the entire data to plot a dendrogram, albeit unreadable, 
remains open. '''


def prompt_user():
    print (colored('Select subject', 'green'))
    subject=int(input())
    print (colored('Select method:', 'green'))   
    dist_method=input()
    print (colored(' You have selected subject ', 'green'), subject, 
           colored('\n The method used to calculate the cluster distances will be', 'green'), 
           dist_method)
    
    matrix=process_data(subject)
    dendr(matrix,dist_method)
    print (colored('The dendrogram has been plotted. If you would like to plot another dendrogram, press y','green'))
    cont=input()
    if ((cont=='y')or(cont=='Y')):
        prompt_user()

print (colored('\n\nThis program will generate a dendrogram accompanied by a visualisation of the cophenet matrix.\n','green'))
print (colored('You will be asked to choose the subject number', 'green'), colored('(from 1 to 6)','red'),colored('and a method to compute the distance between clusters.\n','green'))
print (colored('The different methods to compute the distance are:', 'green'),colored('single, average, complete, ward, centroid, median, and weighted.\n','red'))
print (colored('More information about the formulae used to calculate the distances used by these different methods visit the documentation available on docs.scipy.org\n\n','green'))
       

prompt_user()
 
