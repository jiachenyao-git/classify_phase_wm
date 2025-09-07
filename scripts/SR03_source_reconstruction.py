'''
Compute inverse models

'''
globals().clear()
import numpy as np
import mne
import os
import os.path as op
import matplotlib.pyplot as plt

# silence mne
mne.set_log_level('warning')

# define subject and session
subject ='01'
session = '01'  ### NOTE change sessions accordingly -- one unique forward model for each session

# define data directory
data_dir = '/Users/jiachenyao/Desktop/PhaseCode/data'

# path to raw EEG epochs
epo_clas_fname = f"{data_dir}/raw_epo/sub-{subject}_clas_ses-{session}_epo.fif"
epo_phas_fname = f"{data_dir}/raw_epo/sub-{subject}_phas_ses-{session}_epo.fif"

'''
Read epochs
'''
# load raw EEG epochs
epochs_clas = mne.read_epochs(epo_clas_fname)
epochs_clas.info
epochs_clas.pick_types(meg=False, eeg=True)

epochs_phas = mne.read_epochs(epo_phas_fname)
epochs_phas.info
epochs_phas.pick_types(meg=False, eeg=True)

'''
Prepare beamforming of data

For beamforming, we need a data covariance matrix.

Since we have two independent time windows that does not form any contrasts, we will compute two separate data covariance matrices, one for each of them
'''
# compute two data covariance matrices
data_cov_clas = mne.compute_covariance(epochs_clas, 
                                       tmin=0., tmax=0.8,method='empirical', rank='info')
data_cov_phas = mne.compute_covariance(epochs_phas, 
                                       tmin=-1.25, tmax=0.,method='empirical', rank='info')

# visualize data covariance matrices
#mne.viz.plot_cov(data_cov_clas, info=epochs_clas.info)
#input("Check data covariance matrix for classification, then close visualization by pressing ENTER ...")

#mne.viz.plot_cov(data_cov_phas, info=epochs_phas.info)
#input("Check data covariance matrix for phase estimation, then close visualization by pressing ENTER ...")

'''
from mne import compute_rank
ranks = compute_rank(data_cov_clas, rank='full', info = epochs_clas.info)
'''

'''
We also need to compute a noise covariance matrix. That will be used for pre-whitening the data, data covariance matrix, and forward model. 
'''
# compute two noise covariance matrixs to pre-whiten the data, data covariance matrix, and forward model.
noise_cov_clas = mne.compute_covariance(epochs_clas, tmin=-.2, tmax=0.,  # use the baseline
                                        method='empirical', 
                                        rank='info')

noise_cov_phas = mne.compute_covariance(epochs_phas, tmin=-1.45, tmax=-1.25,  # use the baseline
                                        method='empirical', 
                                        rank='info')

'''
Lastly, we also need to read the forward model that we had saved!
'''
# read the forward models we saved
fwd_fname = f"{data_dir}/source_recon/sub-{subject}/sub-{subject}_ses-{session}_eeg_fwd.fif"
fwd = mne.read_forward_solution(fwd_fname)
fwd = mne.pick_types_forward(fwd, meg=False, eeg=True)
print('forward models loaded!')

'''
Compute beamformer and apply to epoch data
'''
from mne.beamformer import make_lcmv, apply_lcmv_epochs

# compute the beamformer
# unit-noise gain beamformers handle depth bias by normalizing the weights of the spatial filter.
def compute_beamformer(epochs, data_cov, noise_cov):

    filters = make_lcmv(
        epochs.info, 
        fwd,
        data_cov=data_cov, 
        reg=0.05, 
        noise_cov=noise_cov,
        pick_ori='max-power', 
        rank=None,
        weight_norm='unit-noise-gain',
    )
    
    return filters

filters_clas = compute_beamformer(epochs_clas, data_cov_clas, noise_cov_clas)
filters_phas = compute_beamformer(epochs_phas, data_cov_phas, noise_cov_phas)
print('beamformers computed!')

# apply the spatial filters
# data shape: vertices * times
# generator type data can be accessed by next()
stc_clas = apply_lcmv_epochs(epochs=epochs_clas, filters=filters_clas, return_generator=True, )
stc_phas = apply_lcmv_epochs(epochs=epochs_phas, filters=filters_phas, return_generator=True, )
print('beamformers applied!')

'''
#plot the brain and time course
mri_dir = op.join(data_dir, 'mri_reconst')
next(stc_clas).plot(subjects_dir=mri_dir, subject=subject, clim=dict(kind="value", lims=[3, 6, 9]), hemi='both', smoothing_steps=7,)
input("Check brain and time course for classification, then close visualization by pressing ENTER ...")

next(stc_phas).plot(subjects_dir=mri_dir, subject=subject, clim=dict(kind="value", lims=[3, 6, 9]), hemi='both', smoothing_steps=7,)
input("Check brain and time course for phase estimation, then close visualization by pressing ENTER ...")
'''

