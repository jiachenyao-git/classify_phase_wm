'''
Compute inverse models

'''
globals().clear()
import numpy as np
import mne
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import os
import os.path as op
import matplotlib
#matplotlib.use('Agg')   # use non-interactive backend
import matplotlib.pyplot as plt

# silence mne
mne.set_log_level('warning')

# define subject and session
subject ='02'
session_01 = '01'
session_02 = '02'

# define data directory
data_dir = '/Users/jiachenyao/Desktop/PhaseCode/data'

def load_epochs(data_dir, subject, session):
    '''
    Load and preprocess classification and phase epochs,
    keeping only overlapping impulse trials.
    
    Parameters:
    data_dir : str
        Base directory where the epochs files are stored.
    subject : str or int
        Subject identifier (without 'sub-' prefix).
    session : str or int
        Session identifier (without 'ses-' prefix).
    
    Returns:
    epochs_adj : mne.Epochs
        Adjective epochs for decoding.
    epochs_n : mne.Epochs
        Noun epochs for decoding.
    epochs_imp_nb : mne.Epochs
        Impulse epochs (non-binding).
    epochs_imp_b : mne.Epochs
        Impulse epochs (binding).
    epochs_phas_nb : mne.Epochs
        Phase estimation epochs (non-binding).
    epochs_phas_b : mne.Epochs
        Phase estimation epochs (binding).
    '''

    # path to raw EEG epochs
    epo_clas_fname = f"{data_dir}/raw_epo/sub-{subject}_clas_ses-{session}_epo.fif"
    epo_phas_fname = f"{data_dir}/raw_epo/sub-{subject}_phas_ses-{session}_epo.fif"

    # define event
    event_adj = ['w1_T-B-AN', 'w2_T-B-NA', 'w1_T-NB-AN', 'w2_T-NB-NA',]
    event_n = ['w2_T-B-AN', 'w1_T-B-NA', 'w2_T-NB-AN', 'w1_T-NB-NA',]
    event_imp_nb = ['imp_T-NB-AN','imp_T-NB-NA',]
    event_imp_b = ['imp_T-B-AN', 'imp_T-B-NA', ]

    # load raw EEG epochs
    epochs_clas = mne.read_epochs(epo_clas_fname)
    epochs_clas.info
    epochs_clas.pick_types(meg=False, eeg=True)
    epochs_clas = epochs_clas[event_adj + event_n + event_imp_nb + event_imp_b] ### drop the control condition

    epochs_phas = mne.read_epochs(epo_phas_fname)
    epochs_phas.info
    epochs_phas.pick_types(meg=False, eeg=True)
    epochs_phas[event_imp_nb + event_imp_b] ### drop the control condition

    epochs_adj = epochs_clas[event_adj]
    epochs_n = epochs_clas[event_n]
    epochs_imp_nb = epochs_clas[event_imp_nb]
    epochs_imp_b = epochs_clas[event_imp_b]

    # match the epochs between impulse and phase epochs
    # select common epochs indices based on the trial number
    epochs_comm_nb = np.intersect1d(epochs_phas.selection, epochs_imp_nb.selection)
    epochs_comm_b = np.intersect1d(epochs_phas.selection, epochs_imp_b.selection)

    # assign epochs for phase estimation
    epochs_phas_nb = epochs_phas[np.isin(epochs_phas.selection, epochs_comm_nb)]
    epochs_phas_b = epochs_phas[np.isin(epochs_phas.selection, epochs_comm_b)]

    # assign epochs for decoding
    epochs_imp_nb = epochs_imp_nb[np.isin(epochs_imp_nb.selection, epochs_comm_nb)]
    epochs_imp_b = epochs_imp_b[np.isin(epochs_imp_b.selection, epochs_comm_b)]

    return epochs_adj, epochs_n, epochs_imp_nb, epochs_imp_b, epochs_phas_nb, epochs_phas_b

epochs_adj_ses01, epochs_n_ses01, epochs_imp_nb_ses01, epochs_imp_b_ses01, epochs_phas_nb_ses01, epochs_phas_b_ses01 = load_epochs(data_dir, subject, session_01)

epochs_adj_ses02, epochs_n_ses02, epochs_imp_nb_ses02, epochs_imp_b_ses02, epochs_phas_nb_ses02, epochs_phas_b_ses02 = load_epochs(data_dir, subject, session_02)

'''
Prepare beamforming of data

For beamforming, we need a data covariance matrix.

Since we have two independent time windows that does not form any contrasts, we will compute two separate data covariance matrices, one for each of them
'''

'''
# Concatenate relevant epochs for decoding
'''
def modify_event(epochs, old_code1, old_code2, new_event1, new_event2, new_code1, new_code2):
    '''
    Parameters:
    epochs
    old_code1: list, the old codes for event 1
    old_code2: list, the old codes for event 2
    new_event1: string, the new name for event 1
    new_event2: string, the new name for event 2
    new_code1: int, the new code for event1
    new_code2: int, the new code for event2

    Returns:
    epochs, in which events are modified
    '''
    events = epochs.events.copy()
    for code in old_code1:
        events[events[:, 2] == code, 2] = new_code1
    for code in old_code2:
        events[events[:, 2] == code, 2] = new_code2
    
    new_event_id = {new_event1: new_code1,
                    new_event2: new_code2}
    
    epochs_new = mne.EpochsArray(
    epochs.get_data(),
    info=epochs.info,
    events=events,
    event_id=new_event_id,
    tmin=epochs.tmin,
    baseline=epochs.baseline,
)
    return epochs_new

# concatenate epochs for training and validation
def concatenate_epochs(epochs_adj, epochs_n, epochs_imp_nb, epochs_imp_b, epochs_phas_nb, epochs_phas_b):
    '''
    Parameters:
    epochs_adj : mne.Epochs
        Adjective epochs for decoding.
    epochs_n : mne.Epochs
        Noun epochs for decoding.
    epochs_imp_nb : mne.Epochs
        Impulse epochs (non-binding).
    epochs_imp_b : mne.Epochs
        Impulse epochs (binding).
    epochs_phas_nb : mne.Epochs
        Phase estimation epochs (non-binding).
    epochs_phas_b : mne.Epochs
        Phase estimation epochs (binding).

    Returns:
    epochs_train: mne.Epochs
        Epochs used to train the classifier (Adjective and Noun)
    epochs_test: mne.Epochs
        Epochs used to test the classifier (Impulse)
    epochs_clas_lcmv: mne.Epochs
        Epochs used to compute the data & noise covariance matrix for the source reconstructing the decoding epochs
    epochs_phas_lcmv: mne.Epochs
        Epochs used to compute the data & noise covariance matrix for the source reconstructing the phase estimation epochs

    '''

    epochs_train = mne.concatenate_epochs([epochs_adj, epochs_n])

    epochs_train = modify_event(epochs_train,
                                [12, 32, 23, 43],
                                [22, 42, 13, 33],
                                'Adjective',
                                'Noun',
                                1,
                                2,)

    # concatenate epochs for testing
    epochs_test = mne.concatenate_epochs([epochs_imp_nb, epochs_imp_b])

    epochs_test = modify_event(epochs_test,
                                [34, 44],
                                [14, 24],
                                'Nonbinding',
                                'Binding',
                                1,
                                2,)

    '''
    # Concatenate relevant epochs for beamforming
    '''
    # concatenate epochs for classification
    epochs_clas_lcmv = mne.concatenate_epochs([epochs_train, epochs_test])

    # concatenate epochs for phase estimation
    epochs_phas_lcmv= mne.concatenate_epochs([epochs_phas_nb, epochs_phas_b])

    epochs_phas_lcmv = modify_event(epochs_phas_lcmv,
                                [34, 44],
                                [14, 24],
                                'Nonbinding',
                                'Binding',
                                1,
                                2,)

    return epochs_train, epochs_test, epochs_clas_lcmv, epochs_phas_lcmv

epochs_train_ses01, epochs_test_ses01, epochs_clas_lcmv_ses01, epochs_phas_lcmv_ses01 = concatenate_epochs(epochs_adj_ses01, epochs_n_ses01, epochs_imp_nb_ses01, epochs_imp_b_ses01, epochs_phas_nb_ses01, epochs_phas_b_ses01)

epochs_train_ses02, epochs_test_ses02, epochs_clas_lcmv_ses02, epochs_phas_lcmv_ses02 = concatenate_epochs(epochs_adj_ses02, epochs_n_ses02, epochs_imp_nb_ses02, epochs_imp_b_ses02, epochs_phas_nb_ses02, epochs_phas_b_ses02)

# compute two data covariance matrices
# data_cov_clas will be used to compute a common spatial filter for the classification epochs (adj + n + imp)
data_cov_clas_ses01 = mne.compute_covariance(epochs_clas_lcmv_ses01, 
                                       tmin=0., tmax=0.8,method='empirical', rank='info')
data_cov_clas_ses02 = mne.compute_covariance(epochs_clas_lcmv_ses02, 
                                       tmin=0., tmax=0.8,method='empirical', rank='info')
data_cov_phas_ses01 = mne.compute_covariance(epochs_phas_lcmv_ses01, 
                                       tmin=-1.25, tmax=0.,method='empirical', rank='info')
data_cov_phas_ses02 = mne.compute_covariance(epochs_phas_lcmv_ses02, 
                                       tmin=-1.25, tmax=0.,method='empirical', rank='info')

# visualize data covariance matrices
mne.viz.plot_cov(data_cov_clas_ses01, info=epochs_clas_lcmv_ses01.info)
input("Check session 1 data covariance matrix for classification, then close visualization by pressing ENTER ...")
mne.viz.plot_cov(data_cov_clas_ses02, info=epochs_clas_lcmv_ses02.info)
input("Check session 2 data covariance matrix for classification, then close visualization by pressing ENTER ...")
mne.viz.plot_cov(data_cov_phas_ses01, info=epochs_phas_lcmv_ses01.info)
input("Check session 1 data covariance matrix for phase estimation, then close visualization by pressing ENTER ...")
mne.viz.plot_cov(data_cov_phas_ses02, info=epochs_phas_lcmv_ses02.info)
input("Check session 2 data covariance matrix for phase estimation, then close visualization by pressing ENTER ...")

'''
from mne import compute_rank
ranks = compute_rank(data_cov_clas, rank='full', info = epochs_clas.info)
'''

'''
We also need to compute a noise covariance matrix. That will be used for pre-whitening the data, data covariance matrix, and forward model. 
'''
# compute two noise covariance matrixs to pre-whiten the data, data covariance matrix, and forward model.
noise_cov_clas_ses01 = mne.compute_covariance(epochs_clas_lcmv_ses01, tmin=-.2, tmax=0.,  # use the baseline
                                        method='empirical', 
                                        rank='info')
noise_cov_clas_ses02 = mne.compute_covariance(epochs_clas_lcmv_ses02, tmin=-.2, tmax=0.,  # use the baseline
                                        method='empirical', 
                                        rank='info')

noise_cov_phas_ses01 = mne.compute_covariance(epochs_phas_lcmv_ses01, tmin=-1.45, tmax=-1.25,  # use the baseline
                                        method='empirical', 
                                        rank='info')
noise_cov_phas_ses02 = mne.compute_covariance(epochs_phas_lcmv_ses02, tmin=-1.45, tmax=-1.25,  # use the baseline
                                        method='empirical', 
                                        rank='info')

'''
Lastly, we also need to read the forward model that we had saved!
'''
# read the forward models we saved
fwd_ses01_fname = f"{data_dir}/source_recon/sub-{subject}/sub-{subject}_ses-{session_01}_eeg_fwd.fif"
fwd_ses02_fname = f"{data_dir}/source_recon/sub-{subject}/sub-{subject}_ses-{session_02}_eeg_fwd.fif"

fwd_ses01 = mne.read_forward_solution(fwd_ses01_fname)
fwd_ses01 = mne.pick_types_forward(fwd_ses01, meg=False, eeg=True)
print('forward models loaded for session 1!')

fwd_ses02 = mne.read_forward_solution(fwd_ses02_fname)
fwd_ses02 = mne.pick_types_forward(fwd_ses02, meg=False, eeg=True)
print('forward models loaded for session 2!')

del epochs_adj_ses01
del epochs_adj_ses02
del epochs_n_ses01
del epochs_n_ses02
del epochs_imp_nb_ses01
del epochs_imp_nb_ses02
del epochs_imp_b_ses01
del epochs_imp_b_ses02

'''
Compute beamformer and apply to epoch data
'''
# compute the beamformer
# unit-noise gain beamformers handle depth bias by normalizing the weights of the spatial filter.
def compute_beamformer(epochs, fwd, data_cov, noise_cov):

    filters = make_lcmv(
        epochs.info, 
        fwd,
        data_cov=data_cov, 
        reg=0.05, 
        noise_cov=noise_cov,
        pick_ori='max-power', 
        rank='info',
        weight_norm='unit-noise-gain',
    )
    
    return filters

filters_clas_ses01 = compute_beamformer(epochs_clas_lcmv_ses01, fwd_ses01, data_cov_clas_ses01, noise_cov_clas_ses01)
filters_phas_ses01 = compute_beamformer(epochs_phas_lcmv_ses01, fwd_ses01, data_cov_phas_ses01, noise_cov_phas_ses01)
print('beamformers computed for session 1!')

input('wait')

filters_clas_ses02 = compute_beamformer(epochs_clas_lcmv_ses02, fwd_ses02, data_cov_clas_ses02, noise_cov_clas_ses02)
filters_phas_ses02 = compute_beamformer(epochs_phas_lcmv_ses02, fwd_ses02, data_cov_phas_ses02, noise_cov_phas_ses02)
print('beamformers computed for session 2!')

del epochs_clas_lcmv_ses01
del epochs_clas_lcmv_ses02

input('wait')

# drop baseline data
epochs_train_ses01 = epochs_train_ses01.crop(tmin=0)
epochs_train_ses02 = epochs_train_ses02.crop(tmin=0)
epochs_test_ses01 = epochs_test_ses01.crop(tmin=0)
epochs_test_ses02 = epochs_test_ses02.crop(tmin=0)
epochs_phas_lcmv_ses01 = epochs_phas_lcmv_ses01.crop(tmin=-1.25, tmax=0)
epochs_phas_lcmv_ses02 = epochs_phas_lcmv_ses02.crop(tmin=-1.25, tmax=0)

# apply the spatial filters
# data shape: n_epochs * n_vertices * n_times
# generator type data can be accessed by next()
stc_train_ses01 = apply_lcmv_epochs(epochs=epochs_train_ses01, filters=filters_clas_ses01, return_generator=False, )
stc_train_ses02 = apply_lcmv_epochs(epochs=epochs_train_ses02, filters=filters_clas_ses02, return_generator=False, )
print('beamformers applied for training set!')

stc_test_ses01 = apply_lcmv_epochs(epochs=epochs_test_ses01, filters=filters_clas_ses01, return_generator=False, )
stc_test_ses02 = apply_lcmv_epochs(epochs=epochs_test_ses02, filters=filters_clas_ses02, return_generator=False, )
print('beamformers applied for testing set!')

stc_phas_ses01 = apply_lcmv_epochs(epochs=epochs_phas_lcmv_ses01, filters=filters_phas_ses01, return_generator=False, )
stc_phas_ses02 = apply_lcmv_epochs(epochs=epochs_phas_lcmv_ses02, filters=filters_phas_ses02, return_generator=False, )
print('beamformers applied for phase estimate set!')

'''
#plot the brain and time course
mri_dir = op.join(data_dir, 'mri_reconst')
next(stc_clas).plot(subjects_dir=mri_dir, subject=subject, clim=dict(kind="value", lims=[3, 6, 9]), hemi='both', smoothing_steps=7,)
input("Check brain and time course for classification, then close visualization by pressing ENTER ...")

next(stc_phas).plot(subjects_dir=mri_dir, subject=subject, clim=dict(kind="value", lims=[3, 6, 9]), hemi='both', smoothing_steps=7,)
input("Check brain and time course for phase estimation, then close visualization by pressing ENTER ...")
'''

'''
Extract ROI source data
'''
# path to src
source_dir = op.join(data_dir, 'source_recon')

# load src
src_fname = op.join(source_dir, f'sub-{subject}',f'sub-{subject}_src.fif')
src = mne.read_source_spaces(src_fname)
print('source space loaded!')

from mne.label import Label
from mne import extract_label_time_course

# path to individual freesurfer reconstruction
mri_dir = op.join(data_dir, 'mri_reconst')

# define rois
STG = ["G_temp_sup-Lateral", 
       "G_temp_sup-Plan_polar", 
       "G_temp_sup-Plan_tempo",]
MTG = ["G_temporal_middle"]
ATL = ["Pole_temporal"]
IFG = ["G_front_inf-Opercular", 
       "G_front_inf-Triangul", 
       "G_front_inf-Orbital"]
IPL = ["G_pariet_inf-Angular", 
       "G_pariet_inf-Supramar"]

def read_labels_from_rois(subject, subjects_dir, parc, roi_list):
    """
    subject: str, FreeSurfer subject
    subjects_dir: str
    parc: str, parcellation name, e.g., 'aparc.a2009s'
    roi_list: list of strings, ROI names or regexp
    hemi: str, left hemisphere
    """
    labels = []
    for roi in roi_list:
        matched = mne.read_labels_from_annot(subject, parc=parc,
                                             subjects_dir=subjects_dir,
                                             regexp=roi, hemi='lh')
        labels.extend(matched)
    return labels

labels_stg = read_labels_from_rois(subject, mri_dir, "aparc.a2009s", STG)    
labels_mtg = read_labels_from_rois(subject, mri_dir, "aparc.a2009s", MTG)
labels_atl = read_labels_from_rois(subject, mri_dir, "aparc.a2009s", ATL)
labels_ifg = read_labels_from_rois(subject, mri_dir, "aparc.a2009s", IFG)
labels_ipl = read_labels_from_rois(subject, mri_dir, "aparc.a2009s", IPL)

labels_rois = labels_stg + labels_mtg + labels_atl + labels_ifg + labels_ipl

# extract ROI time courses
# data shape: n_epoch * n_labels (anatomical) * n_vertices * n_times
def extract_roi_stc(stc, labels, src):
    stc_roi = mne.extract_label_time_course(stc, 
                                            labels, 
                                            src=src, 
                                            mode = None, 
                                            return_generator=False, allow_empty=False)
    return stc_roi

# concatenate ROI time courses across sessions
def concatenate_roi_stc(stc_rois_ses01, stc_rois_ses02):
    stc_concat = []
    for lbl_idx in range(len(stc_rois_ses01[0])):
        data1 = np.stack([stc_rois_ses01[epo_idx][lbl_idx]
                         for epo_idx in range(len(stc_rois_ses01))], axis=0)
        data2 = np.stack([stc_rois_ses02[epo_idx][lbl_idx]
                         for epo_idx in range(len(stc_rois_ses02))], axis=0)
        data_concat = np.concatenate([data1, data2], axis=0)
        stc_concat.append(data_concat)
    
    return stc_concat

'''
Training
'''
#import sys
#sys.path.append(os.path.join(os.getcwd(), "scripts"))

from decoding import sliding_logreg_source
from decoding import pca
from decoding import get_fft_phase
from decoding import get_preds
from decoding import get_phase_per_time
from decoding import itpc_over_time
from decoding import plot_itpc_over_time
from decoding import categorize_phase_by_label
from decoding import compute_phase_vector
from decoding import plot_vl
from decoding import compute_relative_phase
from decoding import plot_va

def prepare_X(stc_roi):
    '''
    Parameters:
    stc_roi: list, roi souce space time course

    Returns:
    X: np.array, data to be fitted, shape: n_epochs, sum_vertices_across_labels, n_times
    '''
    X = np.concatenate([lbl.reshape(lbl.shape[0], -1, lbl.shape[2]) for lbl in stc_roi], axis=1)
    return X

def prepare_y(epochs_ses01, epochs_ses02):
    '''
    Parameters:
    epochs_ses01: mne.Epochs
    epochs_ses02: mne.Epochs

    Returns:
    y: np.array, response vector, contains epochs event id
    '''
    y = np.concatenate([epochs_ses01.events[:, 2], epochs_ses02.events[:, 2]])
    return y

'''
Loop over ROIs
'''
roi_dict = {
    "rois": labels_rois,
    "stg": labels_stg,
    "mtg": labels_mtg,
    "atl": labels_atl,
    "ifg": labels_ifg,
    "ipl": labels_ipl
}

def extract_concat(stc_ses01, stc_ses02, labels, src):
    roi_ses01 = extract_roi_stc(stc_ses01, labels, src)
    roi_ses02 = extract_roi_stc(stc_ses02, labels, src)
    return concatenate_roi_stc(roi_ses01, roi_ses02)

train_rois, test_rois, phas_rois = {}, {}, {}
for roi_name, roi_labels in roi_dict.items():
    train_rois[roi_name] = extract_concat(stc_train_ses01, stc_train_ses02, roi_labels, src)
    test_rois[roi_name]  = extract_concat(stc_test_ses01,  stc_test_ses02,  roi_labels, src)
    phas_rois[roi_name]  = extract_concat(stc_phas_ses01,  stc_phas_ses02,  roi_labels, src)


y_train = prepare_y(epochs_train_ses01, epochs_train_ses02)
y_test  = prepare_y(epochs_test_ses01,  epochs_test_ses02)
y_phase = prepare_y(epochs_phas_lcmv_ses01, epochs_phas_lcmv_ses02)

foi = np.arange(4, 20, 0.1)
sfreq = 500

decode_results = {}
for roi_name in roi_dict.keys():
    # prepare data for decoding
    # returns n_epochs, sum_vertices_across_labels, n_times
    X_train = prepare_X(train_rois[roi_name])
    X_test  = prepare_X(test_rois[roi_name])
    X_phase = prepare_X(phas_rois[roi_name])

    print(f"Decoding {roi_name} ...")
    decode_results[roi_name] = sliding_logreg_source(X_train, y_train, X_test, y_test, folds=5)

    # get preds
    imp_lbl_nb, imp_proba_nb, imp_lbl_b, imp_proba_b = get_preds(decode_results[roi_name])

    # sort phase epochs by event id
    X_phase_nb = X_phase[y_phase==1]
    X_phase_b = X_phase[y_phase==2]

    # run pca within roi
    # return shape: n_epochs, n_times
    X_phase_nb_pc = pca(X_phase_nb)
    X_phase_b_pc = pca(X_phase_b)

    # estimate phase angle
    # return shape: n_foi, n_epochs
    ang_nb = get_fft_phase(X_phase_nb_pc, foi, sfreq)
    ang_b = get_fft_phase(X_phase_b_pc, foi, sfreq)

    # at each time point, categorize trials based on decoded label
    # then assign the phase for those trials to that time point
    ppt_nb_adj, ppt_nb_n = get_phase_per_time(imp_lbl_nb, ang_nb, 42)
    ppt_b_adj, ppt_b_n = get_phase_per_time(imp_lbl_b, ang_b, 42)

    # compute itpc over time
    itpc_nb_adj = itpc_over_time(ppt_nb_adj, foi)
    itpc_nb_n = itpc_over_time(ppt_nb_n, foi)

    itpc_b_adj = itpc_over_time(ppt_b_adj, foi)
    itpc_b_n = itpc_over_time(ppt_b_n, foi)

    # plot and save itpc
    plot_itpc_over_time(itpc_nb_adj, itpc_nb_n, sfreq, foi,
                    save_path=f"subject-{subject}_itpc_{roi_name}_NB.png",
                    title=f"subject-{subject}_{roi_name} - Non-binding")
    plot_itpc_over_time(itpc_b_adj, itpc_b_n, sfreq, foi,
                    save_path=f"subject-{subject}_itpc_{roi_name}_B.png",
                    title=f"subject-{subject}_{roi_name} - Binding")
    
    # categorize phase estimation data based on predicted labels
    X_phase_nb_adj, X_phase_nb_n, X_phase_b_adj, X_phase_b_n, X_phase_baseline = categorize_phase_by_label(imp_lbl_nb, imp_lbl_b, X_phase_nb, X_phase_b, X_phase, 42)

    # run pca within roi
    # return shape: n_epochs, n_times
    X_phase_nb_adj_pc = pca(X_phase_nb_adj)
    X_phase_nb_n_pc = pca(X_phase_nb_n)
    X_phase_b_adj_pc = pca(X_phase_b_adj)
    X_phase_b_n_pc = pca(X_phase_b_n)
    X_phase_baseline_pc = pca(X_phase_baseline)

    # estimate phase angle
    # return shape: n_foi, n_epochs
    ang_nb_adj = get_fft_phase(X_phase_nb_adj_pc, foi, sfreq)
    ang_nb_n = get_fft_phase(X_phase_nb_n_pc, foi, sfreq)
    ang_b_adj = get_fft_phase(X_phase_b_adj_pc, foi, sfreq)
    ang_b_n = get_fft_phase(X_phase_b_n_pc, foi, sfreq)
    ang_baseline = get_fft_phase(X_phase_baseline_pc, foi, sfreq)

    # compute static itpc and phase angle
    vl_nb_adj, va_nb_adj = compute_phase_vector(ang_nb_adj)
    vl_nb_n, va_nb_n = compute_phase_vector(ang_nb_n)
    vl_b_adj, va_b_adj= compute_phase_vector(ang_b_adj)
    vl_b_n, va_b_n = compute_phase_vector(ang_b_n)
    vl_base, va_base = compute_phase_vector(ang_baseline)

    # plot and save static itpc
    plot_vl(vl_nb_adj, 
            vl_nb_n, 
            vl_b_adj, 
            vl_b_n, 
            vl_base, 
            foi, 
            title=f'subject-{subject}-{roi_name}: Phase Consistency across Frequencies',
            save_path=f"subject-{subject}_vl_{roi_name}.png")

    # compute relative phase
    rva_nb_adj_n = compute_relative_phase(va_nb_adj, va_nb_n)
    rva_b_adj_n = compute_relative_phase(va_b_adj, va_b_n)

    # plot and save relative phase
    plot_va(rva_b_adj_n,
            rva_nb_adj_n,
            foi, 
            labels=['B: Adjective vs. Noun', 
                    'NB: Adjective vs. Noun'],
            colors=[ '#FF8C00', '#4682B4'],
            highlight_freqs=[(4, 7), (7, 12), (13, 20)],
            highlight_colors=['black', 'red', 'gray'],
            highlight_markers=["^", "s", "D"],
            title=f'subject-{subject}-{roi_name}: Relative Phase Distribution',
            save_path=f"subject-{subject}_rva_{roi_name}.png")



