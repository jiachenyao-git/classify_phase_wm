'''
Compute inverse models

'''
globals().clear()
import numpy as np
import mne
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import os
import os.path as op
import matplotlib.pyplot as plt

# silence mne
mne.set_log_level('warning')

# define subject and session
subject ='01'
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
        rank=None,
        weight_norm='unit-noise-gain',
    )
    
    return filters

filters_clas_ses01 = compute_beamformer(epochs_clas_lcmv_ses01, fwd_ses01, data_cov_clas_ses01, noise_cov_clas_ses01)
filters_phas_ses01 = compute_beamformer(epochs_phas_lcmv_ses01, fwd_ses01, data_cov_phas_ses01, noise_cov_phas_ses01)
print('beamformers computed for session 1!')

filters_clas_ses02 = compute_beamformer(epochs_clas_lcmv_ses02, fwd_ses02, data_cov_clas_ses02, noise_cov_clas_ses02)
filters_phas_ses02 = compute_beamformer(epochs_phas_lcmv_ses02, fwd_ses02, data_cov_phas_ses02, noise_cov_phas_ses02)
print('beamformers computed for session 2!')

del epochs_clas_lcmv_ses01
del epochs_clas_lcmv_ses02

# apply the spatial filters
# data shape: n_epochs * n_vertices * n_times
# generator type data can be accessed by next()
stc_train_ses01 = apply_lcmv_epochs(epochs=epochs_train_ses01, filters=filters_clas_ses01, return_generator=False, )
stc_train_ses02 = apply_lcmv_epochs(epochs=epochs_train_ses02, filters=filters_clas_ses02, return_generator=False, )

stc_test_ses01 = apply_lcmv_epochs(epochs=epochs_test_ses01, filters=filters_clas_ses01, return_generator=False, )
stc_test_ses02 = apply_lcmv_epochs(epochs=epochs_test_ses02, filters=filters_clas_ses02, return_generator=False, )

stc_phas_ses01 = apply_lcmv_epochs(epochs=epochs_phas_lcmv_ses01, filters=filters_phas_ses01, return_generator=False, )
stc_phas_ses02 = apply_lcmv_epochs(epochs=epochs_phas_lcmv_ses02, filters=filters_phas_ses02, return_generator=False, )
print('beamformers applied!')

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
stc_train_ses01_rois = mne.extract_label_time_course(stc_train_ses01, labels_rois, src, mode = None, return_generator=False, allow_empty=False)
stc_train_ses02_rois = mne.extract_label_time_course(stc_train_ses02, labels_rois, src, mode = None, return_generator=False, allow_empty=False)

stc_test_ses01_rois = mne.extract_label_time_course(stc_test_ses01, labels_rois, src, mode = None, return_generator=False, allow_empty=False)
stc_test_ses02_rois = mne.extract_label_time_course(stc_test_ses02, labels_rois, src, mode = None, return_generator=False, allow_empty=False)

del stc_train_ses01
del stc_train_ses02
del stc_test_ses01
del stc_test_ses02

# concatenate ROI time courses across sessions
stc_train_roi = []

for lbl_idx in range(len(stc_train_ses01_rois[0])): ### loop over each ROI label
    ### for each label, get the data for all the epochs
    data1 = np.stack([stc_train_ses01_rois[epo_idx][lbl_idx]
                      for epo_idx in range(len(stc_train_ses01_rois))], axis=0)
    data2 = np.stack([stc_train_ses02_rois[epo_idx][lbl_idx]
                      for epo_idx in range(len(stc_train_ses02_rois))], axis=0)
    data_concat = np.concatenate([data1, data2], axis=0)
    stc_train_roi.append(data_concat)   ### shape: n_labels * n_epochs * n_vertices * n_times


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

stc_test_roi = concatenate_roi_stc(stc_test_ses01_rois, stc_test_ses02_rois)

# check for the same number of epochs across ROIs
n_epochs_train = [lbl.shape[0] for lbl in stc_train_roi]
n_epochs_test = [lbl.shape[0] for lbl in stc_test_roi]
assert len(set(n_epochs_train)) == 1, "All labels must have same number of epochs for the training set"
assert len(set(n_epochs_test)) == 1, "All labels must have same number of epochs for the testing set"

'''
Decoding
'''
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mne.decoding import LinearModel, SlidingEstimator, cross_val_multiscore, get_coef

'''
Training
'''
# prepare data for decoding (n_epochs, sum_vertices_across_labels, n_times)
X = np.concatenate([lbl.reshape(lbl.shape[0], -1, lbl.shape[2]) for lbl in stc_train_roi], axis=1)
print(X.shape)  # (n_epochs, sum_vertices_across_labels, n_times)
# concatenate event labels across session
y = np.concatenate([epochs_train_ses01.events[:, 2], epochs_train_ses02.events[:, 2]])

# prepare a series of classifier applied at each time sample
clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    SelectKBest(f_classif, k=500),  # select features for speed
    LinearModel(LogisticRegression(C=1, solver="liblinear")),
)
time_decod = SlidingEstimator(clf, scoring="roc_auc")

# run cross-validated decoding analyses:
scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=None)

# plot average decoding scores of 5 splits
fig, ax = plt.subplots(1)
ax.plot(epochs_train_ses01.times, scores.mean(0), label="score")
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.axvline(0, color="k")
plt.legend()
plt.show()

'''
Testing
'''
# prepare data for decoding (n_epochs, sum_vertices_across_labels, n_times)
X_test = np.concatenate([lbl.reshape(lbl.shape[0], -1, lbl.shape[2]) for lbl in stc_test_roi], axis=1)
print(X_test.shape)  # (n_epochs, sum_vertices_across_labels, n_times)

assert X.shape[1] == X_test.shape[1], 'Training and Testing set should have the same vertices'

y_pred = time_decod.predict(X_test)

