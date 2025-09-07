'''Create epochs
This script creates stimulus-locked and impulse locked epochs using auto rejection criterials and robust z-score values.

The following pipeline is followed:
    1. 

'''
globals().clear()

import mne
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import Counter

filt_dir = '/Users/jiachenyao/Desktop/PhaseCode/data/filt_reconst'
raw_dir =  '/Users/jiachenyao/Desktop/PhaseCode/data/raw_reconst'
sub = 'sub-02'

### loop through all sessions and all segments
def load_and_print_data(file_names, data_dir):
    data_dict = {fname: mne.io.read_raw_fif(op.join(data_dir, fname), preload=True, verbose=True) for fname in file_names}
    for fname, data in data_dict.items():
        print(f"\nLoaded: {fname}")
        print(data.info)
    return data_dict

file_names = [f"{sub}_ses-{ses:02d}.fif" for ses in range(1, 3)]

raw_dict = load_and_print_data(file_names, raw_dir)
filt_dict = load_and_print_data(file_names, filt_dir)

### seperate the data by session
raw_ses01 = raw_dict[f'{sub}_ses-01.fif']
raw_ses02 = raw_dict[f'{sub}_ses-02.fif']
#filt_ses01 = filt_dict[f'{sub}_ses-01.fif']
#filt_ses02 = filt_dict[f'{sub}_ses-02.fif']

######################################################################
'''
Step 1. Epoch the raw data

Here we define the event code for two time windows:
    classification window : [-0.2, 0.8] time-locked to audio and impulse
    phase estimation window : [-1.45, 1.0] time-locked to impulse

Creating epochs involves baseline correction.

'''
# epoch raw data
# NOTE the shortest wav is 0.47s, the longest wav is 1.14s
# define event for epoching
event_classify_dict = {
    'w1_T-B-AN': 12,
    'w1_T-B-NA': 22,
    'w1_T-NB-AN': 32,
    'w1_T-NB-NA': 42,
    'w1_T-C-AN': 52,
    'w1_T-C-NA': 62,

    'w2_T-B-AN': 13,
    'w2_T-B-NA': 23,
    'w2_T-NB-AN': 33,
    'w2_T-NB-NA': 43,
    'w2_T-C-AN': 53,
    'w2_T-C-NA': 63,

    'imp_T-B-AN': 14,
    'imp_T-B-NA': 24,
    'imp_T-NB-AN': 34,
    'imp_T-NB-NA': 44,
    'imp_T-C-AN': 54,
    'imp_T-C-NA': 64,
}

event_phase_dict ={
    'imp_T-B-AN': 14,
    'imp_T-B-NA': 24,
    'imp_T-NB-AN': 34,
    'imp_T-NB-NA': 44,
    'imp_T-C-AN': 54,
    'imp_T-C-NA': 64,
}

# define auto-epoch-rejection criteria
reject_criteria = dict(
    eeg=150e-6,  # 100 µV
)
flat_criteria = dict(
    eeg=1e-6
)  # 1 µV

# create epochs on raw data
# cut epochs used for classification ([-0.2, 0.8] time-locked to audio and impulse)
def create_epoch_classify (data, event_dict, chan='TRIGGER', prestim=-0.2, poststim=0.8):

    events = mne.find_events(data, stim_channel=chan, min_duration=2 / data.info['sfreq'])
    epochs = mne.Epochs(
        data, 
        events, 
        tmin=prestim, 
        tmax=poststim, 
        event_id=event_dict,
        reject=reject_criteria,
        flat=flat_criteria, 
        preload=True, 
        reject_by_annotation=True,
        baseline = [-0.2, 0],)  # baseline correction
     
    return epochs

# cut epochs used for phase estimation ([-1.45, 1.0] time-locked to impulse)
def create_epoch_phase (data, event_dict, chan='TRIGGER', prestim=-1.45, poststim=1):

    events = mne.find_events(data, stim_channel=chan, min_duration=2 / data.info['sfreq'])
    epochs = mne.Epochs(
        data, 
        events, 
        tmin=prestim, 
        tmax=poststim, 
        event_id=event_dict,
        reject=reject_criteria,
        flat=flat_criteria,  
        preload=True, 
        reject_by_annotation=True,
        baseline = [-1.45, -1.25])  # baseline corrected
     
    return epochs

######################################################################
'''
Procedure:
    1. create epochs with auto-rejection criteria
    2. check, for session 1 and 2 respectively, if rejection is mainly due to specific channels
    3. reject these channels on unepoched data
    4. then re-create epochs with auto-rejection again
    5. calculate z-scores for epochs
    6. manually reject trials that surpass the z-scores threshold
    7. plot psd and raw epoch data to inspect trial by trial

NOTE the same channels should be kept for the epochs comming from the same session
'''
######################################################################
'''
Step 1. Create epochs with auto-rejection criteria

Here we inspect which channels lead to more epoch rejection and decide on channel-trial trade-offs
'''

'''
Epoching data from measurement session 1
'''
# define functions
def check_bad_channel(epochs):
    rejected_chans = []
    n_rejected_epochs = 0   ### count the number of epochs rejected

    for log in epochs.drop_log:
        ### filter the channel names that causes the rejection, which cannot be 'IGNORED'
        ### if log = ['IGNORED'], then chs will be []
        ### if log = ['any real channel'], chs = ['channel name']
        chs = [ch for ch in log if ch != 'IGNORED']
        if chs: ### if chs != []
            rejected_chans.extend(chs) ### log the channel name
            n_rejected_epochs += 1 ### count the epoch that is rejected due to channel
    
    return rejected_chans, n_rejected_epochs

def print_bad_channel (rejected_chans, n_rejected_epochs):
    ch_freq = Counter(rejected_chans)
    for ch, freq in ch_freq.most_common():
        print(f"{ch}: {freq} ({freq/n_rejected_epochs*100:.1f}%)")

def drop_channels (data, drop_chans):
    data.drop_channels(drop_chans)
    print(len(data.info['ch_names'])-3)

def compute_z_score(epo, z_thresh):
    """
    Compute robust z-score per channel and mark bad samples

    Parameters:
    epo : np.ndarray
        EEG epochdata, shape (n_channels, n_times)
    z_thresh : float
        Threshold for robust z-score

    Returns:
    bad_epochs_idx : np.ndarray
        Indices of epochs exceeding the robust z-score threshold
    robust_z : np.ndarray
        The computed robust z-scores, shape (n_epochs, n_channels, n_times)

    """
    # only pick EEG channels
    picks = mne.pick_types(epo.info, eeg=True)
    data = epo.get_data(picks)  # shape: (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape
    
    # initialize robust z-score matrix
    robust_z = np.zeros_like(data)
    
    # per-channel loop
    for ch in range(n_channels):
        # channel data, shape: (n_epochs, n_times)
        ch_data = data[:, ch, :]
        
        # median & MAD (or IQR)
        med = np.median(ch_data)   # shape: (n_epochs, n_times)
        mad = np.median(np.abs(ch_data - med))
        if mad == 0:
            mad = 1e-6
        '''
        iqr = np.percentile(ch_data, 75) - np.percentile(ch_data, 25)
        if iqr == 0:
            iqr = 1e-6  # avoid dividing by 0
        '''

        # robust z-score
        '''
        robust_z[:, ch, :] = (ch_data - med) / (0.7413 * iqr)
        '''
        robust_z[:, ch, :] = 0.6745 * (ch_data - med) / mad
    
    # mark bad epochs if any time point in any channel exceeds the threshold
    bad_epochs_idx = np.where(np.any(np.abs(robust_z) > z_thresh, axis=(1, 2)))[0]
    
    print(f"{len(bad_epochs_idx)} / {n_epochs} epochs ({len(bad_epochs_idx)/n_epochs*100:.1f}%) will be rejected")
    
    return bad_epochs_idx, robust_z

# plot psd
def plot_psd(epo):
    epo.plot_psd()
    plt.show()

# plot epochs
def plot_epo(epo, event_id, n_epochs=2, picks=['eeg', 'eog']):
    epo.plot(n_epochs=n_epochs, event_id=event_id, events=True, picks=picks, scalings=None)
    plt.show()

'''
Session 1
'''
# step 0. create a deep copy of raw data
raw_ses01_copy = copy.deepcopy(raw_ses01)

# step 1. create epochs
raw_epo_ses01_clas = create_epoch_classify(raw_ses01_copy, event_classify_dict)
raw_epo_ses01_phas = create_epoch_phase(raw_ses01_copy, event_phase_dict)

# step 2. check & print bad channel
# * unpack the tuple to two variables
print_bad_channel(*check_bad_channel(raw_epo_ses01_clas)) 
print_bad_channel(*check_bad_channel(raw_epo_ses01_phas))

# step 3. drop 'bad' channels that consistently cause auto-rejection
# NOTE this should be done on the UNEPOCHED data
drop_chans = ['FT7', 'AF4', 'AF3']
drop_channels(raw_ses01_copy, drop_chans)

# step 4. second-time run after rejecting channels that causes auto epoch rejections
# NOTE log the rejected trial numbers in the logbook
raw_epo_ses01_clas = create_epoch_classify(raw_ses01_copy, event_classify_dict)
raw_epo_ses01_phas = create_epoch_phase(raw_ses01_copy, event_phase_dict)

# step 5. calculate z-scores for epochs
# NOTE log the rejected trial numbers in the logbook
bad_idx_ses01_clas, rz_scores_ses01_clas = compute_z_score(raw_epo_ses01_clas, z_thresh=8.5)
bad_idx_ses01_phas, rz_scores_ses01_phas = compute_z_score(raw_epo_ses01_phas, z_thresh=8.5)

# step 6. reject trials based on z-scores
raw_epo_ses01_clas.drop(bad_idx_ses01_clas)
raw_epo_ses01_phas.drop(bad_idx_ses01_phas)

# step 7. plot psd and raw epochs 
plot_psd(raw_epo_ses01_clas)
plot_psd(raw_epo_ses01_phas)

plot_epo(raw_epo_ses01_clas, event_id=event_classify_dict)
plot_epo(raw_epo_ses01_phas, event_id=event_phase_dict)

'''
Session 2
'''
# step 0. create a deep copy of raw data
raw_ses02_copy = copy.deepcopy(raw_ses02)

# step 1. create epochs
raw_epo_ses02_clas = create_epoch_classify(raw_ses02_copy, event_classify_dict)
raw_epo_ses02_phas = create_epoch_phase(raw_ses02_copy, event_phase_dict)

# step 2. check & print bad channel
# * unpack the tuple to two variables
print_bad_channel(*check_bad_channel(raw_epo_ses02_clas)) 
print_bad_channel(*check_bad_channel(raw_epo_ses02_phas))

# step 3. drop 'bad' channels that consistently cause auto-rejection
# NOTE this should be done on the UNEPOCHED data
drop_chans = ['FC3',]
drop_channels(raw_ses02_copy, drop_chans)

# step 4. second-time run after rejecting channels that causes auto epoch rejections
# NOTE log the rejected trial numbers in the logbook
raw_epo_ses02_clas = create_epoch_classify(raw_ses02_copy, event_classify_dict)
raw_epo_ses02_phas = create_epoch_phase(raw_ses02_copy, event_phase_dict)

# step 5. calculate z-scores for epochs
# NOTE log the rejected trial numbers in the logbook
bad_idx_ses02_clas, rz_scores_ses02_clas = compute_z_score(raw_epo_ses02_clas, z_thresh=7)
bad_idx_ses02_phas, rz_scores_ses02_phas = compute_z_score(raw_epo_ses02_phas, z_thresh=7)

# step 6. reject trials based on z-scores
raw_epo_ses02_clas.drop(bad_idx_ses02_clas)
raw_epo_ses02_phas.drop(bad_idx_ses02_phas)

# step 7. plot psd and raw epochs 
plot_psd(raw_epo_ses02_clas)
plot_psd(raw_epo_ses02_phas)

plot_epo(raw_epo_ses02_clas, event_id=event_classify_dict)
plot_epo(raw_epo_ses02_phas, event_id=event_phase_dict)

'''
Save epochs
'''
save_epo_dir = '/Users/jiachenyao/Desktop/PhaseCode/data/raw_epo'

raw_epo_ses01_clas.save(f"{save_epo_dir}/{sub}_clas_ses-01.fif", overwrite=True)
raw_epo_ses01_phas.save(f"{save_epo_dir}/{sub}_phas_ses-01.fif", overwrite=True)

raw_epo_ses02_clas.save(f"{save_epo_dir}/{sub}_clas_ses-02.fif", overwrite=True)
raw_epo_ses02_phas.save(f"{save_epo_dir}/{sub}_phas_ses-02.fif", overwrite=True)


######################################################################
######################################################################
######################################################################
# NOTE 2025.09.02 11:36 -- THE FOLLOWING CODE HASN'T BEEN CHANGED
# epoch filt data
data_dict = filt_dict

def create_epoch_stim_filt (data_dict, event_dict, chan='TRIGGER', prestim=-0.2, poststim=0.8):
    epochs_dict = {}

    for fname, data in data_dict.items():
        events = mne.find_events(data, stim_channel=chan, min_duration=2 / data.info['sfreq'])
        epochs = mne.Epochs(
            data, 
            events, 
            tmin=prestim, 
            tmax=poststim, 
            event_id=event_dict,
            preload=True, 
            reject_by_annotation=False,
            baseline = [-0.2, 0],)
        epochs_dict[fname] = epochs   
    return epochs_dict

def create_epoch_imp_filt (data_dict, event_dict, chan='TRIGGER', prestim=-1.45, poststim=1):
    epochs_dict = {}

    for fname, data in data_dict.items():
        events = mne.find_events(data, stim_channel=chan, min_duration=2 / data.info['sfreq'])
        epochs = mne.Epochs(
            data, 
            events, 
            tmin=prestim, 
            tmax=poststim, 
            event_id=event_dict,
            preload=True, 
            reject_by_annotation=False,
            baseline = [-1.45, -1.25])
        epochs_dict[fname] = epochs    
    return epochs_dict

filt_epo_stim = create_epoch_stim_filt(data_dict, event_classify_dict)
filt_epo_imp = create_epoch_imp_filt(data_dict, event_phase_dict)

filt_epo_stim_ses01 = filt_epo_stim[f'{sub}_ses-01.fif']
filt_epo_stim_ses02 = filt_epo_stim[f'{sub}_ses-02.fif']
filt_epo_imp_ses01 = filt_epo_imp[f'{sub}_ses-01.fif']
filt_epo_imp_ses02 = filt_epo_imp[f'{sub}_ses-02.fif']

del data_dict
del filt_epo_stim
del filt_epo_imp

# reject the same epochs as  raw for filt
# drop_idx copied from external record file
drop_epo = [17, 45, 104, 134, 193, 224, 265, 402, 408, 414, 426, 451, 457, 468, 480, 526, 568, 586, 609, 625, 636, 659, 665, 683, 713, 738, 750, 761, 767, 779, 790, 832, 838, 844, 863, 875, 881, 917, 1023, 1041, 1078, 1090, 1102, 1166, 1184, 1202, 1208, 1267, 1291, 1320, 1326, 1354, 1360, 1366, 1477, 1508, 1526, 1532, 1538, 1544, 1585, 1621, 1663, 1736, 1748, 1760, 1766, 1806, 1848, 1872, 1918, 1925, 1931, 1937, 1955, 1985, 2003, 2009, 2086, 2133, 2141, 2147, 2153, 2183, 2224, 2300, 2329, 2335, 2341, 2353, 2371, 2383, 2423, 2429, 2441, 2459, 2483, 2517, 2553, 2559]
trial_idx = filt_epo_imp_ses01.selection
drop_idx = [i for i, t in enumerate(trial_idx) if t in drop_epo]
filt_epo_imp_ses01.drop(drop_idx)

######################################################################
# save raw epochs
save_epo_dir = '/Users/jiachenyao/Desktop/PhaseCode/data/raw_epo'

raw_epo_stim_ses01.save(f"{save_epo_dir}/{sub}_stim_ses-01.fif", overwrite=True)
raw_epo_stim_ses02.save(f"{save_epo_dir}/{sub}_stim_ses-02.fif", overwrite=True)

raw_epo_imp_ses01.save(f"{save_epo_dir}/{sub}_imp_ses-01.fif", overwrite=True)
raw_epo_imp_ses02.save(f"{save_epo_dir}/{sub}_imp_ses-02.fif", overwrite=True)

# save filt epochs
save_epo_dir = '/Users/jiachenyao/Desktop/PhaseCode/data/filt_epo'

filt_epo_stim_ses01.save(f"{save_epo_dir}/sub-01_stim_ses-01.fif", overwrite=True)
filt_epo_stim_ses02.save(f"{save_epo_dir}/sub-01_stim_ses-02.fif", overwrite=True)

filt_epo_imp_ses01.save(f"{save_epo_dir}/sub-01_imp_ses-01.fif", overwrite=True)
filt_epo_imp_ses02.save(f"{save_epo_dir}/sub-01_imp_ses-02.fif", overwrite=True)


'''
np.save(f"{save_epo_dir}/epo_cla_enc_ses01.npy", epo_cla_enc_ses01.get_data())
np.save(f"{save_epo_dir}/epo_cla_enc_ses01.npy", epo_cla_dec_ses01.get_data())

np.save
np.save(())

np.save(f"{save_epo_dir}/epo_pha_ses01.npy", epo_pha_ses01.get_data())
np.save(f"{save_epo_dir}/epo_pha_ses02.npy", epo_pha_ses02.get_data())
'''