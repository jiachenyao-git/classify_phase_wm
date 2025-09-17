'''Preprocess EEG signal. Check signal quality and mark bad channels

AUTHOR: Jiachen Yao <jasonyao0703[at]gmail.com>
LICENCE: BSD 3-clause
'''
globals().clear()

import mne
import os.path as op
import matplotlib.pyplot as plt
from mne.channels import read_custom_montage
from collections import defaultdict

### define data directory
sub = 'sub-03'
sensor_dir = f'/Users/jiachenyao/Desktop/PhaseCode/data/{sub}/eeg'

### read in raw data
### loop through all sessions and all segments
file_names = [f"{sub}_ses-{ses:02d}_seg-{seg:02d}.fif" for ses in range(1, 3) for seg in range(1, 3)]
raw_dict = {fname: mne.io.read_raw_fif(op.join(sensor_dir, fname), preload=True, verbose=True) for fname in file_names}

for fname, raw in raw_dict.items():
    print(f"\nLoaded: {fname}")
    print(raw.info)
    print(raw._data.min(), raw._data.max())

######################################################################
### convert unit into volts and update channel types for all raw files
for fname, raw in raw_dict.items():
    print(f"\nProcessing {fname}...")

    ### convert unit to volts
    for i, ch_name in enumerate(raw.ch_names):
        if ch_name != "TRIGGER":  # skip the TRIGGER channel
            raw._data[i, :] /= 1e6

    ### update channel units
    for ch in raw.info['chs']:
        if ch['ch_name'] in raw.info['ch_names']: 
            ch['unit'] = 107  

    ### print updated channel units
    for ch in raw.info['chs']:
        print(f"Channel: {ch['ch_name']}, Unit: {ch['unit']}")

    ### define channel type mapping
    channel_mapping = {
        'EOGH': 'eog',  # horizontal EOG
        'EOGV': 'eog',  # vertical EOG
        'TRIGGER': 'stim'  # triggers
    }

    ### assign EEG type to all other channels
    for ch_name in raw.info['ch_names']:
        if ch_name not in channel_mapping:  
            channel_mapping[ch_name] = 'eeg'

    ### update channel types
    raw.set_channel_types(channel_mapping)

    ### print updated channel types
    for ch in raw.info['chs']:
        print(f"Channel: {ch['ch_name']}, Type: {ch['kind']}")

######################################################################
### concatenate data across segments per session
raw_ses_dict = defaultdict(list)
raw_concat_dict = {}

for fname, raw in raw_dict.items():
    session_key = '_'.join(fname.split('_')[:2]) # extract 'sub-xx_ses-xx'
    raw_ses_dict[session_key].append(raw)

for session, raws in raw_ses_dict.items():
    raw_concat_dict[session] = mne.concatenate_raws(raws)

######################################################################
### plot psd and raw data
### here mark the bad channels on raw data
for fname, raw in raw_concat_dict.items():
    print(f"\nPlotting for {fname}...")

    # plot power spectral density
    spectrum = raw.compute_psd()
    spectrum.plot(average=False, picks="eeg", exclude=["Fz"], amplitude=False)
    plt.title(f"Power Spectral Density - {fname}")
    plt.show()

    # plot raw data
    raw.plot(title=f"Raw Data - {fname}")
    plt.show()

######################################################################
### drop bad channels in raw data for each session separately
### bad channels
bad_chan_ses01 = ['T7', 'T8', 'Oz', 'AF7', 'FCz']
bad_chan_ses02 = ['T7', 'T8', 'FT8', 'AF4', 'AF8']
rem_chan = ['A1', 'A2']

raw_concat_dict[f'{sub}_ses-01'].drop_channels(bad_chan_ses01 + rem_chan)
raw_concat_dict[f'{sub}_ses-02'].drop_channels(bad_chan_ses02 + rem_chan)

######################################################################
### check trigger
for fname, raw in raw_concat_dict.items():
    events = mne.find_events(raw, stim_channel='TRIGGER', min_duration=2 / raw.info['sfreq'])
    print(f"\nEvents in {fname}:")
    print(events)

### label trigger as stimulus_TrialType-Condition-Order
event_dict = {
    'fix_T-B-AN': 11,
    'w1_T-B-AN': 12,
    'w2_T-B-AN': 13,
    'imp_T-B-AN': 14,
    'p_T-B-AN': 15,
    'miss_T-B-AN': 16,

    'fix_T-B-NA': 21,
    'w1_T-B-NA': 22,
    'w2_T-B-NA': 23,
    'imp_T-B-NA': 24,
    'p_T-B-NA': 25,
    'miss_T-B-NA': 26,

    'fix_T-NB-AN': 31,
    'w1_T-NB-AN': 32,
    'w2_T-NB-AN': 33,
    'imp_T-NB-AN': 34,
    'p_T-NB-AN': 35,
    'miss_T-NB-AN': 36,

    'fix_T-NB-NA': 41,
    'w1_T-NB-NA': 42,
    'w2_T-NB-NA': 43,
    'imp_T-NB-NA': 44,
    'p_T-NB-NA': 45,
    'miss_T-NB-NA': 46,

    'fix_T-C-AN': 51,
    'w1_T-C-AN': 52,
    'w2_T-C-AN': 53,
    'imp_T-C-AN': 54,
    'p_T-C-AN': 55,
    'miss_T-C-AN': 56,

    'fix_T-C-NA': 61,
    'w1_T-C-NA': 62,
    'w2_T-C-NA': 63,
    'imp_T-C-NA': 64,
    'p_T-C-NA': 65,
    'miss_T-C-NA': 66,

    'fix_C-B-AN': 71,
    'w1_C-B-AN': 72,
    'w2_C-B-AN': 73,
    'p_C-B-AN': 75,
    'miss_C-B-AN': 76,

    'fix_C-B-NA': 81,
    'w1_C-B-NA': 82,
    'w2_C-B-NA': 83,
    'p_C-B-NA': 85,
    'miss_C-B-NA': 86,

    'fix_C-NB-AN': 91,
    'w1_C-NB-AN': 92,
    'w2_C-NB-AN': 93,
    'p_C-NB-AN': 95,
    'miss_C-NB-AN': 96,

    'fix_C-NB-NA': 101,
    'w1_C-NB-NA': 102,
    'w2_C-NB-NA': 103,
    'p_C-NB-NA': 105,
    'miss_C-NB-NA': 106,

    'fix_C-C-AN': 111,
    'w1_C-C-AN': 112,
    'w2_C-C-AN': 113,
    'p_C-C-AN': 115,
    'miss_C-C-AN': 116,

    'fix_C-C-NA': 121,
    'w1_C-C-NA': 122,
    'w2_C-C-NA': 123,
    'p_C-C-NA': 125,
    'miss_C-C-NA': 126,

    'correct_match': 40,
    'correct_nomatch': 50,
    'wrong_match': 4,
    'wrong_nomatch': 5
}

exclude_event_code = {1, 2, 3, 4, 5, 6}

### plot event codes
for fname, raw in raw_concat_dict.items():
    events = mne.find_events(raw, min_duration=0, shortest_event=1)
    event_codes = set(events[:, 2])
    main_event_codes = event_codes - exclude_event_code

    sub_event_dict = {key: val for key, val in event_dict.items() if val in main_event_codes}

    try:
        fig = mne.viz.plot_events(
            events, sfreq = raw.info['sfreq'], first_samp = raw.first_samp, event_id = sub_event_dict
        )
    except Exception as e:
        print(f'Failed to plot events for {fname}: {e}')

######################################################################
### set montage and rename A1 and A2 to match the montage
montage = mne.channels.make_standard_montage("easycap-M1") 

### re-ref to common average reference
raw_avgref = {}

for fname, raw in raw_concat_dict.items():
    raw.set_montage(montage)
    
    eeg_data = raw.copy().pick(picks = 'eeg')
    eog_data = raw.copy().pick(picks='eog')
    stim_data = raw.copy().pick(picks='stim')

    eeg_data.set_eeg_reference(ref_channels='average') 
    
    eeg_data.add_channels([eog_data, stim_data], force_update_info=True) 
    raw_avgref[fname] = eeg_data   

######################################################################
### filter
### keep a version of raw data with only notch filters
### keep a version of filtered data with high-/low-pass and notch filters
raw_np_dict = {}
filt_dict = {}

for fname, raw in raw_avgref.items():
    raw_copy = raw.copy()
    raw_copy.notch_filter([50, 100, 150, 200], picks='eeg', fir_design='firwin')
    raw_np_dict[fname] = raw_copy.copy()

    raw_copy.filter(l_freq=0.1, h_freq=40, picks='eeg', fir_design='firwin')
    filt_dict[fname] = raw_copy

### check bad channels and bad span on filt data 
for fname, filt in filt_dict.items():
    spectrum = filt.compute_psd()
    spectrum.plot(average=False, picks="eeg", exclude=["Fz"], amplitude=False)
    plt.title(f"Power Spectral Density - {fname}")
    plt.show()

for fname, filt in filt_dict.items():
    filt.plot(title=f"Filtered Data - {fname}", scalings={'eeg': 20e-6})
    plt.show()


######################################################################
# save raw and filt
raw_output_dir = '/Users/jiachenyao/Desktop/PhaseCode/data/raw_firstcheck'
filt_output_dir = '/Users/jiachenyao/Desktop/PhaseCode/data/filt_firstcheck'

for fname, raw in raw_np_dict.items():
    output_path = f"{raw_output_dir}/{fname}.fif"
    raw.save(output_path, fmt = 'single', overwrite=True)
    print(f"Saved cleaned raw data for {fname} to {output_path}")

for fname, filt in filt_dict.items():
    output_path = f"{filt_output_dir}/{fname}.fif"
    filt.save(output_path, fmt = 'single', overwrite=True)
    print(f"Saved cleaned raw data for {fname} to {output_path}")
