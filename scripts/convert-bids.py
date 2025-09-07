'''Convert data from custom format to BIDS format

'''
globals().clear()
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.io.constants import FIFF
from mne_bids import BIDSPath, write_raw_bids

# define data directory
sub = 'sub-01'
data_dir = f'/Users/jiachenyao/Desktop/PhaseCode/data/{sub}/eeg/'

### read in raw data
### loop through all sessions and all segments
### NOTE 'preload=True' because channel names, types and data unit need to be changed
file_names = [f'{sub}_ses-{ses:02d}_seg-{seg:02d}.fif' for ses in range(1, 3) for seg in range(1, 3)]
raw_dict = {fname: mne.io.read_raw_fif(op.join(data_dir, fname), preload=True, verbose=True) for fname in file_names}

### specify power line noise
for fnames, raw in raw_dict.items():
    raw.info["line_freq"] = 50

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
### make montage with electrode position and head shape
dig_dir = f'/Users/jiachenyao/Desktop/PhaseCode/data/{sub}'
p3d_paths = {
    'ses-01': f'{dig_dir}/dig/{sub}_ses-01_elec.p3d',
    'ses-02': f'{dig_dir}/dig/{sub}_ses-02_elec.p3d',
}
sha_paths = {
    'ses-01': f'{dig_dir}/dig/{sub}_ses-01_head.sha',
    'ses-02': f'{dig_dir}/dig/{sub}_ses-02_head.sha',
}

### read electrode position
label_dict = {'FPZ': 'Fpz','FP1': 'Fp1', 'FP2':'Fp2', 'AFZ':'AFz', 'FZ':'Fz', 'FCZ':'FCz','CZ': 'Cz',
              'CPZ':'CPz', 'Cp5':'CP5', 'PZ':'Pz','POZ': 'POz','OZ': 'Oz'}

def read_p3d(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    nas = []
    lpa = []
    rpa = []
    
    coords = {}
    for line in lines[1:]:
        parts = line.split()
        label = parts[0]
        ### mne format [-anterior/+posterior, +left/-right, +superior/-inferior]
        ### p3d format [-left/+right, +anterior/-posterior, +superior/-inferior]
        ### so mne.[x,y,z]=p3d.[-y,-x,z]
        if label.startswith("NAS"):
            x, y, z = map(float, parts[1:])
            nas.append(np.array([-y,-x,z]))
            
        elif label.startswith("LPA"):
            x, y, z = map(float, parts[1:])
            lpa.append(np.array([-y,-x,z]))

        elif label.startswith("RPA"):
            x, y, z = map(float, parts[1:])
            rpa.append(np.array([-y,-x,z]))

        else:
            if label in label_dict.keys():
                label = label_dict[label]
               
            x, y, z = map(float, parts[1:])
            
            coords[label] = np.array([-y,-x,z])
       
    nas = np.mean(nas, axis = 0)*0.01   # convert to meters
    lpa = np.mean(lpa, axis = 0)*0.01
    rpa = np.mean(rpa, axis = 0)*0.01
    coords = {key: val * 0.01 for key, val in coords.items()}
   
    return coords, nas, lpa, rpa

### read head shape
def read_sha_headshape(sha_path):
    headshape_points = []

    with open(sha_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue  # skip comments and blank lines
            try:
                x, y, z = map(float, line.split())
                headshape_points.append([x, y, z])
            except ValueError:
                continue 

    return np.array(headshape_points) * 0.01 # convert to meters

### make montage from cooridnates
def create_montage_from_coords(coords, nas, lpa, rpa, hsp):
    ch_pos = {label: np.array(pos) for label, pos in coords.items()}

    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,  
        nasion=nas, 
        lpa=lpa, 
        rpa=rpa, 
        coord_frame = 'head',
        hsp=hsp,
        )
    
    return montage

montage_tmp = {}
for fname, raw in raw_dict.items():
    if 'ses-01' in fname:
        ses_label = 'ses-01'
    elif 'ses-02' in fname:
        ses_label = 'ses-02'
    else:
        print(f"Session not recognized in {fname}, skipping...")
        continue
    
    if 'seg-01' in fname:
        seg_label = 'seg-01'
    elif 'seg-02' in fname:
        seg_label = 'ses-02'
    else:
        print(f"Segment not recognized in {fname}, skipping...")
        continue

    if ses_label not in montage_tmp:
        coords, nas, lpa, rpa = read_p3d(p3d_paths[ses_label])
        headshape = read_sha_headshape(sha_paths[ses_label])
        montage = create_montage_from_coords(coords, nas, lpa, rpa, headshape)
        montage_tmp[ses_label] = montage
    else:
        montage = montage_tmp[ses_label]

    raw.set_montage(montage)

    fig = montage.plot()
    plt.title(f'Sensor positions - {ses_label}_{seg_label}')
    plt.show()

    print(f'Montage set for {fname}')

for ses_label, montage in montage_tmp.items():
    fig = montage.plot(kind="3d", show=False)  # 3D
    fig = fig.gca().view_init(azim=70, elev=15)
    plt.title(f'Sensor positions - {ses_label}')
    plt.show()

######################################################################
print(write_raw_bids.__doc__)
print(raw.annotations)

event_id = {
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

### define a task name and a directory where to save the data to
task = 'WorkingMemory'
bids_root = f'/Users/jiachenyao/Desktop/PhaseCode/data_bids'

for fname, raw in raw_dict.items():
    bids_path = BIDSPath(subject=sub, task=task, root=bids_root)
    write_raw_bids(raw, bids_path, overwrite=True)














