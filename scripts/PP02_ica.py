'''Apply ICA

AUTHOR: Jiachen Yao <jasonyao0703[at]gmail.com>
LICENCE: BSD 3-clause

'''
globals().clear()

import mne
from mne.preprocessing import ICA, corrmap, create_eog_epochs
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

filt_dir = '/Users/jiachenyao/Desktop/PhaseCode/data/filt_firstcheck'
raw_dir =  '/Users/jiachenyao/Desktop/PhaseCode/data/raw_firstcheck'
sub = 'sub-02'

### loop through all sessions and all segments
def load_and_print_data(file_names, data_dir):
    data_dict = {fname: mne.io.read_raw_fif(op.join(data_dir, fname), preload=True, verbose=True) for fname in file_names}
    for fname, data in data_dict.items():
        print(f"\nLoaded: {fname}")
        print(data.info)
    return data_dict

file_names = [f"{sub}_ses-{ses:02d}.fif" for ses in range(1, 3)]

filt_dict = load_and_print_data(file_names, filt_dir)
raw_dict = load_and_print_data(file_names, raw_dir)

######################################################################
### separate data per session
montage = mne.channels.make_standard_montage("easycap-M1") 

raw_ses01 = raw_dict[f'{sub}_ses-01.fif'].set_montage(montage)
raw_ses02 = raw_dict[f'{sub}_ses-02.fif'].set_montage(montage)

filt_ses01 = filt_dict[f'{sub}_ses-01.fif'].set_montage(montage)
filt_ses02 = filt_dict[f'{sub}_ses-02.fif'].set_montage(montage)

del filt_dict
del raw_dict

### prepare filt data for ICA
def prepare_ica(data, l_freq=1.0, h_freq=None):
    return data.filter(l_freq=l_freq, h_freq=h_freq)

ica_filt_ses01 = prepare_ica(filt_ses01)
ica_filt_ses02 = prepare_ica(filt_ses02)

######################################################################
### NOTE change target dataset accordingly
ica_data = ica_filt_ses02
filt_data = filt_ses02
raw_data = raw_ses02

### define ICA
ica = ICA(n_components=15,  max_iter="auto", random_state=97)

### apply ICA
ica.fit(ica_data, reject_by_annotation = True)
ica

### check ICA solution
explained_var_ratio = ica.get_explained_variance_ratio(ica_data)
for channel_type, ratio in explained_var_ratio.items():
    print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

### plot ICA topograph components
ica.plot_components(inst = filt_data)

### plot ICA time series on filt data
filt_data.load_data()
ica.plot_sources(filt_data, show_scrollbars=True)
plt.show()

######################################################################
# confirm ICA components to reject
ica.exclude = [0,1,3,9,12]
reconst_raw = raw_data.copy()
reconst_filt = filt_data.copy()

# reject components and reconstruct data
ica.apply(reconst_raw)
ica.apply(reconst_filt)

######################################################################
# save ICA-repaired data
raw_save_dir = '/Users/jiachenyao/Desktop/PhaseCode/data/raw_ica_reconst'
filt_save_dir = '/Users/jiachenyao/Desktop/PhaseCode/data/filt_ica_reconst'
file_save_name = f'{sub}_ses-02.fif' # NOTE change sub and ses accordingly

reconst_raw.save(op.join(raw_save_dir, file_save_name), overwrite=True)
reconst_filt.save(op.join(filt_save_dir, file_save_name), overwrite=True)

del reconst_raw
del reconst_filt
del ica
