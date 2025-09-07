'''Compute time-frequency representation and erp on sensor space

'''
globals().clear()

import mne
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

epo_dir = '/Users/jiachenyao/Desktop/PhaseCode/data/raw_epo'

def load_and_print_data(file_names, data_dir):
    data_dict = {fname: mne.read_epochs(op.join(data_dir, fname), preload=True, verbose=True) for fname in file_names}
    for fname, data in data_dict.items():
        print(f"\nLoaded: {fname}")
        print(data.info)
    return data_dict

fnames_stim = [f"sub-01_stim_ses-{ses:02d}.fif" for ses in range(1, 3)]
fnames_imp = [f"sub-01_imp_ses-{ses:02d}.fif" for ses in range(1, 3)]

epo_stim = load_and_print_data(fnames_stim, epo_dir)
epo_imp = load_and_print_data(fnames_imp, epo_dir)

epo_stim_ses01 = epo_stim['sub-01_stim_ses-01.fif']
epo_imp_ses01 = epo_imp['sub-01_imp_ses-01.fif']

epo_stim_ses02 = epo_stim['sub-01_stim_ses-02.fif']
epo_imp_ses02 = epo_imp['sub-01_imp_ses-02.fif']

######################################################################
### pick channels for analysis
test_chans = ['C6', 'CP6', 'P6',
              'FC2', 'C2', 'CP2', 'P2',
              'FC4', 'C4', 'CP4', 'PO4',
              'FCz', 'Cz', 'CPz', 'Pz', 'POz',
              'FC1', 'C1', 'CP1', 'P1',
              'FC3', 'C3', 'CP3', 'P3', 'PO3',
              'C5', 'CP5', 'P5',]

epo_stim_ses01.pick_channels(test_chans)
epo_stim_ses02.pick_channels(test_chans)
epo_imp_ses01.pick_channels(test_chans)
epo_imp_ses02.pick_channels(test_chans)

### concatenate epoch across sessions (stimulus and phase)
epo_stim = mne.concatenate_epochs([epo_stim_ses01, epo_stim_ses02])
epo_pha = mne.concatenate_epochs([epo_imp_ses01, epo_imp_ses02])

######################################################################
# define foi
def compute_tfr_morlet(epochs, 
                       fmin=1, fmax=35, num_freqs=8, cycle_ratio=2.0, decim=3, average=True):

    freqs = np.logspace(*np.log10([fmin, fmax]), num=num_freqs)
    n_cycles = freqs / cycle_ratio 

    power, itc = epochs.compute_tfr(
        method='morlet', 
        freqs=freqs, 
        n_cycles=n_cycles, 
        average=average, 
        return_itc=True, 
        decim=decim,
    )

    return power, itc

# compute trf
power, itc = compute_tfr_morlet(epo_stim)

######################################################################
### compute event-spcific trf
adj_event = ['w1_T-B-AN', 'w2_T-B-NA']
n_event = ['w2_T-B-AN', 'w1_T-B-NA']

epo_adj = epo_stim[adj_event]
epo_n = epo_stim[n_event]

power_adj, itc_adj = compute_tfr_morlet(epo_adj)
power_n, itc_n = compute_tfr_morlet(epo_n)

### plot
power_adj.plot_joint(
    baseline=(-0.2, 0), mode="mean", tmin=-0.2, tmax=0.8,
)

power_n.plot_joint(
    baseline=(-0.2, 0), mode="mean", tmin=-0.2, tmax=0.8,
)

### compute the difference
power_diff = power_adj - power_n

power_diff.plot_joint(
    baseline=(-0.2, 0), mode="mean", tmin=-0.2, tmax=0.8, timefreqs=None
)

######################################################################
### compute erp
erp_adj = epo_adj.average()
erp_n = epo_n.average()

erp_adj.plot_topomap(times=[-0.2, 0, 0.2, 0.4, 0.6, 0.8], average=0.05)
erp_adj.plot_joint()
plt.show()
erp_n.plot_topomap(times=[-0.2, 0, 0.2, 0.4, 0.6, 0.8], average=0.05)
erp_n.plot_joint()
plt.show()

evokeds = dict(adjective=erp_adj, noun=erp_n)
mne.viz.plot_compare_evokeds(evokeds, picks=test_chans, combine="mean")

evokeds = dict(
    adjective=list(epo_adj.iter_evoked()), 
    noun=list(epo_n.iter_evoked()))
mne.viz.plot_compare_evokeds(evokeds, picks=test_chans, combine="mean")

adj_minus_n = mne.combine_evoked([erp_adj, erp_n], weights=[1, -1])
adj_minus_n.plot_joint()
mne.viz.plot_compare_evokeds(adj_minus_n, picks=test_chans, combine="mean")