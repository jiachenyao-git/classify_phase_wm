'''Demonstration script for classification  rational

AUTHOR: Jiachen Yao <jasonyao0703[at]gmail.com>
LICENCE: BSD 3-clause

'''
globals().clear()

import mne
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
'''
epo_dir = '/Users/jiachenyao/Desktop/PhaseCode/data/raw_epo'

def load_and_print_data(file_names, data_dir):
    data_dict = {fname: mne.read_epochs(op.join(data_dir, fname), preload=True, verbose=True) for fname in file_names}
    for fname, data in data_dict.items():
        print(f"\nLoaded: {fname}")
        print(data.info)
    return data_dict

fnames_cla = [f"epo_cla_ses{ses:02d}_long.fif" for ses in range(1, 3)]
fnames_pha = [f"epo_pha_ses{ses:02d}.fif" for ses in range(1, 3)]

epo_cla = load_and_print_data(fnames_cla, epo_dir)
epo_pha = load_and_print_data(fnames_pha, epo_dir)

epo_cla_ses01 = epo_cla['epo_cla_ses01_long.fif']
#epo_cla_ses02 = epo_cla['epo_cla_ses02_long.fif']
epo_pha_ses01 = epo_pha['epo_pha_ses01.fif']
#epo_pha_ses02 = epo_pha['epo_pha_ses02.fif']
'''

######################################################################
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

### cut stim epochs to post-stim window
epo_stim = epo_stim.crop(tmin=0, tmax=0.8)

######################################################################
### select events
nchan = epo_stim.pick(picks='eeg').info['nchan']

adj_event = ['w1_T-B-AN', 'w2_T-B-NA', 'w1_T-NB-AN', 'w2_T-NB-NA']
n_event = ['w2_T-B-AN', 'w1_T-B-NA', 'w2_T-NB-AN', 'w1_T-NB-NA']
'''
adj_event_b = ['w1_T-B-AN', 'w2_T-B-NA', ]
adj_event_nb = ['w1_T-NB-AN', 'w2_T-NB-NA',]
n_event_b = ['w2_T-B-AN', 'w1_T-B-NA', ]
n_event_nb = ['w2_T-NB-AN', 'w1_T-NB-NA',]
imp_event_b = ['imp_T-B-AN', 'imp_T-B-NA', ]
imp_event_nb = ['imp_T-NB-AN','imp_T-NB-NA',]
'''
epo_adj = epo_stim[adj_event]
epo_n = epo_stim[n_event]
'''
epo_adj_nb = epo_stim[adj_event_nb]
epo_n_nb = epo_stim[n_event_nb]
epo_imp_nb = epo_stim[imp_event_nb]

epo_adj_b = epo_stim[adj_event_b]
epo_n_b = epo_stim[n_event_b]
epo_imp_b = epo_stim[imp_event_b]

### match trials for epo_pha and epo_imp
epo_comm_nb = np.intersect1d(epo_pha.selection, epo_imp_nb.selection)
epo_comm_b = np.intersect1d(epo_pha.selection, epo_imp_b.selection)

### for phase extraction
epo_pha_nb = epo_pha[np.isin(epo_pha.selection, epo_comm_nb)]
epo_pha_b = epo_pha[np.isin(epo_pha.selection, epo_comm_b)]

### for decoding
epo_imp_nb = epo_imp_nb[np.isin(epo_imp_nb.selection, epo_comm_nb)]
epo_imp_b = epo_imp_b[np.isin(epo_imp_b.selection, epo_comm_b)]

### assert same n_trials for data for phase and decoding
assert epo_pha_nb.get_data()[:,:,0].shape == epo_imp_nb.get_data()[:,:,0].shape
assert epo_pha_b.get_data()[:,:,0].shape == epo_imp_b.get_data()[:,:,0].shape

del epo_stim
'''
######################################################################
'''classify_data = epo_cla_ses01
nchan = classify_data.pick(picks='eeg').info['nchan']

adj_event = ['w1_T-B-AN', 'w2_T-B-NA', 'w1_T-NB-AN', 'w2_T-NB-NA']
n_event = ['w2_T-B-AN', 'w1_T-B-NA', 'w2_T-NB-AN', 'w1_T-NB-NA']
#imp_event = ['imp_T-B_AN', 'imp_T-B-NA', 'imp_T-NB-AN','imp_T-NB-NA']

epo_adj = classify_data[adj_event]
epo_n = classify_data[n_event]
#epo_imp = classify_data[imp_event]
'''
ranges = [
          ('erp', 1, 40),
          ('delta', 1, 4),
          ('theta', 4, 8),
          ('alpha', 8, 12),
          ('beta', 12, 30),
          ('gamma', 30, 60),
]

out_folder = 'plots/2-sessions-erp'

import os
os.makedirs(out_folder, exist_ok=True)

for case, start, stop in ranges:
    '''    
    foi = np.arange(start, stop+1, 1)
    cycles = foi / 5

    tfr_adj = epo_adj.compute_tfr(method="morlet", picks='eeg', freqs=foi, n_cycles=cycles, decim=3)
    tfr_n = epo_n.compute_tfr(method="morlet", picks='eeg', freqs=foi, n_cycles=cycles, decim=3)

    adjs = np.mean(tfr_adj.get_data(), axis=2)
    ns = np.mean(tfr_n.get_data(), axis=2)
    '''
       
    adjs = epo_adj.filter(start, stop).pick('eeg').get_data()
    ns = epo_n.filter(start, stop).pick('eeg').get_data()
    #impulse = epo_imp.filter(start, stop).pick('eeg').get_data()
    
    folds = 50
    ### checking timepoints are matched
    assert adjs.shape[2] == ns.shape[2]
    timepoints = adjs.shape[2]
    ### checking electrodes are matched
    assert adjs.shape[1] == ns.shape[1]
    ### collecting variables
    trials_adj = adjs.shape[0]
    n_train_adj = int((trials_adj)*0.8)
    n_test_adj = int((trials_adj)*0.2)
    trials_n = ns.shape[0]
    n_train_n = int((trials_n)*0.8)
    n_test_n = int((trials_n)*0.2)
    train_samples = min(n_train_adj, n_train_n)
    test_samples = min(n_test_adj, n_test_n)

    import random
    import scipy
    import sklearn

    from scipy import stats
    from sklearn import linear_model
    from tqdm import tqdm

    results = list()

    for f in tqdm(range(folds)):
        train_idxs_adjs = random.sample(range(len(adjs)), k=train_samples)
        test_idxs_adjs = [i for i in range(len(adjs)) if i not in train_idxs_adjs]
        train_idxs_ns = random.sample(range(len(ns)), k=train_samples)
        test_idxs_ns = [i for i in range(len(ns)) if i not in train_idxs_ns]
        assert len(train_idxs_ns) == len(train_idxs_adjs)
        assert train_idxs_ns != train_idxs_adjs
        assert test_idxs_ns != test_idxs_adjs
        fold_train = list(adjs[train_idxs_adjs, :, :]) + list(ns[train_idxs_ns, :, :])
        ### labels
        ### adj is 1, noun is 0
        labels_train = [1 for _ in range(len(train_idxs_adjs))] + [0 for _ in range(len(train_idxs_adjs))]
        fold_test = list(adjs[test_idxs_adjs, :, :]) + list(ns[test_idxs_ns, :, :])
        labels_test = [1 for _ in range(len(test_idxs_adjs))] + [0 for _ in range(len(test_idxs_ns))]
        #fold_test_impulse = list(impulse)       
        fold_results = list()
        #preds_impulse_list = list()
        for t in tqdm(range(timepoints)):
            t_train = np.array([erp[:, t] for erp in fold_train])
            t_test = np.array([erp[:, t] for erp in fold_test])
            #t_test_impulse = np.array([erp[:, t] for erp in fold_test_impulse])
            ### samples x electrodes
            assert len(t_train.shape) == 2
            assert len(t_test.shape) == 2   
            #assert len(t_test_impulse.shape) == 2        
            ### checking electrodes
            assert t_train.shape[1] == nchan
            assert t_test.shape[1] == nchan
            #assert t_test_impulse.shape[1] == nchan
            model = linear_model.RidgeClassifierCV(alphas=(0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
            model.fit(t_train, labels_train)
            preds = model.predict(t_test)
            evaluation = [1 if pred==real else 0 for pred, real in zip(preds, labels_test)]
            acc = sum(evaluation) / len(evaluation)
            assert acc > 0. and acc < 1.
            fold_results.append(acc)

            #preds_impulse = model.predict(t_test_impulse)
            #preds_impulse_list.append(preds_impulse)

        assert len(fold_results) == timepoints
        results.append(fold_results)
    results = np.array(results)
    assert results.shape == (folds, timepoints)

    ###
    #prop_adjective_impulse = np.mean(np.array(preds_impulse_list) == 1, axis=1)

    fig, ax = plt.subplots(
        constrained_layout=True
        )
    ax.hlines(
            y=0.5, 
            xmin=min(epo_n.times),
            xmax=max(epo_n.times), 
            color='black',
            )
    ax.vlines(
            x=0., 
            ymin=.45,
            ymax=.55, 
            color='black',
            )
    ax.plot(
            #tfr_n.times,
            epo_n.times, 
            np.average(results, axis=0), color='orange'
            #epo_n.times,
            #prop_adjective_impulse, 
            )
    ax.fill_between(
                #tfr_n.times,
                epo_n.times, 
                np.average(results, axis=0)-scipy.stats.sem(results, axis=0), 
                np.average(results, axis=0)+scipy.stats.sem(results, axis=0),
                color='bisque'
                )
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Proportion of Correct Predictions")
    ax.set_title(f"{case}")

    plt.savefig(os.path.join(out_folder, '{}.jpg'.format(case)))

######################################################################
# NOTE increase epochs to 800 to 1000ms
# NOTE try with clusters of electrodes
# NOTE try different filtering methods (e.g., morlet)
# NOTE try testing on the actual post-impulse period
