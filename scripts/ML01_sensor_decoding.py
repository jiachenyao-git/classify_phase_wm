'''Classification category

'''
globals().clear()

import mne
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

epo_dir = '/Users/jiachenyao/Desktop/PhaseCode/data/raw_epo'
sub = 'sub-01'

def load_and_print_data(file_names, data_dir):
    data_dict = {fname: mne.read_epochs(op.join(data_dir, fname), preload=True, verbose=True) for fname in file_names}
    for fname, data in data_dict.items():
        print(f"\nLoaded: {fname}")
        print(data.info)
    return data_dict

fnames_stim = [f"{sub}_stim_ses-{ses:02d}.fif" for ses in range(1, 3)]
fnames_imp = [f"{sub}_imp_ses-{ses:02d}.fif" for ses in range(1, 3)]

epo_stim = load_and_print_data(fnames_stim, epo_dir)
epo_imp = load_and_print_data(fnames_imp, epo_dir)

epo_stim_ses01 = epo_stim[f'{sub}_stim_ses-01.fif']
epo_imp_ses01 = epo_imp[f'{sub}_imp_ses-01.fif']

epo_stim_ses02 = epo_stim[f'{sub}_stim_ses-02.fif']
epo_imp_ses02 = epo_imp[f'{sub}_imp_ses-02.fif']

######################################################################
### pick channels for analysis
test_chans = ['C6', 'CP6',
              'FC2', 'C2', 'CP2', 'P2',
              'FC4', 'C4', 'CP4', 
              'Cz', 'CPz', 'Pz', 'POz',
              'FC3', 'C3', 'CP3', 'P3',
              'FC1', 'C1', 'CP1', 'P1',
              'C5', 'CP5', 'P5',]

epo_stim_ses01.pick(test_chans)
epo_stim_ses02.pick(test_chans)
epo_imp_ses01.pick(test_chans)
epo_imp_ses02.pick(test_chans)

### concatenate epoch across sessions (stimulus and phase)
epo_stim = mne.concatenate_epochs([epo_stim_ses01, epo_stim_ses02])
epo_pha = mne.concatenate_epochs([epo_imp_ses01, epo_imp_ses02])

### cut stim epochs to post-stim window
epo_stim = epo_stim.crop(tmin=0, tmax=0.8)

######################################################################
### select events
nchan = epo_stim.pick(picks='eeg').info['nchan']

adj_event = ['w1_T-B-AN', 'w2_T-B-NA', 'w1_T-NB-AN', 'w2_T-NB-NA',]
adj_event_b = ['w1_T-B-AN', 'w2_T-B-NA', ]
adj_event_nb = ['w1_T-NB-AN', 'w2_T-NB-NA',]
n_event = ['w2_T-B-AN', 'w1_T-B-NA', 'w2_T-NB-AN', 'w1_T-NB-NA',]
n_event_b = ['w2_T-B-AN', 'w1_T-B-NA', ]
n_event_nb = ['w2_T-NB-AN', 'w1_T-NB-NA',]
imp_event = ['imp_T-B-AN', 'imp_T-B-NA', 'imp_T-NB-AN','imp_T-NB-NA',]
imp_event_b = ['imp_T-B-AN', 'imp_T-B-NA', ]
imp_event_nb = ['imp_T-NB-AN','imp_T-NB-NA',]

epo_adj_nb = epo_stim[adj_event_nb]
epo_n_nb = epo_stim[n_event_nb]
epo_imp_nb = epo_stim[imp_event_nb]

epo_adj_b = epo_stim[adj_event_b]
epo_n_b = epo_stim[n_event_b]
epo_imp_b = epo_stim[imp_event_b]

epo_adj = epo_stim[adj_event]
epo_n = epo_stim[n_event]

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

######################################################################
### NOTE run classification for 'nb' and 'b' separately
adjs = epo_adj.pick('eeg').get_data()
ns = epo_n.pick('eeg').get_data()
imp_nb = epo_imp_nb.pick('eeg').get_data()
imp_b = epo_imp_b.pick('eeg').get_data()

######################################################################
folds = 100
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

results_acc = list()
results_w_label = list()
#results_w_conf = list()
results_imp_nb_lbl = list()
results_imp_b_lbl = list()
#results_imp_nb_conf = list()
#results_imp_b_conf = list()

results_w_proba = list()
results_imp_nb_proba = list()
results_imp_b_proba = list()

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

    fold_test_impulse_nb = list(imp_nb)
    fold_test_impulse_b = list(imp_b)       
    fold_results = list()
    fold_preds_w_label = list()
    #fold_preds_w_conf = list()
    fold_preds_imp_nb_lbl = list()
    fold_preds_imp_b_lbl = list()
    #fold_preds_imp_nb_conf = list()
    #fold_preds_imp_b_conf = list()

    fold_preds_w_proba = list()
    fold_preds_imp_nb_proba = list()
    fold_preds_imp_b_proba = list()

    for t in tqdm(range(timepoints)):
        t_train = np.array([erp[:, t] for erp in fold_train])
        t_test = np.array([erp[:, t] for erp in fold_test])
        t_test_impulse_nb = np.array([erp[:, t] for erp in fold_test_impulse_nb])
        t_test_impulse_b = np.array([erp[:, t] for erp in fold_test_impulse_b])
        ### samples x electrodes
        assert len(t_train.shape) == 2
        assert len(t_test.shape) == 2   
        assert len(t_test_impulse_nb.shape) == 2   
        assert len(t_test_impulse_b.shape) == 2     
        ### checking electrodes
        assert t_train.shape[1] == nchan
        assert t_test.shape[1] == nchan
        assert t_test_impulse_nb.shape[1] == nchan
        assert t_test_impulse_b.shape[1] == nchan
        '''
        model = linear_model.RidgeClassifierCV(
            alphas=(0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
        '''
        
        model = linear_model.LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],  
            cv=5,                     # 5-fold cross-validation
            penalty='l2',             # l2 regularization
            solver='liblinear',       # good for small-medium datasets, supports L2
            scoring='roc_auc',       # can be changed to 'roc_auc', 'neg_log_loss', etc.
            max_iter=1000,
            n_jobs=-1,                # use all cores
            random_state=42
        )
        model.fit(t_train, labels_train)
        preds = model.predict(t_test)
        evaluation = [1 if pred==real else 0 for pred, real in zip(preds, labels_test)]
        acc = sum(evaluation) / len(evaluation)
        assert acc > 0. and acc < 1.
        fold_results.append(acc)
        fold_preds_w_label.append(preds)
        #preds_w_conf = model.decision_function(t_test)
        #fold_preds_w_conf.append(preds_w_conf)

        preds_w_proba = model.predict_proba(t_test)
        fold_preds_w_proba.append(preds_w_proba)

        preds_imp_nb_lbl = model.predict(t_test_impulse_nb)
        fold_preds_imp_nb_lbl.append(preds_imp_nb_lbl)
        preds_imp_b_lbl = model.predict(t_test_impulse_b)
        fold_preds_imp_b_lbl.append(preds_imp_b_lbl)
        '''
        preds_imp_nb_conf = model.decision_function(t_test_impulse_nb)
        fold_preds_imp_nb_conf.append(preds_imp_nb_conf)
        preds_imp_b_conf = model.decision_function(t_test_impulse_b)
        fold_preds_imp_b_conf.append(preds_imp_b_conf)
        '''

        preds_imp_nb_proba = model.predict_proba(t_test_impulse_nb)
        fold_preds_imp_nb_proba.append(preds_imp_nb_proba)
        preds_imp_b_proba = model.predict_proba(t_test_impulse_b)
        fold_preds_imp_b_proba.append(preds_imp_b_proba)


    assert len(fold_results) == timepoints
    assert len(fold_preds_w_label) == timepoints
    assert len(fold_preds_imp_nb_lbl) == timepoints
    assert len(fold_preds_imp_b_lbl) == timepoints
    '''
    assert len(fold_preds_w_conf) == timepoints
    assert len(fold_preds_imp_nb_conf) == timepoints
    assert len(fold_preds_imp_b_conf) == timepoints
    '''
    assert len(fold_preds_w_proba) == timepoints
    assert len(fold_preds_imp_nb_proba) == timepoints
    assert len(fold_preds_imp_b_proba) == timepoints

    results_acc.append(fold_results)
    results_w_label.append(fold_preds_w_label)
    #results_w_conf.append(fold_preds_w_conf)
    results_imp_nb_lbl.append(fold_preds_imp_nb_lbl)
    results_imp_b_lbl.append(fold_preds_imp_b_lbl)
    #results_imp_nb_conf.append(fold_preds_imp_nb_conf)
    #results_imp_b_conf.append(fold_preds_imp_b_conf)

    results_w_proba.append(fold_preds_w_proba)
    results_imp_nb_proba.append(fold_preds_imp_nb_proba)
    results_imp_b_proba.append(fold_preds_imp_b_proba)

all_results = {
    "acc": np.array(results_acc),
    "w_label": np.array(results_w_label),
    #"w_conf": np.array(results_w_conf),
    "imp_nb_label": np.array(results_imp_nb_lbl),
    "imp_b_label": np.array(results_imp_b_lbl),
    #"imp_nb_conf": np.array(results_imp_nb_conf),
    #"imp_b_conf": np.array(results_imp_b_conf),
    "w_proba": np.array(results_w_proba),
    "imp_nb_proba": np.array(results_imp_nb_proba),
    "imp_b_proba": np.array(results_imp_b_proba),
}

assert all_results['acc'].shape == (folds, timepoints)
assert all_results['w_label'].shape == (folds, timepoints, len(labels_test))
#assert all_results['w_conf'].shape == (folds, timepoints, len(labels_test))
assert all_results['imp_nb_label'].shape == (folds, timepoints, len(imp_nb))
assert all_results['imp_b_label'].shape == (folds, timepoints, len(imp_b))
#assert all_results['imp_nb_conf'].shape == (folds, timepoints, len(imp_nb))
#assert all_results['imp_b_conf'].shape == (folds, timepoints, len(imp_b))

assert all_results['w_proba'].shape == (folds, timepoints, len(labels_test), 2)
assert all_results['imp_nb_proba'].shape == (folds, timepoints, len(imp_nb), 2)
assert all_results['imp_b_proba'].shape == (folds, timepoints, len(imp_b), 2)

######################################################################
### get decoding lable and confidence (or probability)
### non-binding condition
imp_lbl_nb = all_results['imp_nb_label'].copy()
#imp_conf_nb = all_results['imp_nb_conf'].copy()
imp_proba_nb = all_results['imp_nb_proba'].copy()
assert imp_lbl_nb.shape == (folds, timepoints, len(epo_imp_nb))
#assert imp_conf_nb.shape == (folds, timepoints, len(epo_imp_nb))
assert imp_proba_nb.shape == (folds, timepoints, len(epo_imp_nb), 2)

### binding condition
imp_lbl_b = all_results['imp_b_label'].copy()
#imp_conf_b = all_results['imp_b_conf'].copy()
imp_proba_b = all_results['imp_b_proba'].copy()
assert imp_lbl_b.shape == (folds, timepoints, len(epo_imp_b))
#assert imp_conf_b.shape == (folds, timepoints, len(epo_imp_b))
assert imp_proba_b.shape == (folds, timepoints, len(epo_imp_b), 2)

del all_results
del adjs, ns, imp_nb, imp_b

### average over folds
imp_lbl_nb = np.mean(imp_lbl_nb, axis=0)
#imp_conf_nb = np.mean(imp_conf_nb, axis=0)
imp_proba_nb = np.mean(imp_proba_nb, axis=0)
imp_lbl_b = np.mean(imp_lbl_b, axis=0)
#imp_conf_b = np.mean(imp_conf_b, axis=0)
imp_proba_b = np.mean(imp_proba_b, axis=0)

######################################################################
### fft on the label oscillations (optional)
data_fft = imp_lbl_nb

n_timepoints, n_trials = data_fft.shape
sfreq = 500
amp_spectra = np.zeros((n_timepoints//2 + 1, n_trials))
freqs = np.fft.rfftfreq(n_timepoints, d=1/sfreq)

for i in range(n_trials):
    signal = data_fft[:, i] - np.mean(data_fft[:, i])
    fft_vals = np.fft.rfft(signal)
    amp_spectra[:, i] = np.abs(fft_vals)  

assert amp_spectra.shape == (len(freqs), n_trials)
mean_spectrum = np.mean(amp_spectra, axis=1)

### plot fft
plt.plot(freqs[freqs <= 50], mean_spectrum[freqs <= 50])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Mean Amplitude')
plt.title('FFT on Decoding Labels [0,1] across Trials')
plt.show()

######################################################################
### categorize trials by prediction labels
avg_lbl_nb = np.mean(imp_lbl_nb, axis=0)
avg_lbl_b = np.mean(imp_lbl_b, axis=0)

def plot_label_proportion(avg_lbl_1, 
                          avg_lbl_2, 
                          threshold=0.5, 
                          bins=20,
                          alpha=0.7,
                          lbl_1='', 
                          lbl_2='',
                          clr_1='skyblue', 
                          clr_2='orange',):
    plt.hist(avg_lbl_1, bins=bins, alpha=alpha, label=lbl_1, color=clr_1)
    plt.hist(avg_lbl_2, bins=bins, alpha=alpha, label=lbl_2, color=clr_2)
    plt.axvline(0.5, color='red', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel('Proportion of "1" Predictions')
    plt.ylabel('Number of Trials')
    plt.title('Distribution of Predicted Class Proportions')
    plt.legend()
    plt.show() 

plot_label_proportion(avg_lbl_b, 
                      avg_lbl_nb,
                      lbl_1 = 'Binding',
                      lbl_2 = 'NBinding')

### create trial labels
lbl_nb = np.where(avg_lbl_nb >0.5, 'ADJ-NB', 'N-NB')
#lbl_b = np.full_like(avg_lbl_b, fill_value='UNK', dtype='<U4') 
lbl_b = np.where(avg_lbl_b >0.5, 'ADJ-B', 'N-B')
assert len(lbl_nb) == len(epo_imp_nb)
assert len(lbl_b) == len(epo_imp_b)

### assign labels back to the phase data
def assign_label(lbl_nb, lbl_b, epo_pha_nb, epo_pha_b):
    onsets_nb = np.arange(len(lbl_nb))
    onsets_b = np.arange(len(lbl_b))
    
    annotations_nb = mne.annotations.Annotations(onset=onsets_nb, duration=np.zeros(len(onsets_nb)), description=lbl_nb)
    annotations_b = mne.annotations.Annotations(onset=onsets_b, duration=np.zeros(len(onsets_b)), description=lbl_b)

    epo_pha_nb.set_annotations(annotations_nb)
    epo_pha_b.set_annotations(annotations_b)
    '''
    adj_idx = np.where(lbl_nb == "ADJ")[0]
    n_idx = np.where(lbl_nb == "N")[0]
    unk_idx = np.where(lbl_b == 'UNK')[0]
    '''
    adj_nb_idx = np.where(lbl_nb == "ADJ-NB")[0]
    n_nb_idx = np.where(lbl_nb == "N-NB")[0]
    adj_b_idx = np.where(lbl_b == "ADJ-B")[0]
    n_b_idx = np.where(lbl_b == "N-B")[0]

    #return adj_idx, n_idx, unk_idx
    return adj_nb_idx, n_nb_idx, adj_b_idx, n_b_idx

#adj_idx, n_idx, unk_idx = assign_label(lbl_nb, lbl_b, epo_pha_nb, epo_pha_b)
adj_nb_idx, n_nb_idx, adj_b_idx, n_b_idx = assign_label(lbl_nb, lbl_b, epo_pha_nb, epo_pha_b)

### separate the decoding data by label
'''
imp_lbl_ADJ = imp_lbl_nb[:, lbl_nb == 'ADJ']
imp_lbl_N = imp_lbl_nb[:, lbl_nb == 'N']
assert imp_lbl_ADJ.shape == (timepoints, len(adj_idx))
assert imp_lbl_N.shape == (timepoints, len(n_idx))
'''
imp_lbl_ADJ_NB = imp_lbl_nb[:, lbl_nb == 'ADJ-NB']
imp_lbl_N_NB = imp_lbl_nb[:, lbl_nb == 'N-NB']
imp_lbl_ADJ_B = imp_lbl_b[:, lbl_b == 'ADJ-B']
imp_lbl_N_B = imp_lbl_b[:, lbl_b == 'N-B']

assert imp_lbl_ADJ_NB.shape == (timepoints, len(adj_nb_idx))
assert imp_lbl_N_NB.shape == (timepoints, len(n_nb_idx))
assert imp_lbl_ADJ_B.shape == (timepoints, len(adj_b_idx))
assert imp_lbl_N_B.shape == (timepoints, len(n_b_idx))

######################################################################
### categorize trials by predition confidence
avg_conf_nb = np.mean(imp_conf_nb, axis=0)
avg_conf_b = np.mean(imp_conf_b, axis=0)

def plot_confidence(avg_1, avg_2,
                    lbl_1='', lbl_2='',
                    clr_1='skyblue', clr_2='orange',
                    mean_clr_1='teal', 
                    mean_clr_2='darkorange',
                    bins=30,
                    title='Avg Confidence Distribution',
                    xlabel='Avg Confidence',
                    ylabel='Number of Trials',
                    figsize=(10, 4)):
    plt.figure(figsize=figsize)
    plt.hist(avg_1, bins=bins, color=clr_1, alpha=0.7, label=lbl_1)
    plt.hist(avg_2, bins=bins, color=clr_2, alpha=0.7, label=lbl_2)
    
    plt.axvline(np.mean(avg_1), color=mean_clr_1, linestyle='--', label=f'Mean ({lbl_1.lower()})')
    plt.axvline(np.mean(avg_2), color=mean_clr_2, linestyle='--', label=f'Mean ({lbl_2.lower()})')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_confidence(avg_conf_b,
                avg_conf_nb,
                lbl_1='Binding',
                lbl_2='NBinding',)

### create trial labels
conf_nb = np.where(avg_conf_nb >0, 'ADJ-NB', 'N-NB')
conf_b = np.where(avg_conf_b >0, 'ADJ-B', 'N-B')
assert len(conf_nb) == len(epo_imp_nb)
assert len(conf_b) == len(epo_imp_b)

### assign labels back to the data
adj_nb_idx, n_nb_idx, adj_b_idx, n_b_idx = assign_label(conf_nb, conf_b, epo_pha_nb, epo_pha_b)

'''
### plot confidence distribution
plt.figure(figsize=(10, 4))
plt.plot(avg_conf, marker='o')
plt.axhline(np.mean(avg_conf), color='gray', linestyle='--', label='Mean')
plt.axhline(np.mean(avg_conf) + np.std(avg_conf), color='green', linestyle='--', label='Mean + SD')
plt.axhline(np.mean(avg_conf) - np.std(avg_conf), color='green', linestyle='--', label='Mean - SD')
plt.axhline(np.mean(avg_conf) + np.std(avg_conf) / np.sqrt(len(avg_conf)), color='red', linestyle='--', label='Mean + SE')
plt.axhline(np.mean(avg_conf) - np.std(avg_conf) / np.sqrt(len(avg_conf)), color='red', linestyle='--', label='Mean - SE')
plt.title('Avg Confidence per Trial')
plt.xlabel('Trial')
plt.ylabel('Avg Confidence')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

percentiles = [5, 15, 25, 35, 65, 75, 85, 95,]
percentile_values = np.percentile(avg_conf, percentiles)

plt.figure(figsize=(10, 4))
plt.hist(avg_conf, bins=30, color='skyblue', edgecolor='black')  
plt.axvline(np.mean(avg_conf), color='gray', linestyle='--', label='Mean')
plt.axvline(np.mean(avg_conf) + np.std(avg_conf), color='green', linestyle='--', label='Mean + SD')
plt.axvline(np.mean(avg_conf) - np.std(avg_conf), color='green', linestyle='--', label='Mean - SD')
plt.axvline(np.mean(avg_conf) + np.std(avg_conf) / np.sqrt(len(avg_conf)), color='red', linestyle='--', label='Mean + SE')
plt.axvline(np.mean(avg_conf) - np.std(avg_conf) / np.sqrt(len(avg_conf)), color='red', linestyle='--', label='Mean - SE')
colors = ['purple', 'purple', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'purple', 'purple']
for perc, val, col in zip(percentiles, percentile_values, colors):
    plt.axvline(val, color=col, linestyle=':', alpha=0.8)
    plt.text(val, plt.ylim()[1]*0.9, f'{perc}%', rotation=90, color=col, ha='right', va='top', fontsize=8)
plt.title('Avg Confidence Distribution')
plt.xlabel('Avg Confidence')
plt.ylabel('Number of Trials')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

### threshold to get labels
labels = np.full_like(avg_conf, fill_value='UNK', dtype='<U4') 

#labels[avg_conf <= np.percentile(avg_conf, 35)] = 'N'
#labels[avg_conf >= np.percentile(avg_conf, 65)] = 'ADJ'
labels = np.where(avg_conf >0, 'ADJ', 'N')
assert len(labels) == len(epo_pha)

### assign labels back to the data
onsets = np.arange(len(labels))
annotations = mne.annotations.Annotations(onset=onsets, duration=np.zeros(len(onsets)), description=labels)
epo_pha.set_annotations(annotations)

adj_idx = np.where(labels == "ADJ")[0]
noun_idx = np.where(labels == "N")[0]
unk_idx = np.where(labels == 'UNK')[0]
'''
######################################################################
'''
### get phase by tfr
ang = []
ang_ADJ = []
ang_N = []
ang_UNK = []

foi = np.arange(4, 13, 0.1)
for f in foi:
    toi = [-5/f, 0]
    epochs = epo_pha.copy().crop(tmin=toi[0], tmax=toi[1])
    
    phase = epochs.compute_tfr(method='multitaper', 
                                freqs=[f], 
                                picks='eeg',
                                output='phase',
                                n_cycles=5,
                            )

    phase = epochs.compute_tfr(method='morlet', 
                                freqs=[f], 
                                picks='eeg',
                                output='phase',
                                n_cycles=3,
                            )
 
    power, itpc = epochs.compute_tfr(method='morlet', 
                                freqs=[f], 
                                picks='eeg',
                                average=True,
                                return_itc=True,
                                n_cycles=3,
                            )
    
    #phase = np.mean(phase, axis=2) # average across tapers (only for multitaper)
    phase = np.median(phase, axis=-1)
    
    ang.append(phase)   # if no median, shape == (foi, trial, channel, taper, timepoint)
    ang_ADJ.append(phase[adj_idx])
    ang_N.append(phase[noun_idx])
    ang_UNK.append(phase[unk_idx])

ang = np.array(ang)
ang_ADJ = np.array(ang_ADJ)  
ang_N = np.array(ang_N) 
ang_UNK = np.array(ang_UNK)

### ang.shape = foi x channels x trials
ang = np.squeeze(ang,axis=-1)
ang_ADJ = np.squeeze(ang_ADJ, axis=-1)    
ang_N = np.squeeze(ang_N, axis=-1)    
ang_UNK = np.squeeze(ang_UNK, axis=-1)
'''
######################################################################
### get phase via fft
def get_phase_fft (data, foi):
    n_trials, n_channels, n_times = data.get_data().shape
    n_freqs = len(foi)
    sfreq = 500
    phase_mat = np.zeros((n_freqs, n_channels, n_trials))

    for fi, f in enumerate(foi):
        tmin, tmax = -5/f, 0
        data_crop = data.copy().crop(tmin=tmin, tmax=tmax).get_data()
        n_samples = data_crop.shape[-1]
        window = np.hanning(n_samples)

        for ch in range(n_channels):
            data_ch = data_crop[:, ch, :] * window
            fft_data = np.fft.rfft(data_ch, axis=1)
            freqs = np.fft.rfftfreq(n_samples, 1 / sfreq)
            f_idx = np.argmin(np.abs(freqs - f))
            phases = np.angle(fft_data[:, f_idx])
            phase_mat[fi, ch, :] = phases
    return phase_mat
foi = np.arange(4, 20, 0.1)

phase_all = get_phase_fft(epo_pha, foi)
phase_nb = get_phase_fft(epo_pha_nb, foi)
phase_b = get_phase_fft(epo_pha_b, foi)

ang_ALL = phase_all     ### all trials
'''
ang_ADJ = phase_nb[:, :, adj_idx]   ### adj trials for nb 
ang_N = phase_nb[:, :, n_idx]       ### n trials for nb
ang_UNK = phase_b[:, :, unk_idx]    ### all b trials
ang_NB = phase_nb.copy()            ### all nb trials

assert ang_ALL.shape == (len(foi), epo_pha.info['nchan'], len(epo_pha))
assert ang_ADJ.shape == (len(foi), epo_pha_nb.info['nchan'], len(adj_idx))
assert ang_N.shape == (len(foi), epo_pha_nb.info['nchan'], len(n_idx))
assert ang_UNK.shape == (len(foi), epo_pha_b.info['nchan'], len(unk_idx))
assert ang_NB.shape == (len(foi), epo_pha_nb.info['nchan'], len(epo_pha_nb))
'''
ang_ADJ_NB = phase_nb[:, :, adj_nb_idx]
ang_N_NB = phase_nb[:, :, n_nb_idx]
ang_ADJ_B = phase_b[:, :, adj_b_idx]
ang_N_B = phase_b[:, :, n_b_idx]
assert ang_ALL.shape == (len(foi), epo_pha.info['nchan'], len(epo_pha))
assert ang_ADJ_NB.shape == (len(foi), epo_pha_nb.info['nchan'], len(adj_nb_idx))
assert ang_N_NB.shape == (len(foi), epo_pha_nb.info['nchan'], len(n_nb_idx))
assert ang_ADJ_B.shape == (len(foi), epo_pha_b.info['nchan'], len(adj_b_idx))
assert ang_N_B.shape == (len(foi), epo_pha_b.info['nchan'], len(n_b_idx))

### sub-sample phase to the same amount
### this avoids the ITPC being biased to smaller values for larger amount of trials
min_trials = min(ang_ADJ_NB.shape[2],
                 ang_N_NB.shape[2],
                 ang_ADJ_B.shape[2],
                 ang_N_B.shape[2])
print("Minimum trial number:", min_trials)

def subsample_trials(arr, n_trials):
    total_trials = arr.shape[2]
    idx = np.random.choice(total_trials, n_trials, replace=False)
    return arr[:, :, idx]

ang_ADJ_NB_sub = subsample_trials(ang_ADJ_NB, min_trials)
ang_N_NB_sub   = subsample_trials(ang_N_NB, min_trials)
ang_ADJ_B_sub  = subsample_trials(ang_ADJ_B, min_trials)
ang_N_B_sub    = subsample_trials(ang_N_B, min_trials)
ang_ALL_sub = subsample_trials(ang_ALL, min_trials)

### plot polar phase
ch_names = epo_pha.ch_names
channel = 'Cz'
freq = 12
ch_idx = ch_names.index(channel)
f_idx = np.argmin(np.abs(foi - freq))
phases = phase_all[f_idx, ch_idx, :]  # shape: (n_trials,)

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.hist(phases, bins=30, density=True, alpha=0.7)
ax.set_title(f'Polar Phase\n{channel} at {foi[f_idx]:.1f} Hz')
plt.show()

######################################################################
'''
### pca over vector
def ang_to_complex (data):
    results = np.concatenate([
        np.cos(data),
        np.sin(data),
    ],
    axis = 1
    )
    return results

ang_ADJ_complex = ang_to_complex(ang_ADJ)
ang_N_complex = ang_to_complex(ang_N)
ang_UNK_complex = ang_to_complex(ang_UNK)

ang_ADJ_complex_pca, explained_ADJ = pca_phase(ang_ADJ_complex)
ang_N_complex_pca, explained_N = pca_phase(ang_N_complex)
ang_UNK_complex_pca, explained_UNK = pca_phase(ang_UNK_complex)

### plot explained variance across frequencies
plt.plot(foi, explained_ADJ * 100, label='ADJ')
plt.plot(foi, explained_N * 100, label='N')
#plt.plot(foi, explained_UNK * 100, label='UNK')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Explained Variance of PC1 (%)")
plt.title("PCA on Phase: Channel Dimensionality Reduction")
plt.legend()
plt.grid(True)
plt.show()
'''
######################################################################
### circular mean
def cir_mean_phase(data):
    complex_phase = np.exp(1j*data)
    mean_vector = np.mean(complex_phase, axis=1)
    mean_phase = np.angle(mean_vector)
    R = np.abs(mean_vector)
    return mean_phase, R

ang_ALL_mean, r_ALL = cir_mean_phase(ang_ALL_sub)
'''
ang_ADJ_mean, r_ADJ = cir_mean_phase(ang_ADJ)
ang_N_mean, r_N = cir_mean_phase(ang_N)
ang_UNK_mean, r_UNK = cir_mean_phase(ang_UNK)
ang_NB_mean, r_NB = cir_mean_phase(ang_NB)

assert ang_ALL_mean.shape == (len(foi), len(epo_pha))
assert ang_ADJ_mean.shape == (len(foi), len(adj_idx))
assert ang_N_mean.shape == (len(foi), len(n_idx))
assert ang_UNK_mean.shape == (len(foi), len(unk_idx))
assert ang_NB_mean.shape == (len(foi), len(epo_pha_nb))
'''
ang_ADJ_NB_mean, r_ADJ_NB = cir_mean_phase(ang_ADJ_NB_sub)
ang_N_NB_mean, r_N_NB = cir_mean_phase(ang_N_NB_sub)
ang_ADJ_B_mean, r_ADJ_B = cir_mean_phase(ang_ADJ_B_sub)
ang_N_B_mean, r_N_B = cir_mean_phase(ang_N_B_sub)

assert ang_ALL_mean.shape == (len(foi), min_trials)
assert ang_ADJ_NB_mean.shape == (len(foi), min_trials)
assert ang_N_NB_mean.shape == (len(foi), min_trials)
assert ang_ADJ_B_mean.shape == (len(foi), min_trials)
assert ang_N_B_mean.shape == (len(foi), min_trials)

######################################################################
def compute_phase_vector(phase_angles):
    """
    phase_angles: shape (foi, trials)
    Returns:
        - vector_lengths: shape (foi,)
        - vector_angles: shape (foi,)
    """
    complex_vectors = np.exp(1j * phase_angles)  # shape: (foi, trials)
    mean_vector = np.mean(complex_vectors, axis=1)  # shape: (foi,)
    vector_lengths = np.abs(mean_vector)           # length = phase concentration
    vector_angles = np.angle(mean_vector)          # angle = mean phase angle
    return vector_lengths, vector_angles
'''
vl_ALL, va_ALL = compute_phase_vector(ang_ALL_mean)
vl_ADJ, va_ADJ = compute_phase_vector(ang_ADJ_mean)
vl_N, va_N = compute_phase_vector(ang_N_mean)
vl_UNK, va_UNK = compute_phase_vector(ang_UNK_mean)

assert vl_ALL.shape == foi.shape
assert vl_ADJ.shape == foi.shape
assert vl_N.shape == foi.shape
assert vl_UNK.shape == foi.shape
assert va_ALL.shape == foi.shape
assert va_ADJ.shape == foi.shape
assert va_N.shape == foi.shape
assert va_UNK.shape == foi.shape
'''

vl_ALL, va_ALL = compute_phase_vector(ang_ALL_mean)
vl_ADJ_NB, va_ADJ_NB = compute_phase_vector(ang_ADJ_NB_mean)
vl_N_NB, va_N_NB = compute_phase_vector(ang_N_NB_mean)
vl_ADJ_B, va_ADJ_B = compute_phase_vector(ang_ADJ_B_mean)
vl_N_B, va_N_B = compute_phase_vector(ang_N_B_mean)

'''
def find_peaks(vl, label, color):
    top_idx = np.argsort(vl)[-3:] 
    for idx in top_idx:
        freq = foi[idx]
        value = vl[idx]
        plt.axvline(x=freq, linestyle='--', color=color, alpha=0.5)
        plt.text(freq, value + 0.02, f'{label}\n{freq:.1f}Hz\n{value:.2f}',
                 color=color, ha='center', fontsize=8)
        
find_peaks(vl_ADJ, 'ADJ', 'tab:blue')
find_peaks(vl_N, 'NOUN', 'tab:orange')
find_peaks(vl_UNK, 'UNK', 'tab:green')
'''  
highlight_freqs = [
    (8.5, 13),
]

### plot itpc
#plt.plot(foi, vl_ADJ_NB, label='ADJ_NB', color='#D55E00')
#plt.plot(foi, vl_N_NB, label='N_NB', color='#F0A500')
plt.plot(foi, vl_ADJ_B, label='ADJ_B', color='#0072B2')
plt.plot(foi, vl_N_B, label='N_B', color='#56B4E9')
plt.plot(foi, vl_ALL, label='ALL', color='gray')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase Consistency')
plt.legend(loc='upper right')
plt.title('Phase Consistency across Frequencies')
'''for start, end in highlight_freqs:
    plt.axvspan(start, end, color='red', alpha=0.2)
    plt.text((start + end) / 2, plt.ylim()[1]*0.95, f'{start}-{end}Hz',
             color='red', ha='center', va='top', fontsize=8)
plt.tight_layout()'''
plt.show()

### plot baseline-corrected itpc
plt.plot(foi, vl_ADJ_NB-vl_ALL, label='ADJ_NB', color='#D55E00')
plt.plot(foi, vl_N_NB-vl_ALL, label='N_NB', color='#F0A500')
plt.plot(foi, vl_ADJ_B-vl_ALL, label='ADJ_B', color='#0072B2')
plt.plot(foi, vl_N_B-vl_ALL, label='N_B', color='#56B4E9')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase Consistency')
plt.legend()
plt.title('Phase Consistency across Frequencies (baseline-corrected)')
'''for start, end in highlight_freqs:
    plt.axvspan(start, end, color='red', alpha=0.2)
    plt.text((start + end) / 2, plt.ylim()[1]*0.95, f'{start}-{end}Hz',
             color='red', ha='center', va='top', fontsize=8)
plt.tight_layout()'''
plt.show()

### plot relative phase in polar plane
def compute_relative_phase(va_1, va_2):
    diff = va_1 - va_2
    return np.arctan2(np.sin(diff), np.cos(diff))

rva_ADJ_N_NB = compute_relative_phase(va_ADJ_NB,va_N_NB)
rva_ADJ_N_B = compute_relative_phase(va_ADJ_B,va_N_B)
rva_ADJ_NB_B = compute_relative_phase(va_ADJ_NB,va_ADJ_B)
rva_N_NB_B = compute_relative_phase(va_N_NB,va_N_B)

def plot_relative_phase(rva_1, rva_2, rva_3, rva_4, 
                        foi,
                        labels=[],
                        colors=[],
                        marker_size=15, alpha=0.7,
                        jitter_strength=0.05,
                        highlight_range=None, 
                        highlight_color='black',
                        seed=42,
                        highlight_marker='',
                        title='',
                        ):
    
    np.random.seed(seed)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    
    phase_sets = [rva_1, rva_2, rva_3, rva_4]
    base_radii = [1.0, 1.2, 1.4, 1.6]
    for phase_vals, label, color, base_radius in zip(phase_sets, labels, colors, base_radii):
        jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(phase_vals))
        radii = base_radius + jitter
        if highlight_range:
            foi = np.array(foi)
            in_range = (foi >= highlight_range[0]) & (foi <= highlight_range[1])
            out_range = ~in_range

            ax.scatter(phase_vals[out_range], 
                       radii[out_range], 
                       color=color, 
                       alpha=alpha, 
                       s=marker_size, 
                       label=label)

            ax.scatter(phase_vals[in_range], 
                       radii[in_range], 
                       color=highlight_color, 
                       edgecolor='black',
                       alpha=1.0, 
                       s=marker_size+10, 
                       label=f'{label} {highlight_range[0]}-{highlight_range[1]} Hz' if label else None,
                       marker=highlight_marker)
        else:
            ax.scatter(phase_vals, radii, color=color, alpha=alpha, s=marker_size, label=label)
    
    ax.set_rticks([])
    ax.set_title(title, va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

plot_relative_phase(rva_ADJ_N_NB,
                 rva_ADJ_N_B,
                 rva_ADJ_NB_B,
                 rva_N_NB_B,
                 foi, 
                 labels=['NB: ADJ vs N', 'B: ADJ vs N', 'ADJ: NB vs B', 'N: NB vs B'],
                 colors=['#3CB371', '#DC143C', '#FF8C00', '#4682B4'],
                 highlight_range=(8, 12),
                 highlight_marker='^',
                 title='Relative Phase Distribution')

######################################################################
### plot va
plt.plot(foi, va_ADJ, label='ADJ')
plt.plot(foi, va_N, label='NOUN')
plt.plot(foi, va_UNK, label='UNK')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Mean Phase Angle (rad)')
plt.legend()
plt.title('Mean Phase Angle across Frequencies')
'''for start, end in highlight_freqs:
    plt.axvspan(start, end, color='red', alpha=0.2)
    plt.text((start + end) / 2, plt.ylim()[1]*0.95, f'{start}-{end}Hz',
             color='red', ha='center', va='top', fontsize=8)
plt.tight_layout()'''
plt.show()

### plot va difference
plt.plot(foi, va_ADJ-va_N, label='ADJ-N')
plt.plot(foi, va_ADJ-va_UNK, label='ADJ-UNK')
plt.plot(foi, va_N-va_UNK, label='N-UNK')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Mean Phase Angle (rad)')
plt.legend()
plt.title('Mean Phase Angle Difference across Frequencies')
'''for start, end in highlight_freqs:
    plt.axvspan(start, end, color='red', alpha=0.2)
    plt.text((start + end) / 2, plt.ylim()[1]*0.95, f'{start}-{end}Hz',
             color='red', ha='center', va='top', fontsize=8)
plt.tight_layout()'''
plt.show()

######################################################################
### function to time-resolved vl
def circ_r(alpha, w=None, d=0, axis=0):
    """
    computes the mean resultant vector length for circular data.

    parameters:
    - alpha : array-like
        sample of angles in radians.
    - w : array-like or None
        optional weights (e.g., number of incidences for binned data).
    - d : float
        spacing of bin centers (in radians), for correction in binned data.
    - axis : int
        axis along which to compute (default: 0).

    returns:
    - r : float or ndarray
        mean resultant vector length.
    """
    alpha = np.asarray(alpha)

    if w is None:
        w = np.ones_like(alpha)
    else:
        w = np.asarray(w)
        if w.shape != alpha.shape:
            raise ValueError("alpha and w must have the same shape")

    # compute weighted complex vector sum
    r_complex = np.sum(w * np.exp(1j * alpha), axis=axis)
    r = np.abs(r_complex) / np.sum(w, axis=axis)

    # apply correction factor if bin spacing d is given
    if d != 0:
        c = d / (2 * np.sin(d / 2))
        r *= c

    return r

print(f'ADJ: max label={imp_lbl_ADJ.max()}, min label={imp_lbl_ADJ.min()}')
print(f'N: max label={imp_lbl_N.max()}, min label={imp_lbl_N.min()}')

### rescale the N decoding data by flipping
### original: closer to 0 -> N
### rescaled: closer to 1 -> ADJ (same as ADJ decoding data)
imp_lbl_N_flp = 1 - imp_lbl_N.copy()

### compute vl per timepoint per frequency per trial for ADJ and N seperately
def compute_vl(ang_mean, dcd_data, n_perm):
    n_freqs, n_trials = ang_mean.shape
    n_timepoints = dcd_data.shape[0]
    n_perm = n_perm

    vl = np.zeros((n_freqs, n_timepoints))
    vl_perm = np.zeros((n_freqs, n_timepoints, n_perm))

    for t in range(n_timepoints):
        w = dcd_data[t, :]  ### decoder confidence at for each trial
        for f in range(n_freqs):
            phase = ang_mean[f, :]  #### phase for each trial
            vl[f, t] = circ_r(phase, w)

            #### permutation
            for p in range(n_perm):
                w_perm = np.random.permutation(w)
                vl_perm[f, t, p] = circ_r(phase, w_perm)
    return vl, vl_perm
n_perm = 1000

vl_adj, vl_perm_adj = compute_vl(ang_ADJ_mean, imp_lbl_ADJ, n_perm)
vl_n, vl_perm_n = compute_vl(ang_N_mean, imp_lbl_N_flp, n_perm)

assert vl_adj.shape == (ang_ADJ_mean.shape[0], n_timepoints)
assert vl_perm_adj.shape == (ang_ADJ_mean.shape[0], n_timepoints, n_perm)
assert vl_n.shape == (ang_N_mean.shape[0], n_timepoints)
assert vl_perm_n.shape == (ang_ADJ_mean.shape[0], n_timepoints, n_perm)

print(f'vl_adj: max={vl_adj.max()}, min={vl_adj.min()}')
print(f'vl_noun: max={vl_n.max()}, min={vl_n.min()}')

### vl difference
vl_adj_diff = vl_adj - vl_perm_adj.mean(axis=2)
vl_n_diff = vl_n - vl_perm_n.mean(axis=2)
assert vl_adj_diff.shape == (ang_ADJ_mean.shape[0], n_timepoints)
assert vl_n_diff.shape == (ang_N_mean.shape[0], n_timepoints)

### plot vl
def plot_vl(vl, 
            title='',
            freq_range=(4, 20), 
            time_range=(0, 0.8),
            ):
    freqs = np.linspace(freq_range[0], freq_range[1], vl.shape[0])
    times = np.linspace(time_range[0], time_range[1], vl.shape[1])

    plt.figure(figsize=(10, 6))
    plt.imshow(vl,
               aspect='auto',
               origin='lower',
               extent=[times[0], times[-1], freqs[0], freqs[-1]],
               cmap='RdYlBu_r',
               vmin=vl.min(), vmax=vl.max())
    
    plt.colorbar(label='vl')
    plt.xlabel('Time (s)')
    plt.ylabel('Freq. (Hz)')
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_vl(vl_adj, 'Vector length (ADJ)')
plot_vl(vl_n, 'Vector length (N)')
plot_vl(vl_adj_diff, 'Vector length difference (ADJ)')
plot_vl(vl_n_diff, 'Vector length difference (N)')

######################################################################
######################################################################
######################################################################
'''
ten Oever et al., (2021) method
'''
### label: timepoints * trials
assert imp_lbl_nb.shape == (timepoints, len(epo_imp_nb))
assert imp_lbl_b.shape == (timepoints, len(epo_imp_b))

### run a PCA for raw data over the selected channels
from sklearn.decomposition import PCA

### epoch: trials * channels * timepoints
# define PCA
def pca(epoch):
    X = epoch.get_data()
    n_trials, n_channels, n_timepoints = X.shape
    PC1_trials = []

    for trial in range(n_trials):
        X_T = X[trial].T    # timepoints * channels
        pca = PCA(n_components=1)
        PC1 = pca.fit_transform(X_T).ravel()    # timepoints
        PC1_trials.append(PC1)
    
    PC1_trials = np.array(PC1_trials)
    assert PC1_trials.shape == (n_trials, n_timepoints)

    return PC1_trials

# run PCA over channels to get trials * timepoints
pha_nb_pc = pca(epo_pha_nb)
pha_b_pc = pca(epo_pha_b)

### extract phases
def fft_phase(data, foi, sfreq=500):
    """
    Compute phase at given frequencies using FFT for trial * timepoints data.
    
    Parameters:
    data : ndarray, shape (n_trials, n_times)
        Time series data.
    foi : list or array
        Frequencies of interest (Hz).
    sfreq : float
        Sampling frequency (Hz).
    
    Returns:
    phase_mat : ndarray, shape (n_freqs, n_trials)
        Phase values (radians) for each frequency and trial.
    """
    n_trials, n_times = data.shape
    n_freqs = len(foi)
    samples = np.arange(n_times) / sfreq
    phase_mat = np.zeros((n_freqs, n_trials))


    for fi, f in enumerate(foi):
        # crop the window of interest per frequency
        # the data (np array, not mne epochs now) is 2.45s long
        # with pre-impulse 1.45s and post-impulse 1s
        # the cropped window should ends at the pre-impulse 0s, which is the sample at 1.45s in the data
        tmin, tmax = round(1.45-5/f, 5), 1.45
        start_idx = int(tmin * sfreq)
        end_idx   = int(tmax * sfreq)
        data_crop = data[:, start_idx:end_idx]
        n_samples = data_crop.shape[-1]
        window = np.hanning(n_samples)

        data_win = data_crop[:, :] * window
        fft_data = np.fft.rfft(data_win, axis=1)
        freqs = np.fft.rfftfreq(n_samples, 1 / sfreq)
        f_idx = np.argmin(np.abs(freqs - f))
        phases = np.angle(fft_data[:, f_idx])
        phase_mat[fi, :] = phases

    return phase_mat

foi = np.arange(4, 20, 0.1)

ang_nb = fft_phase(pha_nb_pc, foi)
ang_b = fft_phase(pha_b_pc, foi)

# phase: foi * trials
assert ang_nb.shape == (len(foi), len(pha_nb_pc))
assert ang_b.shape == (len(foi), len(pha_b_pc))

'''
Now the decoding label is timepoints * trials, while the extracted pre-impulse phase is foi * trials. We want to have 
'''

phase_per_timepoint_adj = []
phase_per_timepoint_n = []

for t in range(imp_lbl_nb.shape[0]):
    # find adj trials where decoding label > 0.5
    # find n trials where decoding label < 0.5
    adj_trials = np.where(imp_lbl_nb[t, :] > 0.5)[0]
    n_trials = np.where(imp_lbl_nb[t, :] < 0.5)[0]

    # get phases for these trials (shape: n_foi x trials)
    adj_phase = ang_nb[:, adj_trials]
    n_phase = ang_nb[:, n_trials]

    phase_per_timepoint_adj.append(adj_phase)
    phase_per_timepoint_n.append(n_phase) 

'''
phase_per_time is a list of length n_timepoints
each entry is (n_foi x n_selected_trials) array
'''

def itpc(phase_per_timepoint):
    itpc = np.full((len(foi), len(phase_per_timepoint)), np.nan, dtype=float)

    for t in range(len(phase_per_timepoint)):
        phases = phase_per_timepoint[t]  # shape (n_foi, n_trials)

        # skip if nothing is there
        if phases is None:
            continue
        phases = np.asarray(phases)

        # if phases is 1D (happens if there’s only one frequency), reshape so it becomes (1, n_trials) instead of (n_trials,) to keep dimensions consistent
        if phases.ndim == 1:
            phases = phases[np.newaxis, :]

        # if there are no trials at this time point, skip it.
        if phases.shape[1] == 0:
            continue

        plv = np.abs(np.mean(np.exp(1j * phases), axis=1))  # shape = n_foi
        itpc[:, t] = plv    # (n_foi, n_timepoints)
    return itpc

itpc_adj = itpc(phase_per_timepoint_adj)
itpc_n = itpc(phase_per_timepoint_n)

### plot itpc
fig, axs = plt.subplots(2, 1, figsize=(12, 6))

im1 = axs[0].imshow(itpc_adj, aspect='auto', origin='lower',
                   extent=[0, itpc_adj.shape[1]/sfreq, foi[0], foi[-1]],
                   cmap='jet')
axs[0].set_title('Vector Length Adjective')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Frequency (Hz)')
fig.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(itpc_n, aspect='auto', origin='lower',
                   extent=[0, itpc_n.shape[1]/sfreq, foi[0], foi[-1]],
                   cmap='jet')
axs[1].set_title('Vector Length Noun')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Frequency (Hz)')
fig.colorbar(im2, ax=axs[1])

plt.tight_layout()
plt.show()
