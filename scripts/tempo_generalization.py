'''Decoding in temporal generalization fashion

AUTHOR: Jiachen Yao <jasonyao0703[at]gmail.com>
LICENCE: BSD 3-clause

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

### concatenate epoch (stimulus and phase)
epo_stim = mne.concatenate_epochs([epo_stim_ses01, epo_stim_ses02])
epo_pha = mne.concatenate_epochs([epo_imp_ses01, epo_imp_ses02])

### cut epochs to post-stim
epo_stim = epo_stim.crop(tmin=0, tmax=0.8)

######################################################################
nchan = epo_stim.pick(picks='eeg').info['nchan']

adj_event = ['w1_T-B-AN', 'w2_T-B-NA', 'w1_T-NB-AN', 'w2_T-NB-NA',]
#adj_event_nb = ['w1_T-NB-AN', 'w2_T-NB-NA',]
n_event = ['w2_T-B-AN', 'w1_T-B-NA','w2_T-NB-AN', 'w1_T-NB-NA', ]
#n_event_nb = ['w2_T-NB-AN', 'w1_T-NB-NA',]
imp_event = ['imp_T-B-AN', 'imp_T-B-NA', 'imp_T-NB-AN','imp_T-NB-NA']
#imp_event_nb = ['imp_T-NB-AN','imp_T-NB-NA']

epo_adj = epo_stim[adj_event]
epo_n = epo_stim[n_event]
epo_imp = epo_stim[imp_event]
'''
### match trials for epo_pha and epo_imp
epo_comm = np.intersect1d(epo_pha.selection, epo_imp.selection)

epo_pha = epo_pha[np.isin(epo_pha.selection, epo_comm)]
epo_imp = epo_imp[np.isin(epo_imp.selection, epo_comm)]
'''
adjs = epo_adj.pick('eeg').get_data()
ns = epo_n.pick('eeg').get_data()
impulse = epo_imp.pick('eeg').get_data()
'''
### alternative with filtering
adjs = epo_adj.pick('eeg').filter(1, 40).get_data()
ns = epo_n.pick('eeg').filter(1, 40).get_data()
impulse = epo_imp.pick('eeg').filter(1, 40).get_data()
'''
del epo_stim

#epo_pha_b = epo_pha[np.isin(epo_pha.selection, epo_comm)]
#epo_pha_nb = 

######################################################################
folds = 25
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
results_w_conf = list()
results_imp_label = list()
results_imp_conf = list()

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

    fold_test_impulse = list(impulse)
    fold_results = np.zeros((timepoints, timepoints, ))
    fold_preds_w_label = np.zeros((timepoints, timepoints, len(fold_test)))
    fold_preds_imp_label = np.zeros((timepoints, timepoints, len(fold_test_impulse)))
    fold_preds_w_conf = np.zeros((timepoints, timepoints, len(fold_test)))
    fold_preds_imp_conf = np.zeros((timepoints, timepoints, len(fold_test_impulse)))

    for t_train in tqdm(range(timepoints)):
        t_train_data = np.array([erp[:, t_train] for erp in fold_train])
        ### samples x electrodes
        assert len(t_train_data.shape) == 2     
        ### checking electrodes
        assert t_train_data.shape[1] == nchan
        model = linear_model.RidgeClassifierCV(
            alphas=(0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
        model.fit(t_train_data, labels_train)

        for t_test in range(timepoints):
            t_test_data = np.array([erp[:, t_test] for erp in fold_test]) 
            t_impulse_data = np.array([erp[:, t_test] for erp in fold_test_impulse])
            assert len(t_test_data.shape) == 2
            assert len(t_impulse_data.shape) == 2

            preds = model.predict(t_test_data)
            evaluation = [1 if pred==real else 0 for pred, real in zip(preds, labels_test)]
            acc = sum(evaluation) / len(evaluation)
            assert acc > 0. and acc < 1.

            fold_preds_w_label[t_train, t_test, :] = preds

            preds_imp_lbl = model.predict(t_impulse_data)
            fold_preds_imp_label[t_train, t_test, :] = preds_imp_lbl

            preds_w_cf = model.decision_function(t_test_data)
            fold_preds_w_conf[t_train, t_test, :] = preds_w_cf

            preds_imp_cf = model.decision_function(t_impulse_data)
            fold_preds_imp_conf[t_train, t_test, :] = preds_imp_cf

            fold_results[t_train, t_test] = acc
    
    results_acc.append(fold_results)
    results_w_label.append(fold_preds_w_label)
    results_w_conf.append(fold_preds_w_conf)
    results_imp_label.append(fold_preds_imp_label)
    results_imp_conf.append(fold_preds_imp_conf)

all_results = {
    "acc": np.array(results_acc),
    "w_label": np.array(results_w_label),
    "w_conf": np.array(results_w_conf),
    "imp_label": np.array(results_imp_label),
    "imp_conf": np.array(results_imp_conf),
}

### shape = [folds, train_time, test_time]
assert all_results['acc'].shape == (folds, timepoints, timepoints)
assert all_results['w_label'].shape == (folds, timepoints, timepoints, len(labels_test))
assert all_results['w_conf'].shape == (folds, timepoints, timepoints, len(labels_test))
assert all_results['imp_label'].shape == (folds, timepoints, timepoints, len(impulse))
assert all_results['imp_conf'].shape == (folds, timepoints, timepoints, len(impulse))

#np.savez('tg_sub-01.npz', **all_results)

######################################################################
### plot temporal generalization matrix
def plot_tgm(data_avg, times,
            vmin, vmax,
            title='', colorbar_label='',
            cmap='RdBu_r',
            figsize=(6, 5),
            save_path=None,
            show_diagonal=True,  
            diag_color='k', diag_style='--', diag_width=1,
            ):
    
    plt.figure(figsize=figsize)
    im = plt.imshow(data_avg, 
                    origin='lower', 
                    aspect='auto', 
                    cmap=cmap, 
                    vmin=vmin, vmax=vmax,
                    extent=[times[0], times[-1], times[0], times[-1]])
    
    plt.colorbar(im, label=colorbar_label)
    plt.xlabel('Test Time (s)')
    plt.ylabel('Train Time (s)')
    plt.title(title)
    if show_diagonal:
        plt.plot([times[0], times[-1]], [times[0], times[-1]], 
                 color=diag_color, linestyle=diag_style, linewidth=diag_width)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

### acc
acc = all_results['acc']
print(acc.shape)
acc_avg = np.mean(acc, axis=0) # average over folds
n_times = acc_avg.shape[0]
times = np.arange(n_times) / 500
print(acc_avg.shape)
print(acc_avg.min(), acc_avg.max())

plot_tgm(acc_avg, times, 
         vmin=acc_avg.min(), vmax=acc_avg.max(), 
         title='Temporal Generalization Matrix (Stimulus)',
         colorbar_label='Accuracy',
         save_path='tgm_stim_acc.png')
'''
plt.figure(figsize=(6, 5))
im = plt.imshow(data_avg, 
                origin='lower', 
                aspect='auto', 
                cmap='cividis', ### cividis / RdBu_r
                vmin=0.3, vmax=0.7,     ### change accordingly
                extent=[times[0], 
                        times[-1], 
                        times[0], 
                        times[-1]],)

plt.colorbar(im, label='avg_conf')  ### change accordingly
plt.xlabel('Test Time (s)')
plt.ylabel('Train Time (s)')
plt.title('Temporal Generalization Matrix (stimulus)')
plt.show()
'''

### w_label
w_lbl = all_results['w_label']
print(w_lbl.shape)
w_lbl_avg = np.mean(w_lbl, axis=3) # average over trials
w_lbl_avg = np.mean(w_lbl_avg, axis=0) # over folds
n_times = w_lbl_avg.shape[0]
times = np.arange(n_times) / 500
print(w_lbl_avg.shape)
print(w_lbl_avg.min(), w_lbl_avg.max())

plot_tgm(w_lbl_avg, times, 
         vmin=w_lbl_avg.min(), vmax=w_lbl_avg.max(), 
         title='Temporal Generalization Matrix (Stimulus)',
         colorbar_label='Average Label',
         save_path='tgm_stim_lbl.png')

### w_conf
w_conf = all_results['w_conf']
print(w_conf.shape)
w_conf_avg = np.mean(w_conf, axis=3) # average over trials
w_conf_avg = np.mean(w_conf_avg, axis=0) # over folds
n_times = w_conf_avg.shape[0]
times = np.arange(n_times) / 500
print(w_conf_avg.shape)
print(w_conf_avg.min(), w_conf_avg.max())

plot_tgm(w_conf_avg, times, 
         vmin=w_conf_avg.min(), vmax=w_conf_avg.max(), 
         title='Temporal Generalization Matrix (Stimulus)',
         colorbar_label='Average Confidence',
         save_path='tgm_stim_conf.png')

### imp_label
imp_lbl = all_results['imp_label']
print(imp_lbl.shape)
imp_lbl_avg = np.mean(imp_lbl, axis=3) # average over trials
imp_lbl_avg = np.mean(imp_lbl_avg, axis=0) # over folds
n_times = imp_lbl_avg.shape[0]
times = np.arange(n_times) / 500
print(imp_lbl_avg.shape)
print(imp_lbl_avg.min(), imp_lbl_avg.max())

plot_tgm(imp_lbl_avg, times, 
         vmin=imp_lbl_avg.min(), vmax=imp_lbl_avg.max(), 
         title='Temporal Generalization Matrix (Impulse)',
         colorbar_label='Average Label',
         save_path='tgm_imp_lbl.png')

### w_conf
imp_conf = all_results['imp_conf']
print(imp_conf.shape)
imp_conf_avg = np.mean(imp_conf, axis=3) # average over trials
imp_conf_avg = np.mean(imp_conf_avg, axis=0) # over folds
n_times = imp_conf_avg.shape[0]
times = np.arange(n_times) / 500
print(imp_conf_avg.shape)
print(imp_conf_avg.min(), imp_conf_avg.max())

plot_tgm(imp_conf_avg, times, 
         vmin=imp_conf_avg.min(), vmax=imp_conf_avg.max(), 
         title='Temporal Generalization Matrix (Impulse)',
         colorbar_label='Average Confidence',
         save_path='tgm_imp_conf.png')

### plot the anti-diagonal line of TGM
anti_diag_acc = np.fliplr(acc_avg).diagonal()
anti_diag_w_lbl = np.fliplr(w_lbl_avg).diagonal()
anti_diag_w_conf = np.fliplr(w_conf_avg).diagonal()
anti_diag_imp_lbl = np.fliplr(imp_lbl_avg).diagonal()
anti_diag_imp_conf = np.fliplr(imp_conf_avg).diagonal()
assert len(anti_diag_acc) == timepoints
assert len(anti_diag_w_lbl) == timepoints
assert len(anti_diag_w_conf) == timepoints
assert len(anti_diag_imp_lbl) == timepoints
assert len(anti_diag_imp_conf) == timepoints
'''
plt.plot(times, anti_diag_acc)
plt.title("Anti-diagonal of Temporal Generalization Matrix")
plt.xlabel("Time (s)")
plt.ylabel("Acc")
plt.grid(True)
plt.show()
'''
def plot_anti_diagonal(times, anti_diag, 
                       title='',
                       xlabel='Time (s)',
                       ylabel='',
                       figsize=(6, 4),
                       save_path=None,):
    plt.figure(figsize=figsize)
    plt.plot(times, anti_diag, color='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

plot_anti_diagonal(times, anti_diag_acc, 
                   title="TGM Anti-diagonal (Stimulus)",
                   ylabel="Accuracy",
                   save_path="tgm_diag_acc.png")
plot_anti_diagonal(times, anti_diag_w_lbl, 
                   title="TGM Anti-diagonal (Stimulus)",
                   ylabel="Average Label",
                   save_path="tgm_diag_w_lbl.png")
plot_anti_diagonal(times, anti_diag_w_conf, 
                   title="TGM Anti-diagonal (Stimulus)",
                   ylabel="Average Confidence",
                   save_path="tgm_diag_w_conf.png")
plot_anti_diagonal(times, anti_diag_imp_lbl, 
                   title="TGM Anti-diagonal (Impulse)",
                   ylabel="Average Label",
                   save_path="tgm_diag_imp_lbl.png")
plot_anti_diagonal(times, anti_diag_imp_conf, 
                   title="TGM Anti-diagonal (Stimulus)",
                   ylabel="Average Confidence",
                   save_path="tgm_diag_imp_conf.png")