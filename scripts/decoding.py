'''
Decoding

This script contains the functions used for decoding in the source space

AUTHOR: Jiachen Yao <jasonyao0703[at]gmail.com>
LICENCE: BSD 3-clause

'''
import random
import scipy
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sklearn import linear_model
from sklearn.decomposition import PCA
from tqdm import tqdm

def sliding_logreg_source(X_train, y_train, X_test, y_test, folds):

    folds = folds

    X_adj = X_train[y_train==1] # n_epochs * n_vertices * n_times
    X_n = X_train[y_train==2]

    X_imp_nb = X_test[y_test==1]
    X_imp_b = X_test[y_test==2]

    ### checking timepoints are matched
    assert X_adj.shape[2] == X_n.shape[2]
    timepoints = X_adj.shape[2]
    ### checking vertices are matched
    assert X_adj.shape[1] == X_n.shape[1]
    ### collecting variables
    trials_adj = X_adj.shape[0]
    n_train_adj = int((trials_adj)*0.8)
    n_test_adj = int((trials_adj)*0.2)
    trials_n = X_n.shape[0]
    n_train_n = int((trials_n)*0.8)
    n_test_n = int((trials_n)*0.2)
    train_samples = min(n_train_adj, n_train_n)
    test_samples = min(n_test_adj, n_test_n)

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
        train_idxs_adjs = random.sample(range(len(X_adj)), k=train_samples)
        test_idxs_adjs = random.sample([i for i in range(len(X_adj)) if i not in train_idxs_adjs], k=test_samples)
        train_idxs_ns = random.sample(range(len(X_n)), k=train_samples)
        test_idxs_ns = random.sample([i for i in range(len(X_n)) if i not in train_idxs_ns], k=test_samples)
        assert len(train_idxs_ns) == len(train_idxs_adjs)
        assert train_idxs_ns != train_idxs_adjs
        assert test_idxs_ns != test_idxs_adjs
        fold_train = list(X_adj[train_idxs_adjs, :, :]) + list(X_n[train_idxs_ns, :, :])
        ### labels
        ### adj is 1, noun is 0
        labels_train = [1 for _ in range(len(train_idxs_adjs))] + [0 for _ in range(len(train_idxs_ns))]
        fold_test = list(X_adj[test_idxs_adjs, :, :]) + list(X_n[test_idxs_ns, :, :])
        labels_test = [1 for _ in range(len(test_idxs_adjs))] + [0 for _ in range(len(test_idxs_ns))]

        fold_test_impulse_nb = list(X_imp_nb)
        fold_test_impulse_b = list(X_imp_b)       
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
            ### samples x vertices
            assert len(t_train.shape) == 2
            assert len(t_test.shape) == 2   
            assert len(t_test_impulse_nb.shape) == 2   
            assert len(t_test_impulse_b.shape) == 2     
            ### checking vertices
            assert t_train.shape[1] == t_test.shape[1]
            assert t_test_impulse_nb.shape[1] == t_test_impulse_b.shape[1]

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
    assert all_results['imp_nb_label'].shape == (folds, timepoints, len(X_imp_nb))
    assert all_results['imp_b_label'].shape == (folds, timepoints, len(X_imp_b))
    #assert all_results['imp_nb_conf'].shape == (folds, timepoints, len(imp_nb))
    #assert all_results['imp_b_conf'].shape == (folds, timepoints, len(imp_b))

    assert all_results['w_proba'].shape == (folds, timepoints, len(labels_test), 2)
    assert all_results['imp_nb_proba'].shape == (folds, timepoints, len(X_imp_nb), 2)
    assert all_results['imp_b_proba'].shape == (folds, timepoints, len(X_imp_b), 2)

    return all_results

def pca(X):
    n_trials, n_channels, n_timepoints = X.shape
    PC1_trials = []

    for trial in range(n_trials):
        X_T = X[trial].T    # n_times * n_vertices
        pca = PCA(n_components=1)
        PC1 = pca.fit_transform(X_T).ravel()    # n_times
        PC1_trials.append(PC1)
    
    PC1_trials = np.array(PC1_trials)
    assert PC1_trials.shape == (n_trials, n_timepoints)

    return PC1_trials

def get_fft_phase(data, foi, sfreq):
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
        tmin, tmax = round(1.25-5/f, 5), 1.25
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

def get_preds(all_results):
    ### get decoding labels and probabilities
    ### shape: n_folds, n_times, n_epochs
    imp_lbl_nb = all_results['imp_nb_label'].copy()
    imp_proba_nb = all_results['imp_nb_proba'].copy()
    imp_lbl_b = all_results['imp_b_label'].copy()
    imp_proba_b = all_results['imp_b_proba'].copy()
    
    ### average over folders
    ### shape: n_times, n_epochs
    imp_lbl_nb = np.mean(imp_lbl_nb, axis=0)
    imp_proba_nb = np.mean(imp_proba_nb, axis=0)
    imp_lbl_b = np.mean(imp_lbl_b, axis=0)
    imp_proba_b = np.mean(imp_proba_b, axis=0) 

    return imp_lbl_nb, imp_proba_nb, imp_lbl_b, imp_proba_b

def get_phase_per_time(imp_label, phase_angle, random_state):
    '''
    Parameters:
    imp_label: predicted label at each timepoint
    phase_angle: estimate phase per foi per trial
    random_state: int, for replication of the selection

    Returns:
    phase_per_timepoint_adj: at each timepoint, the phase angles of the trials that are labeled adjective (subsampled)
    phase_per_timepoint_n: at each timepoint, the phase angles of the trials that are labeled noun (subsampled)
    '''
    
    rng = np.random.default_rng(random_state)
    phase_per_timepoint_adj = []
    phase_per_timepoint_n = []

    for t in range(imp_label.shape[0]):
        # find adj trials where decoding label > 0.5
        # find n trials where decoding label < 0.5
        adj_trials = np.where(imp_label[t, :] > 0.5)[0]
        n_trials = np.where(imp_label[t, :] < 0.5)[0]

        # find the minimal number of trial
        n_min = min(len(adj_trials), len(n_trials))
        
        # randomly downsample both sets to n_min
        adj_trials_ds = rng.choice(adj_trials, size=n_min, replace=False)
        n_trials_ds = rng.choice(n_trials, size=n_min, replace=False)

        # get phases for these trials (shape: n_foi x trials)
        adj_phase = phase_angle[:, adj_trials_ds]
        n_phase = phase_angle[:, n_trials_ds]

        phase_per_timepoint_adj.append(adj_phase)
        phase_per_timepoint_n.append(n_phase) 
    
    return phase_per_timepoint_adj, phase_per_timepoint_n

def itpc_over_time(phase_per_timepoint, foi):
    '''
    Parameters:
    phase_per_timepoint: at each timepoint, the phase angles of the trials that are classified as certain label
    foi: list, frequencies of interest

    Returns:
    itpc: n_foi * n_times, the itpc of the phases of potentially different trials at each timepoint
    '''
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

def plot_itpc_over_time(itpc_adj, itpc_n, sfreq, foi, save_path=None, title=None):

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
    if title:
        fig.suptitle(title, fontsize=14)
        plt.subplots_adjust(top=0.9)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)  
    else:
        plt.show()

def compute_phase_vector(data):
    '''
    Parameters:
    data: phase angles, shape (foi, epochs)

    Returns:
    vector_lengths, shape (foi,)
    vector_angles, shape (foi,)
    '''
    complex_phase = np.exp(1j*data)
    mean_vector = np.mean(complex_phase, axis=1)
    vector_lengths = np.abs(mean_vector)
    vector_angles = np.angle(mean_vector)  
    return vector_lengths, vector_angles

def categorize_phase_by_label(imp_lbl_nb, imp_lbl_b, X_phase_nb, X_phase_b, X_phase, random_state):
    '''
    Parameters:
    imp_lbl_nb: n_epochs * n_times, the predicted labels for non-binding condition
    imp_lbl_b: n_epochs * n_times, the predicted labels for binding condition
    X_phase_nb: n_epochs * n_vertices * n_times, the source estimates of the time window for phase estimation for non-binding condition
    X_phase_b: n_epochs * n_vertices * n_times, the source estimates of the time window for phase estimation for binding condition
    X_phase: n_epochs * n_vertices * n_times, the source estimates of the time window for phase estimation
    random_state: int, for replication
    

    Returns:
    X_phase_nb_adj_ds, X_phase_nb_n_ds, X_phase_b_adj_ds, X_phase_b_n_ds: the subsampled source estimates of the time window for phase estimation
    
    '''
    # average label over folds
    avg_lbl_nb = np.mean(imp_lbl_nb, axis=0)
    avg_lbl_b = np.mean(imp_lbl_b, axis=0)

    # create trial label based on averaged label
    lbl_nb = np.where(avg_lbl_nb >0.5, 'ADJ-NB', 'N-NB')
    lbl_b = np.where(avg_lbl_b >0.5, 'ADJ-B', 'N-B')

    nb_adj_idx = np.where(lbl_nb == "ADJ-NB")[0]
    nb_n_idx = np.where(lbl_nb == "N-NB")[0]
    b_adj_idx = np.where(lbl_b == "ADJ-B")[0]
    b_n_idx = np.where(lbl_b == "N-B")[0]

    # categorize souce data for phase estimate based on labels
    # return n_epochs, sum_vertices_across_labels, n_times
    X_phase_nb_adj = X_phase_nb[nb_adj_idx, :, :]
    X_phase_nb_n = X_phase_nb[nb_n_idx, :, :]
    X_phase_b_adj = X_phase_b[b_adj_idx, :, :]
    X_phase_b_n = X_phase_b[b_n_idx, :, :]

    # sub-sample 
    rng = np.random.default_rng(random_state)

    # find the minimal number of trial
    n_min = min(len(X_phase_nb_adj), 
                len(X_phase_nb_n),
                len(X_phase_b_adj), 
                len(X_phase_b_n))
    
    # sub-sample trial index
    nb_adj_idx_ds = rng.choice(len(X_phase_nb_adj), size=n_min, replace=False)
    nb_n_idx_ds = rng.choice(len(X_phase_nb_n), size=n_min, replace=False)
    b_adj_idx_ds = rng.choice(len(X_phase_b_adj), size=n_min, replace=False)
    b_n_idx_ds = rng.choice(len(X_phase_b_n), size=n_min, replace=False)
    baseline_idx = rng.choice(len(X_phase), size=n_min, replace=False)

    # sub-sample the phase trials
    X_phase_nb_adj_ds = X_phase_nb_adj[nb_adj_idx_ds, :, :]
    X_phase_nb_n_ds   = X_phase_nb_n[nb_n_idx_ds, :, :]
    X_phase_b_adj_ds  = X_phase_b_adj[b_adj_idx_ds, :, :]
    X_phase_b_n_ds    = X_phase_b_n[b_n_idx_ds, :, :]
    X_phase_ds   = X_phase[baseline_idx, :, :]

    return X_phase_nb_adj_ds, X_phase_nb_n_ds, X_phase_b_adj_ds, X_phase_b_n_ds, X_phase_ds

def plot_vl(vl_nb_adj, vl_nb_n, vl_b_adj, vl_b_n, vl_base, foi, title, save_path=None, ):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(foi, vl_nb_adj, label='NB: Adjecive', color='#D55E00')
    ax.plot(foi, vl_nb_n, label='NB: Noun', color='#F0A500')
    ax.plot(foi, vl_b_adj, label='B: Adjective', color='#0072B2')
    ax.plot(foi, vl_b_n, label='B: Noun', color='#56B4E9')
    ax.plot(foi, vl_base, label='Baseline', color='gray')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase Consistency')
    ax.legend(loc='upper right')
    ax.set_title(title, va='bottom')
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

def compute_relative_phase(va_1, va_2):
    diff = va_1 - va_2
    return np.arctan2(np.sin(diff), np.cos(diff))

'''
def plot_va(rva_1, rva_2, 
            foi,
            labels=[],
            colors=[],
            marker_size=15, alpha=0.7,
            jitter_strength=0.05,
            highlight_freqs=None, 
            highlight_color='black',
            seed=42,
            highlight_marker='',
            title='',
            save_path=None,
            ):
    
    np.random.seed(seed)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111, projection='polar')
    
    phase_sets = [rva_1, rva_2]
    base_radii = [1.0, 1.2]
    for phase_vals, label, color, base_radius in zip(phase_sets, labels, colors, base_radii):
        jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(phase_vals))
        radii = base_radius + jitter
        if highlight_freqs:
            foi = np.array(foi)
            in_range = (foi >= highlight_freqs[0]) & (foi <= highlight_freqs[1])
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
                       label=f'{label} {highlight_freqs[0]}-{highlight_freqs[1]} Hz' if label else None,
                       marker=highlight_marker)
        else:
            ax.scatter(phase_vals, radii, color=color, alpha=alpha, s=marker_size, label=label)
    
    ax.set_rticks([])
    ax.set_title(title, va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.close(fig) 
'''

def plot_va(rva_1, rva_2, 
            foi,
            labels=[],
            colors=[],
            marker_size=15, alpha=0.7,
            jitter_strength=0.05,
            highlight_freqs=None,   # e.g. [(4,7), (8,12), (13,30)]
            highlight_colors=None,  # optional: different colors per band
            seed=42,
            highlight_markers='',
            title='',
            save_path=None,
            ):
    
    np.random.seed(seed)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111, projection='polar')
    
    if highlight_freqs is None:
        highlight_freqs = []
    if highlight_colors is None:
        highlight_colors = ['black'] * len(highlight_freqs)
    if highlight_markers is None:
        highlight_markers = ['o'] * len(highlight_freqs)

    phase_sets = [rva_1, rva_2]
    base_radii = [1.0, 1.2]
    
    for phase_vals, label, color, base_radius in zip(phase_sets, labels, colors, base_radii):
        jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(phase_vals))
        radii = base_radius + jitter
        foi = np.array(foi)
        
        ax.scatter(phase_vals, radii, color=color, alpha=alpha, s=marker_size, label=label)
        
        for (frange, hcolor, hmarker) in zip(highlight_freqs, highlight_colors, highlight_markers):
            in_range = (foi >= frange[0]) & (foi <= frange[1])
            if np.any(in_range):
                ax.scatter(phase_vals[in_range], 
                           radii[in_range], 
                           color=hcolor, 
                           edgecolor='black',
                           alpha=1.0, 
                           s=marker_size+10, 
                           label=f'{label} {frange[0]}-{frange[1]} Hz',
                           marker=hmarker)
    
    ax.set_rticks([])
    ax.set_title(title, va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)