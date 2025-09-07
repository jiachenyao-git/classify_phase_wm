'''Compute data covariance matrix

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

fnames_cla = [f"epo_cla_ses{ses:02d}.fif" for ses in range(1, 3)]
fnames_pha = [f"epo_pha_ses{ses:02d}.fif" for ses in range(1, 3)]

epo_cla = load_and_print_data(fnames_cla, epo_dir)
epo_pha = load_and_print_data(fnames_pha, epo_dir)

epo_cla_ses01 = epo_cla['epo_cla_ses01.fif']
epo_cla_ses02 = epo_cla['epo_cla_ses02.fif']
epo_pha_ses01 = epo_pha['epo_pha_ses01.fif']
epo_pha_ses02 = epo_pha['epo_pha_ses02.fif']

del epo_cla
del epo_pha

######################################################################
# compute data covariance matrix
# classify
def compute_cov_classify(epoch, d_cov_tmin=0, d_cov_tmax=0.6, n_cov_tmin=-0.2, n_cov_tmax=0, method='empirical'):
    print(f"Computing data covariance for tmin={d_cov_tmin}, tmax={d_cov_tmax}")
    data_cov = mne.compute_covariance(epoch, tmin=d_cov_tmin, tmax=d_cov_tmax, method=method)

    print(f"Computing noise covariance for tmin={n_cov_tmin}, tmax={n_cov_tmax}")
    noise_cov = mne.compute_covariance(epoch, tmin=n_cov_tmin, tmax=n_cov_tmax, method=method)

    return data_cov, noise_cov

data_cov_cla_ses01, noise_cov_cla_ses01 = compute_cov_classify(epo_cla_ses01)
data_cov_cla_ses02, noise_cov_cla_ses02 = compute_cov_classify(epo_cla_ses02)

# plot data covariance martix
data_cov_cla_ses01.plot(epo_cla_ses01.info)
data_cov_cla_ses02.plot(epo_cla_ses02.info)

del epo_cla_ses01
del epo_cla_ses02

# phase
def compute_cov_phase(epoch, d_cov_tmin=-1.25, d_cov_tmax=0, n_cov_tmin=-1.45, n_cov_tmax=-1.25, method='empirical'):
    print(f"Computing data covariance for tmin={d_cov_tmin}, tmax={d_cov_tmax}")
    data_cov = mne.compute_covariance(epoch, tmin=d_cov_tmin, tmax=d_cov_tmax, method=method)

    print(f"Computing noise covariance for tmin={n_cov_tmin}, tmax={n_cov_tmax}")
    noise_cov = mne.compute_covariance(epoch, tmin=n_cov_tmin, tmax=n_cov_tmax, method=method)

    return data_cov, noise_cov

data_cov_pha_ses01, noise_cov_pha_ses01 = compute_cov_phase(epo_pha_ses01)
data_cov_pha_ses02, noise_cov_pha_ses02 = compute_cov_phase(epo_pha_ses02)

# plot data covariance martix
data_cov_pha_ses01.plot(epo_pha_ses01.info)
data_cov_pha_ses02.plot(epo_pha_ses02.info)

del epo_pha_ses01
del epo_pha_ses02

######################################################################
