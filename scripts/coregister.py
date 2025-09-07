''' Coregistration

'''
globals().clear()

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import nibabel
import numpy as np

import mne
from mne.io.constants import FIFF
from mne.transforms import apply_trans

subjects_dir = '/Users/jiachenyao/Desktop/PhaseCode/data/mri_reconst'
subject = "01"
t1_fname = subjects_dir + '/' + subject + "/mri/T1.mgz"
t1 = nibabel.load(t1_fname)
t1.orthoview()
plt.show()

data = np.asarray(t1.dataobj)
print(data.shape)

print(t1.affine)
vox = np.array([122, 119, 102])
xyz_ras = apply_trans(t1.affine, vox)
print(
    "Our voxel has real-world coordinates {}, {}, {} (mm)".format(*np.round(xyz_ras, 3))
)

######################################################################
import nibabel as nib

dig_dir = '/Users/jiachenyao/Desktop/PhaseCode/data'
dig_sub = '01'
dig_ses = '01'

dig_fname = f"{dig_dir}/sub-{dig_sub}/dig/sub-{dig_sub}_ses-{dig_ses}_elec.p3d"

dig = mne.channels.read_dig_polhemus_isotrak(dig_fname)