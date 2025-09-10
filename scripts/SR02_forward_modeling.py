'''
Compute forward models

'''
globals().clear()
import numpy as np
import mne
import os
import os.path as op
import matplotlib.pyplot as plt

mne.set_log_level('warning')

'''
Ingredients:
1. individual MRI reconstruction from Freesurfer
2. individual bem from MRI reconstruction (run SR00_make_bem.py)
2. raw EEG epoch data with digitized electrode position (run SR01_make_montage.py)

Procedures
1. compute head model (BEM)
2. coregistration
3. compute source model
4. coregister sensor locations
5. use 1, 2, 3 to compute forward model
6. use 4 and EEG data to compute inverse solution (beamforming)
7. statistics/visualization
'''

# define subject and session
subject ='02'
session = '02'  ### NOTE change sessions accordingly -- one unique forward model for each session

######################################################################
'''
Step 1. Check T1 and head model
Ingredients:
    indiviudal T1 MRI
    BEM (cf. computed by make_bem.py)
'''
# define data director
data_dir = '/Users/jiachenyao/Desktop/PhaseCode/data'

# define mri directory
mri_dir = '/Users/jiachenyao/Desktop/PhaseCode/data/mri_reconst'

# first set the path to the T1
t1_fname = op.join(mri_dir, subject, 'mri', 'T1.mgz')

# check T1
from nilearn import plotting
plotting.plot_anat(t1_fname)
plt.show()

# check BEM
mne.viz.plot_bem(subject=subject, subjects_dir=mri_dir,
                 mri=t1_fname,
                 orientation='coronal')
plt.show()

######################################################################
'''
Step 2. Coregistration
The next step usually would be to coregister the MRI coordinate system with the MEG coordinate system. This is done to get the sensors into the right relation to the head model for the forward model computation.

For this we only need one type of time window, as we only need the digitized electrode positions of the remaining channels in the raw EEG data (which is the same for classification and phase estimation windows within the same session)

NOTE The following part should be done separately for session 1 and session 2 EEG data. For consistency we use the phase extraction time window to compute the forward model. The forward model does NOT depend on the EEG data itself.
'''

# path to raw EEG epochs
raw_fname = f"{data_dir}/raw_epo/sub-{subject}_phas_ses-{session}_epo.fif"

'''
Step 2.2. Coregisteration using GUI (in Mac terminal)
'''
# do the coregistration in GUI (video cf. https://mne.tools/stable/generated/mne.gui.coregistration.html)
mne.gui.coregistration(subject=subject, subjects_dir=mri_dir, inst=raw_fname)

# save the coregistration as 'sub-xx_ses-xx_trans.fif' under mri_recon dir

input("Make sure you saved the coregisrtration as 'sub-xx_ses-xx_trans.fif' under mri_recon dir, then press ENTER to close the GUI and continue with visualization ...")

# visual coregistration
trans_fname = op.join(mri_dir, subject, f'sub-{subject}_ses-{session}_trans.fif')
info = mne.io.read_info(raw_fname)
fig = mne.viz.plot_alignment(info, trans_fname, subject=subject, dig=True, subjects_dir=mri_dir, verbose=True,)

input("Check coregistration, then close visualization by pressing ENTER ...")

######################################################################
'''
Step 3. Compute source space
The source space defines the positions of the candidate source locations. The following code computes such a source space with an OCT-6 resolution.

Note that this is a surface (not volume) source space.
'''
# compute source space
mne.set_log_level('WARNING')
src = mne.setup_source_space(subject=subject,
                             spacing='oct6',
                             subjects_dir=mri_dir,
                             add_dist=False)

# src contains two parts, one for the left hemisphere (4098 locations) and one for the right hemisphere (4098 locations).
src

# visualize src （NOTE run in terminal)
'''
mne.viz.plot_alignment(info, trans_fname, subject=subject, dig=False, src=src,
                             subjects_dir=mri_dir, verbose=True, meg=False,
                             eeg=False)
'''

mne.viz.plot_alignment(info, trans_fname, subject=subject,
                       src=src, subjects_dir=mri_dir, dig=True,
                       surfaces=['head-dense', 'white'], coord_frame='meg')

input("Check source space, then close visualization by pressing ENTER ...")

# save src 
source_dir = op.join(data_dir, 'source_recon')
os.makedirs(source_dir, exist_ok=True)

src_fname = op.join(source_dir, f'sub-{subject}',f'sub-{subject}_ses-{session}_src.fif')
mne.write_source_spaces(src_fname, src, overwrite=True)

######################################################################
'''
Step 3. Compute forward solutions
Now we have all the ingredients to compute the forward solution.

For EEG, we need 3 layers:
    inner skull, outer skull, skin

First, we compute the BEM model using mne.make_bem_solution(), then we compute the forward solution using mne.make_forward_solution().
'''
### compute bem model
conductivity = (0.3, 0.006, 0.3)  # for three layers
# conductivity = (0.3,)  # for single layer (MEG)
model = mne.make_bem_model(subject=subject, ico=4,
                           conductivity=conductivity,
                           subjects_dir=mri_dir)
bem = mne.make_bem_solution(model)

### save bem
bem_fname = op.join(source_dir, f'sub-{subject}',f'sub-{subject}_ses-{session}_bem.fif')
mne.bem.write_bem_solution(bem_fname, bem, overwrite=True)
print(f"BEM saved as sub-{subject}_ses-{session}_bem.fif")

### compute forward model
trans_fname = op.join(mri_dir, subject, f'sub-{subject}_ses-{session}_trans.fif')

fwd = mne.make_forward_solution(raw_fname, trans=trans_fname,
                                src=src, bem=bem,
                                meg=False,  # exclude MEG channels
                                eeg=True,  # include EEG channels
                                mindist=5.0,  # ignore sources <= 5mm from inner skull
                                n_jobs=1)  # number of jobs to run in parallel

fwd

leadfield = fwd['sol']['data']
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

### save forward model
fwd_fname = op.join(source_dir, f'sub-{subject}', f'sub-{subject}_ses-{session}_eeg_fwd.fif')
mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
print(f"Forward model saved as sub-{subject}_ses-{session}_eeg_fwd.fif")

### compute sensitivity map
sens_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')

# enable correct backend for 3d plotting
clim = dict(kind='percent', lims=(0.0, 50, 99), smoothing_steps=3)  # let's see single dipoles
brain = sens_map.plot(subject=subject, time_label='EEG sensitivity',
                      subjects_dir=mri_dir, clim=clim, smoothing_steps=8)
view = 'lat'    # lateral view
brain.show_view(view)

input("Check sensitivity map, then close visualization by pressing ENTER ...")