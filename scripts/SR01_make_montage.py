'''
Make individualized montage for EEG data

This script uses the registered electrode locations (.p3d file) to make individualized montage for each subject's EEG data

For each subject:
    2 session of digitization

'''
import numpy as np
import mne
import os.path as op

# define path to .p3d file, subject and session number
subject = '02'
session = '02'

dir = '/Users/jiachenyao/Desktop/PhaseCode/data'

# rename some electrodes
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

        if label.startswith("NAS"):
            x, y, z = map(float, parts[1:])
            nas.append(np.array([x,y,z]))
            
        elif label.startswith("LPA"):
            x, y, z = map(float, parts[1:])
            lpa.append(np.array([x,y,z]))

        elif label.startswith("RPA"):
            x, y, z = map(float, parts[1:])
            rpa.append(np.array([x,y,z]))

        

        else:
            if label in label_dict.keys():
                
                label = label_dict[label]
    
            
            x, y, z = map(float, parts[1:])
            
            coords[label] = np.array([x, y, z])
       
    nas = np.mean(nas, axis = 0)*0.01
    lpa = np.mean(lpa, axis = 0)*0.01
    rpa = np.mean(rpa, axis = 0)*0.01
    coords = {key: val * 0.01 for key, val in coords.items()}
   
    return coords, nas, lpa, rpa

def create_montage_from_coords(coords, nas, lpa, rpa):
    ch_pos = {label: np.array(pos) for label, pos in coords.items()}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos,  nasion=nas, lpa=lpa, rpa=rpa, coord_frame = 'unknown')
    return montage

'''
Start making individual montage
'''
dig_fname = f"{dir}/sub-{subject}/dig/sub-{subject}_ses-{session}_elec.p3d"

# read the .p3d file and create montage
coords, nas, lpa, rpa = read_p3d(dig_fname)
montage = create_montage_from_coords(coords, nas, lpa, rpa)

# path to raw EEG epochs
raw_clas_fname = f"{dir}/raw_epo/sub-{subject}_clas_ses-{session}_epo.fif"
raw_phas_fname = f"{dir}/raw_epo/sub-{subject}_phas_ses-{session}_epo.fif"

# load raw EEG data
raw_clas = mne.read_epochs(raw_clas_fname)
raw_phas = mne.read_epochs(raw_phas_fname)

# set the montage to the EEG data
raw_clas.set_montage(montage)
raw_phas.set_montage(montage)

# set EEG reference to common average reference, which will be used in souce reconstruction
raw_clas.set_eeg_reference(projection=True) 
raw_phas.set_eeg_reference(projection=True) 

# save EEG data with montage
raw_clas.save(raw_clas_fname, overwrite=True)
raw_phas.save(raw_phas_fname, overwrite=True)