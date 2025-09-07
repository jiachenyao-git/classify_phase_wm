'''
Make BEM from reconstructed MRI

This script runs in terminal. It requires freesurfer installation, and individual MRI structural scans
'''

'''
How to use this code:
1. open mac terminal
2. set shell environment variables of freesurfer, by entering the following
    2.1. export FREESURFER_HOME=/Applications/freesurfer
    2.2. export FS_LICENSE=/Applications/freesurfer/license.txt
    2.3. source $FREESURFER_HOME/SetUpFreeSurfer.sh
3. confirm environment variables, by entering the following:
    3.1. echo $FREESURFER_HOME 
    3.2. which mri_watershed  
    3.3. ls $FREESURFER_HOME/lib/bem/
4. cd to the directory of this current script
5. run, in the terminal:
    python3 SR00_make_bem.py --subject 01 --overwrite
'''

#!/usr/bin/env python3
import os
import argparse
import mne
import mne.bem

parser = argparse.ArgumentParser(description='Generate BEM using MNE and FreeSurfer.')
parser.add_argument('--subject', type=str, required=True,
                    help='Subject folder name (e.g., 01)')
parser.add_argument('--subjects_dir', type=str,
                    default='/Users/jiachenyao/Desktop/PhaseCode/data/mri_reconst',
                    help='Parent folder of FreeSurfer subjects')
parser.add_argument('--freesurfer_home', type=str,
                    default='/Applications/freesurfer',
                    help='Path to FreeSurfer installation')
parser.add_argument('--overwrite', action='store_true',
                    help='Overwrite existing BEM')
args = parser.parse_args()

# set up environment variable
os.environ['FREESURFER_HOME'] = args.freesurfer_home
os.environ['SUBJECTS_DIR'] = args.subjects_dir

# source FreeSurfer setup
# os.system(f'source {args.freesurfer_home}/SetUpFreeSurfer.sh')

# use mne.bem.watershed to make bem
mne.bem.make_watershed_bem(subject=args.subject,
                           subjects_dir=args.subjects_dir,
                           overwrite=args.overwrite)

print(f"BEM generated for subject {args.subject} in {args.subjects_dir}/{args.subject}/bem/")

# make scalp surfaces
mne.bem.make_scalp_surfaces(
    subject=args.subject,
    subjects_dir=args.subjects_dir,
    force=True,
    overwrite=True
)

print(f"Scalp surfaces generated for subject {args.subject}")


