#!/bin/bash
###plumed###
export OMP_NUM_THREADS=8
export PATH=/usr/local/plumed-2.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/plumed-2.9/lib:$LD_LIBRARY_PATH
source /usr/local/gromacs-2022.5_plumed/bin/GMXRC


python compute_contact_largest_cluster_fraction.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/W_cluster/alhx_0/TR_Mdvwhole/pbc.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/W_cluster/alhx_0/TR_Mdvwhole/pbc.xtc -o 01.npy --fraction 0.5 --save_cluster_info cluster_info.txt
wait
python compute_contact_largest_cluster_fraction.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/W_cluster/alhx_0_re1/TR_Mdvwhole/pbc.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/W_cluster/alhx_0_re1/TR_Mdvwhole/pbc.xtc -o 02.npy --fraction 0.5 --save_cluster_info cluster_info.txt
wait
python compute_contact_largest_cluster_fraction.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/W_cluster/alhx_0_re2/TR_Mdvwhole/pbc.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/W_cluster/alhx_0_re2/TR_Mdvwhole/pbc.xtc -o 03.npy --fraction 0.5 --save_cluster_info cluster_info.txt
wait
python compute_contact_largest_cluster_fraction.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/W_cluster/alhx_700/TR_Mdvwhole/pbc.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/W_cluster/alhx_700/TR_Mdvwhole/pbc.xtc -o 71.npy --fraction 0.5 --save_cluster_info cluster_info.txt
wait
python compute_contact_largest_cluster_fraction.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/W_cluster/alhx_700_re1/TR_Mdvwhole/pbc.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/W_cluster/alhx_700_re1/TR_Mdvwhole/pbc.xtc -o 72.npy --fraction 0.5 --save_cluster_info cluster_info.txt
wait
python compute_contact_largest_cluster_fraction.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/W_cluster/alhx_700_re2/TR_Mdvwhole/pbc.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/W_cluster/alhx_700_re2/TR_Mdvwhole/pbc.xtc -o 73.npy --fraction 0.5 --save_cluster_info cluster_info.txt
wait
