#!/bin/bash
###plumed###
export OMP_NUM_THREADS=8
export PATH=/usr/local/plumed-2.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/plumed-2.9/lib:$LD_LIBRARY_PATH
source /usr/local/gromacs-2022.5_plumed/bin/GMXRC


python compute_contact_largest_cluster_fraction.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/Onmem/muti_chain/POPC7-POPG3/alhx_0_plus1/TR_Mdvwhole/pbc.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/Onmem/muti_chain/POPC7-POPG3/alhx_0_plus1/TR_Mdvwhole/pbc.xtc -o 01.npy --fraction 1 --save_cluster_info cluster_info.txt
wait
python compute_contact_largest_cluster_fraction.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/Onmem/muti_chain/POPC7-POPG3/alhx_0_plus2/TR_Mdvwhole/pbc.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/Onmem/muti_chain/POPC7-POPG3/alhx_0_plus2/TR_Mdvwhole/pbc.xtc -o 02.npy --fraction 1 --save_cluster_info cluster_info.txt
wait
python compute_contact_largest_cluster_fraction.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/Onmem/muti_chain/POPC7-POPG3/alhx_0_plus3/TR_Mdvwhole/pbc.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/Onmem/muti_chain/POPC7-POPG3/alhx_0_plus3/TR_Mdvwhole/pbc.xtc -o 03.npy --fraction 1 --save_cluster_info cluster_info.txt
wait
python compute_contact_largest_cluster_fraction.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/Onmem/muti_chain/POPC7-POPG3/alhx_700_WALL1/TR_Mdvwhole/pbc.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/Onmem/muti_chain/POPC7-POPG3/alhx_700_WALL1/TR_Mdvwhole/pbc.xtc -o 71.npy --fraction 0.5 --save_cluster_info cluster_info.txt --last_n_chains 40
wait
python compute_contact_largest_cluster_fraction.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/Onmem/muti_chain/POPC7-POPG3/alhx_700_WALL2/TR_Mdvwhole/pbc.tpr  -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/Onmem/muti_chain/POPC7-POPG3/alhx_700_WALL2/TR_Mdvwhole/pbc.xtc -o 72.npy --fraction 0.5 --save_cluster_info cluster_info.txt --last_n_chains 40
wait
python compute_contact_largest_cluster_fraction.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/Onmem/muti_chain/POPC7-POPG3/alhx_700_WALL3/TR_Mdvwhole/pbc.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/Onmem/muti_chain/POPC7-POPG3/alhx_700_WALL3/TR_Mdvwhole/pbc.xtc -o 73.npy --fraction 0.5 --save_cluster_info cluster_info.txt --last_n_chains 40
wait
