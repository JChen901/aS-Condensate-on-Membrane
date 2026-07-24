#!/bin/bash
###plumed###
export OMP_NUM_THREADS=8
export PATH=/usr/local/plumed-2.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/plumed-2.9/lib:$LD_LIBRARY_PATH
source /usr/local/gromacs-2022.5_plumed/bin/GMXRC


python average_contact_map.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/helix_trendy/alhx_0/step5_production.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/helix_trendy/alhx_0/tj_pbc.xtc -o 01.npy
wait
python average_contact_map.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/helix_trendy/alhx_0/step5_production.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/helix_trendy/alhx_0_re1/tj_pbc.xtc -o 02.npy
wait
python average_contact_map.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/helix_trendy/alhx_0/step5_production.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/helix_trendy/alhx_0_re2/tj_pbc.xtc -o 03.npy
wait
python average_contact_map.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/helix_trendy/alhx_700/step5_production.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/helix_trendy/alhx_700/tj_pbc.xtc -o 71.npy
wait
python average_contact_map.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/helix_trendy/alhx_700/step5_production.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/helix_trendy/alhx_700_re1/tj_pbc.xtc -o 72.npy
wait
python average_contact_map.py -s /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/helix_trendy/alhx_700/step5_production.tpr -f /data/jychen/MD_projects/curvature_phase/Alpha-Synuclein/martini3001/Martini_alhx/mdrun/M3IDP/helix_trendy/alhx_700_re2/tj_pbc.xtc -o 73.npy
