#!/bin/bash
###plumed###
export OMP_NUM_THREADS=8
export PATH=/usr/local/plumed-2.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/plumed-2.9/lib:$LD_LIBRARY_PATH
source /usr/local/gromacs-2022.5_plumed/bin/GMXRC

gro_posre=system_posre.gro
gro=system.gro
top=system.top
'''
# Minimization
setenv GMX_MAXCONSTRWARN -1
# step6.0 - soft-core minimization
# If you encountered "There are 1 perturbed non-bonded pair interaction ......" error message, 
# please modify rvdw and rcoulomb values from 1.1 to 2.0 in the step6.0_minimization.mdp file
gmx grompp -f mdp/step6.0_minimization.mdp -o step6.0_minimization.tpr -c $gro -r $gro -p $top -n index.ndx -maxwarn 2
gmx mdrun -deffnm step6.0_minimization -ntmpi 1 -ntomp 8

# step6.1
gmx grompp -f mdp/step6.1_minimization.mdp -o step6.1_minimization.tpr -c step6.0_minimization.gro -r $gro -p $top -n index.ndx -maxwarn 1
gmx mdrun -deffnm step6.1_minimization -ntmpi 1 -ntomp 8
unsetenv GMX_MAXCONSTRWARN

# step6.2
gmx grompp -f mdp/step6.2_equilibration.mdp -o step6.2_equilibration.tpr -c step6.1_minimization.gro -r $gro -p $top -n index.ndx -maxwarn 2

gmx mdrun -deffnm step6.2_equilibration -nb gpu -bonded gpu -update gpu -gpu_id 0 -ntmpi 1 -ntomp 8 -cpi


# Equilibration

gmx grompp -f mdp/step6.3_equilibration.mdp -o step6.3_equilibration.tpr -c step6.2_equilibration.gro -r $gro_posre -p $top -n index.ndx -maxwarn 2
gmx mdrun -deffnm step6.3_equilibration -nb gpu -bonded gpu -gpu_id 0 -ntmpi 1 -ntomp 8

gmx grompp -f mdp/step6.4_equilibration.mdp -o step6.4_equilibration.tpr -c step6.3_equilibration.gro -r $gro_posre -p $top -n index.ndx -maxwarn 2
gmx mdrun -deffnm step6.4_equilibration -nb gpu -bonded gpu -gpu_id 0 -ntmpi 1 -ntomp 8

gmx grompp -f mdp/step6.5_equilibration.mdp -o step6.5_equilibration.tpr -c step6.4_equilibration.gro -r $gro_posre -p $top -n index.ndx -maxwarn 2
gmx mdrun -deffnm step6.5_equilibration -nb gpu -bonded gpu -gpu_id 0 -ntmpi 1 -ntomp 8

gmx grompp -f mdp/step6.6_equilibration.mdp -o step6.6_equilibration.tpr -c step6.5_equilibration.gro -r $gro_posre -p $top -n index.ndx -maxwarn 2
gmx mdrun -deffnm step6.6_equilibration -nb gpu -bonded gpu -gpu_id 0 -ntmpi 1 -ntomp 8 


# Production
gmx grompp -f mdp/step7_production.mdp -o step7_production.tpr -c step6.6_equilibration.gro -r $gro_posre -p $top -n index.ndx
'''
gmx mdrun -deffnm step7_production -nb gpu -bonded gpu -update gpu -gpu_id 1 -ntmpi 1 -ntomp 8 -cpi
