/bin/bash
###plumed###
export PATH=/usr/local/plumed-2.8.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/plumed-2.8.3/lib:$LD_LIBRARY_PATH

###gromacs2022.5-plumed-2.8.3###
source /usr/local/gromacs-plumed/bin/GMXRC
'''
#Minimization
setenv GMX_MAXCONSTRWARN -1
#step4.0 - soft-core minimization
gmx grompp -f mdp/step4.0_minimization.mdp -o step4.0_minimization.tpr -c system.gro -r system.gro -p system.top -n index.ndx -maxwarn 2
gmx mdrun -deffnm step4.0_minimization -ntmpi 1 -ntomp 8
#step4.1
gmx grompp -f mdp/step4.1_minimization.mdp -o step4.1_minimization.tpr -c step4.0_minimization.gro -r system.gro -p system.top -n index.ndx -maxwarn 1
gmx mdrun -deffnm step4.1_minimization -ntmpi 1 -ntomp 8

unset GMX_MAXCONSTRWARN
#Equilibration
gmx grompp -f mdp/step4.2_equilibration.mdp -o step4.2_equilibration.tpr -c step4.1_minimization.gro -r system.gro -p system.top -n index.ndx -maxwarn 1
gmx mdrun -deffnm step4.2_equilibration -nb gpu -bonded gpu -gpu_id 3 -ntmpi 1 -ntomp 8 -update gpu 

#Production
gmx grompp -f mdp/step5_production.mdp -o step5_production.tpr -c step4.2_equilibration.gro -p system.top -n index.ndx 
'''
gmx mdrun -deffnm step5_production -nb gpu -bonded gpu -gpu_id 1 -ntmpi 1 -ntomp 8 -update gpu -cpi
