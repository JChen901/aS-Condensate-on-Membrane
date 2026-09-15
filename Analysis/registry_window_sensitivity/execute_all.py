"""Run 12 trajectories with bounded concurrency, then summarize, export and plot."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import os
ROOT = Path(__file__).resolve().parent
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
def run(job):
    group, rep = job
    result = subprocess.run(
        [sys.executable, str(ROOT/'run_registry_window_sensitivity.py'), '--group', str(group), '--rep', str(rep)],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        print(result.stdout, end='')
        print(result.stderr, end='', file=sys.stderr)
        result.check_returncode()
    print(f'Completed group {group}, replicate {rep}', flush=True)
if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run, [(g,r) for g in range(4) for r in (1,2,3)]))
    subprocess.run([sys.executable, str(ROOT/'run_registry_window_sensitivity.py'), '--summarise'], check=True)
    subprocess.run([sys.executable, str(ROOT/'export_results.py')], check=True)
