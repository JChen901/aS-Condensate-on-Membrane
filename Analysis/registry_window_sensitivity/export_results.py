"""Validate window 8 and plot IDP-IDP registry-window sensitivity."""
from pathlib import Path
import json
import importlib.util
import pandas as pd
ROOT = Path(__file__).resolve().parent
GROUPS = ('G1_Wcluster_alhx0', 'G2_Wcluster_alhx700', 'G3_Onmem_plus', 'G4_Onmem_wall')
TABLES = ('trajectory_summary.csv', 'event_histograms.csv')
def normalize(table):
    table = table.copy()
    parts = table.variant.str.rsplit('_w', n=1, expand=True)
    table['variant'] = parts[0]
    table['registry_window'] = parts[1].astype(int)
    return table

def main():
    checks = {}
    for group in GROUPS:
        for filename in TABLES:
            table = normalize(pd.read_csv(ROOT/'data'/group/filename))
            baseline = pd.read_csv(ROOT.parent/'data'/group/filename)
            baseline = baseline[baseline.variant.isin(['banded_sym','banded_anti']) & baseline.contact.isin(['one','three','triplet'])]
            current = table[table.registry_window == 8][baseline.columns]
            keys = ['replicate','variant','contact','pair_type']
            if filename == 'event_histograms.csv': keys += ['lifetime_frames']
            pd.testing.assert_frame_equal(current.sort_values(keys).reset_index(drop=True), baseline.sort_values(keys).reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-9)
            checks[f'{group}/{filename}/window8_matches_existing'] = True
        summary = normalize(pd.read_csv(ROOT/'data'/group/'trajectory_summary.csv'))
        expected = 288 if group == 'G4_Onmem_wall' else 72
        assert len(summary) == expected
        assert set(summary.registry_window) == {4,6,8,10}
        assert not summary.duplicated(['replicate','variant','registry_window','contact','pair_type']).any()
        checks[f'{group}/coverage'] = True
    spec = importlib.util.spec_from_file_location('sensitivity_plot', ROOT/'plot_registry_window_sensitivity.py')
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    for metric in ('t_01_ps','t_1e_ps'): plot.plot(metric)
    print(json.dumps({'valid': True, 'checks': checks}, indent=2), flush=True)
    print('IDP-IDP figures with the Hel50 reference plotted; window 8 matches existing persistent results.', flush=True)
if __name__ == '__main__': main()
