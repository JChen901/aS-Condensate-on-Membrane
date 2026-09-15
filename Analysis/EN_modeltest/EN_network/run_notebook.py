"""Execute notebook cells sequentially and retain text/figure outputs without a kernel."""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import traceback

os.environ.setdefault('MPLCONFIGDIR', '/tmp/en_modeltest_mpl')
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
path = HERE / 'EN_modeltest.ipynb'
nb = json.loads(path.read_text())
namespace = {'__name__': '__main__'}
current_outputs = []

def capture_show(*args, **kwargs):
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        current_outputs.append({'output_type': 'display_data', 'metadata': {},
                                'data': {'image/png': base64.b64encode(buf.getvalue()).decode(),
                                         'text/plain': ['<Matplotlib figure>']}})
    plt.close('all')

plt.show = capture_show
os.chdir(HERE)
count = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    count += 1
    current_outputs = []
    cell['outputs'] = current_outputs
    cell['execution_count'] = count
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            exec(compile(''.join(cell['source']), f'{path.name}:cell{count}', 'exec'), namespace)
    except Exception as exc:
        current_outputs.append({'output_type': 'error', 'ename': type(exc).__name__,
                                'evalue': str(exc), 'traceback': traceback.format_exc().splitlines()})
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
        print(stream.getvalue())
        raise
    if stream.getvalue():
        current_outputs.insert(0, {'output_type': 'stream', 'name': 'stdout',
                                   'text': stream.getvalue().splitlines(True)})
        print(stream.getvalue(), end='')
nb['metadata']['execution_method'] = 'Sequential fresh-process execution via run_notebook.py; matplotlib output capture'
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f'Executed {count} cells successfully: {path}')
