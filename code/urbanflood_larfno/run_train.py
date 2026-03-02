import os
import subprocess
import sys

env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '0'
env['PYTHONIOENCODING'] = 'utf-8'

script_dir = os.path.dirname(os.path.abspath(__file__))
train_script = os.path.join(script_dir, 'train.py')
log_file = os.path.join(script_dir, 'train_log.txt')

with open(log_file, 'w', encoding='utf-8') as f:
    result = subprocess.run(
        [sys.executable, train_script],
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd=script_dir,
    )

print(f'exit code: {result.returncode}')
print(f'Log saved to: {log_file}')
