"""
Run the whole knee-bend pipeline: SKEL export -> check -> Blender scene. Works on macOS and Windows.

Run from the repo root, with the project's Python environment:
    python run_knee_bend.py female left
    python run_knee_bend.py all --force          (all four cases)

Output: output/knee_bend/knee_bend_anim_<gender>_<side>.npz and knee_bend_<gender>_<side>.blend
(the ground-truth scene: frame n = SKEL pose n). For the smooth --visual animation, run
scripts/blender_import_knee.py directly with another output name (see README).
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join('output', 'knee_bend')


def find_blender():
    if os.environ.get('BLENDER'):
        return os.environ['BLENDER']
    if shutil.which('blender'):
        return shutil.which('blender')
    if sys.platform == 'darwin':
        candidates = ['/Applications/Blender.app/Contents/MacOS/Blender']
    elif sys.platform == 'win32':
        # newest version last, e.g. C:\Program Files\Blender Foundation\Blender 5.3\blender.exe
        candidates = sorted(glob.glob(r'C:\Program Files\Blender Foundation\Blender*\blender.exe'))[::-1]
    else:
        candidates = []
    for path in candidates:
        if os.path.exists(path):
            return path
    sys.exit('Blender not found. Set the BLENDER environment variable to the blender executable.')


def run(step, cmd):
    """Run one step; stop everything if it fails."""
    print(f'--- {step}', flush=True)
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        sys.exit(f'FAILED at: {step}')


def run_case(gender, side, blender, force):
    npz = os.path.join(OUT_DIR, f'knee_bend_anim_{gender}_{side}.npz')
    blend = os.path.join(OUT_DIR, f'knee_bend_{gender}_{side}.blend')
    print(f'\n===== {gender} {side}')
    # Check this first, so nothing is regenerated when the scene would not be saved anyway.
    if os.path.exists(os.path.join(ROOT, blend)) and not force:
        sys.exit(f'{blend} already exists (maybe edited by a teammate). Use --force to overwrite it.')
    # sys.executable = the Python running this script, i.e. the project's .venv
    run('1/3 export SKEL', [sys.executable, os.path.join('scripts', 'knee_bend_export.py'),
                            '--gender', gender, '--side', side])
    run('2/3 check export', [sys.executable, os.path.join('scripts', 'check_export.py'), npz])
    run('3/3 build Blender scene', [blender, '-b', '--python', os.path.join('scripts', 'blender_import_knee.py'),
                                    '--', npz, blend] + (['--force'] if force else []))


def main():
    parser = argparse.ArgumentParser(description='SKEL knee bend -> Blender, for one or all cases.')
    parser.add_argument('gender', choices=['male', 'female', 'all'])
    parser.add_argument('side', nargs='?', choices=['right', 'left'], default='right')
    parser.add_argument('--force', action='store_true', help='overwrite existing .blend files')
    args = parser.parse_args()

    blender = find_blender()

    if args.gender == 'all':
        cases = [(g, s) for g in ('male', 'female') for s in ('right', 'left')]
    else:
        cases = [(args.gender, args.side)]
    for gender, side in cases:
        run_case(gender, side, blender, args.force)
    print(f'\nDone: {len(cases)} case(s) in {OUT_DIR}')


if __name__ == '__main__':
    main()
