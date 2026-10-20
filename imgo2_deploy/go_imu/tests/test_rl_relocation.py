"""Copy the complete project outside the workspace, build, install, relocate and preflight."""
import pathlib
import shutil
import subprocess
import sys
import tempfile
source, torch_dir = sys.argv[1:]
with tempfile.TemporaryDirectory(prefix='go_rl_relocation_') as tmp:
    root=pathlib.Path(tmp)
    shutil.copytree(source,root/'source',ignore=shutil.ignore_patterns('build','cmake-build-*','__pycache__','.git'))
    def run(args):
        result=subprocess.run(args,cwd=root,capture_output=True,text=True,timeout=180)
        assert result.returncode==0, result.stdout+'\n'+result.stderr
        return result.stdout
    run(['cmake','-S',str(root/'source'),'-B',str(root/'build'),'-DCMAKE_BUILD_TYPE=Release','-DGO_IMU_BUILD_RL=ON','-DGO_IMU_BUILD_APPS=OFF','-DBUILD_TESTING=OFF','-DTorch_DIR='+torch_dir])
    run(['cmake','--build',str(root/'build'),'-j2'])
    run(['cmake','--install',str(root/'build'),'--prefix',str(root/'installed')])
    shutil.move(root/'installed',root/'relocated')
    shutil.rmtree(root/'source');shutil.rmtree(root/'build')
    out=run([str(root/'relocated/bin/rl_real_imgo2'),'--check-policy'])
    assert '45 observations -> 12 actions' in out
print('Copied project built independently; installed executable/resources relocated and policy loaded')
