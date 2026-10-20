"""Reject unsupported deployment contracts before any device is opened."""
import pathlib
import shutil
import subprocess
import sys
import tempfile
exe, policy, config = sys.argv[1:]
with tempfile.TemporaryDirectory(prefix='go_rl_preflight_') as tmp:
    root = pathlib.Path(tmp)
    shutil.copytree(policy, root/'policy')
    original = (root/'policy/ppo/config.yaml').read_text()
    for before, after in [('num_observations: 45', 'num_observations: 46'),
                          ('observations_history: []', 'observations_history: [0]'),
                          ('ang_vel_scale: 0.25', 'ang_vel_scale: .nan'),
                          ('model_name: "policy.pt"', 'model_name: "missing.pt"')]:
        assert before in original
        (root/'policy/ppo/config.yaml').write_text(original.replace(before, after))
        result = subprocess.run([exe, '--check-policy', '--policy-dir', str(root/'policy'), '--config', config], capture_output=True, text=True, timeout=15)
        assert result.returncode==1 and 'deployment error:' in result.stderr, result
        assert 'start failed' not in result.stderr
    (root/'policy/ppo/config.yaml').write_text(original)
    invalid = root/'invalid.yaml'
    for field in ["policy_to_hardware: [0, 0]", "stand_pose: [0, 0]", "crouch_pose: [0]", "crouch_settle_s: 3",
                  "return_stand_s: 0", "return_stand_s: -2", "return_stand_s: .nan",
                  "stand_pose: [9, .87, -1.82, 0, .87, -1.82, 0, .87, -1.82, 0, .87, -1.82]"]:
        invalid.write_text('deployment:\n  '+field+'\n')
        result = subprocess.run([exe, '--check-policy', '--policy-dir', str(root/'policy'), '--config', str(invalid)], capture_output=True, text=True, timeout=15)
        assert result.returncode==1 and 'deployment error:' in result.stderr
print('Invalid dimensions/history/scales/model/mapping rejected before hardware startup')
