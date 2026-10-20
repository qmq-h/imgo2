"""Invalid CLI/config is rejected before opening any real device."""
import os
import subprocess
import sys
monitor, stand = sys.argv[1:]
for exe, args, code in [(monitor, ['-p', 'abc'], 2), (monitor, ['-t', '-1'], 2)]:
    result = subprocess.run([exe] + args, capture_output=True, timeout=3)
    assert result.returncode == code, (exe, result)
    assert result.stderr, result
# 专用测试入口：人工设备仅/dev/null，时长非法/超速应在打开前拒绝。
env = os.environ.copy()
env['GO_STAND_TEST_DEVICES'] = ','.join(['/dev/null'] * 4)
for duration in ['0', 'nan', '.001']:
    env['GO_STAND_DURATION_S'] = duration
    result = subprocess.run([stand], env=env, capture_output=True, timeout=3)
    assert result.returncode == 1, result
    assert b'stand failed' in result.stderr, result.stderr
    assert b'tty' not in result.stderr and b'start failed' not in result.stderr, result.stderr
print('Application CLI and trajectory preflight rejection passed')
