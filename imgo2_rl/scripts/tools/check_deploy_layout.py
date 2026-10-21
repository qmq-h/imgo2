#!/usr/bin/env python3
"""Offline checks for the real/ros2_ws split. Standard library; opens no devices."""
import ast
from pathlib import Path
import re
import subprocess

REPO = Path(__file__).resolve().parents[3]
WS = REPO / 'imgo2_deploy/ros2_ws'
REAL = REPO / 'imgo2_deploy/real'


def main():
    failures = []

    def check(condition, description):
        print(('PASS ' if condition else 'FAIL ') + description)
        if not condition:
            failures.append(description)

    link = WS / 'src/imgo2_description'
    check(link.is_symlink() and link.resolve() == REPO / 'imgo2_description',
          'ROS model link resolves to the unique repository model')
    for relative in ('build.sh', 'policy/imgo2/base.yaml', 'policy/imgo2/ppo/policy.pt',
                     'policy/imgo2/amp/policy.pt', 'scripts/fastdds_udp_only.xml',
                     'src/imgo2_deploy/launch/gazebo.launch.py'):
        check((WS / relative).is_file(), 'ROS resource: ' + relative)
    for relative in ('CMakeLists.txt', 'config/deployment.yaml', 'policy/ppo/config.yaml',
                     'policy/ppo/policy.pt'):
        check((REAL / relative).is_file(), 'real resource: ' + relative)

    cmake = (WS / 'src/imgo2_deploy/CMakeLists.txt').read_text()
    check('${PROJECT_ROOT_DIR}/../../imgo2_description' in cmake,
          'MuJoCo model root accounts for the extra workspace level')
    for name in ('check_policy_trace.py', 'run_gazebo_direction_audit.py', 'run_gazebo_ppo_ramp.py'):
        path = WS / 'scripts' / name
        tree = ast.parse(path.read_text())
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'REPO' for t in n.targets))
        depth = assignment.value.slice.value
        check(path.resolve().parents[depth] == REPO, name + ' derives the repository root')

    for doc in (REPO / 'command.md', REPO / 'docs/xiaoji_cmd/策略导出与ROS2仿真.md'):
        text = doc.read_text()
        check('imgo2_deploy/ros2_ws' in text and not re.search(
            r'imgo2_deploy/(?:policy|scripts|install|src)/', text),
            str(doc.relative_to(REPO)) + ' uses the ROS workspace paths')

    def ignored(path):
        result = subprocess.run(['git', 'check-ignore', '--no-index', '-q', path], cwd=REPO)
        if result.returncode not in (0, 1):
            raise RuntimeError('git check-ignore failed for ' + path)
        return result.returncode == 0

    for path in ('imgo2_deploy/real/library', 'imgo2_deploy/real/build_layout/CMakeCache.txt',
                 'imgo2_deploy/real/log/robot.csv', 'imgo2_deploy/ros2_ws/library',
                 'imgo2_deploy/ros2_ws/build/robot_msgs/CMakeCache.txt',
                 'imgo2_deploy/ros2_ws/install/setup.bash', 'imgo2_deploy/ros2_ws/log/latest'):
        check(ignored(path), 'generated/downloaded path ignored: ' + path)
    for path in ('imgo2_deploy/real/policy/ppo/policy.pt',
                 'imgo2_deploy/ros2_ws/policy/imgo2/amp/policy.pt',
                 'imgo2_deploy/ros2_ws/policy/imgo2/ppo/policy.pt',
                 'imgo2_deploy/ros2_ws/policy/imgo2/himloco/himloco.pt',
                 'imgo2_deploy/ros2_ws/src/imgo2_deploy/library/core/rl_sdk/rl_sdk.cpp'):
        check(not ignored(path), 'deployment source/weight retained: ' + path)
    result = subprocess.check_output(['git', 'ls-files', '-i', '-c', '--exclude-standard'], cwd=REPO)
    check(not result.strip(), 'no tracked paths are ignored')
    check(not (REAL / '.gitignore').exists(), 'ignore rules live only in root .gitignore')
    print(f'\n{len(failures)} failure(s); no hardware accessed')
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
