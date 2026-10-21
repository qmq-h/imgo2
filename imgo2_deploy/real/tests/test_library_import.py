"""Build an independent consumer against the public hardware target without starting devices."""
from pathlib import Path
import subprocess
import tempfile
import sys
source=Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix='imgo2_hardware_import_') as temp:
    root=Path(temp)
    (root/'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.10)
project(hardware_consumer LANGUAGES CXX)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(GO_IMU_BUILD_APPS OFF CACHE BOOL "" FORCE)
add_subdirectory("'''+source.as_posix()+'''" toolkit)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE go_imu::hardware)
''')
    (root/'main.cpp').write_text('''#include "hardware/real_hardware.h"
int main() {
    real_hardware::RealHardware hardware;
    auto state=hardware.readState();
    hardware.stop();
    return state.started || !state.joints.empty();
}
''')
    for command in [['cmake','-S',str(root),'-B',str(root/'build')],
                    ['cmake','--build',str(root/'build'),'--target','consumer','-j2'],
                    [str(root/'build/consumer')]]:
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        if result.returncode:raise RuntimeError(result.stdout)
    link=(root/'build/CMakeFiles/consumer.dir/link.txt').read_text()
    assert 'go_imu_display' not in link and 'go_imu_apps' not in link,link
print('External add_subdirectory consumer passed; no display/app libraries linked')
