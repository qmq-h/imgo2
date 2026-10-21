"""伪终端集成测试共用的协议常量与小工具（test_demo.py / test_stand.py 共用）。

假电机（伪终端另一端）固定汇报 START_Q 里的输出侧位置，所以测试可以直接算出
“某个 ID 应该收到什么目标”，从而验证上层发出的命令序列。
"""
import math
import struct

MOTOR_IDS = (0, 1, 2)
GEAR_RATIO = 6.33
# 每台电机的输出侧起始位置：按 ID 区分，便于检查各总线/各 ID 没有串台。
START_Q = {0: 0.2, 1: -0.35, 2: 0.55}
KP_WIRE = 798  # 输出侧 kp=25 -> 25/6.33^2*1280
KD_WIRE = 15   # 输出侧 kd=0.5 -> 0.5/6.33^2*1280


def crc16(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x8408 if crc & 1 else 0)
    return crc


def reply(motor_id):
    """假电机的 16 字节反馈：FOC 模式、错误码 0、位置 = START_Q[motor_id]。"""
    rotor_q = round(START_Q[motor_id] * GEAR_RATIO / (2 * math.pi) * 32768)
    payload = struct.pack('<BBBhh ibH', 0xFD, 0xEE, motor_id | (1 << 4),
                          0, 0, rotor_q, 25, 0)
    return payload + struct.pack('<H', crc16(payload))


def control_is_zero(frame):
    """这一帧命令是不是全零（= 卸力）。"""
    return not any(frame[3:15])


def target_q(frame):
    """从命令帧解出输出侧目标位置（rad）。"""
    raw = int.from_bytes(frame[7:11], 'little', signed=True)
    return raw / 32768 * 2 * math.pi / GEAR_RATIO


def target_dq(frame):
    """从命令帧解出输出侧目标速度（rad/s）。"""
    raw = int.from_bytes(frame[5:7], 'little', signed=True)
    return raw / 256 * 2 * math.pi / GEAR_RATIO
