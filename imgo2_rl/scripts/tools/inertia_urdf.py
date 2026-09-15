import pinocchio as pin
import numpy as np
import os

from pathlib import Path


# 注意：imgo2_model/imgo2_urdf/urdf/imgo2.urdf 只是腿部件片段——没有 base link、
# 没有 <robot> 起始标签，XML 都无法独立解析，pinocchio 更加载不了；本脚本原先指向它，
# 因此必然在 buildModelFromUrdf 处失败。改用完整的训练模型；四份完整 URDF 的关节与
# 物理参数已统一，用哪一份算出的惯量相同。
URDF_PATH = (
    Path(__file__).resolve().parents[2]
    / "source" / "imgo2_rl" / "data" / "imgo2_model" / "imgo2_urdf" / "urdf" / "imgo2.urdf"
)
FLOATING_BASE = True                     # 四足一般 True
# ===========================================


def main():
    assert os.path.exists(URDF_PATH), f"URDF not found: {URDF_PATH}"

    # ---- build model ----
    if FLOATING_BASE:
        model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    else:
        model = pin.buildModelFromUrdf(str(URDF_PATH))

    data = model.createData()

    print("Model loaded.")
    print("nq =", model.nq, " nv =", model.nv)
    print("Number of joints (incl. universe) =", model.njoints)

    # ---- initial configuration (neutral) ----
    q = pin.neutral(model)

    # floating base: [x,y,z, qx,qy,qz,qw]
    if FLOATING_BASE:
        q[0:3] = np.array([0.0, 0.0, 0.0])   # base position
        q[3:7] = np.array([0.0, 0.0, 0.0, 1.0])  # unit quaternion
    
    # 关节初值
    q[7:19] = np.array([
        0.0, 1.37, -2.74,
        0.0, 1.37, -2.74,
        0.0, 1.37, -2.74,
        0.0, 1.37, -2.74,
    ])

    # ---- forward kinematics + composite inertia ----
    pin.forwardKinematics(model, data, q)
    pin.centerOfMass(model, data, q)
    # 关键：计算整体空间惯性
    v = np.zeros(model.nv)
    pin.ccrba(model, data, q, v)

    total_mass = data.mass[0]
    com = data.com[0]

    spatial_inertia = data.Ig          # 6x6 numpy array
    # inertia_com = spatial_inertia[0:3, 0:3]

    total_mass = data.mass[0]
    com = data.com[0]

    # ---- print results ----
    np.set_printoptions(precision=6, suppress=True)

    print("\n========== RESULT (Initial Configuration) ==========")
    print(f"Total mass [kg]: {total_mass:.6f}")
    print(f"Center of Mass (world) [m]: {com}")

    # print("\nInertia at COM (3x3) [kg*m^2]:")
    # print(inertia_com)

    print("\nSpatial inertia:")
    print(spatial_inertia)
    print("====================================================\n")


if __name__ == "__main__":
    main()

