#!/usr/bin/env python3
"""Replay recorded Gazebo actor inputs and verify action/target/feedback conversion."""
import argparse
import csv
import collections
import hashlib
import io
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
import yaml

REPO = Path(__file__).resolve().parents[3]


def analyze(path, policy, config):
    with path.open() as source: rows=list(csv.DictReader(source))
    raw=torch.tensor([[float(r[f"raw_{i}"]) for i in range(12)] for r in rows])
    target=torch.tensor([[float(r[f"target_{i}"]) for i in range(12)] for r in rows])
    obs=torch.tensor([[float(r[f"obs_{i}"]) for i in range(45)] for r in rows])
    modes=torch.tensor([int(r["joint_target_clip"]) for r in rows]).bool()
    scale=torch.tensor(config["action_scale"]); offset=torch.tensor(config["default_dof_pos"])
    lower=torch.tensor(config["clip_actions_lower"]); upper=torch.tensor(config["clip_actions_upper"])
    clipped_raw=torch.clamp(raw,min=lower,max=upper)
    expected_target=torch.where(modes[:,None],torch.clamp(raw*scale+offset,min=lower,max=upper),clipped_raw*scale+offset)
    expected_feedback=torch.where(modes[:,None],raw,clipped_raw)
    expected_feedback=torch.clamp(expected_feedback,-config["clip_obs"],config["clip_obs"])
    with torch.inference_mode(): predicted=policy(obs)
    result={"trace":str(path),"frames":len(rows),"all_joint_target_clip":bool(modes.all()),
            "actor_max_diff":float((predicted-raw).abs().max()),
            "target_max_diff":float((target-expected_target).abs().max()),
            "last_action_max_diff":float((obs[1:,33:45]-expected_feedback[:-1]).abs().max()),
            "raw_over_3_element_fraction":float((raw.abs()>3).float().mean()),
            "raw_over_3_frame_fraction":float((raw.abs()>3).any(dim=1).float().mean()),
            "max_abs_raw":float(raw.abs().max()),
            "processed_target_clipped_frame_fraction":float(((raw*scale+offset<lower)|(raw*scale+offset>upper)).any(dim=1).float().mean())}
    result["passed"]=result["actor_max_diff"]<2e-5 and result["target_max_diff"]<2e-6 and result["last_action_max_diff"]<2e-6
    model=ET.parse(REPO/"imgo2_description/urdf/imgo2_real.gazebo.urdf").getroot()
    joint_names=[f"{leg}_{joint}_joint" for leg in ("FL","FR","RL","RR") for joint in ("hip","thigh","shank")]
    limits=[model.find(f"joint[@name='{name}']/limit").attrib for name in joint_names]
    hardware_low=torch.tensor([float(limit["lower"]) for limit in limits])
    hardware_high=torch.tensor([float(limit["upper"]) for limit in limits])
    beyond=(target<hardware_low)|(target>hardware_high)
    result["controller_additional_position_limit"]={"affected_elements":int(beyond.sum()),
        "affected_frames":int(beyond.any(dim=1).sum()),
        "limits":"ROS2 controller also clamps target to URDF limits before PD; Isaac JointAction does not explicitly perform this extra clamp. This difference remains."}
    transient=path.parent/"transient.json"
    if transient.is_file():
        telemetry=json.loads(transient.read_text())
        start=telemetry["policy_entry_wall"]+4
        pairs=[(wall,odom[0]) for wall,odom in zip(telemetry["odom_wall"],telemetry["odom"]) if wall>=start]
        if len(pairs)>2:
            low,high=pairs[0][0],pairs[-1][0]
            wall_span=high-low; sim_span=pairs[-1][1]-pairs[0][1]
            selected=[float(row["wall_s"]) for row in rows if low<=float(row["wall_s"])<=high]
            quantized=len(set(selected))<len(selected)/2
            if quantized:
                # Legacy traces rounded 1e9-second epochs to whole seconds.
                # Count complete rounding bins, avoiding the cut boundary bins.
                counts=collections.Counter(float(row['wall_s']) for row in rows)
                bins=range(math.ceil(low+.5),math.floor(high-.5)+1)
                bins=list(bins)
                actor_wall=sum(counts[float(t)] for t in bins)/len(bins) if bins else None
                method='Estimated from complete one-second rounded timestamp bins'
            else:
                actor_wall=(len(selected)-1)/(selected[-1]-selected[0])
                method='First/last actor timestamp in stable received-odom window'
            result["stable_clock"]={"wall_span_s":wall_span,"simulation_span_s":sim_span,
                                    "real_time_factor":sim_span/wall_span,
                                    "actor_hz_wall":actor_wall,"actor_hz_simulation":actor_wall/(sim_span/wall_span) if actor_wall else None,
                                    "actor_timestamp_quantized":quantized,"actor_rate_method":method,
                                    "limits":"Aggregate rate from received odom stamps; not individual feedback latency."}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces",nargs="+",type=Path)
    parser.add_argument("--policy",type=Path,default=REPO/"imgo2_deploy/ros2_ws/policy/imgo2/amp/policy.pt")
    parser.add_argument("--config",type=Path,default=REPO/"imgo2_deploy/ros2_ws/policy/imgo2/amp/config.yaml")
    parser.add_argument("--json",type=Path)
    args=parser.parse_args()
    policy_bytes=args.policy.read_bytes()
    policy=torch.jit.load(io.BytesIO(policy_bytes),map_location="cpu").eval()
    config=yaml.safe_load(args.config.read_text())["imgo2/amp"]
    results=[analyze(p,policy,config) for p in args.traces]
    for result in results:
        result['tested_policy']=str(args.policy.resolve())
        result['tested_policy_sha256']=hashlib.sha256(policy_bytes).hexdigest()
    if args.json: args.json.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
    if not all(r["passed"] for r in results): raise SystemExit(1)


if __name__=="__main__": main()
