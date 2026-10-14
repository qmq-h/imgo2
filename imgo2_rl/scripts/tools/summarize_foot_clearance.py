#!/usr/bin/env python3
"""Reconstruct lowest FOOT collision height on a flat plane from Gazebo telemetry.

Standard library only. Geometric height is not a measured contact signal.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
import audit_amp_dataset as fk

REPO = Path(__file__).resolve().parents[3]
LEGS = ("FL", "FR", "RL", "RR")


def rpy_rotation(values):
    return fk.matmul(fk.axis_rotation([0, 0, 1], values[2]),
                     fk.matmul(fk.axis_rotation([0, 1, 0], values[1]), fk.axis_rotation([1, 0, 0], values[0])))


def pose(chain, angles):
    p, r, index = [0.0, 0.0, 0.0], fk.IDENTITY, 0
    for kind, xyz, origin_r, axis in chain:
        p = [a + b for a, b in zip(p, fk.rotate(r, xyz))]
        r = fk.matmul(r, origin_r)
        if kind != "fixed":
            r = fk.matmul(r, fk.axis_rotation(axis, angles[index]))
            index += 1
    if index != len(angles):
        raise ValueError("Joint count mismatch")
    return p, r


def foot_shapes(root, name):
    shapes = []
    for c in root.find(f"link[@name='{name}']").findall("collision"):
        origin = c.find("origin")
        xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
        rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
        shape = list(c.find("geometry"))[0]
        shapes.append((xyz, rpy_rotation(rpy), shape.tag, {k: tuple(float(x) for x in v.split()) for k, v in shape.attrib.items()}))
    return shapes


def collision_bottom(position_w, rotation_w, shapes):
    values = []
    for xyz, origin_r, kind, params in shapes:
        center_z = position_w[2] + fk.rotate(rotation_w, xyz)[2]
        rot = fk.matmul(rotation_w, origin_r)
        if kind == "sphere":
            extent = params["radius"][0]
        elif kind == "cylinder":
            axis_z = abs(rot[2][2])
            extent = params["length"][0] * 0.5 * axis_z + params["radius"][0] * math.sqrt(max(0, 1 - axis_z * axis_z))
        elif kind == "box":
            extent = sum(abs(rot[2][i]) * params["size"][i] * 0.5 for i in range(3))
        else:
            raise ValueError(f"Unsupported foot collision: {kind}")
        values.append(center_z - extent)
    return min(values)


def quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    at = (len(ordered) - 1) * probability
    lo = int(at)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered)-1)] - ordered[lo]) * (at - lo)


def height_summary(times, heights):
    runs, start = [], None
    for i, height in enumerate(heights):
        if height > 0.005 and start is None:
            start = i
        if height <= 0.005 and start is not None:
            if start > 0 and times[i] - times[start] >= 0.04:
                runs.append((times[start], times[i], max(heights[start:i])))
            start = None
    periods = [b[0]-a[0] for a,b in zip(runs, runs[1:])]
    return {"min_m": min(heights), "p95_m": quantile(heights,.95), "p99_m": quantile(heights,.99), "max_m": max(heights),
            "above_1cm_fraction": sum(v > .01 for v in heights)/len(heights),
            "above_2cm_fraction": sum(v > .02 for v in heights)/len(heights),
            "height_bouts_over_5mm": len(runs),
            "height_bout_mean_peak_m": statistics.mean(r[2] for r in runs) if runs else None,
            "height_bout_mean_duration_s": statistics.mean(r[1]-r[0] for r in runs) if runs else None,
            "height_bout_median_period_s": statistics.median(periods) if periods else None}


def analyze(raw_file, urdf):
    data = json.loads(raw_file.read_text())
    if "Frames" in data:
        return analyze_reference(raw_file, urdf, data)
    odom, joints = data["odom"], data["joint_states"]
    root = ET.parse(urdf).getroot()
    chains = {leg: fk.read_chain(root, leg+"_FOOT") for leg in LEGS}
    shapes = {leg: foot_shapes(root, leg+"_FOOT") for leg in LEGS}
    times, heights = [], {leg: [] for leg in LEGS}
    cursor = 0
    for t, values in joints:
        if not odom[0][0] <= t <= odom[-1][0]:
            continue
        while cursor + 1 < len(odom)-1 and odom[cursor+1][0] < t:
            cursor += 1
        a, b = odom[cursor], odom[cursor+1]
        alpha = (t-a[0])/(b[0]-a[0]) if b[0]>a[0] else 0
        angles=[]
        for index in (4,5,3):
            delta=math.atan2(math.sin(b[index]-a[index]),math.cos(b[index]-a[index]))
            angles.append(a[index]+alpha*delta)
        rotation=rpy_rotation(angles)
        base_z=a[2]+alpha*(b[2]-a[2])
        for leg in LEGS:
            p,r=pose(chains[leg],[values[leg+'_'+j+'_joint'][0] for j in ('hip','thigh','shank')])
            rp=fk.rotate(rotation,p)
            heights[leg].append(collision_bottom([0,0,base_z+rp[2]],fk.matmul(rotation,r),shapes[leg]))
        times.append(t)
    if len(times)<20:
        raise ValueError("Not enough aligned pose/joint samples")
    return {"raw_file":str(raw_file),"urdf":str(urdf),"samples":len(times),
            "duration_s":times[-1]-times[0],"per_foot":{leg:height_summary(times,v) for leg,v in heights.items()},
            "limits":"Lowest primitive collision surface above z=0 flat ground; reconstructed at telemetry rate. 5mm bouts are geometric, not contact/air-time measurement."}


def analyze_reference(raw_file, urdf, data):
    import ast
    root=ET.parse(urdf).getroot()
    chains={leg:fk.read_chain(root,leg+'_FOOT') for leg in LEGS}
    shapes={leg:foot_shapes(root,leg+'_FOOT') for leg in LEGS}
    dt=float(data['FrameDuration'])
    heights={leg:[] for leg in LEGS}
    for frame in data['Frames']:
        rotation=fk.quaternion_rotation(frame[3:7])
        for index,leg in enumerate(LEGS):
            p,r=pose(chains[leg],frame[7+3*index:10+3*index])
            rp=fk.rotate(rotation,p)
            heights[leg].append(collision_bottom([0,0,frame[2]+rp[2]],fk.matmul(rotation,r),shapes[leg]))
    source=REPO/'imgo2_deploy/scripts/eval_gazebo_policy.py'
    fn=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='autocorr_peak')
    ns={'math':math}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
    per_foot={}
    times=[i*dt for i in range(len(data['Frames']))]
    for leg,values in heights.items():
        avg=statistics.mean(values)
        period,score,freq=ns['autocorr_peak']([v-avg for v in values],dt)
        per_foot[leg]={**height_summary(times,values),
                       'p95_minus_p05_lift_m':quantile(values,.95)-quantile(values,.05),
                       'height_autocorr_period_s':period if math.isfinite(period) else None,
                       'height_autocorr_score':score if math.isfinite(score) else None}
    return {'raw_file':str(raw_file),'source':'reference_motion','urdf':str(urdf),'dt':dt,
            'samples':len(times),'per_foot':per_foot,
            'limits':'Recorded root pose and joint FK, no contact force labels. Relative height lift does not establish measured air time.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_files",nargs="+",type=Path)
    parser.add_argument("--urdf",type=Path,default=REPO/'imgo2_description/urdf/imgo2_real.urdf')
    parser.add_argument("--json",type=Path)
    args=parser.parse_args()
    out=[analyze(path,args.urdf) for path in args.raw_files]
    if args.json: args.json.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
