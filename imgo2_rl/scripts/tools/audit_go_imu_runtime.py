#!/usr/bin/env python3
"""Read uploaded go_imu logs and reproduce a receive-deadline signature offline.

Standard library only. Never opens serial ports, runs a deployment executable,
changes source/config, or treats unlabelled quiet windows as confirmed lifting.
"""
import argparse
from array import array
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import tempfile


def stats(values):
    values = sorted(values)
    if not values:
        return None
    return dict(n=len(values), median=statistics.median(values),
                p95=values[int((len(values)-1)*.95)],
                p99=values[int((len(values)-1)*.99)], max=values[-1])


def rms(values):
    values = list(values)
    return math.sqrt(math.fsum(x*x for x in values)/len(values)) if values else None


def read_csv(path):
    with path.open() as stream:
        rows = csv.reader(stream)
        header = next(rows)
        columns = {key: array('d') for key in header}
        bad = 0
        for row in rows:
            if len(row) != len(header):
                bad += 1
                continue
            for key, value in zip(header, row):
                columns[key].append(float(value))
    return columns, bad


def command_class(vx, vy, wz):
    if max(abs(vx), abs(vy), abs(wz)) < .05:
        return 'zero_command'
    if abs(vy) < .05 and abs(wz) < .05:
        return 'forward_low' if abs(vx) < .15 else 'forward_medium_high'
    if abs(vx) < .05 and abs(wz) < .05:
        return 'lateral_only'
    if abs(vx) < .05 and abs(vy) < .05:
        return 'yaw_only'
    return 'mixed_commands'


def audit_log(directory):
    c, bad = read_csv(directory/'robot.csv')
    n = len(c['tick'])
    if not n:
        return {'directory': str(directory), 'rows': 0}
    t = c['t_mono_ms']
    dt = [t[i]-t[i-1] for i in range(1, n)]
    # Physical contact labels are unavailable. These are motion metrics, not diagnoses.
    hp = [0.0]*n
    sums = [0.0]*3
    gyros = [c['gyro_'+axis] for axis in 'xyz']
    left = 0
    for i in range(n):
        for axis in range(3):
            sums[axis] += gyros[axis][i]
        while left < i and t[i]-t[left] > 200:
            for axis in range(3):
                sums[axis] -= gyros[axis][left]
            left += 1
        hp[i] = sum((gyros[axis][i]-sums[axis]/(i-left+1))**2 for axis in range(3))

    def metrics(ids):
        return dict(rows=len(ids),
                    gyro_rms_rad_s=rms(gyros[a][i] for i in ids for a in range(3)),
                    gyro_hp_200ms_rms_rad_s=math.sqrt(statistics.mean(hp[i] for i in ids)),
                    joint_dq_rms_rad_s=rms(c[f'dq_{j:02}'][i] for i in ids for j in range(12)),
                    tracking_rms_rad=rms(c[f'q_des_{j:02}'][i]-c[f'q_{j:02}'][i]
                                         for i in ids for j in range(12)),
                    imu_age_ms=stats(c['imu_age_ms'][i] for i in ids),
                    bus_total_us=stats(c[f'b{b}_total_us'][i] for i in ids for b in range(4)))

    groups = defaultdict(list)
    windows = defaultdict(list)
    for i in range(n):
        if int(c['mode'][i]) == 3:
            groups[command_class(c['cmd_vx'][i], c['cmd_vy'][i], c['cmd_wz'][i])].append(i)
            windows[int(t[i]//2000)].append(i)
    quiet_loud = []
    for window, ids in sorted(windows.items()):
        if len(ids) < 300:
            continue
        commands = Counter((round(c['cmd_vx'][i], 2), round(c['cmd_vy'][i], 2),
                            round(c['cmd_wz'][i], 2)) for i in ids)
        cmd, count = commands.most_common(1)[0]
        cmd = tuple(0.0 if x == 0 else x for x in cmd)
        if count < .95*len(ids):
            continue
        quiet_loud.append(dict(start_s=window*2, command=cmd,
                               hp_rms=math.sqrt(statistics.mean(hp[i] for i in ids)),
                               bus_max_us=max(c[f'b{b}_total_us'][i] for i in ids for b in range(4)),
                               imu_age_max_ms=max(c['imu_age_ms'][i] for i in ids)))
    by_cmd = defaultdict(list)
    for w in quiet_loud:
        by_cmd[str(w['command'])].append(w)
    contrasts = []
    for cmd, ws in by_cmd.items():
        if len(ws) < 2:
            continue
        low, high = min(ws, key=lambda w:w['hp_rms']), max(ws, key=lambda w:w['hp_rms'])
        if low['hp_rms'] > 0 and high['hp_rms']/low['hp_rms'] > 3:
            contrasts.append(dict(command=cmd, quiet=low, active=high,
                                  ratio=high['hp_rms']/low['hp_rms']))
    stalls = sorted(range(1, n), key=lambda i:dt[i-1], reverse=True)[:10]
    report = dict(directory=str(directory), rows=n, columns=len(c), malformed_rows=bad,
                  duration_s=(t[-1]-t[0])/1000, mode_counts=Counter(map(int,c['mode'])),
                  main_dt_ms=stats(dt), imu_age_ms=stats(c['imu_age_ms']),
                  joint_age_ms=stats(c[f'age_{j:02}'][i] for i in range(n) for j in range(12)),
                  buses={f'b{b}':{k:stats(c[f'b{b}_{k}_us']) for k in ['total','flush','write','reply']}
                         for b in range(4)},
                  largest_main_gaps=[dict(time_s=t[i]/1000, dt_ms=dt[i-1],
                                         imu_age_ms=c['imu_age_ms'][i],
                                         min_joint_age_ms=min(c[f'age_{j:02}'][i] for j in range(12)))
                                     for i in stalls],
                  command_groups={k:metrics(ids) for k,ids in groups.items()},
                  same_command_2s_contrasts=contrasts,
                  cautions=['gyro_hp removes trailing 200ms mean; it is not a calibrated vibration band',
                            'quiet/active windows have no lift/contact labels',
                            'robot rows repeat asynchronous bus snapshots; medians are time-weighted',
                            'policy.csv obs is reconstructed at consumption, not captured in infer'])
    # User could not annotate lifting times. Preserve exact windows as unlabelled evidence.
    report['selected_windows'] = []
    for start, end in [(52,54),(170,172),(176,178),(230,232),
                       (244,246),(248,250),(272,274),(280,282),(324,326),(328,330)]:
        ids = [i for i in range(n) if start*1000 <= t[i] < end*1000]
        if not ids:
            continue
        w = metrics(ids)
        w.update(start_s=start,end_s=end,
                 command=[c[key][ids[len(ids)//2]] for key in ['cmd_vx','cmd_vy','cmd_wz']])
        report['selected_windows'].append(w)
    p, bad_p = read_csv(directory/'policy.csv')
    if p['tick']:
        report['policy'] = dict(rows=len(p['tick']), malformed_rows=bad_p,
                                dt_ms=stats(p['t_mono_ms'][i]-p['t_mono_ms'][i-1]
                                            for i in range(1,len(p['tick']))),
                                result_age_ms=stats(p['result_age_ms']),
                                log_obs_age_ms=stats(p['obs_age_ms']))
    trace = directory/'trace.fault.csv'
    if trace.exists():
        f, bad_f = read_csv(trace)
        report['fault_transactions'] = [dict(time_ms=f['t_ms'][i],bus=f['bus'][i],id=f['id'][i],
                                              waited_us=f['waited_us'][i],bytes=f['bytes'][i],
                                              bad=f['bad'][i],stray=f['stray'][i])
                                         for i,ok in enumerate(f['ok']) if not ok]
        transactions = defaultdict(list)
        for i, ok in enumerate(f['ok']):
            if ok:
                transactions[f"bus{int(f['bus'][i])}/id{int(f['id'][i])}"].append(f['waited_us'][i])
        report['recent_success_transaction_us'] = {key:stats(values)
                                                    for key,values in transactions.items()}
    return report


def deadline_probe(source):
    # Replace only SerialPort read/destructor in an isolated test executable.
    code = r'''
#include "motor_transport.h"
#include <algorithm>
#include <iostream>
#include <thread>
bool late_read=false;
namespace go {
SerialPort::~SerialPort() {}
size_t SerialPort::readUntil(uint8_t* data,size_t len,Deadline end) {
    uint8_t frame[kReplyLen]{};
    frame[0]=kReplyHead0;frame[1]=kHead1;frame[2]=0x10;
    const auto crc=crc16(frame,14);frame[14]=uint8_t(crc);frame[15]=uint8_t(crc>>8);
    if(late_read) std::this_thread::sleep_until(end+std::chrono::milliseconds(2));
    std::copy(frame,frame+std::min(len,kReplyLen),data);return std::min(len,kReplyLen);
}}
int main() {
    go::SerialPort port;go::Feedback feedback;go::TransportStats early,late;std::string error;
    const bool a=go::MotorTransport::receive(port,0,go::SerialPort::Clock::now()+std::chrono::milliseconds(5),feedback,early,error);
    late_read=true;
    const bool b=go::MotorTransport::receive(port,0,go::SerialPort::Clock::now()+std::chrono::milliseconds(5),feedback,late,error);
    std::cout<<"early_ok="<<a<<" late_ok="<<b<<" bytes="<<late.bytesIn<<" bad="<<late.badFrames<<" stray="<<late.strayFrames<<" error="<<error<<"\n";
    return !(a && !b && late.bytesIn==16 && late.badFrames==0 && late.strayFrames==0);
}
'''
    with tempfile.TemporaryDirectory(prefix='go_receive_audit_') as tmp:
        root = Path(tmp)
        (root/'probe.cpp').write_text(code)
        subprocess.run(['g++','-std=c++14','-pthread','-I'+str(source/'Motor/inc'),
                        '-I'+str(source/'serialport/inc'),'-I'+str(source),str(root/'probe.cpp'),
                        str(source/'Motor/src/go_protocol.cpp'),'-o',str(root/'probe')],check=True,
                       capture_output=True,text=True)
        return subprocess.check_output([str(root/'probe')],text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--deadline-probe',action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    result = dict(source=str(source), wire_theory_us=(17+16)*10/4_000_000*1e6,
                  logs=[audit_log(p) for p in sorted((source/'build/log').iterdir()) if (p/'robot.csv').exists()],
                  source_sha256={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [source/'Motor/inc/motor_transport.h', source/'rl_real_imgo2.cpp',
                                           source/'config/deployment.yaml', source/'config/real_config.h',
                                           source/'policy/ppo/policy.pt']})
    if args.deadline_probe:
        result['deadline_probe'] = deadline_probe(source)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    for log in result['logs']:
        print(Path(log['directory']).name,log['rows'],'rows',log['main_dt_ms'])
        print('bus0',log['buses']['b0'],'imu_age',log['imu_age_ms'])
        print('motion groups',json.dumps(log['command_groups'],ensure_ascii=False))
        print('same-command contrasts',json.dumps(log['same_command_2s_contrasts'],ensure_ascii=False))
    print(result.get('deadline_probe','deadline probe not requested'))


if __name__ == '__main__':
    main()
