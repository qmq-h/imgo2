#!/usr/bin/env python3
"""Audit a single-joint sine CSV log; standard library, optional matplotlib plot.

Reports recorded control-loop timing, not device sampling/application timing.
tau_set_nm is the controller's total PD torque, not a feed-forward command.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def rms(xs):
    return math.sqrt(statistics.fmean(x*x for x in xs))


def solve(matrix, vector):
    a = [list(row) + [v] for row, v in zip(matrix, vector)]
    n = len(vector)
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(a[i][col]))
        a[col], a[pivot] = a[pivot], a[col]
        if abs(a[col][col]) < 1e-14:
            raise ValueError('Singular fit: insufficient time/frequency coverage')
        scale = a[col][col]
        a[col] = [x/scale for x in a[col]]
        for i in range(n):
            if i != col:
                scale = a[i][col]
                a[i] = [x-scale*y for x, y in zip(a[i], a[col])]
    return [row[-1] for row in a]


def fit(t, y, frequency):
    x = [[1., math.sin(2*math.pi*frequency*v), math.cos(2*math.pi*frequency*v)] for v in t]
    c = solve([[math.fsum(v[i]*v[j] for v in x) for j in range(3)] for i in range(3)],
              [math.fsum(v[i]*u for v, u in zip(x, y)) for i in range(3)])
    errors = [u-sum(a*b for a, b in zip(v, c)) for v, u in zip(x, y)]
    mean_y = statistics.fmean(y)
    return dict(mean=c[0], amplitude=math.hypot(c[1], c[2]),
                phase_deg=math.degrees(math.atan2(c[2], c[1])), residual_rms=rms(errors),
                r_squared=1-sum(e*e for e in errors)/sum((v-mean_y)**2 for v in y))


def phase_difference(a, b):
    return (a-b+180) % 360-180


def integrate(t, y):
    return math.fsum((b-a)*(u+v)/2 for a, b, u, v in zip(t, t[1:], y, y[1:]))


def quantile(values, fraction):
    s = sorted(values)
    x = (len(s)-1)*fraction
    k = int(x)
    return s[k] if k == len(s)-1 else s[k]+(x-k)*(s[k+1]-s[k])


def plot(path, t, d, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = {x.name for x in matplotlib.font_manager.fontManager.ttflist}
    for name in ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei']:
        if name in names:
            plt.rcParams['font.sans-serif'] = [name]
            break
    plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(3, 2, figsize=(15, 11), constrained_layout=True)
    a, b, c, e, f, g = axs.flat
    for ax in axs.flat:
        ax.grid(alpha=.25)
        ax.set_xlabel('Time from log start (s)')
    a.plot(t, d['q_set_rad'], label='q target', color='tab:orange')
    a.plot(t, d['q_actual_rad'], label='q feedback', color='tab:blue')
    a.axvspan(0, report['window']['requested_start_s'], color='gray', alpha=.12)
    a.set(title='Absolute joint position', ylabel='rad')
    a.legend()
    start = report['window']['requested_start_s']
    end = min(start+2, report['window']['requested_end_s'])
    idx = [i for i, v in enumerate(t) if start <= v < end]
    for key, color, label in [('q_set_rad','tab:orange','target minus fitted mean'),
                               ('q_actual_rad','tab:blue','feedback minus fitted mean')]:
        mean = report['steady_fits'][key]['mean']
        b.plot([t[i] for i in idx], [d[key][i]-mean for i in idx], color=color, label=label)
    b.set(title='Steady waveform: mean removed separately; NO time shift', ylabel='rad')
    b.legend(fontsize=8)
    c.plot(t, d['dq_actual_rad_s'], color='tab:blue', alpha=.45, lw=.7, label='dq feedback')
    c.plot(t, d['dq_set_rad_s'], color='tab:orange', label='dq target (actually commanded)')
    c.set(title='Velocity: reference and motor estimate', ylabel='rad/s')
    c.legend(fontsize=8)
    for key, label, color in [('tau_set_nm','Total PD torque','tab:red'),
                               ('tau_actual_nm','Motor estimated torque','tab:green')]:
        e.plot([t[i] for i in idx], [d[key][i] for i in idx], label=label, color=color, lw=1)
    e.set(title='Torque: total PD command / estimate, not calibrated force', ylabel='N m')
    e.legend(fontsize=8)
    f.plot(t, [x-y for x, y in zip(d['q_set_rad'], d['q_actual_rad'])], color='tab:purple')
    f.set(title='Position error: q target - q feedback', ylabel='rad')
    steady_idx = [i for i, v in enumerate(t) if report['window']['requested_start_s'] <= v < report['window']['requested_end_s']]
    ts = [t[i] for i in steady_idx]
    ys = [d['dq_actual_rad_s'][i] for i in steady_idx]
    integral = [0.]
    for j in range(1, len(ts)):
        integral.append(integral[-1]+(ts[j]-ts[j-1])*(ys[j]+ys[j-1])/2)
    q0 = d['q_actual_rad'][steady_idx[0]]
    g.plot(ts, [d['q_actual_rad'][i]-q0 for i in steady_idx], label='q feedback - q at window start')
    g.plot(ts, integral, label='Integral of recorded dq (host timestamps)')
    g.set(title='Consistency check: q change vs integral(dq); timing unresolved', ylabel='rad')
    g.legend(fontsize=8)
    identity = ' / '.join(' '.join(x) for x in report['identities'])
    fig.suptitle(f"{identity}: small sine audit | {report['target_frequency_hz']:.3f} Hz | no Gazebo run", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure', type=Path)
    parser.add_argument('--start', type=float, default=2.)
    parser.add_argument('--end', type=float, default=10.)
    args = parser.parse_args()
    with args.input.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    keys = ['t_mono_ms','q_set_rad','dq_set_rad_s','tau_set_nm',
            'q_actual_rad','dq_actual_rad_s','tau_actual_nm','kp','kd']
    d = {k:[float(row[k]) for row in rows] for k in keys}
    if len(rows) < 10 or not all(math.isfinite(v) for xs in d.values() for v in xs):
        raise ValueError('Too few or nonfinite samples')
    t = [(v-d['t_mono_ms'][0])/1000 for v in d['t_mono_ms']]
    dt = [b-a for a, b in zip(t, t[1:])]
    if min(dt) <= 0:
        raise ValueError('Nonmonotonic timestamps')
    center = (min(d['q_set_rad'])+max(d['q_set_rad']))/2
    crossings = []
    for i in range(1, len(t)):
        a, b = d['q_set_rad'][i-1]-center, d['q_set_rad'][i]-center
        if a <= 0 < b:
            crossings.append(t[i-1]+(t[i]-t[i-1])*(-a)/(b-a))
    if len(crossings) < 3:
        raise ValueError('Insufficient rising zero crossings')
    periods = [b-a for a, b in zip(crossings, crossings[1:])]
    frequency = 1/statistics.median(periods)
    idx = [i for i, v in enumerate(t) if args.start <= v < args.end]
    if len(idx) < 10:
        raise ValueError('Insufficient window samples')
    ts = [t[i] for i in idx]
    steady = {k:[xs[i] for i in idx] for k, xs in d.items()}
    fits = {k:fit(ts, steady[k], frequency) for k in keys if k not in ['t_mono_ms','kp','kd']}
    pd = [kp*(q-z)+kd*(v-w) for kp,q,z,kd,v,w in zip(d['kp'],d['q_set_rad'],d['q_actual_rad'],d['kd'],d['dq_set_rad_s'],d['dq_actual_rad_s'])]
    error = [q-z for q, z in zip(steady['q_set_rad'], steady['q_actual_rad'])]
    qchange = steady['q_actual_rad'][-1]-steady['q_actual_rad'][0]
    dqintegral = integrate(ts, steady['dq_actual_rad_s'])
    cycles = []
    for j in range(math.floor((args.end-args.start)*frequency+1e-4)):
        lo, hi = args.start+j/frequency, args.start+(j+1)/frequency
        ii = [i for i, v in enumerate(t) if lo <= v < hi]
        if len(ii) >= 10:
            ff = fit([t[i] for i in ii], [d['q_actual_rad'][i] for i in ii], frequency)
            cycles.append(dict(cycle=j, start_s=lo, end_s=hi, **ff))
    report = dict(input=str(args.input), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        columns=columns, rows=len(rows), identities=sorted({(r['leg'],r['joint']) for r in rows}), duration_s=t[-1],
        logging_interval_ms={str(q):quantile(dt,q)*1000 for q in [0,.01,.5,.99,1]},
        gaps_over_10ms=sum(v>.01 for v in dt), target_frequency_hz=frequency,
        full_target_fit=fit(t,d['q_set_rad'],frequency),
        unique_kp=sorted(set(d['kp'])), unique_kd=sorted(set(d['kd'])),
        adjacent_target_change_fraction=statistics.fmean(a!=b for a,b in zip(d['q_set_rad'],d['q_set_rad'][1:])),
        adjacent_identical_feedback_q_fraction=statistics.fmean(a==b for a,b in zip(d['q_actual_rad'],d['q_actual_rad'][1:])),
        window=dict(requested_start_s=args.start,requested_end_s=args.end,first_sample_s=ts[0],last_sample_s=ts[-1],samples=len(idx)),
        steady_fits=fits,
        amplitude_ratio=fits['q_actual_rad']['amplitude']/fits['q_set_rad']['amplitude'],
        q_phase_difference_deg=phase_difference(fits['q_actual_rad']['phase_deg'],fits['q_set_rad']['phase_deg']),
        tracking_error=dict(mean=statistics.fmean(error),rms=rms(error),std=statistics.pstdev(error),max_abs=max(map(abs,error))),
        pd_formula_match=dict(max_abs=max(abs(v-w) for v,w in zip(d['tau_set_nm'],pd)),rms=rms([v-w for v,w in zip(d['tau_set_nm'],pd)])),
        dq_consistency=dict(window_q_change_rad=qchange,integral_dq_rad=dqintegral,difference_rad=dqintegral-qchange,
                            mean_dq_rad_s=statistics.fmean(steady['dq_actual_rad_s']),
                            mean_velocity_from_q_change_rad_s=qchange/(ts[-1]-ts[0])),
        cycles=cycles,
        interpretation_limits=['One recorded joint/one run; no device receive/apply timing or independent force reference.',
          'Logging interval is not proven feedback sample period. Identical q can mean quantization or stale data.',
          'dq_set is commanded sinusoidal velocity; tau_set is total PD torque, not feed-forward.',
          'Window/cycle differences are within-run variation, not independent repeated experiments.',
          'Phase is target-to-measured-q phase, not communication delay. No simulation or hardware run.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if args.figure:
        plot(args.figure,t,d,report)
    print(json.dumps({k:report[k] for k in ['rows','duration_s','target_frequency_hz','amplitude_ratio','q_phase_difference_deg','tracking_error','dq_consistency']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
