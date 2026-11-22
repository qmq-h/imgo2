#!/usr/bin/env python3
"""Compare captured actor I/O, with stdlib numerical analysis and optional plots.

--verify-actor uses optional Torch to reconstruct missing real raw outputs.
Saved enriched CSVs allow --prepared analysis without Torch. --plot requires
Matplotlib; statistics, quantiles, distribution distances and PSD use stdlib.
Never controls robots, edits input logs, exports or installs policies.
"""
import argparse
import bisect
import cmath
import csv
import hashlib
import json
import math
import os
import statistics as st
from pathlib import Path

from analyze_command_history import map_host

ROOT = Path(__file__).resolve().parents[3]
JOINTS = [f'{leg}_{kind}' for leg in ['FL', 'FR', 'RL', 'RR'] for kind in ['hip', 'thigh', 'shank']]
SCALE = [.125, .25, .25]*4
DEFAULT = [0, .87, -1.82]*4
LABELS = [f'obs{i:02} gyro {axis} ×0.25' for i, axis in enumerate('XYZ')]
LABELS += [f'obs{i+3:02} gravity {axis}' for i, axis in enumerate('XYZ')]
LABELS += [f'obs{i+6:02} command {axis}' for i, axis in enumerate(['vx', 'vy', 'wz'])]
LABELS += [f'obs{i+9:02} {j} q-default' for i, j in enumerate(JOINTS)]
LABELS += [f'obs{i+21:02} {j} dq ×0.05' for i, j in enumerate(JOINTS)]
LABELS += [f'obs{i+33:02} {j} previous_action' for i, j in enumerate(JOINTS)]
LABELS += [f'action{i:02} {j} clipped' for i, j in enumerate(JOINTS)]
LABELS += [f'raw{i:02} {j} *' for i, j in enumerate(JOINTS)]
LABELS += [f'q_des{i:02} {j} (rad) **' for i, j in enumerate(JOINTS)]
GROUPS = [('body', '机身与速度指令', range(9)), ('q', '12关节位置输入', range(9, 21)),
          ('dq', '12关节速度输入', range(21, 33)), ('previous', '12维上一动作输入', range(33, 45)),
          ('action', '12维裁剪后动作输出', range(45, 57)), ('raw', '12维raw输出：实机同输入重算*', range(57, 69)),
          ('target', '12维动作换算目标角**', range(69, 81))]
REAL_WINDOWS = [(71.9, 74.7), (76.8, 79.6), (97.4, 100.2)]


def read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(a):
    import torch  # Optional, only for paired forward verification/reconstruction.
    torch.set_num_threads(1)
    model = torch.jit.load(str(a.policy), map_location='cpu').eval()
    real_raw, sim_raw = read(a.real/'policy.csv'), read(a.sim/'policy_raw.csv')
    odom = [{k: float(v) for k, v in r.items()} for r in read(a.sim/'odom.csv')]
    mapped = [{'host_s': float(r['host_s'])} for r in sim_raw]
    assert all(b['host_s'] > x['host_s'] for x, b in zip(mapped, mapped[1:]))
    assert odom[0]['host_s'] <= mapped[0]['host_s'] and mapped[-1]['host_s'] <= odom[-1]['host_s'], 'no host-time extrapolation'
    map_host(mapped, odom)
    validation = {}
    for name, source in [('real', real_raw), ('sim', sim_raw)]:
        obs = [[float(r[f'obs_{i:02}' if name == 'real' else f'obs_{i}']) for i in range(45)] for r in source]
        logged = [[float(r[f'act_{i:02}' if name == 'real' else f'action_clipped_{i}']) for i in range(12)] for r in source]
        with torch.no_grad():
            y = model(torch.tensor(obs, dtype=torch.float32))
        assert tuple(y.shape) == (len(source), 12)
        delta = y.clamp(-3, 3)-torch.tensor(logged)
        error = delta.abs().max().item()
        assert error < 2e-5, (name, 'attached policy does not reproduce captured actions', error)
        raw = y.tolist() if name == 'real' else [[float(r[f'raw_{i}']) for i in range(12)] for r in source]
        raw_error = max(abs(u-v) for x, z in zip(y.tolist(), raw) for u, v in zip(x, z))
        assert raw_error < 2e-5
        rows = []
        for k, (r, x, clip, unclipped) in enumerate(zip(source, obs, logged, raw)):
            t = float(r['input_captured_ms'])/1000 if name == 'real' else mapped[k]['t_s']
            target = [d+s*c for d, s, c in zip(DEFAULT, SCALE, clip)]
            if name == 'sim':
                assert max(abs(v-float(r[f'target_{j}'])) for j, v in enumerate(target)) < 2e-5
                assert all(abs(float(r[f'kp_{j}'])-25) < 1e-5 and abs(float(r[f'kd_{j}'])-.8) < 1e-5 for j in range(12))
            rows.append((t, x+clip+unclipped+target))
        assert all(b[0] > r[0] for r, b in zip(rows, rows[1:]))
        with (a.output/f'{name}_enriched.csv').open('w') as f:
            w = csv.writer(f); w.writerow(['t_s']+[f'v{i:02}' for i in range(81)])
            w.writerows([[t]+v for t, v in rows])
        history = max(abs(b[1][33+j]-r[1][45+j]) for r, b in zip(rows, rows[1:]) for j in range(12))
        validation[name] = {'rows': len(rows), 'clipped_reinference_max_error': error,
                            'clipped_reinference_rmse': delta.square().mean().sqrt().item(),
                            'raw_reinference_max_error': raw_error, 'previous_action_adjacent_max_error': history}
    result = {'policy_sha256': digest(a.policy), 'validation': validation,
              'real_raw_origin': 'recomputed with attached policy, not captured raw output',
              'sim_raw_origin': 'captured at actor forward',
              'target_origin': 'default+scale*captured clipped action, not bus application confirmation',
              'sha256': {str(p): digest(p) for p in [a.policy, a.real/'policy.csv', a.sim/'policy_raw.csv',
                                                    a.sim/'odom.csv', a.sim/'collection.json',
                                                    a.output/'real_enriched.csv', a.output/'sim_enriched.csv']}}
    (a.output/'pair_verification.json').write_text(json.dumps(result, indent=2)+'\n')


def quantile(values, p):
    s = sorted(values); x = (len(s)-1)*p; i = int(x)
    return s[i]+(s[min(i+1, len(s)-1)]-s[i])*(x-i)


def distances(x, y):
    a, b = sorted(x), sorted(y)
    values = sorted(set(a+b)); i = j = 0; ks = w1 = 0.; last = values[0]
    for value in values:
        w1 += abs(i/len(a)-j/len(b))*(value-last)
        while i < len(a) and a[i] <= value: i += 1
        while j < len(b) and b[j] <= value: j += 1
        ks = max(ks, abs(i/len(a)-j/len(b))); last = value
    return ks, w1


def uniform(rows, lo, hi):
    times = [r[0] for r in rows]; data = [[] for _ in range(81)]
    for k in range(round((hi-lo)*50)):
        t = lo+k/50; i = bisect.bisect_right(times, t)
        assert 0 < i < len(rows), 'no extrapolation'
        dt = times[i]-times[i-1]
        assert 0 < dt <= .05, f'gap exceeds50ms: {times[i-1:i+1]}'
        frac = (t-times[i-1])/dt
        for c in range(81): data[c].append(rows[i-1][1][c]*(1-frac)+rows[i][1][c]*frac)
    return data


def psd_basis(n):
    weights = [.5-.5*math.cos(2*math.pi*i/n) for i in range(n)]
    basis = [[cmath.exp(-2j*math.pi*k*i/n)*weights[i] for i in range(n)] for k in range(n//2+1)]
    return weights, basis


def psd(values, weights, basis):
    n = len(values); mean = st.mean(values); z = [v-mean for v in values]
    den = 50*sum(w*w for w in weights)
    power = [abs(sum(x*b for x, b in zip(z, row)))**2/den*(1 if k == 0 or (n % 2 == 0 and k == n//2) else 2)
             for k, row in enumerate(basis)]
    expected = sum((v*w)**2 for v, w in zip(z, weights))/sum(w*w for w in weights)
    assert abs(sum(power)*50/n-expected) <= 1e-8*max(1, expected), 'PSD Parseval check failed'
    return power


def analyze(a):
    rows = {}
    for name in ['real', 'sim']:
        rows[name] = [(float(r['t_s']), [float(r[f'v{i:02}']) for i in range(81)]) for r in read(a.output/f'{name}_enriched.csv')]
    collection = json.loads((a.sim/'collection.json').read_text()); t0 = collection['ppo_pd_start_sim_s']
    assert collection['complete'] and collection['hold_seconds'] == 24
    windows = {'real': REAL_WINDOWS, 'sim': [(t0+x, t0+x+2.8) for x in [3, 9, 15]]}
    all_native = {name: [] for name in rows}; waves = {}; gap_info = {}
    for name in rows:
        waves[name] = []; gap_info[name] = []
        for lo, hi in windows[name]:
            native = [r for r in rows[name] if lo <= r[0] < hi]
            assert len(native) > 100
            all_native[name] += native
            gap_info[name].append({'window': [lo, hi], 'samples': len(native),
                                   'max_actor_gap_ms': 1000*max(b[0]-r[0] for r, b in zip(native, native[1:])),
                                   'command_ranges': [[min(r[1][j] for r in native), max(r[1][j] for r in native)] for j in [6, 7, 8]]})
            waves[name].append(uniform(rows[name], lo, hi))
    n = len(waves['real'][0][0]); weights, basis = psd_basis(n)
    frequencies = [50*k/n for k in range(n//2+1)]
    spectra = {}
    for name in rows:
        per_window = [[psd(w[c], weights, basis) for c in range(81)] for w in waves[name]]
        spectra[name] = [[st.mean(p[c][k] for p in per_window) for k in range(len(frequencies))] for c in range(81)]
    stats = []
    for c, label in enumerate(LABELS):
        item = {'channel': c, 'label': label}
        native_values = {name: [r[1][c] for r in all_native[name]] for name in rows}
        for name, values in native_values.items():
            power = spectra[name][c]
            peak = max((k for k, f in enumerate(frequencies) if 1 <= f <= 20), key=lambda k: power[k])
            item[name] = {'n': len(values), 'mean': st.mean(values), 'std': st.pstdev(values),
                          'rms': math.sqrt(st.mean(v*v for v in values)), 'min': min(values), 'max': max(values),
                          'q05': quantile(values, .05), 'median': quantile(values, .5), 'q95': quantile(values, .95),
                          'peak_1_20_hz': frequencies[peak] if st.pstdev(values) > 1e-9 else None,
                          'band_4_10_rms': math.sqrt(sum(p for f, p in zip(frequencies, power) if 4 <= f <= 10)*50/n)}
        item['ks_distance'], item['wasserstein1'] = distances(native_values['real'], native_values['sim'])
        item['std_ratio_real_sim'] = item['real']['std']/item['sim']['std'] if item['sim']['std'] > 1e-9 else None
        item['band_ratio_real_sim'] = item['real']['band_4_10_rms']/item['sim']['band_4_10_rms'] if item['sim']['band_4_10_rms'] > 1e-9 else None
        stats.append(item)
    report = {'policy_sha256': digest(a.policy), 'real_source': str(a.real), 'sim_source': str(a.sim),
              'windows': gap_info, 'main_window_index': {'real': 1, 'sim': 0}, 'channels': stats,
              'psd': {'fs_hz': 50, 'samples_per_window': n, 'frequency_bin_hz': 50/n,
                      'method': 'linear resampling with50ms gap guard; mean removal; periodic Hann; one-sided PSD; average THREE separate periodograms; no concatenation of windows',
                      'spectrum_range_hz': [.5, 20], 'parseval_checks': 'all passed'},
              'limits': ['Real rope/handling intervals unknown.', 'Real lateral command slightly nonzero; physical real speed not measured.',
                         'Independent gait phases; no cross-run pointwise error or phase fitting.',
                         'Marginal KS/W1 are descriptive distances; temporal autocorrelation prevents interpreting them as independent-sample significance tests.',
                         'Frequency peaks alone do not identify causal mechanisms or prove new state distributions cause instability.',
                         'Inputs are actual scaled actor units; real raw output is reconstructed, targets are derived, not bus application evidence.']}
    correlation = {}
    for name in rows:
        columns = [[r[1][c] for r in all_native[name]] for c in range(57)]
        means, stds = [st.mean(v) for v in columns], [st.pstdev(v) for v in columns]
        correlation[name] = [[sum((x-means[i])*(y-means[j]) for x, y in zip(columns[i], columns[j]))/
                              (len(columns[i])*stds[i]*stds[j]) if min(stds[i], stds[j]) > 1e-9 else None
                              for j in range(57)] for i in range(57)]
    report['correlation'] = {'labels': LABELS[:57], 'matrices': correlation,
                             'note': 'Within-side pooled marginal correlation; constant channels omitted; not causal feedback identification.'}
    (a.output/'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (a.output/'channel_statistics.csv').open('w') as f:
        fields = ['channel', 'label']+[f'{name}_{key}' for name in ['real', 'sim'] for key in stats[0]['real']]+['std_ratio_real_sim', 'band_ratio_real_sim', 'ks_distance', 'wasserstein1']
        writer = csv.DictWriter(f, fields); writer.writeheader()
        for item in stats:
            flat = {k: v for k, v in item.items() if k not in ['real', 'sim']}
            for name in ['real', 'sim']: flat.update({f'{name}_{k}': v for k, v in item[name].items()})
            writer.writerow(flat)
    with (a.output/'spectra.csv').open('w') as f:
        writer = csv.writer(f); writer.writerow(['channel', 'label', 'frequency_hz', 'real_psd', 'sim_psd'])
        for c, label in enumerate(LABELS):
            writer.writerows([[c, label, freq, spectra['real'][c][k], spectra['sim'][c][k]] for k, freq in enumerate(frequencies)])
    if a.plot: plot(a.output, rows, all_native, windows, frequencies, spectra, stats, correlation)
    print('81 channels, 3 windows per side, PSD and distribution analysis complete', flush=True)


def plot(out, rows, pooled, windows, freq, spectra, stats, correlation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    override = os.environ.get('IMGO2_PLOT_FONT')
    if override:
        font_manager.fontManager.addfont(override)
        family = font_manager.FontProperties(fname=override).get_name()
    else:
        available = {f.name for f in font_manager.fontManager.ttflist}
        family = next((n for n in ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei'] if n in available), None)
        if family is None: raise RuntimeError('CJK plotting font missing; set IMGO2_PLOT_FONT to a font file')
    plt.rcParams.update({'font.family': family, 'font.size': 8, 'axes.unicode_minus': False})
    colors = {'real': '#D46B32', 'sim': '#287AA7'}; image_index = []
    def figure(kind, group, title, channels, overview=False):
        channels = list(channels); count = len(channels)
        height = count*(.65 if overview else 1.12)+1.45
        fig, axes = plt.subplots(count, 2, figsize=(13, height), squeeze=False)
        heading = {'time': '时域：主窗口，保留均值', 'spectrum': '频谱：三窗分别去均值后的平均PSD', 'distribution': '分布：三窗原生样本，共用分箱'}[kind]
        fig.suptitle(title+' | '+heading, fontsize=13, y=1-.17/height)
        left_header = '实机11月17日 | 76.8-79.6s' if kind == 'time' else '实机11月17日 | 三段直行，共8.4s'
        right_header = 'Gazebo | 14.9-17.7s | 56d6 / 25/.8' if kind == 'time' else 'Gazebo自身反馈 | 三段，共8.4s | 56d6'
        fig.text(.29, 1-.54/height, left_header, ha='center', color=colors['real'], fontsize=10)
        fig.text(.76, 1-.54/height, right_header, ha='center', color=colors['sim'], fontsize=10)
        fig.subplots_adjust(left=.09, right=.985, bottom=.55/height, top=1-.82/height,
                            hspace=.6 if not overview else .68, wspace=.15)
        for row, c in enumerate(channels):
            ax_pair = axes[row]; yy = []
            values = {name: [r[1][c] for r in pooled[name]] for name in ['real', 'sim']}
            lo, hi = min(values['real']+values['sim']), max(values['real']+values['sim'])
            span = max(hi-lo, .001 if c < 6 else .01)
            if kind == 'distribution':
                edges = [lo-.04*span+(hi-lo+.08*span)*k/32 for k in range(33)]
            for col, name in enumerate(['real', 'sim']):
                ax = ax_pair[col]; color = colors[name]; ss = stats[c][name]
                ax.set_title(LABELS[c], loc='left', fontsize=7 if overview else 9, pad=2)
                ax.grid(True, alpha=.2, linewidth=.5)
                if kind == 'time':
                    a, b = windows[name][1 if name == 'real' else 0]
                    native = [r for r in rows[name] if a <= r[0] < b]
                    xx = [r[0]-a for r in native]; vv = [r[1][c] for r in native]
                    ax.plot(xx, vv, color=color, lw=.8)
                    yy += vv; ax.set_xlim(0, 2.8)
                    ax.text(.99, .9, f'本窗均值 {st.mean(vv):.5g}  std {st.pstdev(vv):.3g}', transform=ax.transAxes,
                            ha='right', va='top', fontsize=6 if overview else 7,
                            bbox={'facecolor': 'white', 'alpha': .75, 'edgecolor': 'none', 'pad': 1})
                elif kind == 'spectrum':
                    power = spectra[name][c]; keep = [(f, p) for f, p in zip(freq, power) if .5 <= f <= 20]
                    if ss['std'] < 1e-9:
                        ax.text(.5, .5, '常量：去均值后无振荡谱', transform=ax.transAxes, ha='center', fontsize=7)
                    else: ax.plot([v[0] for v in keep], [max(v[1], 1e-30) for v in keep], color=color, lw=.9)
                    yy += [v[1] for v in keep]; ax.set_xlim(.5, 20)
                    peak = ss['peak_1_20_hz']
                    text = f'主峰 {peak:.2f}Hz' if peak is not None else '主峰 无'
                    ax.text(.99, .9, text+f'  4-10Hz RMS {ss["band_4_10_rms"]:.3g}', transform=ax.transAxes,
                            ha='right', va='top', fontsize=6 if overview else 7,
                            bbox={'facecolor': 'white', 'alpha': .75, 'edgecolor': 'none', 'pad': 1})
                else:
                    density, _, _ = ax.hist(values[name], bins=edges, density=True, color=color, alpha=.68)
                    yy += list(density); ax.set_xlim(edges[0], edges[-1])
                    ax.axvline(ss['mean'], color=color, ls='--', lw=.8)
                    ax.text(.99, .9, f'均值 {ss["mean"]:.5g}  std {ss["std"]:.3g}', transform=ax.transAxes,
                            ha='right', va='top', fontsize=6 if overview else 7,
                            bbox={'facecolor': 'white', 'alpha': .75, 'edgecolor': 'none', 'pad': 1})
                ax.tick_params(labelsize=6 if overview else 7, pad=1)
            if kind == 'time':
                low, high = min(yy), max(yy); padding = max((high-low)*.1, span*.03)
                for ax in ax_pair: ax.set_ylim(low-padding, high+padding)
            elif kind == 'spectrum':
                high = max(max(yy), 1e-12)
                for ax in ax_pair: ax.set_yscale('log'); ax.set_ylim(high*1e-6, high*3)
            else:
                for ax in ax_pair: ax.set_ylim(0, max(yy)*1.3 if max(yy) else 1)
        for ax in axes[-1]: ax.set_xlabel({'time': '窗口内时间(s)，两次运动相位独立', 'spectrum': '频率(Hz)，PSD绝对量级，未归一化', 'distribution': '原始输入/动作数值，密度积分为1'}[kind])
        fig.text(.5, .1/height, '左右同行共用坐标 | obs为Actor缩放值；PSD单位为对应数值²/Hz | *实机raw重算；**q_des动作换算 | 分布差异不单独证明根因', ha='center', fontsize=7)
        path = out/f'{kind}_{group}.png'; fig.savefig(path, dpi=160 if overview else 170, facecolor='white'); plt.close(fig)
        if not overview: image_index.append({'kind': kind, 'group': group, 'title': title, 'path': str(path.resolve()), 'height_inches': height, 'width_inches': 13})
    for kind in ['time', 'spectrum', 'distribution']:
        for group, title, indices in GROUPS: figure(kind, group, title, indices)
        figure(kind, 'overview', '全部45输入 + 12裁剪动作', range(57), True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.2))
    fig.suptitle('45维输入 + 12维动作的相关性：实机与自身反馈Gazebo', fontsize=13)
    fig.subplots_adjust(left=.08, right=.9, top=.87, bottom=.2, wspace=.2)
    for ax, name, title in zip(axes, ['real', 'sim'], ['实机：三窗合并', 'Gazebo：三窗合并']):
        matrix = [[float('nan') if v is None else v for v in row] for row in correlation[name]]
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r', interpolation='nearest')
        ax.set_facecolor('#dddddd'); ax.set_title(title)
        ax.set_xticks([4, 14.5, 26.5, 38.5, 50.5], ['机身/指令', 'q', 'dq', '上一动作', '动作'])
        ax.set_yticks([4, 14.5, 26.5, 38.5, 50.5], ['机身/指令', 'q', 'dq', '上一动作', '动作'])
        for v in [8.5, 20.5, 32.5, 44.5]: ax.axhline(v, color='black', lw=.5); ax.axvline(v, color='black', lw=.5)
    cb = fig.add_axes([.93, .28, .015, .5]); fig.colorbar(im, cax=cb, label='Pearson相关系数')
    fig.text(.5, .07, '灰色为常量通道 | 相关性仅描述同一侧数据中的共同变化，不是因果或闭环稳定性证明', ha='center', fontsize=9)
    path = out/'correlation_two_columns.png'; fig.savefig(path, dpi=170, facecolor='white'); plt.close(fig)
    image_index.append({'kind': 'correlation', 'group': 'all', 'title': '45输入与12动作相关性', 'path': str(path.resolve()), 'height_inches': 7.2, 'width_inches': 13})
    (out/'figure_index.json').write_text(json.dumps(image_index, ensure_ascii=False, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real', type=Path, required=True); p.add_argument('--sim', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify-actor', action='store_true'); p.add_argument('--plot', action='store_true')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    if a.verify_actor: prepare(a)
    else:
        check = json.loads((a.output/'pair_verification.json').read_text())
        assert check['policy_sha256'] == digest(a.policy)
        for filename, value in check['sha256'].items(): assert digest(filename) == value
    analyze(a)


if __name__ == '__main__': main()
