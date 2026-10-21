#!/usr/bin/env python3
"""Offline target/estimated-torque alignment figure; not a latency measurement."""
import argparse
from bisect import bisect_right
import math
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'imgo2_rl/scripts/tools'))
from audit_go_imu_runtime import read_csv


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--log',type=Path,required=True)
    p.add_argument('--joint',type=int,default=0);p.add_argument('--start',type=float,default=100)
    p.add_argument('--end',type=float,default=104);p.add_argument('--alignment-ms',type=float,default=10)
    p.add_argument('--output-prefix',type=Path,required=True);a=p.parse_args()
    if not 0<=a.joint<12 or a.alignment_ms<0:raise ValueError('Invalid joint or alignment')
    c,bad=read_csv(a.log/'robot.csv')
    if bad:raise ValueError('Malformed log')
    for font in font_manager.fontManager.ttflist:
        if font.name in ('Noto Sans CJK JP','Noto Sans CJK SC','WenQuanYi Micro Hei'):
            plt.rcParams['font.family']=font.name;break
    plt.rcParams['axes.unicode_minus']=False
    ids=[i for i,t in enumerate(c['t_mono_ms']) if a.start*1000<=t<a.end*1000]
    source=[max(0,bisect_right(c['t_mono_ms'],c['t_mono_ms'][i]-a.alignment_ms)-1) for i in ids]
    j=f'{a.joint:02}';times=[c['t_mono_ms'][i]/1000 for i in ids]
    fig,axes=plt.subplots(3,1,figsize=(14,10),sharex=True)
    axes[0].plot(times,[c[f'q_des_{j}'][i]*180/math.pi for i in ids],label='主机目标角')
    axes[0].plot(times,[c[f'q_{j}'][i]*180/math.pi for i in ids],label='反馈角')
    axes[0].set_ylabel('角度 (deg)')
    actual=[c[f'tau_{j}'][i] for i in ids]
    def prediction(src):return [c[f'kp_{j}'][i]*(c[f'q_des_{j}'][s]-c[f'q_{j}'][i])+c[f'kd_{j}'][i]*(c[f'dq_des_{j}'][i]-c[f'dq_{j}'][i])+c[f'tau_ff_{j}'][i] for i,s in zip(ids,src)]
    now=prediction(ids);past=prediction(source)
    axes[1].plot(times,actual,label='反馈估算力矩',color='#147fba')
    axes[1].plot(times,now,label='同一行目标计算PD',color='#d35438',alpha=.8)
    axes[1].plot(times,past,label=f'{a.alignment_ms:g}ms前目标计算PD',color='#31934f',alpha=.8)
    axes[1].set_ylabel('力矩 (N·m)')
    axes[2].plot(times,[x-y for x,y in zip(actual,now)],label='同一行目标残差',color='#d35438')
    axes[2].plot(times,[x-y for x,y in zip(actual,past)],label='历史目标残差',color='#31934f')
    axes[2].set_ylabel('力矩残差 (N·m)');axes[2].set_xlabel('日志单调时间 (s)')
    for ax in axes:ax.grid(alpha=.25);ax.legend(loc='upper right');ax.axhline(0,color='gray',linewidth=.5)
    fig.suptitle(f'关节{a.joint}：目标与估算力矩的时间对齐\n历史目标仅用于日志诊断；不能把拟合对齐量直接当作USB延迟',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(str(a.output_prefix)+'.png',dpi=150);fig.savefig(str(a.output_prefix)+'.pdf');plt.close(fig)


if __name__=='__main__':main()
