#!/usr/bin/env python3
"""Plot logged host targets, four hip feedbacks and IMU on a shared time axis.

Offline only. Matplotlib is needed for rendering; no ROS or hardware is started.
"""
import argparse
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
from compare_ppo_sim_real_imu import rpy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log',type=Path,required=True);p.add_argument('--output-prefix',type=Path,required=True)
    p.add_argument('--start',type=float);p.add_argument('--end',type=float)
    args=p.parse_args();c,bad=read_csv(args.log/'robot.csv')
    if bad:raise ValueError('Malformed CSV rows')
    for font in font_manager.fontManager.ttflist:
        if font.name in ('Noto Sans CJK JP','WenQuanYi Micro Hei','Noto Sans CJK SC'):
            plt.rcParams['font.family']=font.name;break
    plt.rcParams['axes.unicode_minus']=False
    ix=[i for i,t in enumerate(c['t_mono_ms']) if (args.start is None or t>=args.start*1000) and (args.end is None or t<=args.end*1000)]
    times=[c['t_mono_ms'][i]/1000 for i in ix]
    angles=[rpy(*(c['quat_'+k][i] for k in 'wxyz')) for i in ix]
    fig,axes=plt.subplots(7,1,figsize=(16,16),sharex=True)
    for key,label in [('cmd_vx','vx m/s'),('cmd_vy','vy m/s'),('cmd_wz','wz rad/s')]:
        axes[0].plot(times,[c[key][i] for i in ix],label=label,linewidth=.8)
    axes[0].set_ylabel('速度指令');axes[0].legend(loc='upper right',ncol=3)
    for k,label in [(0,'Roll'),(1,'Pitch')]:axes[1].plot(times,[a[k]*180/math.pi for a in angles],label=label,linewidth=.7)
    axes[1].set_ylabel('姿态 (deg)');axes[1].legend(loc='upper right',ncol=2)
    for k in 'xyz':axes[2].plot(times,[c['gyro_'+k][i] for i in ix],label=k.upper(),linewidth=.6)
    axes[2].set_ylabel('Gyro (rad/s)');axes[2].legend(loc='upper right',ncol=3)
    for axis,j,label in zip(axes[3:],[0,3,6,9],['FL 左前','FR 右前','BL 左后','BR 右后']):
        axis.plot(times,[c[f'q_des_{j:02}'][i]*180/math.pi for i in ix],label='主机目标角 q_des',color='#dc7c22',linewidth=.85)
        axis.plot(times,[c[f'q_{j:02}'][i]*180/math.pi for i in ix],label='实测角 q',color='#147fba',linewidth=.65,alpha=.9)
        axis.set_ylabel(label+'\n髋关节 (deg)');axis.legend(loc='upper right',ncol=2,fontsize=8)
    for axis in axes:
        axis.grid(alpha=.25);axis.axhline(0,color='gray',linewidth=.5)
    axes[-1].set_xlabel('日志起点后的主机单调时间 (s)')
    fig.suptitle('新实机日志：四髋目标/反馈与IMU\n'+args.log.name+'；原始快照，无显示平滑；接触/托举时刻未标记',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96));args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(str(args.output_prefix)+'.png',dpi=150)
    fig.savefig(str(args.output_prefix)+'.pdf')
    plt.close(fig)


if __name__=='__main__':main()
