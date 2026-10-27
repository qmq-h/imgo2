#!/usr/bin/env python3
"""Plot recorded PPO maneuver comparisons; requires numpy and matplotlib."""
import csv
import math
import os
from pathlib import Path
os.environ['MPLCONFIGDIR']='/tmp/imgo2_mpl_cache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import argparse
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input-root', type=Path, required=True)
p.add_argument('--output-prefix', type=Path, required=True)
args=p.parse_args()
root=args.input_root
def load(path):
    with path.open() as f:
        rows=list(csv.DictReader(f))
    return {k:np.array([float(r[k]) for r in rows]) for k in rows[0] if k!='stage'}
g=load(root/'gazebo_low03/imu.csv')
i=load(root/'isaac_low03/trace.csv')
o=load(root/'gazebo_low03/odom.csv')
t=g['t_s']; ti=i['t_s']
lo=np.clip(np.searchsorted(o['t_s'],t-.1),0,len(o['t_s'])-1)
hi=np.clip(np.searchsorted(o['t_s'],t+.1),0,len(o['t_s'])-1)
dt=np.maximum(o['t_s'][hi]-o['t_s'][lo],1e-9)
vg=((o['x_m'][hi]-o['x_m'][lo])*np.cos(g['yaw_rad'])+(o['y_m'][hi]-o['y_m'][lo])*np.sin(g['yaw_rad']))/dt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(3,2,figsize=(14,10),layout='constrained')
fig.suptitle('Current deployed PPO: turns and fast deceleration\nSame policy 53a57f... | Kp 25, Kd 0.5 | flat ground | headless simulation',fontsize=15)
ax=axs[0,0]
ax.plot(t,g['scheduled_vx_mps'],'k--',lw=1.5,label='command')
ax.plot(t,vg,color='#d97706',lw=.9,label='Gazebo (odom estimate)')
ax.plot(ti,i['vx_body'],color='#0284c7',lw=.8,label='Isaac (body velocity)')
ax.set(xlim=(5,49),ylabel='Forward speed (m/s)',title='0.8 -> 0.3 m/s: fast (0.2s) and slow (3s)')
for a,b in ((24,26.2),(39.2,44.2)):ax.axvspan(a,b,color='#64748b',alpha=.12)
ax.legend(fontsize=8,loc='upper right')
ax=axs[0,1]
ax.plot(t,g['scheduled_wz_rads'],'k--',lw=1.5,label='command')
ax.plot(t,g['gyro_z_rads'],color='#d97706',lw=.65,label='Gazebo')
ax.plot(ti,i['gyro_z_rads'],color='#0284c7',lw=.7,label='Isaac')
ax.set(xlim=(52,95),ylabel='Yaw rate (rad/s)',title='Left / right in-place turns, then walking turn')
ax.legend(fontsize=8)
for ax,data,tt,name in ((axs[1,0],g,t,'Gazebo'),(axs[1,1],i,ti,'Isaac')):
    for end,c,label in ((24.2,'#dc2626','fast decel'),(42.2,'#059669','slow decel')):
        mask=(tt>=end-.5)&(tt<end+2)
        late=(tt>=end+2)&(tt<end+6)
        value=np.degrees(data['pitch_rad']-np.mean(data['pitch_rad'][late]))
        ax.plot(tt[mask]-end,value[mask],color=c,lw=1,label=label)
    ax.axvline(0,color='k',ls=':',lw=.8)
    ax.set(xlim=(-.5,2),ylim=(-2,2),ylabel='Pitch offset (deg)',xlabel='Time after decel ramp ends (s)',title=f'{name}: pitch settles after deceleration')
    ax.legend(fontsize=8)
for ax,data,tt,name in ((axs[2,0],g,t,'Gazebo'),(axs[2,1],i,ti,'Isaac')):
    ax.plot(tt,np.degrees(data['roll_rad']),color='#7c3aed',lw=.65,label='roll')
    ax.plot(tt,np.degrees(data['pitch_rad']),color='#e11d48',lw=.65,label='pitch')
    ax.set(xlim=(54.2,79.2),ylim=(-3,5),ylabel='Body angle (deg)',title=f'{name}: roll / pitch during in-place turns')
    ax.axvspan(55.2,63.2,color='#059669',alpha=.08)
    ax.axvspan(70.2,78.2,color='#0284c7',alpha=.08)
    ax.legend(fontsize=8)
for row in (0,2):
    for ax in axs[row]:ax.set_xlabel('Policy time (s)')
for ax in axs.flat:ax.grid(alpha=.2)
out=args.output_prefix
fig.savefig(out.with_suffix('.png'),dpi=160)
fig.savefig(out.with_suffix('.pdf'))
print(out.with_suffix('.png'))
