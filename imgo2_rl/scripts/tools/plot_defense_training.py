#!/usr/bin/env python3
"""Plot full, unmodified PPO scalar logs for defense figures (optional plotting deps)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prefix',default='ppo_training')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    definitions=[('Train/mean_reward','平均回合回报','记录值'),('Metrics/base_velocity/error_vel_xy','平面速度跟踪误差','日志记录值'),('Metrics/base_velocity/error_vel_yaw','转向角速度跟踪误差','日志记录值'),('Curriculum/terrain_levels','平均地形课程等级','等级')]
    all_values={};sources={}
    for event in sorted(args.run.glob('events.out.tfevents.*')):
        sources[str(event.resolve())]=hashlib.sha256(event.read_bytes()).hexdigest()
        acc=EventAccumulator(str(event),size_guidance={'scalars':0});acc.Reload()
        for tag,_,_ in definitions:
            if tag not in acc.Tags()['scalars']:continue
            points=all_values.setdefault(tag,{})
            for scalar in acc.Scalars(tag):points[scalar.step]=scalar.value
    def draw(ax,tag,title,ylabel):
        points=sorted(all_values[tag].items());x=np.array([p[0] for p in points]);y=np.array([p[1] for p in points]);window=100
        average=np.cumsum(np.insert(y,0,0.));smoothed=(average[window:]-average[:-window])/window
        ax.plot(x,y,color='#a4bfd0',alpha=.48,lw=.5,label='原始日志')
        ax.plot(x[window-1:],smoothed,color='#17658b',lw=2,label='100 轮移动平均')
        for step,color in [(5000,'#14856f'),(6000,'#c37829')]:
            if x[0]<=step<=x[-1]:ax.axvline(step,color=color,lw=1.1,ls='--',label=f'{step} 轮候选')
        ax.set_title(title,loc='left',fontweight='bold',pad=12);ax.set_xlabel('训练迭代');ax.set_ylabel(ylabel);ax.grid(alpha=.18);ax.set_xlim(x[0],x[-1])
    fig,axes=plt.subplots(2,2,figsize=(12.8,7.2))
    for ax,(tag,title,label) in zip(axes.flat,definitions):draw(ax,tag,title,label)
    axes[0,0].legend(fontsize=8,loc='lower right')
    fig.suptitle('Imgo2 PPO 训练过程',fontsize=20,fontweight='bold',x=.075,ha='left')
    fig.text(.075,.01,f'{args.run.name}   完整训练日志；候选策略依据独立运动测试选择。误差采用项目日志口径。',fontsize=9,color='#52606a')
    fig.tight_layout(rect=[.02,.045,1,.93])
    for ext in ('png','pdf','svg'):fig.savefig(args.output/f'{args.prefix}_overview.{ext}',dpi=600,facecolor='white')
    plt.close(fig)
    for n,(tag,title,label) in enumerate(definitions,1):
        fig,ax=plt.subplots(figsize=(9.6,5.4));draw(ax,tag,title,label);ax.legend(fontsize=9)
        fig.tight_layout()
        for ext in ('png','pdf','svg'):fig.savefig(args.output/f'{args.prefix}_{n:02d}.{ext}',dpi=600,facecolor='white')
        plt.close(fig)
    with (args.output/f'{args.prefix}_raw.csv').open('w') as file:
        writer=csv.writer(file);writer.writerow(['tag','iteration','value'])
        for tag in all_values:
            for step,value in sorted(all_values[tag].items()):writer.writerow([tag,step,value])
    meta={'run':str(args.run.resolve()),'sources':sources,'smoothing':'trailing arithmetic mean, 100 logged iterations; original lines also shown','dpi':600,'overview_pixels':[7680,4320],'note':'full scalar logs, no fabricated points or hidden unfavorable tail; distinct task tests select checkpoints; episode return is not proof of terrain passage','tags':{tag:{'n':len(v),'first':min(v),'last':max(v)} for tag,v in all_values.items()}}
    (args.output/f'{args.prefix}_provenance.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta['tags'],indent=2))

if __name__=='__main__':main()
