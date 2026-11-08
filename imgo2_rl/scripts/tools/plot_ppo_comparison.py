#!/usr/bin/env python3
"""Plot recorded policy benchmark summaries (matplotlib optional dependency)."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policies',nargs='+',default=['policy.pt','policy_1.pt','policy_1026.pt','policy_2.pt','policy_latest_1.pt','policy_latest_2.pt'])
    a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads(a.metrics.read_text())
    lookup={(r['policy'],r['case']):r for r in data['results']}
    names=a.policies
    pretty={'policy.pt':'Original 9999','policy_1.pt':'Reference 1','feetheight4_10498.pt':'FHB -4 (10498)','feetheight1_10498.pt':'FHB -1 (10498)'}
    labels=[pretty.get(n,Path(n).stem) for n in names]
    palette=['#df763b','#287fa7','#b7b9bc','#909497','#8882a1','#72a28b']
    colors=[palette[i%len(palette)] for i in range(len(names))]
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    columns=[('height_mean_m',100,'Body height (cm)'),('stride_frequency_hz',1,'Same-foot cadence (Hz)'),('swing_clearance_peak_mean_m',100,'Mean swing peak clearance (cm)')]
    for row,case in enumerate(['forward03','backward03']):
        for col,(key,factor,title) in enumerate(columns):
            ax=axes[row,col];values=[lookup[name,case][key] for name in names]
            ax.bar(range(len(names)),[v*factor if v is not None else float('nan') for v in values],color=colors)
            ax.set_xticks(range(len(names)),labels,rotation=25,ha='right',fontsize=8)
            ax.set_title(('Forward +0.3: ' if row==0 else 'Backward -0.3: ')+title,fontsize=10)
            ax.grid(axis='y',alpha=.2)
            for i,v in enumerate(values):
                if v is not None:ax.annotate(f'{v*factor:.2f}',(i,v*factor),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
            ax.margins(y=.15)
    fig.suptitle(f'{len(names)} PPO policies | identical headless MuJoCo deployment conditions\n20 s per command, first 4 s excluded; flat-ground short tests',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(a.output,dpi=160);plt.close(fig)


if __name__=='__main__':main()
