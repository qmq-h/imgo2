#!/usr/bin/env python3
"""Validate and package native simulator media; standard-library orchestration only."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import struct
import subprocess
from urllib.parse import quote
import zipfile

LABELS={'flat':'平地行走','stairs':'5 cm 台阶上下行','rough':'轻度粗糙地面','slope':'约 5° 分块近似坡面'}

def png_info(path):
    with path.open('rb') as f:
        if f.read(8)!=b'\x89PNG\r\n\x1a\n':raise ValueError('invalid PNG '+str(path))
        result={}
        while True:
            raw=f.read(8)
            if not raw:break
            length,kind=struct.unpack('>I4s',raw);data=f.read(length);f.read(4)
            if kind==b'IHDR':result.update(zip(('width','height'),struct.unpack('>II',data[:8])))
            if kind==b'pHYs':
                x,y,unit=struct.unpack('>IIB',data)
                if unit==1:result['dpi']=[x*.0254,y*.0254]
            if kind==b'IEND':break
        return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--encoder',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();delivery=root/'deliverables'
    folders={'images':delivery/'01_模型与场景图片','videos':delivery/'02_运动视频','curves':delivery/'03_训练曲线'}
    for folder in folders.values():folder.mkdir(exist_ok=True)
    for path in list(delivery.iterdir()):
        if not path.is_file():continue
        if path.name.startswith('ppo_'):target=folders['curves']/path.name
        elif path.suffix=='.mp4':target=folders['videos']/path.name
        elif path.suffix=='.png':target=folders['images']/path.name
        else:continue
        if target.exists():raise FileExistsError(target)
        path.rename(target)
    manifest=json.loads((root/'manifest.json').read_text());validation=json.loads((root/'media_validation.json').read_text())
    images=[]
    for folder in (folders['images'],folders['curves']):
        for path in sorted(folder.glob('*.png')):
            info=png_info(path)
            if folder==folders['images']:assert (info['width'],info['height'])==(3840,2160)
            else:assert min(info['dpi'])>599
            images.append({'file':str(path.relative_to(delivery)),**info})
    for path in sorted(folders['videos'].glob('*.mp4')):
        check=subprocess.run([str(args.encoder),'-v','error','-threads','2','-i',str(path),'-f','null','-'],capture_output=True,text=True,timeout=120)
        (root/(path.stem+'_decode.log')).write_text(check.stdout+check.stderr)
        check.check_returncode();print('Decoded',path.name,flush=True)
    manifest['validation']={'png':images,'videos':validation['videos'],'all_videos_full_decode_passed':True}
    manifest['production_after']={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in manifest['production_before']}
    manifest['production_unchanged']=manifest['production_before']==manifest['production_after']
    manifest['media_scope']='Single deterministic short demonstration per selected case; not multi-seed reliability/real hardware validation. MuJoCo native state rendering; Gazebo native camera. 1x simulation-time playback.'
    manifest['height_probe_interpretation']={'5cm':'completed stairs','8cm':'stopped near first step; not a passage demonstration','18cm':'not tested'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    (delivery/'采集数据与验证.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    guide='''Imgo2 本体运动答辩素材  2026-11-22

先打开“素材总览.html”预览。插入 PPT 时使用分类目录内的原始文件。

01_模型与场景图片
13 张 3840×2160 原生仿真图片，PNG，300 dpi。包括 MuJoCo 站立三视角、运动截图及 Gazebo 站立、平地、台阶、粗糙、坡面截图。
优先使用 mujoco_model_oblique_4k.png 与 gazebo_model_standing_4k.png 介绍模型。

02_运动视频
8 段 H.264 MP4，1920×1080、30 fps，约19–20秒，按仿真时间1倍速播放。
MuJoCo 和 Gazebo 各4段：flat平地、stairs台阶、rough轻度粗糙地面、slope近似坡面。
平地采用11/22 PPO 5000轮；地形采用6000轮。PD 25/0.8，无额外反馈延迟。

场景尺寸
stairs：三阶上行、平台、三阶下行；单阶5 cm，踏面30 cm，平台90 cm。
rough：固定seed42，块高0.5–2 cm，块尺寸25×30 cm；是轻度分块起伏，不等同Isaac高度场。
slope：约5°上下坡，以5 cm宽的小段近似，不是连续斜面的严格复刻。
两引擎本轮均走过这三种地形。8 cm的MuJoCo对照停在首阶附近，不作为通过素材；18 cm未测试。
平地视频有横向/朝向漂移，不能据此写“零漂移直行”。相机只跟随位置，没有约束或托举机器人。

03_训练曲线
ppo_nov22_overview.png：完整10000轮四图总览，7680×4320、600 dpi。
01：回合回报；02：平面跟踪误差；03：转向跟踪误差；04：平均地形等级。
各单图为5760×3240、600 dpi，并提供PDF和SVG矢量版本。
曲线来自2026-11-22_01-09-04原始TensorBoard日志，浅色原始值、深色100轮移动平均。
标记5000/6000轮候选，按独立运动测试分别选择；不存在已证明的全场景最优权重。
误差采用项目日志口径，平均地形等级不是台阶通过率。

证据与边界
全部素材来自本项目模型和真实仿真运行，未用AI生成机器人或伪造运动。
MuJoCo先记录策略闭环轨迹，再用其原生渲染器生成图像；Gazebo由原生相机直接采集。
视频按仿真时间重采样，不将渲染快慢当作机器人运动速度。
本次为固定场景短测与素材采集，不代表多种子成功率、长期可靠性或实物稳定性验收。
实物图片/视频由用户已有素材提供，本包未操作实物，也未启动新训练。
'''
    (delivery/'00_使用说明.txt').write_text(guide)
    def href(path):return quote(str(path.relative_to(delivery)))
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Imgo2 答辩素材</title><style>body{font-family:system-ui,sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;color:#192b35}h1{font-size:32px}h2{margin-top:40px}section{display:flex;flex-wrap:wrap;gap:24px}figure{margin:0;width:590px}img,video{width:100%;height:auto;background:#eef1f3}figcaption{padding:8px 0 20px}p{line-height:1.7}a{color:#17658b}</style><h1>Imgo2 本体运动答辩素材</h1><p>原生仿真截图、运动视频和完整训练曲线。图片 4K，视频 1080p，曲线 600 dpi。<a href="00_%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.txt">使用说明与实验条件</a></p><h2>运动视频</h2><section>']
    for kind,label in LABELS.items():
        for engine in ('gazebo','mujoco'):
            path=folders['videos']/f'{engine}_{kind}_1080p.mp4'
            parts.append(f'<figure><video controls preload="metadata" muted playsinline src="{href(path)}"></video><figcaption>{engine.title()}：{label}</figcaption></figure>')
    parts+=['</section><h2>模型与场景图片</h2><p>点击图片打开 4K 原图。</p><section>']
    for path in sorted(folders['images'].glob('*.png')):
        parts.append(f'<figure><a href="{href(path)}"><img loading="lazy" src="{href(path)}"></a><figcaption>{html.escape(path.stem)}</figcaption></figure>')
    parts+=['</section><h2>训练曲线</h2><section>']
    for path in sorted(folders['curves'].glob('*.png')):
        parts.append(f'<figure><a href="{href(path)}"><img loading="lazy" src="{href(path)}"></a><figcaption>{html.escape(path.stem)}　<a href="{href(path.with_suffix(".pdf"))}">PDF</a>　<a href="{href(path.with_suffix(".svg"))}">SVG</a></figcaption></figure>')
    parts+=['</section></html>'];(delivery/'素材总览.html').write_text(''.join(parts))
    files={str(p.relative_to(delivery)):hashlib.sha256(p.read_bytes()).hexdigest() for p in delivery.rglob('*') if p.is_file()}
    (delivery/'文件校验.json').write_text(json.dumps(files,indent=2,ensure_ascii=False))
    archive=root/'imgo2_defense_media_2026-11-22.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for path in sorted(delivery.rglob('*')):
            if path.is_file():z.write(path,Path('Imgo2答辩素材')/path.relative_to(delivery))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    result={'archive':str(archive),'archive_mb':round(archive.stat().st_size/2**20,1),'files':len(files)+1,'model_scene_pngs':len(list(folders['images'].glob('*.png'))),'videos':len(list(folders['videos'].glob('*.mp4'))),'curve_pngs':len(list(folders['curves'].glob('*.png'))),'production_unchanged':manifest['production_unchanged']}
    (root/'delivery_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
