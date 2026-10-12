"""Organize flat project docs by topic and relocate existing Markdown links.

Standard-library file maintenance only; does not run training or code tests.
Run from either worktree: python imgo2_rl/scripts/tools/organize_docs.py --apply
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

LINK = re.compile(r'(!?\[[^\]\n]*\]\()(<[^>\n]+>|[^\s)\n]+)([^\n)]*\))')
REFERENCE = re.compile(r'^(\[[^\]\n]+\]:[ \t]*)(<[^>\n]+>|[^\s\n]+)([^\n]*)$', re.MULTILINE)
TEXT_SUFFIXES = {'.md', '.py', '.sh', '.ps1', '.toml', '.yaml', '.yml', '.cfg', '.ini', '.rst'}
TOPICS = {
    'amp': 'AMP 对齐、步态、实验与离线数据',
    'cmoe': 'CMoE 网络、训练、地形与部署专属记录',
    'towing': '机器人拖车、连接模型、地形、估计器与测试数据',
    'deployment': '通用部署、Gazebo、MuJoCo 与 sim2sim',
    'training': '训练环境、启动器与依赖兼容性',
    'research': '研究计划、论文与参考实现',
    'maintenance': '代码复审、同步、离线核对与文档维护',
}


def destination(relative):
    path = Path(relative)
    if path.parts[:2] == ('docs', 'towingdata'):
        tail = list(path.parts[2:])
        if tail[-1] == 'README.md':
            tail[-1] = 'overview.md'
        return Path('docs/towing/data', *tail).as_posix()
    if len(path.parts) != 2 or path.parts[0] != 'docs':
        return relative
    name = path.name
    if name.startswith(('amp_', 'gait_')):
        topic = 'amp'
    elif name.startswith('cmoe_'):
        topic = 'cmoe'
    elif name.startswith('towing_'):
        topic = 'towing'
    elif name.startswith(('sim2sim_', 'gazebo_', 'train_vs_deploy_', 'ppo_policy_')):
        topic = 'deployment'
    elif name.startswith(('training_', 'isaaclab_')):
        topic = 'training'
    elif name.startswith(('paper_', 'moe_', 'parkour_')):
        topic = 'research'
    else:
        topic = 'maintenance'
    return Path('docs', topic, *(['data'] if path.suffix != '.md' else []), name).as_posix()


def safe(root, relative):
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f'Path escapes repository: {relative}')
    return resolved


def local_target(root, owner, target):
    raw = target.strip('<>')
    if re.match(r'^[a-zA-Z][\w+.-]*:', raw) or raw.startswith(('#', '/', '\\')):
        return None
    bare = re.split(r'[?#]', raw, maxsplit=1)[0]
    resolved = (root / owner).parent / unquote(bare)
    resolved = resolved.resolve()
    if not resolved.is_relative_to(root):
        return None
    return resolved.relative_to(root).as_posix(), raw[len(bare):]


def relocate_target(root, owner, new_owner, target, mapping, originals):
    found = local_target(root, owner, target)
    if found is None:
        return target
    old_relative, suffix = found
    # Preserve pre-existing missing links as evidence, rather than guessing targets.
    if old_relative not in originals and not (root / old_relative).exists():
        return target
    new_relative = mapping.get(old_relative, old_relative)
    if old_relative == 'docs/towing/data' or old_relative.startswith('docs/towing/data/'):
        new_relative = mapping.get(old_relative, old_relative.replace('docs/towing/data', 'docs/towing/data', 1))
    relative = os.path.relpath(root / new_relative, (root / new_owner).parent).replace('\\', '/')
    value = relative + suffix
    return f'<{value}>' if target.startswith('<') else value


def links(text):
    # Shell diagnostics like [ERROR]: Process inside code fences are not links.
    text = re.sub(r'(?ms)^[ \t]*(?:`{3,}|~{3,})[^\n]*\n.*?^[ \t]*(?:`{3,}|~{3,})[^\n]*(?:\n|$)', '', text)
    for pattern in (LINK, REFERENCE):
        for match in pattern.finditer(text):
            yield match.group(2)


def missing_links(root, documents):
    missing = []
    for owner, text in documents.items():
        for target in links(text):
            found = local_target(root, owner, target)
            if found and not (root / found[0]).exists():
                missing.append({'document': owner, 'target': target})
    return missing


def organize(root, apply):
    root = root.resolve()
    names = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', '.',
         ':(exclude).towing_slope_edit/**'], cwd=root).decode('utf-8').split('\0')
    names = sorted(set(n for n in names if n and (root / n).is_file()))
    originals = set(names)
    mapping = {n: destination(n) for n in names if n.startswith('docs/') and destination(n) != n}
    if not mapping:
        print('Docs already organized; no changes.')
        return
    for old, new in mapping.items():
        safe(root, old)
        if safe(root, new).exists():
            raise FileExistsError(f'Refusing to overwrite {new}')
    texts = {}
    for name in names:
        if Path(name).suffix in TEXT_SUFFIXES:
            try:
                texts[name] = safe(root, name).read_text(encoding='utf-8')
            except UnicodeError:
                continue
    before_docs = {n: t for n,t in texts.items() if n.endswith('.md')}
    old_missing = missing_links(root, before_docs)
    updated = {}
    for old_owner, text in texts.items():
        new_owner = mapping.get(old_owner, old_owner)
        if old_owner.endswith('.md'):
            text = LINK.sub(lambda m: m.group(1) + relocate_target(
                root, old_owner, new_owner, m.group(2), mapping, originals) + m.group(3), text)
            text = REFERENCE.sub(lambda m: m.group(1) + relocate_target(
                root, old_owner, new_owner, m.group(2), mapping, originals) + m.group(3), text)
            text = re.sub(r'(?<!`)`([^`\n]+)`(?!`)', lambda m: '`' + relocate_target(
                root, old_owner, new_owner, m.group(1), mapping, originals) + '`'
                if m.group(1).startswith(('../', './')) and ' ' not in m.group(1) else m.group(0), text)
        # Also update command examples, source comments and hard-coded output paths.
        for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
            text = text.replace(old, new)
        text = text.replace('docs/towing/data', 'docs/towing/data')
        if text != texts[old_owner] or new_owner != old_owner:
            updated[new_owner] = text
    print(f'{root}: move {len(mapping)} files; update {len(updated)} text files')
    if not apply:
        for old,new in mapping.items():
            print(f'{old} -> {new}')
        return
    for old, new in mapping.items():
        target = safe(root, new)
        target.parent.mkdir(parents=True, exist_ok=True)
        safe(root, old).rename(target)
    for name, text in updated.items():
        safe(root, name).write_text(text, encoding='utf-8', newline='\n')
    for topic in TOPICS:
        (root / 'docs' / topic).mkdir(exist_ok=True)
    # Drop only empty old directories, never recursively delete artifacts.
    for path in sorted((root/'docs').rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()) and path.name not in TOPICS:
            safe(root, path.relative_to(root)).rmdir()
    after_docs = {mapping.get(n,n): (root / mapping.get(n,n)).read_text('utf-8') for n in before_docs}
    new_missing = missing_links(root, after_docs)
    known = {(mapping.get(m['document'], m['document']), m['target']) for m in old_missing}
    regressions = [m for m in new_missing if (m['document'], m['target']) not in known]
    report = {
        'date': '2026-10-09', 'mapping': mapping,
        'updated_text_files': sorted(updated),
        'preexisting_missing_links': old_missing,
        'remaining_missing_links': new_missing, 'new_missing_links': regressions,
        'validation_scope': 'File relocation and local Markdown target existence only; no code or simulation tests.'
    }
    report_path = root / 'docs/maintenance/docs_reorganization_2026-10-09.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    note_path = root / 'docs/maintenance/docs_reorganization_2026-10-09.md'
    note_path.write_text(
        '# 文档主题分类（2026-10-09）\n\n'
        '现象：docs 根目录混放算法、部署、牵引和维护记录，数据与说明难以查找。\n\n'
        '已整理：按主题移动原文件，保留原文件名与历史正文；数据进入主题 data/；'
        '牵引数据归档内部 README.md 改名 overview.md，根 README 保持唯一项目入口。'
        '同步重写有效的 Markdown 相对引用、命令示例、脚本输出路径和源码中的文档路径。\n\n'
        f'文件路径核对：移动 {len(mapping)} 个文件，更新 {len(updated)} 个文本文件；'
        f'原有本地缺失链接 {len(old_missing)} 项，新增缺失 {len(regressions)} 项。'
        '完整迁移表和缺失列表见 [记录](docs_reorganization_2026-10-09.json)。\n\n'
        '待修复／待确认：历史缺失目标需原始归档或维护者判断替代文档，'
        '本轮不凭名称臆造文件；日志、外部路径和版本化实验 JSON/CSV/TXT 内容不改。'
        '未运行代码测试、训练或仿真；既有本地修改保留，分类仅在 main 进行。\n',
        encoding='utf-8', newline='\n')
    readme = root / 'README.md'
    content = readme.read_text('utf-8')
    summary = ('> 最后核对：2026-10-09（文档分类）。docs 已按主题建目录，数据归到主题 data/，'
               '文档引用与脚本中的路径同步调整。已完成文件迁移；历史缺失引用待补齐，'
               '本轮未运行代码或仿真测试。见 [分类记录](docs/maintenance/docs_reorganization_2026-10-09.md)。\n\n')
    content = content.replace('# Imgo2 项目说明与维护记录\n\n', '# Imgo2 项目说明与维护记录\n\n'+summary, 1)
    directory = '### docs 分类\n\n| 目录 | 内容 |\n|---|---|\n'
    directory += ''.join(f'| [docs/{k}/](docs/{k}/) | {v} |\n' for k,v in TOPICS.items())
    directory += '\n历史记录保留原文件名；机器可读结果放各主题 `data/`，不另建子目录 README。\n\n'
    pos = content.find('## 2.')
    content = content[:pos] + directory + content[pos:] if pos >= 0 else content + '\n' + directory
    if old_missing or regressions:
        row = ('| DOC-CLASS-01 | P2 | **待确认**：历史 Markdown 引用存在缺失目标；'
               '缺原始归档或替代关系，详见 [分类记录](docs/maintenance/docs_reorganization_2026-10-09.md)。 |\n')
        marker = '| 本次新增问题 |'
        if marker in content:
            where = content.find('\n', content.find('\n', content.index(marker))+1)+1
            content = content[:where] + row + content[where:]
        else:
            marker = '## 7. 待办与已知问题'
            if marker in content:
                content = content.replace(marker, marker+'\n\n| 本次新增问题 | 优先级 | 状态与完成条件 |\n|---|---|---|\n'+row, 1)
    content += ('\n| 2026-10-09 | 按主题分类 docs 并同步路径 | 已移动文档/数据，'
                '更新根 README、AGENTS 与源码引用；历史缺失链接保留待确认。'
                '只做文件路径核对，未运行代码或仿真测试。见 [分类记录](docs/maintenance/docs_reorganization_2026-10-09.md)。 |\n')
    readme.write_text(content, encoding='utf-8', newline='\n')
    agents = root / 'AGENTS.md'
    if agents.exists():
        content = agents.read_text('utf-8')
        marker = '- 顶层目录命名统一'
        policy = ('- `docs/` 按主题归档：`amp/`、`cmoe/`、`towing/`、`deployment/`、'
                  '`training/`、`research/`、`maintenance/`；机器可读结果放对应 `data/`。'
                  '新增记录放对应主题，根 README 是统一入口；移动文档时同步修正相对引用和脚本路径。\n')
        if marker in content:
            content = content.replace(marker, policy+marker, 1)
        agents.write_text(content, encoding='utf-8', newline='\n')
    print(f'Existing missing links: {len(old_missing)}; newly broken links: {len(regressions)}')
    if regressions:
        raise RuntimeError('New broken local links; consult migration report before finishing.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    organize(args.repo_root, args.apply)
