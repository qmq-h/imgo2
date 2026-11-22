#!/usr/bin/env python3
"""Package inspected comparison PNGs as a PDF atlas; optional reportlab only."""
import argparse
import json
from pathlib import Path


def main():
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    entries = json.loads((a.figures/'figure_index.json').read_text())
    c = canvas.Canvas(str(a.output), pageCompression=1)
    c.setTitle('Imgo2 real and Gazebo policy I/O: time, PSD, distributions')
    c.setAuthor('Imgo2 diagnostics, 2026-11-19')
    for i, item in enumerate(entries):
        image = ImageReader(item['path'])
        iw, ih = image.getSize()
        width = 14*72; draw_width = 13*72; draw_height = draw_width*ih/iw
        height = draw_height+76
        c.setPageSize((width, height))
        c.setFont('Helvetica', 10)
        c.drawString(36, height-20, f'Imgo2 | {item["kind"]} | {item["group"]} | policy 56d6d043...')
        c.drawRightString(width-36, height-20, f'{i+1} / {len(entries)}')
        c.drawImage(image, 36, 38, width=draw_width, height=draw_height)
        c.setFont('Helvetica', 8)
        c.drawString(36, 19, 'LEFT: real 2026-11-17. RIGHT: new own-feedback Gazebo, vx=0.5, Kp/Kd=25/0.8.')
        c.drawString(36, 8, 'Real raw outputs are reconstructed; angle targets are derived. No hardware run or causal root-cause claim.')
        key = f'page_{i+1}'
        c.bookmarkPage(key)
        c.addOutlineEntry(f'{item["kind"]} / {item["group"]}', key, level=0)
        c.showPage()
    c.save()
    print(a.output)


if __name__ == '__main__': main()
