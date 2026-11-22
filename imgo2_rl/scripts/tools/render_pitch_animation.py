"""Render the audited pitch trace as a simple quadruped GIF and preview PNG.

Rendering only; requires Pillow. Numerical checking is audit_pitch_trace.py (stdlib).
"""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from PIL import Image, ImageDraw, ImageFont
    data = json.loads(args.data.read_text())
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    small = ImageFont.truetype("DejaVuSans.ttf", 14)
    colors = {"bg": "#f6f7fa", "ink": "#26334b", "line": "#2464a9", "far": "#a7bcd5",
              "camera": "#b45618", "body": "#dce5ef", "grid": "#d4dce5"}
    ground = data["anchors"][0][2]-.022

    def project(p):
        return (round(285+450*(p[0]+.4*p[1])), round(270-450*(p[2]-.23*p[1])))

    def rotate(p, a):
        c, s = math.cos(a), math.sin(a)
        return [c*p[0]-s*p[2], p[1], s*p[0]+c*p[2]]

    def frame(pose):
        image = Image.new("RGB", (820, 500), colors["bg"])
        draw = ImageDraw.Draw(image)
        angle = pose["angle"]
        a = math.radians(angle)
        for k in range(8):
            x = -.45+.2*k
            draw.line([project([x, -.27, ground]), project([x, .27, ground])], fill=colors["grid"])
        for k in range(5):
            y = -.27+.135*k
            draw.line([project([-.45, y, ground]), project([.98, y, ground])], fill=colors["grid"])

        def leg(index, color):
            pts = [project(p) for p in pose["legs"][index]]
            draw.line(pts, fill=color, width=6, joint="curve")
            for i, p in enumerate(pts):
                r = 7 if i == 3 else 4
                draw.ellipse((p[0]-r, p[1]-r, p[0]+r, p[1]+r), fill=color)

        leg(0, colors["far"])
        leg(2, colors["far"])
        corners = [rotate([x, y, z], a) for x in [-.1575, .1575]
                   for y in [-.1215, .1215] for z in [-.0735, .0735]]
        for face in ((0, 1, 3, 2), (2, 3, 7, 6), (1, 3, 7, 5), (0, 1, 5, 4)):
            draw.polygon([project(corners[i]) for i in face], fill=colors["body"], outline=colors["ink"])
        leg(1, colors["line"])
        leg(3, colors["line"])
        camera, ray = project(rotate([.174, 0, .05], a)), project(rotate([.75, 0, .05], a))
        draw.line([camera, ray], fill=colors["camera"], width=3)
        draw.ellipse((camera[0]-6, camera[1]-6, camera[0]+6, camera[1]+6), fill=colors["ink"])
        center = project([0, 0, 0])
        draw.ellipse((center[0]-7, center[1]-7, center[0]+7, center[1]+7), fill=colors["bg"], outline=colors["ink"], width=2)
        draw.line((center[0]-10, center[1], center[0]+10, center[1]), fill=colors["ink"])
        draw.line((center[0], center[1]-10, center[0], center[1]+10), fill=colors["ink"])
        draw.text((120, 145), "Fixed body center", font=small, fill=colors["ink"])
        draw.text((540, 145), "Front camera sightline", font=small, fill=colors["camera"])
        name = {7: "Pitch", 8: "1-second return", 2: "Standing"}[pose["mode"]]
        draw.text((28, 22), f"Imgo2 | {name} | Nose-up {angle:+.1f} deg", font=font, fill=colors["ink"])
        draw.text((28, 51), "C++ FSM targets + canonical URDF FK | Camera mount schematic", font=small, fill=colors["ink"])
        draw.text((28, 470), f"t = {pose['t']:.2f} s | Feet fixed | Range -20 to +30 deg | Kinematics, not dynamics", font=small, fill=colors["ink"])
        return image

    poses = data["frames"][::2]
    if poses[-1]!=data["frames"][-1]:poses.append(data["frames"][-1])
    frames = [frame(p) for p in poses]
    durations = [max(20, round((b["t"]-a["t"])*1000)) for a, b in zip(poses, poses[1:])]+[800]
    palette = frames[0].quantize(colors=32)
    indexed = [image.quantize(palette=palette, dither=0) for image in frames]
    indexed[0].save(args.output, save_all=True, append_images=indexed[1:], duration=durations, loop=0, optimize=True)
    index = min(range(len(poses)), key=lambda i: abs(poses[i]["angle"]-30))
    preview = args.output.with_suffix(".png")
    frames[index].save(preview)
    print(f"{len(frames)} frames: {args.output}; +30deg preview: {preview}")


if __name__ == "__main__":
    main()
