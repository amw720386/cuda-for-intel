#!/usr/bin/env python3
"""Renders the demo videos (MP4 + GIF) into docs/demo/ : python3 render_videos.py
Needs: ffmpeg, playwright (uses installed Google Chrome), Pillow not required.
SVGs from make_assets.py are build intermediates (temp dir, not committed)."""
import os, subprocess, tempfile, shutil
from playwright.sync_api import sync_playwright
HERE = os.path.dirname(os.path.abspath(__file__))
FPS, W, H = 30, 1280, 720
# name: (clip seconds, SVG loop period seconds)
SCENES = {"pipeline": (10, 5), "architecture": (12, 6), "prefill-chart": (12, 6)}
PAGE = ("<html><body style='margin:0;width:%dpx;height:%dpx;background:#0d1117;display:flex;"
        "align-items:center;justify-content:center;overflow:hidden'>%s</body></html>")

def run(*a): subprocess.run(a, check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def main():
    tmp = tempfile.mkdtemp()
    svgs = os.path.join(tmp, "svg"); run("python3", os.path.join(HERE, "make_assets.py"), svgs)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome")
        pg = b.new_page(viewport={"width": W, "height": H})
        for name, (secs, period) in SCENES.items():
            svg = open(os.path.join(svgs, name + ".svg"), encoding="utf-8").read()
            vw, vh = [float(x) for x in svg.split('viewBox="0 0 ')[1].split('"')[0].split()]
            sc = min((W - 40) / vw, (H - 20) / vh)
            svg = svg.replace('width="%d" height="%d"' % (vw, vh), 'width="%d" height="%d"' % (vw * sc, vh * sc), 1)
            pg.set_content(PAGE % (W, H, svg)); pg.wait_for_timeout(300)
            fd = os.path.join(tmp, name); os.makedirs(fd)
            for i in range(secs * FPS):
                t = (i / FPS) % period
                pg.evaluate("t=>{const s=document.querySelector('svg');s.pauseAnimations();s.setCurrentTime(t)}", t)
                pg.screenshot(path=f"{fd}/{i:04d}.png")
            mp4 = os.path.join(HERE, name + ".mp4")
            run("ffmpeg", "-y", "-framerate", str(FPS), "-i", f"{fd}/%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "20", "-preset", "slow", "-movflags", "+faststart", "-vf", "scale=1280:720", mp4)
            pal = os.path.join(tmp, name + "_pal.png")
            vf = "fps=15,scale=800:-2:flags=lanczos"
            run("ffmpeg", "-y", "-i", mp4, "-vf", vf + ",palettegen=max_colors=128", pal)
            run("ffmpeg", "-y", "-i", mp4, "-i", pal, "-lavfi", vf + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
                "-loop", "0", os.path.join(HERE, name + ".gif"))
        b.close()
    # hero: chain the three clips with 0.5 s crossfades
    names = list(SCENES); cl = [SCENES[n][0] for n in names]; f = 0.5
    ins = sum([["-i", os.path.join(HERE, n + ".mp4")] for n in names], [])
    fc = (f"[0][1]xfade=transition=fade:duration={f}:offset={cl[0]-f}[a];"
          f"[a][2]xfade=transition=fade:duration={f}:offset={cl[0]+cl[1]-2*f}[v]")
    run("ffmpeg", "-y", *ins, "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-movflags", "+faststart", os.path.join(HERE, "hero.mp4"))
    shutil.rmtree(tmp)

if __name__ == "__main__": main()
