#!/usr/bin/env python3
"""Renders docs/demo/explainer.mp4 (+ explainer-teaser.gif) from explainer_scenes.html with headless Chrome.
Frames are piped straight into ffmpeg: python3 render_explainer.py"""
import os, subprocess, tempfile
from playwright.sync_api import sync_playwright
HERE = os.path.dirname(os.path.abspath(__file__))
FPS, DUR = 30, 80
out = os.path.join(HERE, "explainer.mp4")
ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "image2pipe", "-framerate", str(FPS), "-i", "-",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart", out],
    stdin=subprocess.PIPE)
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1280, "height": 720})
    pg.goto("file://" + os.path.join(HERE, "explainer_scenes.html")); pg.wait_for_timeout(300)
    for i in range(DUR * FPS):
        pg.evaluate("t=>render(t)", i / FPS)
        ff.stdin.write(pg.screenshot(type="png"))
    b.close()
ff.stdin.close(); ff.wait()
# teaser: the measured-result scene, palette-optimised GIF
vf = "fps=12,scale=720:-2:flags=lanczos"; pal = os.path.join(tempfile.mkdtemp(), "pal.png")
sel = ["-ss", "59.8", "-t", "10", "-i", out]
subprocess.run(["ffmpeg", "-y", "-v", "error", *sel, "-vf", vf + ",palettegen=max_colors=96", pal], check=True)
subprocess.run(["ffmpeg", "-y", "-v", "error", *sel, "-i", pal, "-lavfi", vf + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
    "-loop", "0", os.path.join(HERE, "explainer-teaser.gif")], check=True)
