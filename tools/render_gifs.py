#!/usr/bin/env python3
"""Render validated saved MuJoCo states, without synthesizing trajectories."""
import argparse, hashlib, json, os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'simulation/results_demo/main/nominal/proposed/trajectory.npz')
    p.add_argument('--output',type=Path,default=ROOT/'media')
    args=p.parse_args(); directory=args.input.parent
    audit=json.loads((directory/'comparison_validation.json').read_text())
    if not audit['validity_passed']: raise ValueError('Independent validation must pass first')
    summary=json.loads((directory/'summary.json').read_text())
    with np.load(args.input,allow_pickle=False) as d:
        t=d['t'];q=d['qpos'];v=d['qvel'];err=d['error']
    if summary['run_status']!='completed':raise ValueError('Demo requires completed run')
    model=mujoco.MjModel.from_xml_path(str(ROOT/'simulation/models/dual_cr7.xml'))
    data=mujoco.MjData(model);args.output.mkdir(parents=True,exist_ok=True)
    # Exactly 10 fps, real-time 0–24 s; final frame adds 0.1 s of endpoint hold.
    requests=np.arange(0,t[-1]+1e-9,.1)
    right=np.minimum(np.searchsorted(t, requests), len(t)-1)
    left=np.maximum(right-1, 0)
    idx=np.where(np.abs(t[left]-requests)<=np.abs(t[right]-requests), left, right)
    out={'purpose':'Offline rendering of a newly rerun and independently validated simulation',
         'method':'Proposed','simulation_duration_s':float(t[-1]),'fps':10,'playback_speed':1,
         'state_sampling':'nearest saved state; no trajectory interpolation',
         'trajectory_sha256':sha(args.input),'source_manifest':json.loads((directory/'source_manifest.json').read_text()),
         'validation_passed':True,'render_script_sha256':sha(Path(__file__)), 'views':[]}
    with mujoco.Renderer(model,height=360,width=540) as renderer:
        for camera in ('overview','front'):
            frames=[]
            for k in idx:
                data.qpos[:]=q[k];data.qvel[:]=v[k];data.time=float(t[k]);mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera=camera)
                im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,540,37),fill=(18,22,27))
                draw.text((9,5),f'Proposed | MuJoCo rerun | {camera} | t={t[k]:05.1f} s',fill='white')
                draw.text((9,21),f'Position error: {np.linalg.norm(err[k,:3])*1000:.3f} mm | actual saved states',fill='white')
                frames.append(im)
            dest=args.output/f'proposed_{camera}.gif'
            frames[0].save(dest,save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
            with Image.open(dest) as im:
                assert im.n_frames==len(frames)
                for n in range(im.n_frames): im.seek(n);im.load()
            frames[len(frames)//2].save(args.output/f'proposed_{camera}_preview.png')
            out['views'].append({'camera':camera,'file':dest.name,'frames':len(frames),'sha256':sha(dest),'bytes':dest.stat().st_size})
            print(camera,len(frames),'frames',dest.stat().st_size,'bytes',flush=True)
    (args.output/'provenance.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
