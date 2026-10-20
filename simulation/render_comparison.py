#!/usr/bin/env python3
"""Labelled offline replay of a validated comparison, using saved states only."""
import argparse
import json
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import cv2

import render_results as renderer
from plot_comparison import LABELS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();directory=args.input.parent
    validation=json.loads((directory/'comparison_validation.json').read_text())
    if not validation.get('validity_passed'):raise ValueError('Comparison must pass independent validity checks before video production.')
    summary=json.loads((directory/'summary.json').read_text());method=summary['method']
    completed=summary['run_status']=='completed'
    original=renderer._caption
    def caption(bgr,case,timestamp,position_error_mm,orientation_error_deg,trails,mode_caption=''):
        label=('FULL ORIGINAL JAS | SIGN + COMPLETE COMPOSITE LEARNING' if method=='proposed'
               else 'INDEPENDENT VERGINIS PPC | OWN TWO-LAYER CONTROL' if method=='external_ppc'
               else 'LABELLED STRUCTURAL ABLATION | '+LABELS[method].upper())
        if not completed:label='RECORDED PREFIX | '+label
        out=original(bgr,case,timestamp,position_error_mm,orientation_error_deg,trails,label)
        out[:37]=(17,21,27)
        headline=('Dual CR7 comparison | ' if completed else 'RECORDED PREFIX | ')+LABELS[method]
        cv2.putText(out,headline,(20,27),cv2.FONT_HERSHEY_SIMPLEX,
                    .65,(245,245,245),1,cv2.LINE_AA)
        return out
    renderer._caption=caption
    try:
        result=renderer.render_results(args.input,args.output,width=960,height=640,fps=30)
    finally:renderer._caption=original
    metadata=args.output.with_suffix('.metadata.json')
    if metadata.exists():
        content=json.loads(metadata.read_text());content.update(method=method,method_label=LABELS[method],
            algorithm_variant=summary['algorithm_variant'],comparison_validation=str(directory/'comparison_validation.json'),
            shared_force_target_N=summary['settings']['internal_force'],
            robust_mode=None if method=='external_ppc' else summary['settings']['robust_mode'],
            run_status=summary['run_status'],termination_reason=summary['termination_reason'],
            requested_duration_s=summary['requested_duration_s'],actual_end_s=summary['actual_end_s'],
            recorded_prefix_only=not completed)
        content['mode_caption']='FULL ORIGINAL JAS' if method=='proposed' else LABELS[method]
        if not completed:content['mode_caption']='RECORDED PREFIX | '+content['mode_caption']
        metadata.write_text(json.dumps(content,indent=2)+'\n')
    print(json.dumps(result,default=str),flush=True)


if __name__=='__main__':main()
