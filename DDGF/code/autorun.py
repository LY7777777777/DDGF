# coding: utf-8

import subprocess
import argparse
import sys
import os
import shlex


def parse_args():
    parser = argparse.ArgumentParser(description='Auto run script for Base.py')
    parser.add_argument('--datasets', nargs='+', default=['KLSG'])
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone model name')
    parser.add_argument('--trails', type=int, default=10, help='Number of trails (p_value max)')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds (k_value max)')
    parser.add_argument('--code_file', type=str, default='DDGF.py',help='Training script filename')
    parser.add_argument('--save_results', type=str, default='True')
    parser.add_argument('--save_models', type=str, default='False')
    parser.add_argument('--start_p', type=int, default=0)
    parser.add_argument('--start_k', type=int, default=0)

    parser.add_argument(
        '--extra_args',
        type=str,
        default='',
        help="Extra args passed to training script, e.g. "
             "\"--stage2_mode fatl --split_mode runtime_stratified --runtime_folds 5\""
    )

    return parser.parse_args()

def main():
    args = parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    code_path = os.path.join(curr_dir, args.code_file)

    if not os.path.exists(code_path):
        print(f"Error: Could not find training script at: {code_path}")
        return

    print("Starting Auto-Run Session")
    print(f"   - Script:    {args.code_file}")
    print(f"   - Datasets:  {args.datasets}")
    print(f"   - Backbone:  {args.backbone}")
    print(f"   - Loops:     Trails x Folds = {args.trails * args.folds} runs")
    print(f"   - Resume:    start from P={args.start_p}, K={args.start_k}")
    print("--------------------------------------------------\n")

    for dataset in args.datasets:
        print(f"[Processing Dataset]: {dataset}")

        for p in range(args.trails):
            for k in range(args.folds):
                # 跳过断点前的任务
                if (p < args.start_p) or (p == args.start_p and k < args.start_k):
                    continue

                print(f"Task: {dataset} | P={p} | K={k} ...")

                cmd = [
                    sys.executable, code_path,
                    '--dataset', dataset,
                    '--backbone', args.backbone,
                    '--p_value', str(p),
                    '--k_value', str(k),
                    '--save_results', args.save_results,
                    '--save_models', args.save_models
                ]

                if args.extra_args and args.extra_args.strip():
                    cmd.extend(shlex.split(args.extra_args, posix=False))

                try:
                    subprocess.run(cmd, check=True)
                    print(f"Finished P={p}, K={k}.\n")
                except subprocess.CalledProcessError as e:
                    print(f"Failed P={p}, K={k} (Exit Code: {e.returncode})")
                    print("Skipping to next task...\n")

    print("All scheduled tasks completed.")


if __name__ == '__main__':
    main()