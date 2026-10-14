#!/usr/bin/env python3
"""Generic benchmark experiment runner and execution harness.

Runs or grades an agent submission in an isolated sandbox workspace
against precomputed fixtures using evaluator/grade.py.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
EVALUATOR = BENCH / 'evaluator'
DEFAULT_FIXTURES = EVALUATOR / 'fixtures.json'
GRADER = EVALUATOR / 'grade.py'


def grade_submission(task: int, consumer_dir: Path, fixtures_file: Path) -> dict:
    """Run evaluator/grade.py against consumer_dir."""
    py_exec = sys.executable
    cmd = [py_exec, str(GRADER), str(task), str(consumer_dir), str(fixtures_file)]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=300)
    try:
        gres = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        gres = {
            'success': False,
            'error': f'Grader emitted no JSON (exit code {proc.returncode})',
            'stderr': proc.stderr[-2000:],
            'stdout': proc.stdout[-1000:],
        }
    return gres


def run_benchmark(task: int, workspace: Path, fixtures: Path,
                  command: str = None, timeout: int = 1800, output_file: Path = None) -> dict:
    workspace = workspace.resolve()
    consumer = workspace / 'consumer'
    fixtures = fixtures.resolve()

    rec = {
        'task': task,
        'workspace': str(workspace),
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'running',
    }

    start_time = time.time()

    if command:
        try:
            proc = subprocess.run(command, shell=True, cwd=str(workspace),
                                  text=True, capture_output=True, timeout=timeout)
            elapsed = time.time() - start_time
            rec['command_returncode'] = proc.returncode
            rec['command_stdout'] = proc.stdout[-2000:]
            rec['command_stderr'] = proc.stderr[-2000:]
            rec['elapsed_seconds'] = round(elapsed, 2)
            if proc.returncode != 0:
                rec['status'] = 'command_failed'
        except subprocess.TimeoutExpired:
            rec['status'] = 'timeout'
            rec['elapsed_seconds'] = timeout
            rec['task_success'] = False
            rec['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
            if output_file:
                output_file.write_text(json.dumps(rec, indent=2) + '\n')
            return rec
    else:
        elapsed = 0.0
        rec['elapsed_seconds'] = 0.0

    # Grade consumer directory
    gres = grade_submission(task, consumer, fixtures)
    task_success = bool(gres.get('success', False))

    rec.update({
        'status': 'completed',
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'task_success': task_success,
        'grader': gres,
    })

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(rec, indent=2) + '\n')

    return rec


def main():
    parser = argparse.ArgumentParser(description='Generic benchmark task runner & grader')
    parser.add_argument('--task', type=int, required=True, choices=[1, 2, 3],
                        help='Task number (1, 2, or 3)')
    parser.add_argument('--workspace', type=Path, required=True,
                        help='Path to sandbox workspace')
    parser.add_argument('--fixtures', type=Path, default=DEFAULT_FIXTURES,
                        help=f'Path to fixtures JSON (default: {DEFAULT_FIXTURES})')
    parser.add_argument('--command', type=str, default=None,
                        help='Optional agent execution command to run inside workspace')
    parser.add_argument('--timeout', type=int, default=1800,
                        help='Timeout in seconds for command execution (default: 1800)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Optional path to write result JSON')

    args = parser.parse_args()

    res = run_benchmark(
        task=args.task,
        workspace=args.workspace,
        fixtures=args.fixtures,
        command=args.command,
        timeout=args.timeout,
        output_file=args.output,
    )
    print(json.dumps(res, indent=2))
    sys.exit(0 if res.get('task_success') else 1)


if __name__ == '__main__':
    main()
