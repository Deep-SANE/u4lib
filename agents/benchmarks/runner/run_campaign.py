#!/usr/bin/env python3
"""Run benchmark campaign trials in isolated sandboxes for Claude Sonnet 3.5 on u4lib."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
EVALUATOR = BENCH / 'evaluator'
SNAPSHOTS = BENCH / 'snapshots'
DEFAULT_FIXTURES = EVALUATOR / 'fixtures.json'
GRADER = EVALUATOR / 'grade.py'
U3LIB_PATH = Path(os.environ.get('U3LIB_DIR', str(HERE.parents[3] / 'u3lib')))
PY_EXEC = os.environ.get('BENCH_PYTHON', sys.executable)
TIMEOUT = 1800

TASK_NAMES = {
    1: '01-u6z6recoupling.md',
    2: '02-u9recoupling.md',
    3: '03-coupling-omp.md',
}


def build_env(workspace: Path) -> dict:
    env = os.environ.copy()
    py_lib = workspace / 'u4lib/python'
    env['PATH'] = f"{os.path.dirname(PY_EXEC)}:{env.get('PATH', '')}"
    env['PYTHONPATH'] = f"{py_lib}:{U3LIB_PATH}/python:{env.get('PYTHONPATH', '')}"
    env['U4LIB_DIR'] = str(workspace / 'u4lib')
    env['U3LIB_DIR'] = str(U3LIB_PATH)
    return env


def run_claude(workspace: Path, timeout: int = TIMEOUT) -> dict:
    prompt = "Implement the task specified in common.md and prompt.md. Write all deliverables only in consumer/."
    cmd = [
        "claude", "-p", prompt,
        "--model", "sonnet",
        "--output-format", "json",
        "--permission-mode", "bypassPermissions",
    ]
    env = build_env(workspace)
    start_time = time.time()
    proc = subprocess.run(cmd, cwd=str(workspace), env=env, stdin=subprocess.DEVNULL,
                          text=True, capture_output=True, timeout=timeout)
    elapsed = time.time() - start_time
    
    try:
        obj = json.loads(proc.stdout.strip())
    except Exception:
        obj = None

    if obj and isinstance(obj, dict):
        usage = obj.get('usage', {})
        if usage:
            it = usage.get('input_tokens', 0)
            ot = usage.get('output_tokens', 0)
            cc = usage.get('cache_creation_input_tokens', 0)
            cr = usage.get('cache_read_input_tokens', 0)
            usage['total_tokens'] = it + ot + cc + cr
        cost = obj.get('total_cost_usd')
        duration = obj.get('duration_ms', int(elapsed * 1000)) / 1000.0
        is_error = obj.get('is_error', False)
        rate_limited = (obj.get('api_error_status') == 429 or
                        'session limit' in str(obj.get('result', '')).lower())
    else:
        usage = None
        cost = None
        duration = round(elapsed, 2)
        is_error = proc.returncode != 0
        rate_limited = 'session limit' in proc.stdout.lower() or 'session limit' in proc.stderr.lower()

    return {
        'returncode': proc.returncode,
        'stdout': proc.stdout,
        'stderr': proc.stderr,
        'elapsed_seconds': round(duration, 2),
        'usage': usage,
        'cost_usd': cost,
        'is_error': is_error,
        'rate_limited': rate_limited,
    }


def grade(task: int, consumer_dir: Path, fixtures: Path, workspace: Path) -> dict:
    env = build_env(workspace)
    cmd = [PY_EXEC, str(GRADER), str(task), str(consumer_dir), str(fixtures)]
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=300)
    try:
        gres = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        gres = {
            'success': False,
            'error': f'Grader emitted no valid JSON (code {proc.returncode})',
            'stdout': proc.stdout[-2000:],
            'stderr': proc.stderr[-2000:],
        }
    return gres


def run_trial(item: dict, results_dir: Path, fixtures: Path) -> dict:
    run_id = item['run_id']
    condition = item['condition']
    task = item['task']
    model_name = item['model']

    run_out = results_dir / run_id
    run_out.mkdir(parents=True, exist_ok=True)

    snap_ws = SNAPSHOTS / condition / 'workspace'
    if not snap_ws.is_dir():
        raise RuntimeError(f"Snapshot workspace missing: {snap_ws}. Run freeze_inputs.py first.")

    tmp_dir = Path(tempfile.mkdtemp(prefix=f'{run_id}-'))
    ws = tmp_dir / 'workspace'
    shutil.copytree(snap_ws, ws)

    # Copy task specification
    task_file = BENCH / 'tasks' / TASK_NAMES[task]
    (ws / 'prompt.md').write_text(task_file.read_text())

    rec = {
        'run_id': run_id,
        'task': task,
        'condition': condition,
        'model': model_name,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'running',
    }

    try:
        while True:
            if 'claude' in model_name.lower():
                exec_res = run_claude(ws, timeout=TIMEOUT)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            if exec_res.get('rate_limited'):
                print(f"⏳ Rate/session limit active for {model_name}. Pausing 60s before retry...", flush=True)
                time.sleep(60)
                continue
            break

        (run_out / 'agent.stdout.txt').write_text(exec_res['stdout'])
        (run_out / 'agent.stderr.txt').write_text(exec_res['stderr'])

        # Grade consumer deliverable inside workspace
        gres = grade(task, ws / 'consumer', fixtures, ws)
        (run_out / 'grader.json').write_text(json.dumps(gres, indent=2) + '\n')

        # Save consumer deliverable
        consumer_dst = run_out / 'consumer'
        if (ws / 'consumer').is_dir():
            shutil.copytree(ws / 'consumer', consumer_dst, dirs_exist_ok=True)
        else:
            consumer_dst.mkdir(parents=True, exist_ok=True)

        task_success = bool(gres.get('success', False) and exec_res['elapsed_seconds'] <= TIMEOUT)

        rec.update({
            'status': 'completed',
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': exec_res['elapsed_seconds'],
            'within_time_cap': exec_res['elapsed_seconds'] <= TIMEOUT,
            'agent_is_error': exec_res['is_error'],
            'task_success': task_success,
            'grader': gres,
            'token_usage': exec_res['usage'],
            'total_cost_usd': exec_res['cost_usd'],
        })

    except subprocess.TimeoutExpired:
        rec.update({
            'status': 'timeout',
            'task_success': False,
            'elapsed_seconds': TIMEOUT,
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        })
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    (run_out / 'run.json').write_text(json.dumps(rec, indent=2) + '\n')
    return rec


def update_summary(results_dir: Path):
    runs = []
    for p in sorted(results_dir.glob('*/run.json')):
        runs.append(json.loads(p.read_text()))

    (results_dir / 'results.json').write_text(json.dumps(runs, indent=2) + '\n')

    csv_path = results_dir / 'results.csv'
    with csv_path.open('w') as f:
        f.write('run_id,task,condition,model,status,task_success,elapsed_seconds,input_tokens,output_tokens,total_tokens,cost_usd\n')
        for r in runs:
            u = r.get('token_usage') or {}
            f.write(','.join(str(x if x is not None else '') for x in [
                r.get('run_id'), r.get('task'), r.get('condition'), r.get('model'),
                r.get('status'), r.get('task_success'), r.get('elapsed_seconds'),
                u.get('input_tokens'), u.get('output_tokens'), u.get('total_tokens'),
                r.get('total_cost_usd')
            ]) + '\n')


def main():
    parser = argparse.ArgumentParser(description="u4lib Sandbox Campaign Runner")
    parser.add_argument('--model', type=str, default='claude-sonnet-3.5',
                        choices=['claude-sonnet-3.5'],
                        help="Model to run")
    parser.add_argument('--task', type=int, default=1, choices=[1, 2, 3],
                        help="Task number (default: 1)")
    parser.add_argument('--repeats', type=int, default=5,
                        help="Number of runs per condition (default: 5)")
    parser.add_argument('--results-dir', type=Path, default=None,
                        help="Output directory (default: agents/benchmarks/{model})")
    parser.add_argument('--fixtures', type=Path, default=DEFAULT_FIXTURES,
                        help="Fixtures JSON path")
    parser.add_argument('--only', type=str, default=None,
                        help="Run only a specific run_id")
    parser.add_argument('--force', action='store_true',
                        help="Force re-run even if run.json already exists")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print scheduled runs without executing")

    args = parser.parse_args()

    results_dir = args.results_dir or (BENCH / args.model)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate interleaved schedule: C1, T1, C2, T2, ...
    schedule = []
    for r in range(1, args.repeats + 1):
        schedule.append({
            'run_id': f"task{args.task}-control-{r}",
            'task': args.task,
            'condition': 'control',
            'repeat': r,
            'model': args.model,
            'status': 'pending',
        })
        schedule.append({
            'run_id': f"task{args.task}-treatment-{r}",
            'task': args.task,
            'condition': 'treatment',
            'repeat': r,
            'model': args.model,
            'status': 'pending',
        })

    (results_dir / 'schedule.json').write_text(json.dumps({'runs': schedule}, indent=2) + '\n')
    (results_dir / f'schedule-task{args.task}.json').write_text(json.dumps({'runs': schedule}, indent=2) + '\n')

    if args.dry_run:
        print(f"Scheduled {len(schedule)} runs for {args.model} on Task {args.task}:")
        for item in schedule:
            print(f"  - {item['run_id']} ({item['condition']})")
        return

    todo = [item for item in schedule if not args.only or item['run_id'] == args.only]
    print(f"Checking {len(todo)} scheduled runs for {args.model} on Task {args.task} in {results_dir}...", flush=True)

    for item in todo:
        run_json = results_dir / item['run_id'] / 'run.json'
        if run_json.exists() and not args.force:
            try:
                prev = json.loads(run_json.read_text())
                if prev.get('status') == 'completed':
                    print(f"⏭ SKIPPING already completed: {item['run_id']} (Success: {prev.get('task_success')})", flush=True)
                    continue
            except Exception:
                pass

        print(f"\n==========================================", flush=True)
        print(f"▶ RUN START: {item['run_id']} [{item['condition'].upper()}]", flush=True)
        print(f"==========================================", flush=True)
        res = run_trial(item, results_dir, args.fixtures.resolve())
        update_summary(results_dir)
        u = res.get('token_usage') or {}
        print(f"✔ RUN COMPLETE: {item['run_id']} | Success: {res.get('task_success')} | Time: {res.get('elapsed_seconds')}s | Tokens: {u.get('total_tokens')} | Cost: ${res.get('total_cost_usd')}", flush=True)


if __name__ == '__main__':
    main()
