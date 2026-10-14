#!/usr/bin/env python3
"""Host-mode Claude pilot for u4lib. Adapted from u3lib's run_claude_host.py.

Each attempt runs the `claude` CLI non-interactively in a throwaway temp copy of
the frozen condition workspace (runner/snapshots/{condition}/workspace), with a
host environment.sh (identical in both conditions). u4lib layers on u3lib, so the
runtime exposes BOTH u4lib.so and u3lib.so. The grader is u3lib's grade.py, which
is library-agnostic for tasks 4 and 5 (it runs the consumer and compares to
fixtures); only the fixtures (expected values) are SU(4)-specific.

Token metric: input + cache_creation + cache_read + output (cached input added,
not double-counted). Breakdown + total_cost_usd recorded.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
QCX = BENCH.parents[2]                      # u4lib/agents/benchmarks -> QCX
RESULTS = BENCH / 'results/pilot-claude-u4'
SNAP = HERE / 'snapshots'
RUNTIME = Path('/tmp/qcx-claude-runtime-u4')
HOST_ENV = RUNTIME / 'host-environment.sh'
FIXTURES = BENCH / 'evaluator/fixtures-pilot.json'
GRADER = QCX / 'u3lib/agents/benchmarks/evaluator/grade.py'   # shared, library-agnostic
PY = str(Path.home() / 'miniforge3/bin/python3')
MODEL = 'claude-sonnet-5-5'
PROMPT = 'Implement the task specified in common.md and prompt.md. Write all deliverables only in consumer/.'
TIMEOUT = 1800


def agent_cmd():
    return (". ./environment.sh && claude -p " + repr(PROMPT) +
            " --model " + MODEL + " --output-format json "
            "--permission-mode bypassPermissions < /dev/null")


def token_usage(obj):
    u = obj.get('usage') if isinstance(obj, dict) else None
    if not isinstance(u, dict):
        return None
    it = u.get('input_tokens', 0); ot = u.get('output_tokens', 0)
    cc = u.get('cache_creation_input_tokens', 0); cr = u.get('cache_read_input_tokens', 0)
    return {'input_tokens': it, 'output_tokens': ot,
            'cache_creation_input_tokens': cc, 'cache_read_input_tokens': cr,
            'total_tokens': it + ot + cc + cr}


def grade(task, consumer):
    env = os.environ.copy()
    env['PATH'] = f"{Path.home()}/miniforge3/bin:/opt/homebrew/bin:" + env.get('PATH', '')
    env['PYTHONPATH'] = str(RUNTIME) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    wig = QCX / 'External_libraries/wigxjpf-1.13'
    snap = SNAP / 'control/workspace'
    env.update({
        'QCX_ROOT': str(snap), 'U4LIB_DIR': str(snap / 'u4lib'),
        'U3LIB_DIR': str(snap / 'u3lib'), 'QCX_UTILS_DIR': str(snap / 'qcx-utils'),
        'WIGXJPF_DIR': str(wig), 'USE_WIGXJPF': '1', 'NULLSPACE_SOLVER': 'qr',
        'CXX': '/opt/homebrew/bin/g++-16', 'CC': '/opt/homebrew/bin/gcc-16',
        'CXXFLAGS': ('-std=c++20 -O2 -DUSE_WIGXJPF -I/opt/homebrew/opt/boost/include '
                     '-I/opt/homebrew/opt/eigen/include/eigen3 '
                     f"-I{snap/'u4lib/include'} -I{snap/'u4lib/include/internal'} "
                     f"-I{snap/'u3lib/include'} -I{snap/'u3lib/include/internal'} "
                     f"-I{snap/'qcx-utils/include'} -I{wig/'inc'}"),
        'LDFLAGS': f"-L{wig/'lib'} -lwigxjpf", 'OMP_NUM_THREADS': '1',
    })
    return subprocess.run([PY, str(GRADER), str(task), str(consumer), str(FIXTURES)],
                          text=True, capture_output=True, env=env, timeout=600)


def run_one(item):
    rid, cond, task = item['run_id'], item['condition'], item['task']
    out = RESULTS / rid
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f'{rid}-'))
    ws = tmp / 'workspace'
    shutil.copytree(SNAP / cond / 'workspace', ws)
    shutil.copy2(HOST_ENV, ws / 'environment.sh')
    (ws / 'prompt.md').write_text(next((HERE / 'tasks').glob(f"{task:02d}-*.md")).read_text())

    rec = {**item, 'model': MODEL, 'status': 'running'}
    try:
        proc = subprocess.run(['sh', '-lc', agent_cmd()], cwd=ws, stdin=subprocess.DEVNULL,
                              text=True, capture_output=True, timeout=TIMEOUT)
        (out / 'exec.json').write_text(proc.stdout)
        (out / 'exec.stderr.txt').write_text(proc.stderr)
        try:
            obj = json.loads(proc.stdout.strip())
        except ValueError:
            obj = None
        usage = token_usage(obj) if obj else None
        elapsed = int(obj['duration_ms'] / 1000) if isinstance(obj, dict) and obj.get('duration_ms') else None
        cost = obj.get('total_cost_usd') if isinstance(obj, dict) else None
        agent_error = (obj or {}).get('is_error') if isinstance(obj, dict) else True

        shutil.copytree(ws / 'consumer', out / 'consumer', dirs_exist_ok=True)
        g = grade(task, ws / 'consumer')
        (out / 'grader.stdout.txt').write_text(g.stdout)
        (out / 'grader.stderr.txt').write_text(g.stderr)
        try:
            gres = json.loads(g.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            gres = {'success': False, 'error': 'grader emitted no JSON'}
        within = (elapsed is not None and elapsed <= TIMEOUT)
        grade_pass = bool(g.returncode == 0 and gres.get('success'))
        rec.update({
            'status': 'completed',
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'agent_is_error': agent_error,
            'elapsed_seconds': elapsed, 'within_time_cap': within,
            'grade_pass': grade_pass,
            'task_success': bool(grade_pass and within),
            'grader': gres, 'token_usage': usage, 'token_usage_missing': usage is None,
            'total_cost_usd': cost,
        })
    except subprocess.TimeoutExpired:
        rec.update({'status': 'timeout', 'task_success': False,
                    'completed_at_utc': datetime.now(timezone.utc).isoformat()})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    (out / 'run.json').write_text(json.dumps(rec, indent=2) + '\n')
    return rec


def update_summary():
    recs = []
    for p in sorted(RESULTS.glob('claude-*/run.json')):
        r = json.loads(p.read_text())
        recs.append({k: r.get(k) for k in (
            'run_id', 'task', 'condition', 'repeat', 'status', 'task_success',
            'token_usage', 'total_cost_usd', 'elapsed_seconds', 'within_time_cap',
            'agent_is_error', 'completed_at_utc')})
    (RESULTS / 'results.json').write_text(json.dumps(recs, indent=2) + '\n')
    with (RESULTS / 'results.csv').open('w') as f:
        f.write('run_id,task,condition,repeat,status,task_success,elapsed_seconds,'
                'within_time_cap,input_tokens,cache_creation,cache_read,output_tokens,'
                'total_tokens,total_cost_usd\n')
        for r in recs:
            u = r.get('token_usage') or {}
            f.write(','.join(str(x if x is not None else '') for x in (
                r['run_id'], r['task'], r['condition'], r['repeat'], r['status'],
                r['task_success'], r['elapsed_seconds'], r['within_time_cap'],
                u.get('input_tokens'), u.get('cache_creation_input_tokens'),
                u.get('cache_read_input_tokens'), u.get('output_tokens'),
                u.get('total_tokens'), r.get('total_cost_usd'))) + '\n')


def main():
    global RESULTS, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--only')
    ap.add_argument('--results', default='pilot-claude-u4', help='results dir under results/')
    ap.add_argument('--model', default=MODEL, help='claude model id to pin')
    a = ap.parse_args()
    RESULTS = BENCH / 'results' / a.results
    MODEL = a.model
    sched = json.loads((RESULTS / 'schedule.json').read_text())
    todo = [r for r in sched['runs']
            if (a.only and r['run_id'] == a.only) or (a.all and r['status'] != 'completed')]
    if not todo:
        ap.error('nothing to run (use --all or --only RUN_ID)')
    for item in todo:
        print('START', item['run_id'], flush=True)
        rec = run_one(item)
        for r in sched['runs']:
            if r['run_id'] == item['run_id']:
                r.update({'status': rec['status'], 'task_success': rec.get('task_success'),
                          'elapsed_seconds': rec.get('elapsed_seconds')})
        (RESULTS / 'schedule.json').write_text(json.dumps(sched, indent=2) + '\n')
        update_summary()
        u = rec.get('token_usage') or {}
        print('DONE', item['run_id'], 'success=', rec.get('task_success'),
              'total_tokens=', u.get('total_tokens'), 'cost=', rec.get('total_cost_usd'), flush=True)


if __name__ == '__main__':
    main()
