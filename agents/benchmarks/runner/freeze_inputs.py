#!/usr/bin/env python3
"""Freeze u4lib inputs into snapshots/{control,treatment}/workspace.

Treatment keeps u4lib/agents + the current README; control removes the agents
package and strips the README's agent references. u3lib/include and
qcx-utils/include are copied (code-only, no agents) in both conditions, since
u4lib layers on them. The only control/treatment difference is u4lib/agents and
the u4lib README."""
import hashlib
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
LIB = BENCH.parents[1]          # u4lib
QCX = LIB.parent
SNAP = HERE / 'snapshots'


def main():
    common = (SNAP / 'control/workspace/common.md').read_bytes()
    environment = (SNAP / 'control/workspace/environment.sh').read_bytes()
    allowed = {'.h', '.hpp', '.cpp', '.cc', '.c', '.md', '.sh', '.ipynb', '.tex', '.bib', '.pdf', '.dat', '.txt'}
    for condition in ('control', 'treatment'):
        root = SNAP / condition / 'workspace'
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        (root / 'common.md').write_bytes(common)
        (root / 'environment.sh').write_bytes(environment)
        (root / 'consumer').mkdir()
        for p in LIB.rglob('*'):
            if not p.is_file():
                continue
            rel = p.relative_to(LIB)
            if any(x in rel.parts for x in ('.git', 'benchmarks', '__pycache__', '.ipynb_checkpoints', 'bin')):
                continue
            if rel.parts[0] not in ('include', 'src', 'python', 'test', 'doc', 'agents') and len(rel.parts) != 1:
                continue
            if condition == 'control' and rel.parts[0] == 'agents':
                continue
            if p.suffix not in allowed and p.name not in ('Makefile', 'LICENSE', '.gitignore', '.gitlab-ci.yml'):
                continue
            dst = root / 'u4lib' / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
        shutil.copytree(QCX / 'u3lib/include', root / 'u3lib/include', ignore=shutil.ignore_patterns('.DS_Store', '._*'))
        shutil.copytree(QCX / 'qcx-utils/include', root / 'qcx-utils/include', ignore=shutil.ignore_patterns('.DS_Store', '._*'))
    control = SNAP / 'control/workspace/u4lib/README.md'
    text = control.read_text()
    text = text.replace('AI-assisted computational framework', 'computational framework')
    text = text.replace('exposed through a C++ API, Python bindings, and an\nagent-driven skill layer over all of it.',
                        'exposed through a C++ API and Python bindings.')
    text = re.sub(r'> \*\*Note for AI Agents:\*\*.*?\n\n---\n\n', '', text, flags=re.DOTALL)
    text = re.sub(r'- `agents/`[^\n]+\n', '', text)
    if '## Documentation & AI Agent Skills' in text:
        text = text.split('## Documentation & AI Agent Skills')[0].rstrip() + '\n\n## Documentation\n\n- **Full Reference Manual:** For exhaustive mathematical derivations, phase conventions, getter signatures, and solver backends, consult [doc/Manual.pdf](doc/Manual.pdf).\n'
    elif '## Agent-assisted calculations' in text:
        text = text.split('## Agent-assisted calculations')[0]
    assert not re.search(r'agent|skill', text, re.I), f'Unremoved agent/skill ref in control README: {re.findall(r"agent|skill", text, re.I)}'
    control.write_text(text.rstrip() + '\n')

    # Strip agent references from the control manual too, for zero leak.
    manual = SNAP / 'control/workspace/u4lib/doc/Manual.tex'
    if manual.is_file():
        mt = manual.read_text()
        mt = re.sub(r'\\section\{Agent-assisted interface\}.*?(?=\\section\{Build and installation\})',
                    '', mt, flags=re.DOTALL)
        mt = mt.replace(
            'Python bindings, and a repository-local catalog of agent skills lets a\n'
            'code-executing AI assistant drive any capability from a natural-language request.\n'
            'This document is the reference manual for the public release.',
            'Python bindings. This document is the reference manual for the public release.')
        mt = mt.replace(
            'C++ API is header-only, Python bindings mirror it one-to-one, and an agent-skill\n'
            'catalog exposes every capability to an AI assistant. The remainder documents the\n'
            'repository structure (Sec.~\\ref{sec:structure}), the classes and interface\n'
            '(Sec.~\\ref{sec:classes}), the Python bindings (Sec.~\\ref{sec:python}), the\n'
            'agent-assisted interface (Sec.~\\ref{sec:agents}), and the build\n'
            '(Sec.~\\ref{sec:build}).',
            'C++ API is header-only and Python bindings mirror it one-to-one. The remainder documents the\n'
            'repository structure (Sec.~\\ref{sec:structure}), the classes and interface\n'
            '(Sec.~\\ref{sec:classes}), the Python bindings (Sec.~\\ref{sec:python}), and the build\n'
            '(Sec.~\\ref{sec:build}).')
        mt = re.sub(r'    \\item \\verb\|agents/\|[^\n]*\n', '', mt)
        assert not re.search(r'\bagents?\b|\bskills?\b', mt, re.I), \
            f'Unremoved agent/skill ref in control manual: {re.findall(r"[A-Za-z]*agent[A-Za-z-]*|[A-Za-z]*skill[A-Za-z]*", mt, re.I)}'
        manual.write_text(mt)
    manifest = {'version': 'split-v1', 'conditions': {}, 'files': {},
                'condition_difference': 'treatment has u4lib/agents + unchanged README; control removes agents and README agent refs'}
    for condition in ('control', 'treatment'):
        root = SNAP / condition / 'workspace'
        hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(root.rglob('*')) if p.is_file()}
        manifest['conditions'][condition] = hashes
        manifest['files'][condition] = list(hashes)
    a, b = manifest['conditions']['control'], manifest['conditions']['treatment']
    assert all(a[k] == b[k] for k in a if k not in ('u4lib/README.md', 'u4lib/doc/Manual.tex'))
    assert all(k.startswith('u4lib/agents/') for k in b.keys() - a.keys())
    (SNAP / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Frozen u4 sources; differences restricted to agents/, README, and manual.')


if __name__ == '__main__':
    main()
