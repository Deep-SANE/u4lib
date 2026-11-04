#!/usr/bin/env python3
"""Freeze u4lib inputs into snapshots/{control,treatment}/workspace.

Treatment keeps u4lib/agents + the current README and manual.
Control removes the agents package and strips agent references from README and Manual.tex.
The only control/treatment difference is u4lib/agents and agent-related documentation.
"""
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
LIB = BENCH.parents[1]          # u4lib
SNAP = BENCH / 'snapshots'
COMMON = BENCH / 'tasks/common.md'


def make_environment_sh(root: Path):
    content = """#!/usr/bin/env bash
export U4LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/u4lib" && pwd)"
export U3LIB_DIR="${U3LIB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../u3lib" 2>/dev/null && pwd)}"
export PYTHONPATH="$U4LIB_DIR/python:$U3LIB_DIR/python:${PYTHONPATH:-}"
export PATH="${PATH}"
"""
    (root / 'environment.sh').write_text(content)
    (root / 'environment.sh').chmod(0o755)


def main():
    if not COMMON.is_file():
        raise SystemExit(f"Missing {COMMON}")
    common_bytes = COMMON.read_bytes()

    allowed_exts = {'.h', '.hpp', '.cpp', '.cc', '.c', '.md', '.sh', '.ipynb', '.tex', '.bib', '.pdf', '.dat', '.txt', '.so', '.mk'}

    for condition in ('control', 'treatment'):
        root = SNAP / condition / 'workspace'
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        (root / 'common.md').write_bytes(common_bytes)
        (root / 'consumer').mkdir(parents=True, exist_ok=True)
        make_environment_sh(root)

        for p in LIB.rglob('*'):
            if not p.is_file():
                continue
            rel = p.relative_to(LIB)
            if any(x in rel.parts for x in ('.git', 'benchmarks', '__pycache__', '.ipynb_checkpoints')):
                continue
            if rel.parts[0] not in ('include', 'src', 'python', 'test', 'doc', 'agents', 'utils', 'config', 'bin') and len(rel.parts) != 1:
                continue
            if condition == 'control' and rel.parts[0] == 'agents':
                continue
            if p.suffix not in allowed_exts and p.name not in ('Makefile', 'LICENSE', '.gitignore'):
                continue
            dst = root / 'u4lib' / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
            shutil.copymode(p, dst)

    # Clean control README.md
    control_readme = SNAP / 'control/workspace/u4lib/README.md'
    text = control_readme.read_text()
    text = text.replace('**AI-ready computational framework for U(4)/SU(4) algebra**',
                        'computational framework for U(4)/SU(4) algebra')
    text = text.replace('exposed through a C++ API, Python bindings, and an\nagent-driven skill layer over all of it.',
                        'exposed through a C++ API and Python bindings.')
    text = text.replace('exposed through a C++ API, Python bindings, and an agent-driven skill layer over all of it.',
                        'exposed through a C++ API and Python bindings.')
    text = re.sub(r'> \*\*Note for AI Agents:\*\*.*?\n\n---\n\n', '', text, flags=re.DOTALL)
    text = re.sub(r'\*\*Installing with an AI agent\.\*\*.*?(?=- \*\*Regression Tests)', '', text, flags=re.DOTALL)
    text = re.sub(r'- `agents/`[^\n]+\n', '', text)
    if '## Documentation & AI Agent Skills' in text:
        text = text.split('## Documentation & AI Agent Skills')[0].rstrip() + '\n\n## Documentation\n\n- **Full Reference Manual:** For exhaustive mathematical derivations, phase conventions, getter signatures, and solver backends, consult [doc/Manual.pdf](doc/Manual.pdf).\n'
    elif '## Agent-assisted calculations' in text:
        text = text.split('## Agent-assisted calculations')[0]
    assert not re.search(r'\bagents?\b|\bskills?\b', text, re.I), f'Unremoved agent/skill ref in control README: {re.findall(r"agents?|skills?", text, re.I)}'
    control_readme.write_text(text.rstrip() + '\n')

    # Clean control Manual.tex
    control_manual = SNAP / 'control/workspace/u4lib/doc/Manual.tex'
    if control_manual.is_file():
        mt = control_manual.read_text()
        mt = re.sub(r'\\section\{(?:Agent-assisted|AI-ready) interface\}.*?(?=\\section\{Build and installation\}|\\section\{Conclusions\}|\\section\{References\}|\\bibliographystyle)',
                    '', mt, flags=re.DOTALL)
        mt = mt.replace(
            'Python bindings, and a repository-local catalog of agent skills lets a\ncode-executing AI assistant drive any of these capabilities from a\nnatural-language request.',
            'Python bindings.')
        mt = mt.replace(
            'C++ API is header-only, Python bindings mirror it one-to-one, and an agent-skill\ncatalog exposes every capability to an AI assistant. The remainder documents',
            'C++ API is header-only, and Python bindings mirror it one-to-one. The remainder documents')
        mt = re.sub(r'    \\item \\verb\|agents/\|[^\n]*\n', '', mt)
        mt = re.sub(r'the\s+(?:agent-assisted|AI-ready)\s+interface\s+\(Sec\.~\\ref\{sec:agents\}\),?\s*', '', mt)
        control_manual.write_text(mt)

    print("Successfully froze u4lib control and treatment snapshots.")


if __name__ == '__main__':
    main()
