"""Exercise the hidden grader with deliberately broken consumer submissions."""
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("grader", HERE / "grade.py")
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)


def fail(label, fn):
    try:
        fn()
    except (AssertionError, StopIteration, ValueError) as exc:
        print(f"PASS: grader rejects {label} ({type(exc).__name__})")
    else:
        raise SystemExit(f"FAIL: grader accepted mutation: {label}")


def rejected_submission(task, fixture_data, write):
    with tempfile.TemporaryDirectory(prefix=f"u3-task{task}-mutation-") as td:
        consumer = Path(td) / "consumer"
        consumer.mkdir()
        (consumer / "README.md").write_text("Deliberately broken grader fixture.\n")
        (consumer / "test_mutation.py").write_text("# Deliberately broken fixture.\n")
        write(consumer, fixture_data)
        fail(f"task {task} broken submission", lambda: grader.grade(task, consumer, fixture_data))


def python_mutation(task, module, source):
    def write(consumer, _):
        (consumer / f"{module}.py").write_text(source)
    return write


def make_cli_mutation(task, name, body):
    def write(consumer, _):
        (consumer / "Makefile").write_text("all:\n\t@true\n")
        exe = consumer / name
        exe.write_text("#!/bin/sh\n" + body)
        exe.chmod(0o755)
    return write


if __name__ == "__main__":
    import sys
    fixtures = json.loads(Path(sys.argv[1]).read_text())
    # Task 1: channel labels and orthogonal but incorrect values must fail.
    first = fixtures['1'][0]
    src = "import numpy as np\ndef recoupling(kind,*labels):\n return " + repr(first['left']) + ", " + repr(first['right']) + ", np.ones((" + str(len(first['left'])) + "," + str(len(first['right'])) + "),dtype=np.float64)\n"
    rejected_submission(1, fixtures, python_mutation(1, "recouple", src))
    # Task 2: syntactically complete output with every coefficient copied from
    # the final scratch block must fail the signed value comparisons.
    def stale_u9(consumer, data):
        (consumer / "Makefile").write_text("all:\n\t@true\n")
        cases = repr(data['2'])
        code = f'''#!/usr/bin/env python3
import sys
cases = {cases}
requests = [line for line in sys.stdin if line.strip()]
if len(requests) != len(cases): raise SystemExit(2)
stale = cases[-1]["entries"][0][-1]
for q, case in enumerate(cases):
 print("block", q, *case["shape"])
 for row in case["entries"]:
  print("entry", q, *row[:6], stale)
'''
        exe = consumer / "u9_batch"
        exe.write_text(code)
        exe.chmod(0o755)
    rejected_submission(2, fixtures, stale_u9)
    # Task 3: incorrect output or premature EOF must fail.
    rejected_submission(3, fixtures, make_cli_mutation(3, "coupling_batch", "echo 'canon 0 0'\n"))
    # Shared numerical/ownership guards: wrong sign, swapped/missing axes,
    # and a non-owning view must be rejected.
    ref = np.array([0.25, -0.5], dtype=np.float64)
    fail("wrong sign", lambda: grader.close(-ref, ref))
    mat = np.arange(6, dtype=np.float64).reshape(2, 3)
    fail("swapped axes", lambda: grader.close(mat.T, mat))
    fail("missing multiplicity axis", lambda: grader.close(mat[:, 0], mat))
    borrowed = np.arange(8, dtype=np.float64)[::2]
    fail("borrowed coefficient buffer", lambda: grader.array(borrowed, 1))
    print("PASS: deliberate grader mutation suite")
