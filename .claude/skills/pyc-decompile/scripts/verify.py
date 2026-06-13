"""THE SUCCESS GATE. Compile a reconstructed .py and compare its opcode stream
to the original .pyc (line numbers/addresses ignored; sets normalized).

    <python> verify.py <file.pyc> <file.py>

Prints "✅ LOGICAL MATCH" when functionally exact (done), else the first
mismatching code object with a per-instruction diff. Iterate until match.
Comments/blank lines are NOT in bytecode, so they never need to match — but a
LOAD_CONST int mismatch inside a class body is a `__firstlineno__`: move the
class statement to its exact original source line (pad blank lines/comments).

Batch mode over a directory tree (audit everything that has a .py sibling):
    <python> verify.py --audit <dir> [<dir> ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pyc import verify  # noqa: E402


def audit(roots):
    total = ok = missing = bad = 0
    for root in roots:
        for dirpath, _, files in os.walk(root):
            if "__pycache__" in dirpath.split(os.sep):
                continue
            for fn in sorted(files):
                if not fn.endswith(".pyc"):
                    continue
                pyc = os.path.join(dirpath, fn)
                py = pyc[:-1]
                total += 1
                if not os.path.exists(py):
                    missing += 1
                    print("  MISSING .py:", pyc)
                    continue
                r = verify(pyc, py)
                if r is True:
                    ok += 1
                else:
                    bad += 1
                    print("  MISMATCH:", py)
    print(f"=== {total} .pyc | verified={ok} missing={missing} mismatch={bad} ===")
    return 0 if (missing == 0 and bad == 0) else 1


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--audit":
        return audit(sys.argv[2:])
    if len(sys.argv) != 3:
        print("usage: <python> verify.py <file.pyc> <file.py>")
        print("   or: <python> verify.py --audit <dir> ...")
        return 2
    pyc, py = sys.argv[1], sys.argv[2]
    r = verify(pyc, py)
    if r is True:
        print(f"✅ LOGICAL MATCH (opcode stream identical): {py}")
        return 0
    print(f"⚠️  MISMATCH for {py}")
    print(r)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
