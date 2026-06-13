"""Check that the running interpreter matches a .pyc's CPython version.

Decompilation/verification is only reliable under the exact CPython version that
produced the .pyc. Run this FIRST with a candidate interpreter:

    <python> preflight.py <file.pyc>

Exit 0 = this interpreter matches (use it). Exit 1 = mismatch (find another).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pyc import pyc_matches_current, pyc_version_hint  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("usage: <python> preflight.py <file.pyc>")
        return 2
    pyc = sys.argv[1]
    cur = ".".join(map(str, sys.version_info[:3]))
    need = pyc_version_hint(pyc)
    if pyc_matches_current(pyc):
        print(f"✅ interpreter MATCHES: this is CPython {cur}; .pyc is {need}.")
        print(f"   Use: {sys.executable}")
        return 0
    print(f"❌ interpreter MISMATCH: running CPython {cur}, but .pyc needs ~{need}.")
    print("   Find a matching CPython (often a project's .venv/bin/python, pyenv,")
    print("   uv, or conda env) and re-run preflight with it before proceeding.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
