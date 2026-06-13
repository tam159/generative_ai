"""Full disassembly of a .pyc, recursing into nested code objects.

    <python> disasm.py <file.pyc>

Includes line numbers and the ExceptionTable (control-flow ground truth).
For big files, redirect to a file and read it in ranges:
    <python> disasm.py big.pyc > /tmp/big.dis.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pyc import load_pyc  # noqa: E402

import dis  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("usage: <python> disasm.py <file.pyc>")
        return 2
    dis.dis(load_pyc(sys.argv[1]), depth=40)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
