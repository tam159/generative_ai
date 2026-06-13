"""Dump the code-object tree of a .pyc: the skeleton to reconstruct from.

    <python> introspect.py <file.pyc>

Shows, recursively, each function/class/comprehension's argcounts, varnames,
cell/free vars, names (attrs/globals), and consts (docstrings, literals,
nested code). Read this FIRST to map imports, signatures, and structure.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pyc import load_pyc  # noqa: E402


def summarize(code, indent=0):
    pad = "  " * indent
    print(f"{pad}CODE <{code.co_name}> args={code.co_argcount} "
          f"kwonly={code.co_kwonlyargcount} posonly={code.co_posonlyargcount} "
          f"flags=0x{code.co_flags:x}")
    if code.co_varnames:
        print(f"{pad}  varnames={code.co_varnames}")
    if code.co_cellvars:
        print(f"{pad}  cellvars={code.co_cellvars}")
    if code.co_freevars:
        print(f"{pad}  freevars={code.co_freevars}")
    print(f"{pad}  names={code.co_names}")
    consts = []
    for c in code.co_consts:
        consts.append(f"<code {c.co_name}>" if isinstance(c, types.CodeType) else repr(c))
    print(f"{pad}  consts={consts}")
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            summarize(c, indent + 1)


def main():
    if len(sys.argv) < 2:
        print("usage: <python> introspect.py <file.pyc>")
        return 2
    path = sys.argv[1]
    print("=" * 80)
    print("FILE:", path)
    print("=" * 80)
    summarize(load_pyc(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
