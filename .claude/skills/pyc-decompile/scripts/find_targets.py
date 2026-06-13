"""List .pyc files that need decompiling, with complexity for triage/batching.

    <python> find_targets.py [--all] <dir-or-file> [<dir-or-file> ...]

By default lists only .pyc that LACK a .py sibling (what needs work).
--all lists every .pyc. Skips __pycache__. Sorted by code-object count
(a good proxy for reconstruction effort). Use the counts to bin-pack files
into balanced batches for parallel subagents (~50-60 objs/batch works well).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pyc import count_code_objects, load_pyc  # noqa: E402


def iter_pyc(paths):
    for p in paths:
        if os.path.isfile(p) and p.endswith(".pyc"):
            yield p
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                if "__pycache__" in root.split(os.sep):
                    continue
                for fn in sorted(files):
                    if fn.endswith(".pyc"):
                        yield os.path.join(root, fn)


def main():
    args = sys.argv[1:]
    show_all = False
    if args and args[0] == "--all":
        show_all = True
        args = args[1:]
    if not args:
        print("usage: <python> find_targets.py [--all] <dir-or-file> ...")
        return 2

    rows = []
    for pyc in iter_pyc(args):
        py = pyc[:-1]  # strip trailing 'c'
        has_py = os.path.exists(py)
        if has_py and not show_all:
            continue
        try:
            n = count_code_objects(load_pyc(pyc))
        except Exception as e:
            n = -1
            pyc = f"{pyc}  (LOAD ERROR: {e})"
        rows.append((n, os.path.getsize(pyc) if n >= 0 else 0, pyc, has_py))

    rows.sort(key=lambda r: (r[0], r[2]))
    for n, sz, pyc, has_py in rows:
        flag = "[has .py] " if (show_all and has_py) else ""
        print(f"{n:4d} objs  {sz:8d}B  {flag}{pyc}")
    total = sum(r[0] for r in rows if r[0] > 0)
    print(f"--- {len(rows)} file(s), {total} total code objects ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
