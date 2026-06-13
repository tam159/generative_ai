"""Fast path: find an existing original .py that VERIFIES against a .pyc.

Compiled trees often ship alongside (or mirror) the real source. If a .py
exists somewhere whose opcode stream matches the .pyc exactly, copying it is
strictly better than reconstructing (keeps real comments/formatting).

    <python> find_source.py <file.pyc> <search-dir> [<search-dir> ...]

Prints any verifying .py paths found (best candidate first). Matches by
basename, then confirms by full opcode-stream equality. Exit 0 if found.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pyc import verify  # noqa: E402


def main():
    if len(sys.argv) < 3:
        print("usage: <python> find_source.py <file.pyc> <search-dir> ...")
        return 2
    pyc = sys.argv[1]
    roots = sys.argv[2:]
    target_base = os.path.basename(pyc)[:-1]  # foo.py

    # Prefer candidates whose tail path matches the .pyc's tail (more specific).
    pyc_parts = os.path.normpath(pyc).split(os.sep)
    candidates = []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            if "__pycache__" in dirpath.split(os.sep):
                continue
            for fn in files:
                if fn == target_base:
                    cand = os.path.join(dirpath, fn)
                    cand_parts = os.path.normpath(cand).split(os.sep)
                    # score: how many trailing path components match
                    score = 0
                    for a, b in zip(reversed(pyc_parts[:-1]), reversed(cand_parts[:-1])):
                        if a == b:
                            score += 1
                        else:
                            break
                    candidates.append((-score, cand))
    candidates.sort()

    found = []
    for _, cand in candidates:
        if verify(pyc, cand) is True:
            found.append(cand)

    if found:
        print(f"✅ {len(found)} verifying source(s) for {pyc}:")
        for c in found:
            print(f"   COPY: {c}")
        return 0
    print(f"— no verifying source found under {roots} for {pyc} (reconstruct it)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
