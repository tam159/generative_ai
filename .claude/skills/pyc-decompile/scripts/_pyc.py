"""Shared helpers for the pyc-decompile skill.

Pure stdlib. Works on CPython 3.7+ (.pyc header is 16 bytes since PEP 552).
The verify gate REQUIRES running under the same CPython version that produced
the .pyc — see preflight.py.
"""
import dis
import importlib.util
import marshal
import struct
import sys
import types

# First 2 bytes of a .pyc (little-endian) identify the CPython release.
# Authoritative check is always MAGIC_NUMBER comparison (see pyc_matches_current);
# this map is only a human-friendly hint.
MAGIC_HINT = {
    3379: "3.6", 3394: "3.7", 3413: "3.8", 3425: "3.9", 3439: "3.10",
    3495: "3.11", 3531: "3.12", 3571: "3.13", 3621: "3.14",
}


def load_pyc(path):
    """Unmarshal the top-level code object from a .pyc (skips 16-byte header)."""
    with open(path, "rb") as f:
        return marshal.loads(f.read()[16:])


def pyc_magic(path):
    with open(path, "rb") as f:
        return f.read(4)


def pyc_version_hint(path):
    n = struct.unpack("<H", pyc_magic(path)[:2])[0]
    return MAGIC_HINT.get(n, f"unknown (magic short={n})")


def pyc_matches_current(path):
    """True iff the running interpreter can faithfully (dis-)assemble this .pyc."""
    return pyc_magic(path) == importlib.util.MAGIC_NUMBER


def _norm_const(v):
    # set/frozenset repr order is hash-seed dependent; normalize to compare by value.
    if isinstance(v, (set, frozenset)):
        return (type(v).__name__, tuple(sorted(map(repr, v))))
    if isinstance(v, tuple):
        return tuple(_norm_const(x) for x in v)
    return v


def instr_stream(code):
    """Recursive (opname, argrepr) stream over all nested code objects.

    Ignores line numbers and memory addresses; normalizes set/frozenset consts.
    Two code objects with identical streams are functionally identical.
    """
    out = []

    def walk(c, prefix):
        ops = []
        for ins in dis.get_instructions(c):
            if isinstance(ins.argval, types.CodeType):
                arg = f"<code {ins.argval.co_name}>"
            elif ins.opname == "LOAD_CONST":
                arg = repr(_norm_const(ins.argval))
            else:
                arg = ins.argrepr
            ops.append((ins.opname, arg))
        out.append((prefix, tuple(ops)))
        for const in c.co_consts:
            if isinstance(const, types.CodeType):
                walk(const, prefix + "." + const.co_name)

    walk(code, code.co_name)
    return out


def verify(pyc_path, py_path):
    """Compile py_path and compare its opcode stream to pyc_path's.

    Returns True on exact match, else a short human-readable diff string.
    """
    orig = load_pyc(pyc_path)
    try:
        with open(py_path) as f:
            comp = compile(f.read(), orig.co_filename, "exec")
    except SyntaxError as e:
        return f"SYNTAX ERROR: {e}"
    A, B = instr_stream(orig), instr_stream(comp)
    if A == B:
        return True
    import difflib
    amap, bmap = dict(A), dict(B)
    msgs = []
    if [k for k, _ in A] != [k for k, _ in B]:
        oa, ob = set(amap) - set(bmap), set(bmap) - set(amap)
        if oa:
            msgs.append(f"code objects only in ORIGINAL: {sorted(oa)[:8]}")
        if ob:
            msgs.append(f"code objects only in RECONSTRUCTED: {sorted(ob)[:8]}")
    for k in amap:
        if k in bmap and amap[k] != bmap[k]:
            msgs.append(f"--- first mismatch in {k}: {len(amap[k])} vs {len(bmap[k])} instrs")
            d = difflib.unified_diff(
                [str(x) for x in amap[k]], [str(x) for x in bmap[k]],
                "orig", "recon", lineterm="", n=0,
            )
            msgs.append("\n".join(list(d)[:30]))
            break
    return "\n".join(msgs) if msgs else "MISMATCH (structure differs)"


def count_code_objects(code):
    n = 1
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            n += count_code_objects(c)
    return n
