# Bytecode → source cheat-sheet (modern CPython 3.11–3.13)

`dis` already decodes most operands in the `(...)` argrepr. Read `disasm.py`
output alongside `introspect.py` output. Key patterns:

## Calls, attributes, operators
- `LOAD_GLOBAL X + NULL` … `CALL n` → `X(arg1..argn)`.
- `LOAD_ATTR (name + NULL|self)` … `CALL n` → bound method call `obj.name(...)`.
- `CALL_KW n` with a trailing const tuple of names → the last *k* args are keyword args named by that tuple, e.g. `f(a, x=1, y=2)`.
- `CALL_FUNCTION_EX` + `BUILD_MAP 0`/`DICT_MERGE`/`LIST_EXTEND` → `f(*args, **kwargs)`.
- `BINARY_OP` argrepr shows the operator: `(+) (-) (*) (/) (//) (%) (**) (|) (&) (^) (<<) (>>) (@)` and in-place `(+=)` etc.
- `COMPARE_OP (bool(==))` → `==`; `CONTAINS_OP 0`→`in`, `1`→`not in`; `IS_OP 0`→`is`, `1`→`is not`.
- `BUILD_LIST/TUPLE/SET/MAP`, `BUILD_CONST_KEY_MAP` → literal containers (a `dict(...)` *call* is `LOAD_GLOBAL dict; CALL`, a dict *literal* is `BUILD_(CONST_KEY_)MAP` — they differ!).

## Stores / names / scope
- `STORE_FAST`→local, `STORE_NAME`/`STORE_GLOBAL`→module/global, `STORE_DEREF`→closure cell, `STORE_ATTR`→`obj.x = …`, `STORE_SUBSCR`→`obj[k] = …`.
- `LOAD_FAST_LOAD_FAST` / `STORE_FAST_STORE_FAST` → two locals fused; this only happens when both are on the **same source line** — if verify shows a fused-vs-unfused diff, merge/split the source line.
- `MAKE_CELL` / `COPY_FREE_VARS` → variable is a closure cell (nested function reads/writes it).

## Functions / classes / defs
- `MAKE_FUNCTION` + `SET_FUNCTION_ATTRIBUTE` flags: `0x01` defaults (tuple), `0x02` kwdefaults, `0x04` annotations, `0x08` closure.
- Annotations: a flattened const tuple like `('arg', T, 'return', R)` → `def f(arg: T) -> R`. Missing `0x04` flag → no annotations.
- `LOAD_BUILD_CLASS … MAKE_FUNCTION … CALL 3/4` → `class Name(Base, …):`; the body is the nested code object.
- In a class body: `SETUP_ANNOTATIONS` + `STORE_SUBSCR` into `__annotations__` → bare `k: v` annotations (TypedDict/dataclass fields).
- `__firstlineno__` (a `STORE_NAME` of an int const in the class body) is baked in: the `class` statement **must** sit on that exact physical source line. If verify reports a `LOAD_CONST <int>` mismatch in a class body, pad blank lines/comments above the class to land it on the right line.
- Decorators: the decorator expression is evaluated, then `MAKE_FUNCTION`, then `CALL` applies it. `@functools.wraps(func)` etc.

## Control flow
- `RESUME 0` at the top → ignore (preamble).
- `RETURN_GENERATOR; POP_TOP; RESUME` at the top → generator OR `async def`. If you see `GET_AWAITABLE/SEND/END_SEND` it's `async def`; plain `YIELD_VALUE` without await machinery is a generator.
- `GET_AWAITABLE 0; LOAD_CONST None; SEND…; YIELD_VALUE; …; END_SEND` → one `await <expr>`. Collapse the whole block.
- `FOR_ITER` (preceded by `GET_ITER`) → `for`; body runs to the jump target, `END_FOR` ends it.
- `POP_JUMP_IF_FALSE/TRUE` → `if`/`while` branches. A trailing `NOP` right after `RESUME` often marks a `try:` line.
- `BEFORE_WITH`/`WITH_EXCEPT_START` → `with`; `BEFORE_ASYNC_WITH` → `async with`. Two context managers fused (`with a, b:`) share one block.
- `CHECK_EXC_MATCH` → an `except SomeType:` arm; `PUSH_EXC_INFO` opens a handler; the **ExceptionTable** maps try-body ranges → handlers (read it to get try/except/else/finally shape exactly).
- `RAISE_VARARGS 0`→bare `raise`, `1`→`raise X`, `2`→`raise X from Y`. `RERAISE` is the implicit re-raise at a handler's end.
- `return None`: `RETURN_CONST None` is a *bare* `return`/fall-off; `LOAD_CONST None; RETURN_VALUE` is an explicit `return None` produced by a ternary/expression. They differ in bytecode — match what's there.

## Comprehensions, strings, imports
- Comprehensions are nested code objects named `<listcomp>/<setcomp>/<dictcomp>/<genexpr>`. A `list(... )`-style call with no extra code object is a list *comprehension inlined*; a genexpr passed to a function has its own code object. `asyncio.gather(*[...])` (listcomp) vs `(...)` (genexpr) produce different bytecode.
- f-strings: `FORMAT_SIMPLE/FORMAT_WITH_SPEC/CONVERT_VALUE/BUILD_STRING`. `CONVERT_VALUE (str)`→`!s`, `(repr)`→`!r`, `(ascii)`→`!a`.
- `IMPORT_NAME` + `IMPORT_FROM` → `from mod import a, b` (const level 0/1…, names tuple lists imports). `import a.b` does `IMPORT_NAME a.b; STORE_NAME a`. `import x.y as z` uses fromlist `None` and `IMPORT_FROM`/`STORE` — distinct from `from x import y`.
- Walrus `:=` shows as a `COPY` + `STORE_FAST` keeping the value on the stack (e.g. `if x := f():`).

## Tactics
- Reconstruct top-down: module level first, then each class, then each method.
- After every edit run `verify.py`; it points at the **first** mismatching code object — fix that one thing, re-run. Most diffs are: operator, positional-vs-keyword arg, list-comp-vs-genexpr, `and`-chain vs sequential guards, ternary vs if/else, or a `__firstlineno__` line offset.
- You can't recover comments or exact blank lines — that's expected and fine. Only the opcode stream must match.
