"""Vurze extractor for ``.vz`` files (github.com/VurzeReBooted/Vurze).

No tree-sitter grammar exists for Vurze, so this is a hand-written lexer plus a
declaration-level parser that follows the language's own front end (the token
regex and ``Parser.program`` of ``vurzec.py``; ``selfhost/lexer.vz`` is the same
lexer in Vurze). Vurze has no block comments and no character literals:
``//`` comments, ``"..."`` strings with backslash escapes, identifiers, numbers
and a fixed operator set are the whole token language.

What one file yields (all ``EXTRACTED``):

* the file node; ``import "path.vz"`` -> ``imports_from`` to the imported
  file's node (resolved like the compiler does: next to the importing file,
  then each folder above it up to the scan root, then ``$VURZE_PATH`` and
  ``$VURZE_HOME``); an import that cannot be found on disk gets a
  module stub labelled with the import string (``std/c.vz``).
* ``fn`` (``const fn``, ``det fn``, ``export fn``, ``unsafe fn``), methods
  ``fn Type.name(self: *Type, ...)`` (label ``.name()``, ``method`` edge from the
  type when it is declared in the same file), ``struct``, ``enum`` (variants as
  ``case_of`` children), ``trait``, module-level ``let``/``var``/``const``,
  ``extern fn`` / ``extern var`` (one shared node per C symbol, so every file
  that declares ``sqrtf`` points at the same node) and the functions
  ``@derive(...)`` generates (``Type_eq``, ``Type_to_json``, ...).
* attributes (``@fp_exact``, ``@vertex``, ``@fragment``, ``@compute``,
  ``@kernel``, ``@inline``, ``@naked``, ``@packed``, ``@nocopy``, ``@safe``,
  ``@section``, ``@align``, ``@derive``, ``@vertex_input``) as metadata on the
  declaration plus a ``references`` edge (context ``attribute``) to one shared
  ``@name`` node; ``@fp_exact(module)`` / ``@safe(module)`` link the FILE to
  ``@fp_exact(module)`` and mark every later function ``fp_exact``.
* calls, type uses (parameter, return, field, local, cast, ``alloc[T]``,
  ``sizeof[T]``, explicit type arguments), struct construction ``T(...)``,
  function names used as values (callbacks, hooks, dispatch tables) and reads
  of module-level ``let``/``var``/``const``. A name the file itself declares is
  bound here (Vurze has one program-wide namespace in which a file's own
  definition, and its ``private`` ones, win); every other name is left in
  ``raw_calls`` for :func:`resolve_vurze_references`, which binds it through the
  file's transitive imports.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from graphify.extractors.base import _file_stem, _make_id, _read_source_text, active_scan_root

# vurzec.py's TOK, verbatim apart from the names of the groups.
_TOKEN_RE = re.compile(r'''
 (?P<ws>[ \t\r]+)|(?P<com>//[^\n]*)|(?P<nl>\n)|
 (?P<float>\d+\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)|(?P<int>0x[0-9a-fA-F]+|\d+)|
 (?P<str>"(?:\\.|[^"\\])*")|(?P<id>[A-Za-z_]\w*)|
 (?P<op>=>|->|\.\.\.|\.\.|==|!=|<=|>=|&&|\|\||<<|>>|\+=|-=|\*=|/=|[-+*/%<>=!&|^(){}\[\],:;@.?])''', re.X)

_KEYWORDS = frozenset({
    "fn", "let", "var", "const", "struct", "enum", "trait", "import", "extern", "export",
    "private", "det", "unsafe", "if", "else", "while", "for", "par", "in", "match", "return",
    "break", "continue", "defer", "as", "true", "false", "null", "requires", "ensures",
    "effects", "static_assert", "self", "Self", "_",
})
# Compiler builtins: never a reference to a definition in the corpus unless the
# file declares one of that name itself (in-file binding runs first).
_BUILTINS = frozenset({
    "alloc", "sizeof", "offsetof", "len", "select", "heap", "include_str", "reduce_add",
    "reduce_min", "reduce_max", "reduce_mul", "atomic_load", "atomic_store", "atomic_add",
    "atomic_cas", "atomic_sub", "atomic_xchg", "wrap_add", "wrap_sub", "wrap_mul", "sat_add",
    "sat_sub", "add_overflows", "sub_overflows", "mul_overflows", "mul_hi", "asm", "assert",
    "fail", "unreachable", "hot_sym", "zero", "conv", "printf",
})
_PRIMITIVES = frozenset({
    "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "f32", "f64",
    "bool", "usize", "isize", "void", "closure", "Tex2D", "Tex2DArray", "Tex3D", "TexCube",
})
_VEC_RE = re.compile(r"^(?:i8|i16|i32|i64|u8|u16|u32|u64|f16|f32|f64|bool)x(?:2|3|4|8|16|32)$")
_TOP_STARTS = frozenset({
    "fn", "let", "var", "const", "struct", "enum", "trait", "import", "extern", "export",
    "private", "det", "unsafe", "static_assert",
})
_DERIVE_FNS = {
    "eq": ("_eq",), "hash": ("_hash",), "clone": ("_clone",), "debug": ("_debug",),
    "json": ("_to_json", "_from_json", "_from_json_str"),
}
_SHADER_STAGES = {"vertex": "vertex", "fragment": "fragment", "compute": "compute", "kernel": "kernel"}

_LANGUAGE = "vurze"


def _lex(src: str) -> list[tuple[str, str, int]]:
    """Tokens (kind, text, line); comments and blanks dropped, every newline kept.

    A character the language does not have (a stray non-ASCII byte outside a
    string or comment) is skipped instead of aborting the file.
    """
    toks: list[tuple[str, str, int]] = []
    pos, line, n = 0, 1, len(src)
    match = _TOKEN_RE.match
    while pos < n:
        m = match(src, pos)
        if m is None:
            if src[pos] == '"':  # an unterminated string: skip to the end of the line
                end = src.find("\n", pos)
                pos = n if end < 0 else end
                continue
            pos += 1
            continue
        kind = m.lastgroup or ""
        text = m.group()
        pos = m.end()
        if kind == "nl":
            toks.append(("nl", "\n", line))
            line += 1
            continue
        if kind in ("ws", "com"):
            continue
        if kind == "str":
            toks.append(("str", text, line))
            line += text.count("\n")
            continue
        toks.append((kind, text, line))
    toks.append(("eof", "", line))
    return toks


def _is_type_name(name: str) -> bool:
    return not (name in _PRIMITIVES or name in _KEYWORDS or _VEC_RE.match(name))


class _TypeScan:
    """Parse one Vurze type at ``i``: returns its end and the names it uses.

    type := '*' type | '[' type (';' expr)? ']' | ('fn'|'closure') '(' types ')' ('->' type)?
          | '(' types ')' | NAME ('.' NAME)* ('<' types '>')? ('!' NAME?)?
    """

    def __init__(self, toks: list[tuple[str, str, int]]):
        self.t = toks
        # `Box<Pair<A, B>>` lexes its two closing '>' as one `>>`: the inner list
        # closes on its first half and leaves the second half to the outer list.
        self.half_closed: int | None = None

    def scan(self, i: int, names: list[tuple[str, int]], depth: int = 0) -> int:
        t = self.t
        if depth > 40 or i >= len(t):
            return i
        k, v, line = t[i]
        if k == "op" and v == "*":
            return self.scan(i + 1, names, depth + 1)
        if k == "op" and v == "&":
            return self.scan(i + 1, names, depth + 1)
        if k == "op" and v == "[":
            j = self.scan(i + 1, names, depth + 1)
            # the length (`; N`, `; N * 2`): skip to the matching ']'
            d = 0
            while j < len(t):
                kk, vv, _ = t[j]
                if kk == "eof":
                    return j
                if kk == "op" and vv in ("[", "("):
                    d += 1
                elif kk == "op" and vv in ("]", ")"):
                    if d == 0:
                        return j + 1 if vv == "]" else j
                    d -= 1
                elif kk == "op" and vv in ("{", "}"):
                    return j
                j += 1
            return j
        if k == "id" and v in ("fn", "closure") and i + 1 < len(t) and t[i + 1][1] == "(":
            j = self._list(i + 2, ")", names, depth)
            if j < len(t) and t[j][1] == "->":
                j = self.scan(j + 1, names, depth + 1)
            return j
        if k == "op" and v == "(":
            return self._list(i + 1, ")", names, depth)
        if k == "id":
            if _is_type_name(v):
                names.append((v, line))
            j = i + 1
            while j + 1 < len(t) and t[j][1] == "." and t[j + 1][0] == "id":
                j += 2
            if j < len(t) and t[j][1] == "<":
                j = self._list(j + 1, ">", names, depth)
            if j < len(t) and t[j][1] == "!":
                j += 1
                # `T!E`: the error enum, unless a block / an expression follows
                if j < len(t) and t[j][0] == "id" and t[j][1] not in _KEYWORDS:
                    if _is_type_name(t[j][1]):
                        names.append((t[j][1], t[j][2]))
                    j += 1
            return j
        return i

    def _list(self, i: int, close: str, names: list[tuple[str, int]], depth: int) -> int:
        """Comma-separated types up to ``close`` (``>`` also closes on ``>>``)."""
        t = self.t
        j = i
        guard = 0
        while j < len(t) and guard < 400:
            guard += 1
            k, v, _ = t[j]
            if k == "eof":
                return j
            if k == "nl":
                j += 1
                continue
            if v == close:
                return j + 1
            if close == ">" and v == ">>":
                if self.half_closed == j:
                    self.half_closed = None
                    return j + 1
                self.half_closed = j
                return j
            if v in (",", ":", "const", "..."):
                j += 1
                continue
            if k == "id" and j + 1 < len(t) and t[j + 1][1] == ":":
                j += 2  # a named parameter in a fn type, or `T: Bound`
                continue
            if k in ("int", "float"):
                j += 1
                continue
            if k == "op" and v in ("+",):
                j += 1
                continue
            nj = self.scan(j, names, depth + 1)
            if nj == j:
                return j
            j = nj
        return j


def _match_close(toks: list[tuple[str, str, int]], i: int) -> int:
    """Index of the token closing the bracket at ``i`` (or the last token)."""
    opener = toks[i][1]
    closer = {"(": ")", "[": "]", "{": "}"}[opener]
    d = 0
    for j in range(i, len(toks)):
        k, v, _ = toks[j]
        if k != "op":
            continue
        if v == opener:
            d += 1
        elif v == closer:
            d -= 1
            if d == 0:
                return j
    return len(toks) - 1


def _next_sig(toks: list[tuple[str, str, int]], i: int) -> int:
    while i < len(toks) and toks[i][0] == "nl":
        i += 1
    return i


def _strip_str(s: str) -> str:
    return s[1:-1] if len(s) >= 2 and s[0] == '"' and s[-1] == '"' else s


def resolve_vurze_import(raw: str, source_path: Path) -> Path | None:
    """The file ``import "raw"`` loads, searched the way the compiler searches.

    The importing file's folder first, then each folder above it (the build's
    working directory is usually one of them: ``import "re/chunks.vz"`` from
    ``re/chunks_demo.vz`` is relative to the engine's root) up to the scan root,
    then ``$VURZE_PATH`` (``:``-separated) and ``$VURZE_HOME`` (where ``std/``
    lives). The result keeps the relative/absolute form of ``source_path`` so its
    id matches the imported file's own node id.
    """
    if not raw:
        return None
    rel = Path(raw)
    if rel.is_absolute():
        return Path(os.path.normpath(rel)) if rel.is_file() else None
    here = source_path.parent
    candidates: list[Path] = [here / rel]
    scan_root = active_scan_root()
    try:
        abs_up = here.resolve()
    except OSError:
        abs_up = here.absolute()
    up = here
    for _ in range(12 if scan_root is not None else 6):
        if scan_root is not None and abs_up == scan_root:
            break
        if abs_up.parent == abs_up:
            break
        up = up.parent if str(up) not in ("", ".") and up.name not in ("..",) else up / ".."
        abs_up = abs_up.parent
        candidates.append(up / rel)
    for d in os.environ.get("VURZE_PATH", "").split(os.pathsep):
        if d:
            candidates.append(Path(d) / rel)
    home = os.environ.get("VURZE_HOME")
    if home:
        candidates.append(Path(home) / rel)
    for cand in candidates:
        try:
            if cand.is_file():
                return Path(os.path.normpath(cand))
        except OSError:
            continue
    return None


def extract_vurze(path: Path) -> dict:
    """Extract declarations, imports, calls and type uses from one ``.vz`` file."""
    try:
        src = _read_source_text(path)
    except OSError as exc:
        return {"nodes": [], "edges": [], "error": str(exc)}

    toks = _lex(src)
    types = _TypeScan(toks)
    stem = _file_stem(path)
    str_path = str(path)
    file_nid = _make_id(str_path)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    raw_calls: list[dict[str, Any]] = []
    seen_ids: dict[str, str] = {}
    seen_edges: set[tuple[str, str, str]] = set()

    def add_node(nid: str, label: str, line: int, kind: str, *, owner_key: str,
                 source_backed: bool = True, callable_node: bool = False,
                 meta: dict | None = None) -> str:
        """Add a node; a different declaration whose id collides (ids fold case:
        ``fn v3`` and ``struct V3``) gets the kind, then the line, appended."""
        if nid in seen_ids and seen_ids[nid] != owner_key:
            alt = _make_id(nid, kind)
            if alt in seen_ids and seen_ids[alt] != owner_key:
                alt = _make_id(nid, kind, f"l{line}")
            nid = alt
        if nid in seen_ids:
            return nid
        seen_ids[nid] = owner_key
        metadata = {"language": _LANGUAGE, "kind": kind}
        if meta:
            metadata.update(meta)
        item: dict[str, Any] = {
            "id": nid, "label": label, "file_type": "code",
            "source_location": f"L{line}", "metadata": metadata,
        }
        if source_backed:
            item["source_file"] = str_path
        if callable_node:
            item["_callable"] = True
        nodes.append(item)
        return nid

    def add_edge(src_id: str, tgt_id: str, relation: str, line: int, *,
                 context: str | None = None, target_file: str | None = None,
                 confidence: str = "EXTRACTED") -> None:
        if not src_id or not tgt_id or src_id == tgt_id:
            return
        key = (src_id, tgt_id, relation)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edge: dict[str, Any] = {
            "source": src_id, "target": tgt_id, "relation": relation,
            "confidence": confidence, "confidence_score": 1.0 if confidence == "EXTRACTED" else 0.85,
            "source_file": str_path, "source_location": f"L{line}", "weight": 1.0,
        }
        if context:
            edge["context"] = context
        if target_file:
            edge["target_file"] = target_file
        edges.append(edge)

    add_node(file_nid, path.name, 1, "file", owner_key="file")

    def attr_node(name: str, line: int) -> str:
        # one shared node per attribute across the corpus (no source file, like a stub)
        nid = _make_id("vurze_attribute", name)
        if nid not in seen_ids:
            seen_ids[nid] = "attr:" + name
            nodes.append({"id": nid, "label": "@" + name, "file_type": "code",
                          "source_file": "", "source_location": f"L{line}",
                          "metadata": {"language": _LANGUAGE, "kind": "attribute"}})
        return nid

    def extern_node(name: str, line: int, kind: str) -> str:
        nid = _make_id("vurze_extern", name)
        if nid not in seen_ids:
            seen_ids[nid] = "extern:" + name
            label = f"{name}()" if kind == "extern_fn" else name
            nodes.append({"id": nid, "label": label, "file_type": "code",
                          "source_file": str_path, "source_location": f"L{line}",
                          "metadata": {"language": _LANGUAGE, "kind": kind, "extern": True},
                          **({"_callable": True} if kind == "extern_fn" else {})})
        return nid

    # ---- pass 1: the declarations at brace depth 0 ----
    decls: list[dict[str, Any]] = []
    imports: list[tuple[str, int]] = []
    module_attrs: list[tuple[str, int]] = []
    fp_exact_module = False
    safe_module = False
    n = len(toks)
    i = 0
    pending_attrs: list[tuple[str, list[str], int]] = []
    private = unsafe = det = extern = export = const_fn = False

    def reset_flags() -> None:
        nonlocal private, unsafe, det, extern, export, const_fn
        private = unsafe = det = extern = export = const_fn = False

    def skip_statement(j: int) -> int:
        """End of a module-level `let`/`const` initialiser: the next newline at
        depth 0 that is followed by the start of another declaration."""
        d = 0
        while j < n:
            k, v, _ = toks[j]
            if k == "eof":
                return j
            if k == "op" and v in ("(", "[", "{"):
                d += 1
            elif k == "op" and v in (")", "]", "}"):
                d -= 1
                if d < 0:
                    return j
            elif k == "nl" and d <= 0:
                nj = _next_sig(toks, j)
                nk, nv, _ = toks[nj]
                if nk == "eof" or (nk == "id" and nv in _TOP_STARTS) or (nk == "op" and nv == "@"):
                    return nj
            j += 1
        return j

    while i < n:
        k, v, line = toks[i]
        if k == "eof":
            break
        if k == "nl":
            i += 1
            continue
        if k == "id" and v == "import" and i + 1 < n and toks[i + 1][0] == "str":
            imports.append((_strip_str(toks[i + 1][1]), line))
            i += 2
            if i < n and toks[i][1] == "effects" and i + 1 < n and toks[i + 1][1] == "(":
                i = _match_close(toks, i + 1) + 1
            continue
        if k == "op" and v == "@" and i + 1 < n and toks[i + 1][0] == "id":
            aname = toks[i + 1][1]
            args: list[str] = []
            j = i + 2
            if j < n and toks[j][1] == "(":
                close = _match_close(toks, j)
                args = [tv for tk, tv, _ in toks[j + 1:close] if tk in ("id", "int", "str")]
                j = close + 1
            if aname in ("fp_exact", "safe") and args == ["module"]:
                module_attrs.append((aname + "(module)", line))
                if aname == "fp_exact":
                    fp_exact_module = True
                else:
                    safe_module = True
            else:
                pending_attrs.append((aname, args, line))
            i = j
            continue
        if k != "id":
            i += 1
            continue
        if v == "private":
            private = True
            i += 1
            continue
        if v == "unsafe" and i + 1 < n and toks[i + 1][1] in ("fn", "det", "extern", "export"):
            unsafe = True
            i += 1
            continue
        if v == "det" and i + 1 < n and toks[i + 1][1] in ("fn", "extern", "export"):
            det = True
            i += 1
            continue
        if v == "extern" and i + 1 < n and toks[i + 1][1] in ("var", "let"):
            if i + 2 < n and toks[i + 2][0] == "id":
                name = toks[i + 2][1]
                tnames: list[tuple[str, int]] = []
                j = i + 3
                if j < n and toks[j][1] == ":":
                    j = types.scan(j + 1, tnames)
                decls.append({"kind": "extern_var", "name": name, "line": line, "type_refs": tnames,
                              "private": private, "attrs": pending_attrs})
                i = skip_statement(j)
            else:
                i += 2
            pending_attrs = []
            reset_flags()
            continue
        if v == "extern":
            extern = True
            i += 1
            continue
        if v == "export" and i + 1 < n and toks[i + 1][1] in ("fn", "var", "let", "det", "extern", "unsafe", "const"):
            export = True
            i += 1
            continue
        if v == "const" and i + 1 < n and toks[i + 1][1] == "fn":
            const_fn = True
            i += 1
            continue
        if v in ("let", "var", "const") and i + 1 < n and toks[i + 1][0] == "id":
            name = toks[i + 1][1]
            tnames = []
            j = i + 2
            if j < n and toks[j][1] == ":":
                j = types.scan(j + 1, tnames)
            init_start = j + 1 if j < n and toks[j][1] == "=" else j
            end = skip_statement(init_start)
            decls.append({"kind": "const" if v == "const" else "global", "mut": v == "var",
                          "name": name, "line": line, "type_refs": tnames, "body": (init_start, end),
                          "private": private, "export": export, "attrs": pending_attrs})
            pending_attrs = []
            reset_flags()
            i = end
            continue
        if v == "static_assert" and i + 1 < n and toks[i + 1][1] == "(":
            i = _match_close(toks, i + 1) + 1
            pending_attrs = []
            reset_flags()
            continue
        if v == "trait" and i + 2 < n and toks[i + 1][0] == "id" and toks[i + 2][1] == "{":
            name = toks[i + 1][1]
            close = _match_close(toks, i + 2)
            methods = []
            for j in range(i + 3, close):
                if toks[j][1] == "fn" and j + 1 < close and toks[j + 1][0] == "id":
                    methods.append(toks[j + 1][1])
            decls.append({"kind": "trait", "name": name, "line": line, "methods": methods,
                          "private": private, "attrs": pending_attrs})
            pending_attrs = []
            reset_flags()
            i = close + 1
            continue
        if v in ("struct", "enum") and i + 1 < n and toks[i + 1][0] == "id":
            name = toks[i + 1][1]
            j = i + 2
            tparams: list[str] = []
            if j < n and toks[j][1] == "<":
                d = 0
                while j < n:
                    tv = toks[j][1]
                    if tv == "<":
                        d += 1
                    elif tv == ">":
                        d -= 1
                    elif tv == ">>":
                        d -= 2
                    elif toks[j][0] == "id" and toks[j - 1][1] in ("<", ",", "const") and tv != "const":
                        tparams.append(tv)
                    j += 1
                    if d <= 0:
                        break
            while j < n and toks[j][1] != "{" and toks[j][0] != "nl":
                j += 1
            if j >= n or toks[j][1] != "{":
                i = j
                pending_attrs = []
                reset_flags()
                continue
            close = _match_close(toks, j)
            decl: dict[str, Any] = {"kind": v, "name": name, "line": line, "tparams": tparams,
                                    "private": private, "attrs": pending_attrs}
            if v == "struct":
                # `name: Type` separated by ',' or newlines; a type never holds a
                # `{`, so field starts are the names right after `{`, `,` or a newline
                fields: list[tuple[str, list[tuple[str, int]]]] = []
                q = j + 1
                while q < close:
                    if toks[q][0] == "id" and q + 1 < close and toks[q + 1][1] == ":" \
                            and toks[q - 1][1] in ("{", ",", "\n"):
                        fts: list[tuple[str, int]] = []
                        nq = types.scan(q + 2, fts)
                        fields.append((toks[q][1], fts))
                        q = max(nq, q + 2)
                        continue
                    q += 1
                decl["fields"] = fields
            else:
                variants: list[tuple[str, int, list[tuple[str, int]]]] = []
                q = j + 1
                while q < close:
                    tk, tv, tl = toks[q]
                    if tk == "id" and toks[q - 1][1] in ("{", ",", "\n"):
                        payload: list[tuple[str, int]] = []
                        nq = q + 1
                        if nq < close and toks[nq][1] == "(":
                            pc = _match_close(toks, nq)
                            types._list(nq + 1, ")", payload, 0)
                            nq = pc + 1
                        variants.append((tv, tl, payload))
                        q = nq
                        continue
                    q += 1
                decl["variants"] = variants
            decls.append(decl)
            pending_attrs = []
            reset_flags()
            i = close + 1
            continue
        if v == "fn" and i + 1 < n and toks[i + 1][0] == "id":
            name = toks[i + 1][1]
            recv: str | None = None
            j = i + 2
            if j + 1 < n and toks[j][1] == "." and toks[j + 1][0] == "id":
                recv, name = name, toks[j + 1][1]
                j += 2
            tparams = []
            if j < n and toks[j][1] == "<":
                d = 0
                while j < n:
                    tv = toks[j][1]
                    if tv == "<":
                        d += 1
                    elif tv == ">":
                        d -= 1
                    elif tv == ">>":
                        d -= 2
                    elif toks[j][0] == "id" and toks[j - 1][1] in ("<", ",", "const"):
                        if tv != "const":
                            tparams.append(tv)
                    j += 1
                    if d <= 0:
                        break
            params: list[tuple[str, list[tuple[str, int]]]] = []
            if j < n and toks[j][1] == "(":
                pclose = _match_close(toks, j)
                q = j + 1
                while q < pclose:
                    if toks[q][0] == "id" and q + 1 < pclose and toks[q + 1][1] == ":":
                        pnames: list[tuple[str, int]] = []
                        nq = types.scan(q + 2, pnames)
                        params.append((toks[q][1], pnames))
                        q = max(nq, q + 2)
                        continue
                    q += 1
                j = pclose + 1
            ret: list[tuple[str, int]] = []
            if j < n and toks[j][1] == "->":
                j = types.scan(j + 1, ret)
            # contracts / effects: up to the body's '{' (or the end of an extern line)
            body: tuple[int, int] | None = None
            q = j
            pd = 0
            while q < n:
                tk, tv, _ = toks[q]
                if tk == "eof":
                    break
                if tk == "op" and tv in ("(", "["):
                    pd += 1
                elif tk == "op" and tv in (")", "]"):
                    pd -= 1
                elif tk == "op" and tv == "{" and pd == 0:
                    if extern:
                        break
                    close = _match_close(toks, q)
                    body = (q + 1, close)
                    q = close + 1
                    break
                elif tk == "nl" and pd <= 0:
                    nq = _next_sig(toks, q)
                    if extern or toks[nq][0] == "eof" or (toks[nq][0] == "id" and toks[nq][1] in _TOP_STARTS) \
                            or toks[nq][1] == "@":
                        break
                q += 1
            attrs = pending_attrs
            decls.append({"kind": "fn", "name": name, "recv": recv, "line": line, "params": params,
                          "ret": ret, "body": body, "tparams": tparams, "private": private,
                          "extern": extern, "export": export, "det": det, "const": const_fn,
                          "unsafe": unsafe, "attrs": attrs,
                          "fp_exact": fp_exact_module or any(a[0] == "fp_exact" for a in attrs),
                          "safe": (safe_module and body is not None) or any(a[0] == "safe" for a in attrs)})
            pending_attrs = []
            reset_flags()
            i = q if body is not None else max(q, i + 2)
            continue
        # anything else at the top level (a stray token): step over it
        i += 1

    # ---- the file's imports ----
    for raw, line in imports:
        target = resolve_vurze_import(raw, path)
        if target is not None:
            add_edge(file_nid, _make_id(str(target)), "imports_from", line, target_file=str(target))
        else:
            stub = _make_id("vurze_module", raw)
            if stub not in seen_ids:
                seen_ids[stub] = "module:" + raw
                nodes.append({"id": stub, "label": raw, "file_type": "code",
                              "source_file": "", "source_location": f"L{line}",
                              "metadata": {"language": _LANGUAGE, "kind": "external_module"}})
            add_edge(file_nid, stub, "imports_from", line)
    for aname, line in module_attrs:
        add_edge(file_nid, attr_node(aname, line), "references", line, context="attribute")

    # ---- nodes for the declarations ----
    # name -> node id of what this file declares (its own definitions win: one
    # program-wide namespace, and a private name hides a public one in its file)
    local_defs: dict[str, tuple[str, str]] = {}       # name -> (nid, kind)
    local_methods: dict[tuple[str, str], str] = {}     # (type, method) -> nid
    local_method_names: dict[str, list[str]] = {}      # method -> nids
    extern_names: dict[str, str] = {}                  # name -> shared extern nid
    struct_fields: dict[str, dict[str, str]] = {}      # struct -> field -> first type name
    fn_ret: dict[str, str] = {}                        # fn name -> first named return type
    pending_methods: list[tuple[str, str, int]] = []   # (method nid, type name, line)

    def decl_meta(d: dict) -> dict:
        meta: dict[str, Any] = {}
        attrs = [a for a, _args, _l in d.get("attrs", [])]
        if attrs:
            meta["attributes"] = attrs
        for flag in ("private", "export", "extern", "det", "const", "unsafe", "fp_exact", "safe"):
            if d.get(flag):
                meta[flag] = True
        for a, args, _l in d.get("attrs", []):
            if a in _SHADER_STAGES:
                meta["shader_stage"] = _SHADER_STAGES[a]
                if args:
                    meta["workgroup_size"] = [int(x, 0) for x in args if x[:1].isdigit()]
            if a == "derive":
                meta["derive"] = list(args)
        return meta

    type_kinds = ("struct", "enum", "trait")
    # types first, so methods can hang off a type declared later in the file
    for d in decls:
        if d["kind"] in type_kinds:
            tmeta = decl_meta(d)
            if d["kind"] == "trait":
                tmeta["trait_methods"] = list(d.get("methods", []))
            nid = add_node(_make_id(stem, d["name"]), d["name"], d["line"], d["kind"],
                           owner_key=f"{d['kind']}:{d['name']}", meta=tmeta)
            add_edge(file_nid, nid, "contains", d["line"])
            local_defs.setdefault(d["name"], (nid, d["kind"]))
            d["nid"] = nid
            if d["kind"] == "struct":
                struct_fields[d["name"]] = {}
                for fname, fts in d.get("fields", []):
                    named = [x for x, _ in fts]
                    if named:
                        struct_fields[d["name"]][fname] = named[0]
    for d in decls:
        kind = d["kind"]
        if kind in type_kinds:
            nid = d["nid"]
            if kind == "enum":
                for vname, vline, payload in d.get("variants", []):
                    vid = add_node(_make_id(nid, vname), vname, vline, "enum_variant",
                                   owner_key=f"variant:{d['name']}.{vname}")
                    add_edge(nid, vid, "case_of", vline)
                    d.setdefault("type_refs_ctx", []).extend((x, ln, "field") for x, ln in payload)
            elif kind == "struct":
                for _fname, fts in d.get("fields", []):
                    d.setdefault("type_refs_ctx", []).extend((x, ln, "field") for x, ln in fts)
            for derive_kind in decl_meta(d).get("derive", []):
                for sfx in _DERIVE_FNS.get(derive_kind, ()):
                    dname = d["name"] + sfx
                    did = add_node(_make_id(stem, dname), f"{dname}()", d["line"], "derived_fn",
                                   owner_key=f"fn:{dname}", callable_node=True,
                                   meta={"derived_from": d["name"], **({"private": True} if d.get("private") else {})})
                    add_edge(nid, did, "defines", d["line"], context="derive")
                    local_defs.setdefault(dname, (did, "fn"))
            continue
        if kind == "fn":
            name, recv = d["name"], d.get("recv")
            meta = decl_meta(d)
            if d.get("ret"):
                rnamed = [x for x, _ in d["ret"]]
                if rnamed:
                    meta["return_type"] = rnamed[0]
                    if not recv:
                        fn_ret.setdefault(name, rnamed[0])
            if d.get("extern") and d.get("body") is None:
                nid = extern_node(name, d["line"], "extern_fn")
                add_edge(file_nid, nid, "references", d["line"], context="extern_decl")
                extern_names.setdefault(name, nid)
                d["nid"] = nid
                continue
            if recv:
                meta["receiver_type"] = recv
                nid = add_node(_make_id(stem, recv, name), f".{name}()", d["line"], "method",
                               owner_key=f"method:{recv}.{name}", callable_node=True, meta=meta)
                owner = local_defs.get(recv)
                if owner is not None and owner[1] in type_kinds:
                    add_edge(owner[0], nid, "method", d["line"])
                else:
                    add_edge(file_nid, nid, "contains", d["line"])
                    pending_methods.append((nid, recv, d["line"]))
                local_methods[(recv, name)] = nid
                local_method_names.setdefault(name, []).append(nid)
            else:
                nid = add_node(_make_id(stem, name), f"{name}()", d["line"], "fn",
                               owner_key=f"fn:{name}", callable_node=True, meta=meta)
                add_edge(file_nid, nid, "contains", d["line"])
                if name not in local_defs or local_defs[name][1] not in ("fn",):
                    local_defs[name] = (nid, "fn")
            d["nid"] = nid
            for aname, _args, aline in d.get("attrs", []):
                add_edge(nid, attr_node(aname, aline), "references", aline, context="attribute")
            continue
        if kind in ("global", "const"):
            meta = decl_meta(d)
            if d.get("mut"):
                meta["mutable"] = True
            nid = add_node(_make_id(stem, d["name"]), d["name"], d["line"], kind,
                           owner_key=f"{kind}:{d['name']}", meta=meta)
            add_edge(file_nid, nid, "contains", d["line"])
            local_defs.setdefault(d["name"], (nid, kind))
            d["nid"] = nid
            continue
        if kind == "extern_var":
            nid = extern_node(d["name"], d["line"], "extern_var")
            add_edge(file_nid, nid, "references", d["line"], context="extern_decl")
            extern_names.setdefault(d["name"], nid)
            d["nid"] = nid
    # attributes of types (@packed, @derive, @nocopy, @vertex_input ...)
    for d in decls:
        if d["kind"] in type_kinds or d["kind"] in ("global", "const"):
            for aname, _args, aline in d.get("attrs", []):
                add_edge(d["nid"], attr_node(aname, aline), "references", aline, context="attribute")

    # ---- references: in-file binding, the rest deferred to the resolver ----
    def bind_value(caller: str, name: str, line: int, ctx: str, scope_types: dict[str, str]) -> None:
        """A use of `name` as a type / value / callee from `caller` (ctx: the
        edge context: call, constructor, value, parameter_type, ...)."""
        hit = local_defs.get(name)
        if hit is not None:
            tnid, tkind = hit
            if ctx == "call":
                if tkind in ("fn",):
                    add_edge(caller, tnid, "calls", line, context="call")
                elif tkind == "struct":
                    add_edge(caller, tnid, "references", line, context="constructor")
                elif tkind in ("global", "const"):
                    # a call through a fn-pointer global: the global is read
                    add_edge(caller, tnid, "references", line, context="value")
                return
            if ctx == "value":
                if tkind == "fn":
                    add_edge(caller, tnid, "indirect_call", line, context="function_value",
                             confidence="INFERRED")
                elif tkind in ("global", "const"):
                    add_edge(caller, tnid, "references", line, context="value")
                elif tkind in type_kinds:
                    add_edge(caller, tnid, "references", line, context="type")
                return
            if tkind in type_kinds:
                add_edge(caller, tnid, "references", line, context=ctx)
            return
        if ctx in ("call", "value") and name in _BUILTINS and name not in extern_names:
            return
        raw_calls.append({
            "caller_nid": caller, "callee": name, "language": _LANGUAGE, "vurze_kind": ctx,
            "fallback_nid": extern_names.get(name), "source_file": str_path,
            "source_location": f"L{line}",
        })

    def bind_method(caller: str, recv_type: str | None, method: str, line: int) -> None:
        if recv_type is not None:
            hit = local_methods.get((recv_type, method))
            if hit is not None:
                add_edge(caller, hit, "calls", line, context="call")
                return
            if recv_type in local_defs and local_defs[recv_type][1] == "enum":
                add_edge(caller, local_defs[recv_type][0], "references", line, context="variant")
                return
        raw_calls.append({
            "caller_nid": caller, "callee": method, "language": _LANGUAGE,
            "vurze_kind": "method", "receiver_type": recv_type, "is_member_call": True,
            "source_file": str_path, "source_location": f"L{line}",
        })

    def scan_body(caller: str, b0: int, b1: int, scope_types: dict[str, str],
                  locals_: set[str], self_type: str | None) -> None:
        # pre-pass: every name the body declares (flow-insensitive is enough to
        # keep a local `f` from being read as a call to a global `f`)
        j = b0
        while j < b1:
            tk, tv, _tl = toks[j]
            if tk == "id" and tv in ("let", "var") and j + 1 < b1 and toks[j + 1][0] == "id":
                lname = toks[j + 1][1]
                locals_.add(lname)
                if j + 2 < b1 and toks[j + 2][1] == ":":
                    tn: list[tuple[str, int]] = []
                    types.scan(j + 3, tn)
                    if tn:
                        scope_types.setdefault(lname, tn[0][0])
                elif j + 3 < b1 and toks[j + 2][1] == "=" and toks[j + 3][0] == "id" \
                        and j + 4 < b1 and toks[j + 4][1] in ("(", "["):
                    ctor = toks[j + 3][1]
                    if ctor in struct_fields or (ctor[:1].isupper() and ctor not in fn_ret):
                        scope_types.setdefault(lname, ctor)
                    elif ctor in fn_ret:
                        scope_types.setdefault(lname, fn_ret[ctor])
                    else:
                        scope_types.setdefault(lname, "=" + ctor)  # the result of a call: typed by the resolver
            elif tk == "id" and tv == "for" and j + 1 < b1 and toks[j + 1][0] == "id":
                locals_.add(toks[j + 1][1])
            elif tk == "op" and tv in ("|", "||"):
                pk, pv, _ = toks[j - 1] if j > 0 else ("nl", "", 0)
                if tv == "|" and (pv in ("(", ",", "=", "]", "{", "return", "\n", "heap") or pk == "nl"):
                    q = j + 1
                    while q < b1 and toks[q][1] != "|" and toks[q][0] != "nl":
                        if toks[q][0] == "id" and toks[q - 1][1] in ("|", ","):
                            locals_.add(toks[q][1])
                        q += 1
            elif tk == "id" and j + 1 < b1 and toks[j + 1][1] == "(":
                # match arm `Variant(a, b) =>`: a and b are bindings
                close = _match_close(toks, j + 1)
                if close + 1 < len(toks) and toks[close + 1][1] == "=>":
                    for q in range(j + 2, close):
                        if toks[q][0] == "id":
                            locals_.add(toks[q][1])
            j += 1
        # main pass
        j = b0
        while j < b1:
            tk, tv, tl = toks[j]
            if tk != "id":
                j += 1
                continue
            prev = toks[j - 1][1] if j > 0 else ""
            nxt = toks[j + 1][1] if j + 1 < n else ""
            if tv in ("let", "var") and j + 2 < b1 and toks[j + 2][1] == ":":
                tn = []
                end = types.scan(j + 3, tn)
                for x, xl in tn:
                    bind_value(caller, x, xl, "local_type", scope_types)
                j = end
                continue
            if tv == "as":
                tn = []
                end = types.scan(j + 1, tn)
                for x, xl in tn:
                    bind_value(caller, x, xl, "cast", scope_types)
                j = max(end, j + 1)
                continue
            if tv in ("alloc", "sizeof", "offsetof") and nxt == "[" and tv not in local_defs:
                close = _match_close(toks, j + 1)
                tn = []
                types._list(j + 2, "]", tn, 0)
                for x, xl in tn:
                    bind_value(caller, x, xl, "generic_arg", scope_types)
                j = close + 1
                continue
            if prev == ".":
                # a field or method name; a method call is handled here
                if nxt == "(" and j >= 2:
                    rk, rv, _ = toks[j - 2]
                    recv_type: str | None = None
                    if rk == "id" and (j < 3 or toks[j - 3][1] != "."):
                        if rv == "self":
                            recv_type = self_type
                        elif rv in scope_types and not scope_types[rv].startswith("="):
                            recv_type = scope_types[rv]
                        elif rv not in locals_ and rv in local_defs and local_defs[rv][1] in type_kinds:
                            recv_type = rv
                        elif rv not in locals_ and rv[:1].isupper() and rv not in local_defs:
                            recv_type = rv  # `Type.method(...)` / `Enum.Variant(...)` of a type elsewhere
                        elif rv in scope_types:
                            recv_type = scope_types[rv]  # "=fn": the resolver types it by fn's return
                    elif rk == "id" and j >= 4 and toks[j - 3][1] == "." and toks[j - 4][0] == "id":
                        # `a.b.m(...)`: a's type's field b
                        base = toks[j - 4][1]
                        bt = self_type if base == "self" else scope_types.get(base)
                        if bt and not bt.startswith("=") and bt in struct_fields:
                            recv_type = struct_fields[bt].get(rv)
                    bind_method(caller, recv_type, tv, tl)
                j += 1
                continue
            if tv in _KEYWORDS or tv in _PRIMITIVES or _VEC_RE.match(tv):
                j += 1
                continue
            if nxt == "=>":
                j += 1
                continue
            if nxt == "(":
                close = _match_close(toks, j + 1)
                if close + 1 < len(toks) and toks[close + 1][1] == "=>":
                    j = close + 1  # a match arm's pattern
                    continue
                if tv in locals_:
                    j += 1
                    continue
                bind_value(caller, tv, tl, "call", scope_types)
                j += 1
                continue
            if nxt == "[" and tv not in locals_:
                close = _match_close(toks, j + 1)
                if close + 1 < len(toks) and toks[close + 1][1] == "(":
                    # `name[T, U](args)`: explicit type arguments (an index is never called)
                    tn = []
                    types._list(j + 2, "]", tn, 0)
                    for x, xl in tn:
                        bind_value(caller, x, xl, "generic_arg", scope_types)
                    bind_value(caller, tv, tl, "call", scope_types)
                    j = close + 1
                    continue
            if tv in locals_:
                j += 1
                continue
            if nxt == "." and j + 2 < n and toks[j + 2][0] == "id" and j + 3 < n and toks[j + 3][1] == "(":
                j += 1  # `Type.method(` / `Enum.Variant(`: the method token handles it
                continue
            if nxt == ":" and prev in ("{", ",", "(", "|"):
                j += 1
                continue
            bind_value(caller, tv, tl, "value", scope_types)
            j += 1

    for d in decls:
        nid = d.get("nid")
        if not nid:
            continue
        kind = d["kind"]
        for x, xl, ctx in d.get("type_refs_ctx", []):
            bind_value(nid, x, xl, ctx, {})
        if kind == "fn" and not (d.get("extern") and d.get("body") is None):
            tparams = set(d.get("tparams", []))
            scope_types: dict[str, str] = {}
            locals_: set[str] = set(tparams)
            for pname, pts in d.get("params", []):
                locals_.add(pname)
                named = [x for x, _ in pts if x not in tparams]
                if named:
                    scope_types[pname] = named[0]
                for x, xl in pts:
                    if x not in tparams:
                        bind_value(nid, x, xl, "parameter_type", scope_types)
            for x, xl in d.get("ret", []):
                if x not in tparams:
                    bind_value(nid, x, xl, "return_type", scope_types)
            if d.get("body"):
                b0, b1 = d["body"]
                scan_body(nid, b0, b1, scope_types, locals_, d.get("recv"))
        elif kind in ("global", "const", "extern_var"):
            for x, xl in d.get("type_refs", []):
                bind_value(nid, x, xl, "type", {})
            if d.get("body"):
                b0, b1 = d["body"]
                scan_body(nid, b0, b1, {}, set(), None)

    for mnid, tname, line in pending_methods:
        raw_calls.append({
            "caller_nid": mnid, "callee": tname, "language": _LANGUAGE, "vurze_kind": "method_of",
            "source_file": str_path, "source_location": f"L{line}",
        })

    clean_edges = [
        e for e in edges
        if e["source"] in seen_ids and (e["target"] in seen_ids or e["relation"] == "imports_from")
    ]
    return {"nodes": nodes, "edges": clean_edges, "raw_calls": raw_calls}


# ---------------------------------------------------------------------------
# cross-file resolution
# ---------------------------------------------------------------------------

_DEF_KINDS = frozenset({"fn", "derived_fn", "struct", "enum", "trait", "global", "const",
                        "extern_fn", "extern_var"})
_TYPE_KINDS = frozenset({"struct", "enum", "trait"})


def resolve_vurze_references(
    per_file: list[dict], all_nodes: list[dict], all_edges: list[dict]
) -> None:
    """Bind the names a ``.vz`` file uses but does not declare.

    Vurze compiles a program as the transitive closure of its imports with one
    program-wide namespace (``private`` names excepted). So a name is looked up
    first among the public definitions of the files the caller's file imports,
    directly or through other imports: a unique hit in a DIRECTLY imported file
    is ``EXTRACTED`` (the import names the file the definition lives in), a
    unique hit further down the import chain is ``INFERRED`` (0.95). A name no
    import reaches but defined exactly once in the corpus (a library file using
    what its program's main imports) is ``INFERRED`` (0.65). Two or more
    candidates at the same level bind nothing. A name only ever declared
    ``extern`` lands on the shared extern node. A method call binds through its
    receiver's type when the extractor knew it (``self``, a typed parameter or
    local, a local initialised by a call whose return type is known), else only
    when one method of that name is visible through the imports (``INFERRED``
    0.75).
    """
    vz_calls = [rc for result in per_file for rc in (result.get("raw_calls") or [])
                if rc.get("language") == _LANGUAGE]
    if not vz_calls:
        return
    node_by_id = {n.get("id"): n for n in all_nodes if n.get("id")}

    def is_vz(n: dict) -> bool:
        return (n.get("metadata") or {}).get("language") == _LANGUAGE

    # file nodes
    sf_to_file: dict[str, str] = {}
    for n in all_nodes:
        sf = n.get("source_file")
        if sf and is_vz(n) and (n.get("metadata") or {}).get("kind") == "file":
            sf_to_file.setdefault(str(sf), n["id"])
    file_ids = set(sf_to_file.values())
    # owner file of every node: contains / method parents
    parent: dict[str, str] = {}
    for e in all_edges:
        if e.get("relation") in ("contains", "method", "case_of", "defines"):
            parent.setdefault(str(e.get("target")), str(e.get("source")))

    def file_of(nid: str) -> str | None:
        seen = set()
        cur = nid
        while cur and cur not in seen:
            if cur in file_ids:
                return cur
            seen.add(cur)
            cur = parent.get(cur, "")
        n = node_by_id.get(nid)
        if n and n.get("source_file"):
            return sf_to_file.get(str(n["source_file"]))
        return None

    imports: dict[str, set[str]] = {}
    for e in all_edges:
        if e.get("relation") == "imports_from" and e.get("source") in file_ids and e.get("target") in file_ids:
            imports.setdefault(str(e["source"]), set()).add(str(e["target"]))

    closure_memo: dict[str, set[str]] = {}

    def closure(fid: str) -> set[str]:
        hit = closure_memo.get(fid)
        if hit is not None:
            return hit
        out: set[str] = set()
        stack = list(imports.get(fid, ()))
        while stack:
            f = stack.pop()
            if f in out:
                continue
            out.add(f)
            stack.extend(imports.get(f, ()))
        out.discard(fid)
        closure_memo[fid] = out
        return out

    # definitions: name -> [(nid, file nid, kind)] (public only; private ones were
    # bound in their own file by the extractor)
    defs: dict[str, list[tuple[str, str | None, str]]] = {}
    methods: dict[tuple[str, str], list[tuple[str, str | None]]] = {}
    method_names: dict[str, list[tuple[str, str | None]]] = {}
    ret_type: dict[str, str] = {}
    for n in all_nodes:
        if not is_vz(n):
            continue
        meta = n.get("metadata") or {}
        kind = meta.get("kind")
        label = str(n.get("label", ""))
        if kind == "method":
            name = label.strip(".").removesuffix("()")
            recv = str(meta.get("receiver_type", ""))
            fid = file_of(n["id"])
            methods.setdefault((recv, name), []).append((n["id"], fid))
            if not meta.get("private"):
                method_names.setdefault(name, []).append((n["id"], fid))
            continue
        if kind not in _DEF_KINDS or meta.get("private"):
            continue
        name = label.removesuffix("()")
        fid = None if kind in ("extern_fn", "extern_var") else file_of(n["id"])
        defs.setdefault(name, []).append((n["id"], fid, kind))
        if kind == "fn" and meta.get("return_type"):
            ret_type.setdefault(name, str(meta["return_type"]))

    existing = {(e.get("source"), e.get("target"), e.get("relation")) for e in all_edges}

    def emit(src: str, tgt: str, relation: str, rc: dict, confidence: str, score: float,
             context: str) -> None:
        if not src or not tgt or src == tgt:
            return
        key = (src, tgt, relation)
        if key in existing:
            return
        existing.add(key)
        all_edges.append({
            "source": src, "target": tgt, "relation": relation, "context": context,
            "confidence": confidence, "confidence_score": score,
            "source_file": rc.get("source_file", ""), "source_location": rc.get("source_location"),
            "weight": 1.0,
        })

    def pick(cands: list[tuple[str, str | None]], caller_file: str | None,
             allow_global: bool = True) -> tuple[str, str, float] | None:
        """(nid, confidence, score) of the one candidate the caller's imports reach."""
        if not cands:
            return None
        direct = imports.get(caller_file or "", set())
        reach = closure(caller_file) if caller_file else set()
        own = [c for c in cands if c[1] is not None and c[1] == caller_file]
        if len(own) == 1:
            return own[0][0], "EXTRACTED", 1.0
        d = [c for c in cands if c[1] in direct]
        if len(d) == 1:
            return d[0][0], "EXTRACTED", 1.0
        if len(d) > 1:
            return None
        t = [c for c in cands if c[1] in reach]
        if len(t) == 1:
            return t[0][0], "INFERRED", 0.95
        if len(t) > 1:
            return None
        if allow_global and len(cands) == 1 and cands[0][1] is not None:
            return cands[0][0], "INFERRED", 0.65
        return None

    # structural traits: a type satisfies `trait T { fn m(...) }` when `Type.m`
    # exists for every m (no impl blocks in Vurze), INFERRED (signatures unchecked)
    by_type: dict[str, set[str]] = {}
    for (recv, mname) in methods:
        by_type.setdefault(recv, set()).add(mname)
    for n in all_nodes:
        meta = n.get("metadata") or {}
        if not is_vz(n) or meta.get("kind") != "trait" or not meta.get("trait_methods"):
            continue
        need = set(meta["trait_methods"])
        tfile = file_of(n["id"])
        for tname, have in by_type.items():
            if not need <= have:
                continue
            got = pick([(nid, f) for nid, f, k in defs.get(tname, []) if k in ("struct", "enum")], tfile)
            if got is None:
                cands = [(nid, f) for nid, f, k in defs.get(tname, []) if k in ("struct", "enum")]
                got = (cands[0][0], "INFERRED", 0.65) if len(cands) == 1 else None
            if got:
                emit(got[0], n["id"], "implements", {"source_file": n.get("source_file", ""),
                     "source_location": n.get("source_location")}, "INFERRED", 0.75, "structural_trait")

    # an `extern fn` that a Vurze file exports itself (the runtime's backends:
    # std/gfx.vz declares gfx_time_s, runtime/gfx_vk_core.vz and gfx_sw.vz
    # export it): each exporting definition `implements` the shared extern node
    for n in all_nodes:
        meta = n.get("metadata") or {}
        if not is_vz(n) or meta.get("kind") != "extern_fn":
            continue
        name = str(n.get("label", "")).removesuffix("()")
        for nid, _f, k in defs.get(name, []):
            if k == "fn" and (node_by_id.get(nid, {}).get("metadata") or {}).get("export"):
                emit(nid, n["id"], "implements", {"source_file": node_by_id[nid].get("source_file", ""),
                     "source_location": node_by_id[nid].get("source_location")},
                     "INFERRED", 0.85, "extern_definition")

    for rc in vz_calls:
        caller = rc.get("caller_nid")
        name = str(rc.get("callee", ""))
        if not caller or not name:
            continue
        caller_file = sf_to_file.get(str(rc.get("source_file", ""))) or file_of(str(caller))
        kind = rc.get("vurze_kind")
        if kind == "method_of":
            cands = [(nid, f) for nid, f, k in defs.get(name, []) if k in _TYPE_KINDS]
            got = pick(cands, caller_file)
            if got:
                emit(got[0], str(caller), "method", rc, got[1], got[2], "method")
            continue
        if kind == "method":
            recv = rc.get("receiver_type")
            if recv and str(recv).startswith("="):
                recv = ret_type.get(str(recv)[1:])
            if recv:
                got = pick(methods.get((str(recv), name), []), caller_file)
                if got:
                    emit(str(caller), got[0], "calls", rc, got[1], got[2], "call")
                    continue
                tcands = [(nid, f) for nid, f, k in defs.get(str(recv), []) if k == "enum"]
                got = pick(tcands, caller_file)
                if got:
                    emit(str(caller), got[0], "references", rc, got[1], got[2], "variant")
                continue
            cands = method_names.get(name, [])
            reach = closure(caller_file) if caller_file else set()
            vis = [c for c in cands if c[1] == caller_file or c[1] in reach]
            if len(vis) == 1:
                emit(str(caller), vis[0][0], "calls", rc, "INFERRED", 0.75, "call")
            continue
        cands_all = defs.get(name, [])
        if not cands_all:
            continue
        # a name declared `extern` (here or in an import) is the C symbol unless
        # a Vurze definition of it is reachable through the imports
        has_extern = bool(rc.get("fallback_nid")) or any(k in ("extern_fn", "extern_var") for _n, _f, k in cands_all)
        if kind == "call":
            fns = [(nid, f) for nid, f, k in cands_all if k in ("fn", "derived_fn")]
            got = pick(fns, caller_file, allow_global=not has_extern)
            if got:
                emit(str(caller), got[0], "calls", rc, got[1], got[2], "call")
                continue
            structs = [(nid, f) for nid, f, k in cands_all if k == "struct"]
            got = pick(structs, caller_file)
            if got:
                emit(str(caller), got[0], "references", rc, got[1], got[2], "constructor")
                continue
            fallback = rc.get("fallback_nid") or next(
                (nid for nid, _f, k in cands_all if k == "extern_fn"), None)
            if fallback and fallback in node_by_id:
                emit(str(caller), str(fallback), "calls", rc, "EXTRACTED", 1.0, "extern_call")
                continue
            globs = [(nid, f) for nid, f, k in cands_all if k in ("global", "const")]
            got = pick(globs, caller_file)
            if got:
                emit(str(caller), got[0], "references", rc, got[1], got[2], "value")
            continue
        if kind == "value":
            got = pick([(nid, f) for nid, f, k in cands_all if k in ("global", "const")], caller_file)
            if got:
                emit(str(caller), got[0], "references", rc, got[1], got[2], "value")
                continue
            got = pick([(nid, f) for nid, f, k in cands_all if k in ("fn", "derived_fn")], caller_file)
            if got:
                emit(str(caller), got[0], "indirect_call", rc, "INFERRED", min(got[2], 0.85), "function_value")
                continue
            got = pick([(nid, f) for nid, f, k in cands_all if k in _TYPE_KINDS], caller_file)
            if got:
                emit(str(caller), got[0], "references", rc, got[1], got[2], "type")
                continue
            ext = next((nid for nid, _f, k in cands_all if k == "extern_var"), None)
            if ext:
                emit(str(caller), ext, "references", rc, "EXTRACTED", 1.0, "value")
            continue
        # a type position
        got = pick([(nid, f) for nid, f, k in cands_all if k in _TYPE_KINDS], caller_file)
        if got:
            emit(str(caller), got[0], "references", rc, got[1], got[2], str(kind))
