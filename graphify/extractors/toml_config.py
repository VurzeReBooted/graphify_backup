"""TOML structural extractor for ``.toml`` files.

Hand-written and stdlib-only (``tomllib`` keeps no line numbers, so it cannot
place nodes), in the shape of ``extract_yaml``: a file node, one node per table
(``[a.b]``, label ``[a.b]``), one per entry of an array of tables
(``[[block]]``, labelled with the entry's ``id`` / ``name`` / ``key`` when it
has one), and keys as nodes under their table (label the dotted path, value
preview in metadata), all linked by ``contains``. Implicit parent tables
(``[a]`` of ``[a.b]``, the tables of a dotted key ``a.b = 1``) get nodes too,
so the tree has no holes.

Mod layering (Buddy, Run! / Voxhell, and any project laid out the same way):
a file ``<root>/mods/<mod>/<rel>.toml`` merges over
``<root>/content/<layer>/<rel>.toml``, and ``content/<layer>/<rel>.toml`` merges
over ``content/base/<rel>.toml``. Each such pair gets an ``overrides`` edge
(context ``mod_merge``) from the overriding file to the file it overrides, and
from every table / key of the overriding file to the same table / key of the
overridden one.

Package manifests (``pyproject.toml``, ``Cargo.toml``) never reach this
extractor: they are routed by name to the manifest extractor first.
Package-manifest routing, the sensitive-file filter (``secrets.toml``...) and
the ignore files run before suffix dispatch, as for every other format.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from graphify.extractors.base import _file_stem, _make_id

_TOML_MAX_BYTES = 4 * 1_048_576
_MAX_TABLES = 20000
_DEFAULT_MAX_KEYS = 500
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
_NAME_KEYS = ("id", "name", "key", "title")
_INDEX = re.compile(r"\[\d+\]")


def _max_keys() -> int:
    raw = os.environ.get("GRAPHIFY_TOML_MAX_KEYS", "")
    try:
        return max(0, int(raw)) if raw else _DEFAULT_MAX_KEYS
    except ValueError:
        return _DEFAULT_MAX_KEYS


class _Scanner:
    """A small TOML reader that keeps lines; tolerant (an unreadable line is
    skipped, the rest of the file is still read)."""

    def __init__(self, text: str):
        self.s = text
        self.n = len(text)
        self.i = 0
        self.line = 1

    def _adv(self, k: int = 1) -> None:
        end = min(self.n, self.i + k)
        self.line += self.s.count("\n", self.i, end)
        self.i = end

    def skip_ws(self) -> None:
        while self.i < self.n and self.s[self.i] in " \t":
            self.i += 1

    def skip_to_eol(self) -> None:
        j = self.s.find("\n", self.i)
        self.i = self.n if j < 0 else j

    def skip_blank(self) -> None:
        """Spaces, newlines and comments."""
        while self.i < self.n:
            c = self.s[self.i]
            if c in " \t\r":
                self.i += 1
            elif c == "\n":
                self.line += 1
                self.i += 1
            elif c == "#":
                self.skip_to_eol()
            else:
                return

    def string(self) -> str | None:
        """A string at i (basic, literal, multi-line); returns its text."""
        s = self.s
        if s.startswith('"""', self.i) or s.startswith("'''", self.i):
            q = s[self.i:self.i + 3]
            start = self.i + 3
            j = start
            while True:
                j = s.find(q, j)
                if j < 0:
                    self._adv(self.n - self.i)
                    return s[start:]
                if q == '"""':
                    # an escaped quote does not close; count the backslashes before it
                    bs = 0
                    k = j - 1
                    while k >= start and s[k] == "\\":
                        bs += 1
                        k -= 1
                    if bs % 2 == 1:
                        j += 1
                        continue
                # `""""` / `"""""`: up to two extra quotes belong to the content
                extra = 0
                while extra < 2 and j + 3 < self.n and s[j + 3] == q[0]:
                    j += 1
                    extra += 1
                text = s[start:j]
                self._adv(j + 3 - self.i)
                return text
        c = s[self.i] if self.i < self.n else ""
        if c == '"':
            j = self.i + 1
            out = []
            while j < self.n and s[j] not in '"\n':
                if s[j] == "\\" and j + 1 < self.n:
                    out.append(s[j:j + 2])
                    j += 2
                    continue
                out.append(s[j])
                j += 1
            self.i = j + 1 if j < self.n and s[j] == '"' else j
            raw = "".join(out)
            try:
                return raw.encode("utf-8").decode("unicode_escape") if "\\" in raw else raw
            except UnicodeDecodeError:
                return raw
        if c == "'":
            j = s.find("'", self.i + 1)
            nl = s.find("\n", self.i + 1)
            if j < 0 or (0 <= nl < j):
                j = nl if nl >= 0 else self.n
                text = s[self.i + 1:j]
                self.i = j
                return text
            text = s[self.i + 1:j]
            self.i = j + 1
            return text
        return None

    def key(self) -> list[str] | None:
        """A (dotted) key at i: bare, "basic" or 'literal' parts joined by '.'."""
        parts: list[str] = []
        while True:
            self.skip_ws()
            if self.i >= self.n:
                return parts or None
            c = self.s[self.i]
            if c in "\"'":
                part = self.string()
                if part is None:
                    return None
                parts.append(part)
            else:
                m = _BARE_KEY.match(self.s, self.i)
                if not m:
                    return parts or None
                parts.append(m.group())
                self.i = m.end()
            self.skip_ws()
            if self.i < self.n and self.s[self.i] == ".":
                self.i += 1
                continue
            return parts

    def value(self) -> tuple[str, str]:
        """Skip one value; returns (kind, preview)."""
        self.skip_ws()
        start = self.i
        if self.i >= self.n:
            return "empty", ""
        c = self.s[self.i]
        if c in "\"'":
            text = self.string() or ""
            return "string", text
        if c in "[{":
            close = "]" if c == "[" else "}"
            depth = 0
            while self.i < self.n:
                ch = self.s[self.i]
                if ch in "\"'":
                    self.string()
                    continue
                if ch == "#" and c == "[":
                    self.skip_to_eol()
                    continue
                if ch == "\n":
                    self.line += 1
                    self.i += 1
                    if c == "{":  # an inline table ends at its line
                        break
                    continue
                if ch in "[{":
                    depth += 1
                elif ch in "]}":
                    depth -= 1
                    if depth == 0:
                        self.i += 1
                        break
                self.i += 1
            preview = self.s[start:self.i]
            return ("array" if c == "[" else "inline_table"), " ".join(preview.split())
        j = self.i
        while j < self.n and self.s[j] not in "\n#":
            j += 1
        text = self.s[self.i:j].strip()
        self.i = j
        return "scalar", text


def _norm_path(p: Path) -> Path:
    return Path(os.path.normpath(p))


def toml_layer_targets(path: Path) -> list[Path]:
    """The files ``path`` merges over, nearest layer first.

    ``.../mods/<mod>/<rel>`` -> ``.../content/<layer>/<rel>`` for every layer
    that has the file (``game`` before ``base``); ``.../content/<layer>/<rel>``
    (layer other than ``base``) -> ``.../content/base/<rel>``.
    """
    parts = path.parts
    out: list[Path] = []
    for idx in range(len(parts) - 2, -1, -1):
        name = parts[idx]
        if name == "mods" and idx + 2 < len(parts):
            root = Path(*parts[:idx]) if idx > 0 else Path(".")
            rel = Path(*parts[idx + 2:])
            if rel.name == "mod.toml" and len(rel.parts) == 1:
                return []
            content = root / "content"
            try:
                layers = sorted((d.name for d in content.iterdir() if d.is_dir()),
                                key=lambda nm: (nm == "base", nm))
            except OSError:
                layers = []
            for layer in layers:
                cand = content / layer / rel
                if cand.is_file():
                    out.append(_norm_path(cand))
            return out
        if name == "content" and idx + 2 < len(parts):
            layer = parts[idx + 1]
            if layer == "base":
                return []
            root = Path(*parts[:idx + 1])
            rel = Path(*parts[idx + 2:])
            cand = root / "base" / rel
            if cand.is_file():
                out.append(_norm_path(cand))
            return out
    return out


def _parse(text: str, max_keys: int) -> tuple[list[dict], str | None]:
    """The file's tables / entries / keys as records in source order.

    Each record: kind (table, array_table, array_entry, key), path (resolved,
    array entries as ``name[i]``), label, line, parent path, value preview.
    """
    sc = _Scanner(text)
    records: list[dict] = []
    known: dict[tuple[str, ...], dict] = {}
    aot_count: dict[tuple[str, ...], int] = {}
    error: str | None = None
    keys_seen = 0
    current: tuple[str, ...] = ()

    def resolve(parts: list[str]) -> tuple[str, ...]:
        """Map a written path onto the latest array-of-tables entries."""
        out: list[str] = []
        for p in parts:
            out.append(p)
            k = tuple(out)
            if k in aot_count:
                out[-1] = f"{p}[{aot_count[k] - 1}]"
        return tuple(out)

    def ensure_table(path: tuple[str, ...], line: int, kind: str = "table") -> None:
        for d in range(1, len(path)):
            sub = path[:d]
            if sub not in known:
                rec = {"kind": "table", "path": sub, "line": line, "implicit": True}
                known[sub] = rec
                records.append(rec)
        if path and path not in known:
            rec = {"kind": kind, "path": path, "line": line}
            known[path] = rec
            records.append(rec)

    while True:
        sc.skip_blank()
        if sc.i >= sc.n:
            break
        line = sc.line
        c = sc.s[sc.i]
        if c == "[":
            aot = sc.s.startswith("[[", sc.i)
            sc.i += 2 if aot else 1
            parts = sc.key()
            sc.skip_ws()
            if parts is None:
                error = error or f"bad table header at line {line}"
                sc.skip_to_eol()
                continue
            closer = "]]" if aot else "]"
            if sc.s.startswith(closer, sc.i):
                sc.i += len(closer)
            if len(known) >= _MAX_TABLES:
                sc.skip_to_eol()
                current = ()
                continue
            if aot:
                written = tuple(parts)
                parent = resolve(parts[:-1])
                base = parent + (parts[-1],)
                if base not in known:
                    ensure_table(base, line, "array_table")
                elif known[base]["kind"] != "array_table":
                    known[base]["kind"] = "array_table"
                idx = aot_count.get(written, 0)
                aot_count[written] = idx + 1
                entry = parent + (f"{parts[-1]}[{idx}]",)
                rec = {"kind": "array_entry", "path": entry, "line": line, "array": base}
                known[entry] = rec
                records.append(rec)
                current = entry
            else:
                path = resolve(parts)
                ensure_table(path, line)
                if known[path].get("implicit"):
                    known[path]["implicit"] = False
                    known[path]["line"] = line
                current = path
            sc.skip_ws()
            if sc.i < sc.n and sc.s[sc.i] == "#":
                sc.skip_to_eol()
            continue
        parts = sc.key()
        sc.skip_ws()
        if not parts or sc.i >= sc.n or sc.s[sc.i] != "=":
            error = error or f"unreadable line {line}"
            sc.skip_to_eol()
            continue
        sc.i += 1
        vkind, preview = sc.value()
        sc.skip_ws()
        if sc.i < sc.n and sc.s[sc.i] == "#":
            sc.skip_to_eol()
        holder = current + tuple(parts[:-1])
        if len(parts) > 1 and len(known) < _MAX_TABLES:
            ensure_table(holder, line)
        owner = known.get(current)
        if owner is not None and owner["kind"] == "array_entry" and len(parts) == 1 \
                and parts[0] in _NAME_KEYS and vkind == "string" and "name_hint" not in owner:
            owner["name_hint"] = preview
        if keys_seen >= max_keys:
            continue
        keys_seen += 1
        records.append({"kind": "key", "path": holder + (parts[-1],), "line": line,
                        "parent": holder, "value_kind": vkind, "value": preview[:120]})
    return records, error


def _ids(stem: str, rec_kind: str, path: tuple[str, ...]) -> str:
    return _make_id(stem, "key" if rec_kind == "key" else "tbl", *path)


def _dotted(path: tuple[str, ...]) -> str:
    return ".".join(path)


def extract_toml(path: Path) -> dict:
    """Extract tables, array-of-tables entries and keys from a ``.toml`` file,
    plus ``overrides`` edges for mod / content layering."""
    try:
        with path.open("rb") as fh:
            raw = fh.read(_TOML_MAX_BYTES + 1)
    except OSError as exc:
        return {"nodes": [], "edges": [], "error": str(exc)}
    if len(raw) > _TOML_MAX_BYTES:
        return {"nodes": [], "edges": [], "error": "toml file too large to index"}
    # line endings must not change the graph (CRLF checkouts on Windows)
    text = raw.decode("utf-8", errors="replace").lstrip("\ufeff").replace("\r\n", "\n")
    max_keys = _max_keys()
    records, error = _parse(text, max_keys)

    stem = _file_stem(path)
    str_path = str(path)
    file_nid = _make_id(str_path)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()

    def add_node(nid: str, label: str, line: int, kind: str, extra: dict | None = None) -> None:
        if not nid or nid in seen:
            return
        seen.add(nid)
        meta = {"language": "toml", "kind": kind}
        if extra:
            meta.update(extra)
        nodes.append({"id": nid, "label": label, "file_type": "code", "source_file": str_path,
                      "source_location": f"L{line}", "metadata": meta})

    def add_edge(src: str, tgt: str, relation: str, line: int, *, context: str | None = None,
                 target_file: str | None = None) -> None:
        if not src or not tgt or src == tgt or (src, tgt, relation) in seen_edges:
            return
        seen_edges.add((src, tgt, relation))
        edge = {"source": src, "target": tgt, "relation": relation, "confidence": "EXTRACTED",
                "confidence_score": 1.0, "source_file": str_path, "source_location": f"L{line}",
                "weight": 1.0}
        if context:
            edge["context"] = context
        if target_file:
            edge["target_file"] = target_file
        edges.append(edge)

    add_node(file_nid, path.name, 1, "file")
    table_ids: dict[tuple[str, ...], str] = {}
    for rec in records:
        p = rec["path"]
        if rec["kind"] == "key":
            continue
        nid = _ids(stem, rec["kind"], p)
        if rec["kind"] == "array_entry":
            hint = rec.get("name_hint")
            written = _INDEX.sub("", _dotted(rec["array"]))
            label = f"[[{written}]] {hint}" if hint else \
                f"[[{written}]] #{p[-1].rsplit('[', 1)[-1].rstrip(']')}"
        elif rec["kind"] == "array_table":
            label = f"[[{_INDEX.sub('', _dotted(p))}]]"
        else:
            label = f"[{_INDEX.sub('', _dotted(p))}]"
        add_node(nid, label, rec["line"], rec["kind"],
                 {"path": _dotted(p), **({"implicit": True} if rec.get("implicit") else {})})
        table_ids[p] = nid
    for rec in records:
        if rec["kind"] == "key":
            continue
        p = rec["path"]
        parent = table_ids.get(rec.get("array") or p[:-1]) if rec["kind"] == "array_entry" \
            else table_ids.get(p[:-1])
        add_edge(parent or file_nid, table_ids[p], "contains", rec["line"])
    key_ids: dict[tuple[str, ...], str] = {}
    for rec in records:
        if rec["kind"] != "key":
            continue
        p = rec["path"]
        nid = _ids(stem, "key", p)
        add_node(nid, _dotted(p), rec["line"], "key",
                 {"value_kind": rec["value_kind"], "value": rec["value"]})
        key_ids[p] = nid
        add_edge(table_ids.get(rec["parent"], file_nid), nid, "contains", rec["line"])

    # mod / content layering: the file merges over the nearest layer that has
    # it; each table / key over the nearest layer that has that table / key
    matched: set[tuple[bool, tuple[str, ...]]] = set()
    for depth, target in enumerate(toml_layer_targets(path)):
        if depth == 0:
            add_edge(file_nid, _make_id(str(target)), "overrides", 1, context="mod_merge",
                     target_file=str(target))
        try:
            with target.open("rb") as fh:
                traw = fh.read(_TOML_MAX_BYTES + 1)
        except OSError:
            continue
        if len(traw) > _TOML_MAX_BYTES:
            continue
        ttext = traw.decode("utf-8", errors="replace").lstrip("\ufeff").replace("\r\n", "\n")
        trecords, _terr = _parse(ttext, max_keys)
        tstem = _file_stem(target)
        tpaths = {(r["kind"] == "key", r["path"]): r for r in trecords}
        for is_key, ids in ((False, table_ids), (True, key_ids)):
            for p, nid in ids.items():
                hit = tpaths.get((is_key, p))
                if hit is None or (is_key, p) in matched:
                    continue
                matched.add((is_key, p))
                add_edge(nid, _ids(tstem, hit["kind"], p), "overrides", hit["line"],
                         context="mod_merge", target_file=str(target))

    del error  # a partly unreadable file still yields what could be read
    return {"nodes": nodes, "edges": edges}
