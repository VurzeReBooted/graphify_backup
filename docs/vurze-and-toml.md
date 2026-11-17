# Vurze and TOML

graphify reads [Vurze](https://github.com/VurzeReBooted/Vurze) (`.vz`) and TOML (`.toml`) in its local AST pass:
no tree-sitter grammar, no LLM, no API key. Both extractors are hand-written and use only the standard library
(`graphify/extractors/vurze.py`, `graphify/extractors/toml_config.py`).

## Vurze (`.vz`)

The lexer follows the compiler's own token rules (`vurzec.py`'s `TOK`, the same as `selfhost/lexer.vz`):
`//` comments, `"..."` strings with backslash escapes, no block comments, no character literals. On top of it a
declaration parser follows `Parser.program`.

**Nodes** (each carries `metadata.language = "vurze"` and `metadata.kind`):

| kind | from | label |
|---|---|---|
| `file` | the file | `chunks.vz` |
| `fn` | `fn`, `const fn`, `det fn`, `export fn`, `unsafe fn`, generic `fn f<T>` | `name()` |
| `method` | `fn Type.name(self: *Type, ...)` | `.name()` (`metadata.receiver_type`) |
| `struct`, `enum`, `trait` | `struct Name<T> { ... }`, `enum`, `trait` | `Name` |
| `enum_variant` | each variant of an enum | `Variant` |
| `global`, `const` | module-level `let` / `var` / `const` | `NAME` |
| `extern_fn`, `extern_var` | `extern fn` / `extern var`: ONE shared node per C symbol | `sqrtf()` |
| `derived_fn` | what `@derive(eq, hash, clone, debug, json)` generates | `Vec3_eq()` |
| `attribute` | one shared node per attribute name | `@vertex`, `@fp_exact(module)` |
| `external_module` | an `import` that is not found on disk | `std/c.vz` |

Function metadata records `private`, `export`, `extern`, `det`, `const`, `unsafe`, `fp_exact` (also when it comes
from `@fp_exact(module)`), `safe`, `attributes`, `shader_stage` (`vertex`, `fragment`, `compute`, `kernel`),
`workgroup_size` (`@compute(64)`) and `return_type`.

**Edges**

- `contains`: file -> its declarations; `method`: type -> method; `case_of`: enum -> variant;
  `defines` (context `derive`): type -> its derived functions.
- `imports_from`: file -> imported file, resolved as the compiler searches: next to the importing file, then each
  folder above it up to the scan root (the build's working directory), then `$VURZE_PATH` (`:`-separated) and
  `$VURZE_HOME` (where `std/` lives).
- `calls`: a call `f(...)`, `name[T](...)` (explicit type arguments), `obj.m(...)`, `Type.m(...)`.
- `references` with a context: `parameter_type`, `return_type`, `field`, `local_type`, `cast` (`as T`),
  `generic_arg` (`alloc[T]`, `sizeof[T]`, `f[T]()`), `type`, `constructor` (`Vec3(1.0, 2.0, 3.0)`), `variant`
  (`Shape.Circle(r)`), `value` (a read of a module-level `let` / `var` / `const`), `attribute`, `extern_decl`.
- `indirect_call` (context `function_value`): a function named as a value: a hook (`hud_hook = my_fn`), a
  callback argument, a dispatch table entry.
- `implements`: a type -> a trait it satisfies structurally (it has every method the trait lists), and an
  `export fn` -> the shared extern node of the symbol it defines (`gfx_time_s` declared in `std/gfx.vz`, exported
  by each graphics backend).

**How names are bound.** Vurze compiles a program as the transitive closure of its imports and has one
program-wide namespace (a `private` name is visible only in its file). So:

1. A name the file declares itself binds in the file (`EXTRACTED`); its own and its `private` definitions win, as
   in the compiler. A call to a name the file only declares `extern` goes to step 2 first.
2. Every other name is bound after all files are read (`resolve_vurze_references`): a unique public definition in
   a directly imported file is `EXTRACTED`; a unique one further down the import chain is `INFERRED` (0.95); a
   name that no import reaches but that is defined exactly once in the scanned corpus (a library file using what
   its program's `main` imports) is `INFERRED` (0.65). Two candidates at the same level bind nothing. A name only
   declared `extern` (and not defined by any reachable Vurze file) lands on the shared extern node.
3. A method call binds through its receiver's type when it is known: `self`, a typed parameter or local, a local
   made by `Type(...)` or by a call whose return type is known (`var v = v3(...)` then `v.unit()`), or one field
   deep (`self.post.exposure()`) when that struct is declared in the same file. Otherwise it binds only when
   exactly one method of that name is visible through the imports (`INFERRED` 0.75).

**Limits**

- Names, not types: overloads do not exist in Vurze, but a call through a struct field of function type
  (`op.f(3)`) or a local function pointer is not followed, and a method on an untyped receiver chain longer than
  one field is bound only by the unique-name rule.
- Generic instantiation is not modelled (`Box.get<T>` is one node, not one per `T`).
- `include_str("file")`, `hot_sym("name")` and `asm` strings are not followed.
- graphify's graph is undirected and keeps one edge per node pair: mutual recursion (`a` calls `b`, `b` calls `a`)
  shows as one `calls` edge.
- Imports of another repository resolve only when it is reachable: scan the folder that holds all the checkouts,
  or set `VURZE_PATH` / `VURZE_HOME` (see below). A per-repository graph keeps such imports as stub nodes.

## TOML (`.toml`)

- `file`; `table` per `[a.b]` (label `[a.b]`; the implicit `[a]` too, and the tables a dotted key `a.b = 1`
  creates); `array_table` per `[[name]]` and `array_entry` per entry (labelled `[[name]] <id>` when the entry has an
  `id`, `name`, `key` or `title` string, else `[[name]] #3`); `key` per key, labelled by its full dotted path, with
  `metadata.value` (a preview) and `metadata.value_kind`. All joined by `contains`.
- Multi-line strings, multi-line arrays, inline tables, quoted and dotted keys are read correctly; a line that is
  not TOML is skipped and the rest of the file is still read.
- **Mod layering** (`overrides`, context `mod_merge`): `<root>/mods/<mod>/<rel>.toml` merges over
  `<root>/content/<layer>/<rel>.toml` (the nearest layer that has the file: `content/game` before `content/base`),
  and `content/<layer>/<rel>.toml` over `content/base/<rel>.toml`. The file gets an edge to the file it overrides,
  and each of its tables and keys an edge to the same table or key of the nearest layer that has it.
  `mods/<mod>/mod.toml` (the mod's own manifest) overrides nothing.
- Caps: 500 key nodes per file (`GRAPHIFY_TOML_MAX_KEYS=N`, `0` for tables only), 20000 tables per file, files up
  to 4 MiB. `pyproject.toml` and `Cargo.toml` keep their package-manifest extractor; files with sensitive names
  (`secrets.toml`, `token.toml`, ...) are skipped as before.

## Building a graph of Vurze projects

```
graphify extract path/to/project --code-only
graphify cluster-only path/to/project --no-label
```

`--code-only` keeps the run local (no LLM); `--no-label` names communities by their hub nodes instead of asking a
model. The graph is in `path/to/project/graphify-out/` (`--out DIR` writes it to `DIR/graphify-out/` instead).
For a game, its engine and Vurze in one graph, scan the folder that holds the three checkouts and tell the
resolver where the engine and the standard library are, exactly as the build does:

```
VURZE_PATH=path/to/RealEngine1 VURZE_HOME=path/to/Vurze graphify extract path/to/parent --code-only --exclude other_folder
```
