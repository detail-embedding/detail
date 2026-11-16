"""mkdocs-gen-files — generate the Manifest format reference from code (§7).

Every field table comes from the pydantic models in ``detail.storage.manifest``
(``describe_manifest``), and the per-importer stream inputs come from the
``STREAM_IMPORTERS`` registry — the same models and registry the bulk-import
validator uses, so the reference cannot drift from what the tool accepts. Only the
lead-in prose and the worked example are hand-written here.
"""

import mkdocs_gen_files

from detail.core.registries import STREAM_IMPORTERS
from detail.importers import import_stream_importers
from detail.storage.manifest import MANIFEST_NAMES, describe_manifest

_TITLE = {
    "Manifest": "Manifest",
    "RecordingSpec": "Recording",
    "StreamSpec": "Stream",
}


def _default_cell(field: dict) -> str:
    if field["required"]:
        return "required"
    default = field["default"]
    if default in ({}, []):
        return "—"
    return f"`{default}`"


def _field_table(section: dict) -> list[str]:
    rows = [
        f"## {_TITLE[section['model']]}",
        "",
        section["summary"],
        "",
        "| Field | Type | Default | Description |",
        "| --- | --- | --- | --- |",
    ]
    for field in section["fields"]:
        rows.append(
            f"| `{field['name']}` | {field['type']} | {_default_cell(field)} | {field['help']} |"
        )
    rows.append("")
    return rows


def _importer_table() -> list[str]:
    import_stream_importers()
    rows = [
        "## Stream inputs by importer",
        "",
        "Every stream needs a primary `raw` file. Each importer also declares the "
        "extra `aux` inputs it needs, keyed by name:",
        "",
        "| Importer | Name | Required `aux` inputs |",
        "| --- | --- | --- |",
    ]
    for importer_id, cls in sorted(STREAM_IMPORTERS.items()):
        aux = ", ".join(
            f"`{spec.key}` ({'/'.join(spec.extensions) or 'any'})"
            for spec in cls().aux_inputs
        )
        rows.append(f"| `{importer_id}` | {cls.display_name} | {aux or '—'} |")
    rows.append("")
    return rows


_EXAMPLE = """```yaml
recordings:
  - id: boxA
    video:                     # ordered parts; must share fps + resolution
      - boxA_am.mp4
      - boxA_pm.mp4
    metadata:                  # grouping variables (any keys)
      treatment: drug_A
      sex: M
    streams:
      - name: medpc
        importer: medpc
        raw: PX22.txt
        aux:
          program: Golden_Liq_SelfAdmin.MPC
  - id: boxB                   # a video-only recording
    video: [boxB.mp4]
```"""

_lines = [
    "# Manifest format",
    "",
    "A bundle is a folder with a `" + "` or `".join(MANIFEST_NAMES) + "` plus the "
    "video and stream files it references; paths are relative to the folder. Import "
    "it with `detail recording import --bundle` or the GUI's **Import bundle…** "
    "(see [Many recordings at once](../user-guides/single-animal/create-an-experiment.md#5-optional-many-recordings-at-once)). "
    "The whole manifest is validated before anything is created.",
    "",
]
for _section in describe_manifest():
    _lines += _field_table(_section)
_lines += _importer_table()
_lines += ["## Example", "", _EXAMPLE, ""]

with mkdocs_gen_files.open("reference/manifest.md", "w", encoding="utf-8") as _handle:
    _handle.write("\n".join(_lines) + "\n")
