"""The ``detail skeleton`` command group (§26 *The Keypoints manager*): the machine-wide
skeleton library, the CLI twin of the GUI's Keypoints manager.

Machine-wide, so no ``--experiment``. ``edit`` (of the points) and ``delete`` are refused while
anything in a known experiment uses the skeleton, with what uses it listed; ``save-as-new``
is the way through, as on the screen. Like ``cli_keypoints.py`` it imports nothing from
``detail.cli`` at the top: ``cli.py`` imports it.
"""

from pathlib import Path

import click

_RECOVERABLE = (ValueError, FileNotFoundError, FileExistsError, KeyError)


def _fail(exc: BaseException) -> click.ClickException:
    """A clean message: ``str(KeyError("x"))`` is ``"'x'"``, so take its argument. Any other
    error's first argument is not its message (a ``UnicodeDecodeError``'s is the codec).
    """
    if isinstance(exc, KeyError) and exc.args:
        return click.ClickException(str(exc.args[0]))
    return click.ClickException(str(exc))


def _summary(entry: dict) -> str:
    sk = entry["skeleton"]
    kind = "preset" if entry["builtin"] else "library"
    return (
        f"{entry['id']}  {entry['name']}  {len(sk)} points, {len(sk.edges)} edges, "
        f"{len(sk.pairs)} pairs  ({kind})"
    )


@click.group("skeleton")
def skeleton() -> None:
    """Skeletons (§26): the points a keypoint model finds, the edges between them and the
    left/right pairs. One library per machine, shared by every experiment; the four presets
    are read-only. Whatever uses a skeleton keeps its own copy."""


@skeleton.command("list")
def skeleton_list() -> None:
    """List the presets and the library's skeletons."""
    from detail.api import skeletons as library

    for entry in library.skeletons():
        click.echo(_summary(entry))


@skeleton.command("show")
@click.argument("skeleton_id", type=str)
def skeleton_show(skeleton_id: str) -> None:
    """Print one skeleton's points, edges and pairs, and what uses it."""
    from detail.api import skeletons as library

    try:
        entry = library.skeleton(skeleton_id)
    except _RECOVERABLE as e:
        raise _fail(e)
    sk = entry["skeleton"]
    click.echo(_summary(entry))
    click.echo(f"points: {', '.join(sk.names) or '(none)'}")
    click.echo(f"edges:  {', '.join(f'{a}-{b}' for a, b in sk.edges) or '(none)'}")
    click.echo(f"pairs:  {', '.join(f'{a}/{b}' for a, b in sk.pairs) or '(none)'}")
    found = library.dependents(skeleton_id)
    if not found:
        click.echo("used by: nothing")
    for dep in found:
        click.echo(f"used by: {dep['kind']} {dep['record']} ({dep['experiment']})")


@skeleton.command("create")
@click.argument("name", type=str)
@click.option(
    "--from",
    "source",
    required=True,
    type=str,
    help="A preset or library id to copy, or a skeleton YAML file.",
)
def skeleton_create(name: str, source: str) -> None:
    """Add a skeleton to the library."""
    from detail.api import skeletons as library

    try:
        made = library.create_skeleton(name, source)
    except _RECOVERABLE as e:
        raise _fail(e)
    click.echo(f"created skeleton: {made['id']}")


@skeleton.command("edit")
@click.argument("skeleton_id", type=str)
@click.option("--name", default=None, type=str, help="Rename it (always allowed).")
@click.option(
    "--skeleton",
    "skeleton_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Replace its points, edges, pairs and layout with this YAML file's; refused while "
    "anything uses it (use save-as-new).",
)
def skeleton_edit(
    skeleton_id: str, name: str | None, skeleton_file: Path | None
) -> None:
    """Rename a library skeleton or replace its points."""
    from detail.api import skeletons as library

    if name is None and skeleton_file is None:
        raise click.ClickException("nothing to change: give --name or --skeleton")
    try:
        library.update_skeleton(skeleton_id, name=name, skeleton=skeleton_file)
    except _RECOVERABLE as e:
        raise _fail(e)
    click.echo(f"updated skeleton: {skeleton_id}")


@skeleton.command("save-as-new")
@click.argument("skeleton_id", type=str)
@click.argument("name", type=str)
@click.option(
    "--skeleton",
    "skeleton_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The edited skeleton (YAML).",
)
def skeleton_save_as_new(skeleton_id: str, name: str, skeleton_file: Path) -> None:
    """Keep an edit of SKELETON_ID as a new skeleton NAME (how a used one changes)."""
    from detail.api import skeletons as library

    try:
        made = library.save_as_new(skeleton_id, name, skeleton_file)
    except _RECOVERABLE as e:
        raise _fail(e)
    click.echo(f"created skeleton: {made['id']}")


@skeleton.command("duplicate")
@click.argument("skeleton_id", type=str)
@click.option("--name", default=None, type=str, help="The copy's name.")
def skeleton_duplicate(skeleton_id: str, name: str | None) -> None:
    """Copy a preset or library skeleton into a new library skeleton."""
    from detail.api import skeletons as library

    try:
        made = library.duplicate_skeleton(skeleton_id, name=name)
    except _RECOVERABLE as e:
        raise _fail(e)
    click.echo(f"created skeleton: {made['id']}")


@skeleton.command("export")
@click.argument("skeleton_id", type=str)
@click.argument("out_path", type=click.Path(dir_okay=False, path_type=Path))
def skeleton_export(skeleton_id: str, out_path: Path) -> None:
    """Write a skeleton to a YAML file another machine can import."""
    from detail.api import skeletons as library

    try:
        written = library.export_skeleton(skeleton_id, out_path)
    except _RECOVERABLE as e:
        raise _fail(e)
    click.echo(f"exported skeleton: {written}")


@skeleton.command("import")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--name", default=None, type=str, help="Its name (default: the file's).")
def skeleton_import(path: Path, name: str | None) -> None:
    """Add a skeleton file to the library."""
    from detail.api import skeletons as library

    try:
        made = library.import_skeleton(path, name=name)
    except _RECOVERABLE as e:
        raise _fail(e)
    click.echo(f"imported skeleton: {made['id']}")


@skeleton.command("delete")
@click.argument("skeleton_id", type=str)
def skeleton_delete(skeleton_id: str) -> None:
    """Delete a library skeleton nothing uses."""
    from detail.api import skeletons as library

    try:
        library.delete_skeleton(skeleton_id)
    except _RECOVERABLE as e:
        raise _fail(e)
    click.echo(f"deleted skeleton: {skeleton_id}")
