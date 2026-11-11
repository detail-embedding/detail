"""Named starting points shipped with the tool — "Mouse open field", "Side view".

A baseline is a **file in this package**, not a row a user creates. Every scientific tool that
shipped a user-managed preset library watched it decay: under 2% of Office 2003 sessions had
customised command bars across ~100M users, and 7 of 20 participants in McGrenere's study
stopped personalising within six weeks *inside a tool built to make it easy*. What does work is
a list the user never maintains — SLEAP's dropdown of shipped configurations plus previous runs.
So these are authored here, reviewed, and versioned with the code.

**A baseline carries settings and nothing else**, and that is what makes it portable at all.
The `scope` declaration (`configs/describe.py`) separates a *setting* — a number, a choice, a
threshold — from a *selection*, which names an artifact inside one experiment. A preset system
dies on the second: ilastik's `.ilp`, TrackMate's `ImagePlus`, a SLEAP config that cannot find
its `models/` folder on another machine. `load_baselines` refuses a file that names one, so the
failure mode is a build error rather than a run bound to somebody else's data.

Adding one is a YAML file beside this module::

    schema_version: 1
    name: Mouse open field
    settings:
      detection: {conf_threshold: 0.25}
      tracking:  {max_missed_frames: 30}

Keyed by ``module_id``, so one baseline spans a pipeline and each launch form offers it only
for the modules it actually names.
"""

from dataclasses import dataclass
from importlib.resources import files

import yaml
from pydantic import BaseModel, ValidationError

BASELINES_PACKAGE = "detail.configs"
BASELINES_DIR = "baselines"


class BaselineError(ValueError):
    """A shipped baseline that cannot be trusted to configure a run."""


@dataclass(frozen=True)
class Baseline:
    """One named starting point: a display name, and settings per module."""

    name: str
    settings: dict[str, dict]

    def for_module(self, module_id: str) -> dict | None:
        """This baseline's settings for one module, or ``None`` if it says nothing about it."""
        values = self.settings.get(module_id)
        return dict(values) if values else None


def _config_models() -> dict[str, type[BaseModel]]:
    from detail.configs.describe import iter_config_models

    return dict(iter_config_models())


def _check(baseline_name: str, module_id: str, values: dict) -> None:
    """Every rule a baseline has to satisfy, with the file and field named in the message.

    Raised at load time rather than at launch: a baseline is our own data, so a wrong one is a
    build error, and finding it in a dropdown at the moment somebody wanted to start a run is
    the wrong moment.
    """
    from detail.configs.describe import describe_config

    model_cls = _config_models().get(module_id)
    if model_cls is None:
        raise BaselineError(
            f"baseline {baseline_name!r} configures {module_id!r}, which is not a module that "
            "declares a config model"
        )
    fields = {f["name"]: f for f in describe_config(model_cls)}
    for key in values:
        field = fields.get(key)
        if field is None:
            raise BaselineError(
                f"baseline {baseline_name!r}: {module_id}.{key} is not a parameter of "
                f"{model_cls.__name__}"
            )
        if field["scope"] != "setting":
            raise BaselineError(
                f"baseline {baseline_name!r}: {module_id}.{key} is a *selection* — it names an "
                "artifact inside one experiment, so a baseline carrying it would bind to "
                "whatever happens to share that id elsewhere. Baselines hold settings only."
            )
    try:
        model_cls.model_validate({**_required_stubs(fields), **values})
    except ValidationError as exc:
        raise BaselineError(
            f"baseline {baseline_name!r}: {module_id} values are outside what "
            f"{model_cls.__name__} allows — {exc.errors()[0].get('msg', exc)}"
        ) from exc


def _required_stubs(fields: dict) -> dict:
    """Placeholders for the model's required fields, so a partial baseline can be validated.

    A baseline names a handful of knobs; the model still demands its required ones (an
    ``embed_run_id``, a recording list). Those are selections a baseline must not carry, so
    they are stubbed here purely to get the *bounds* of what it does carry checked.
    """
    stubs: dict = {}
    for name, field in fields.items():
        if not field["required"]:
            continue
        stubs[name] = {
            "array": [],
            "string": "",
            "integer": 0,
            "number": 0.0,
            "boolean": False,
        }.get(field["type"], None)
    return stubs


def load_baselines() -> list[Baseline]:
    """Every shipped baseline, sorted by name.

    Read through ``importlib.resources`` rather than a filesystem path, so it works from an
    installed wheel — the idiom ``gui/__main__.py`` already uses for the app icon.
    """
    directory = files(BASELINES_PACKAGE) / BASELINES_DIR
    if not directory.is_dir():
        return []
    baselines = []
    for entry in sorted(directory.iterdir(), key=lambda p: p.name):
        if not entry.name.endswith((".yaml", ".yml")):
            continue
        data = yaml.safe_load(entry.read_text(encoding="utf-8-sig")) or {}
        name = str(data.get("name") or entry.name)
        settings = data.get("settings") or {}
        if not isinstance(settings, dict):
            raise BaselineError(f"baseline {name!r}: `settings` must be a mapping")
        for module_id, values in settings.items():
            if not isinstance(values, dict):
                raise BaselineError(
                    f"baseline {name!r}: settings for {module_id!r} must be a mapping"
                )
            _check(name, module_id, values)
        baselines.append(
            Baseline(name=name, settings={k: dict(v) for k, v in settings.items()})
        )
    return sorted(baselines, key=lambda b: b.name)


def baselines_for(module_id: str) -> list[Baseline]:
    """The shipped baselines that say something about ``module_id``."""
    return [b for b in load_baselines() if b.for_module(module_id)]
