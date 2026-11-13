"""The two directions of a class name (§1 *Class names*): display through ``label``, input
through ``resolve`` / ``resolve_config``, and the refusals that keep input unambiguous.

**The class this guards:** a config field that names an existing class but is missing from
``CLASS_REF_FIELDS``. Nothing fails at the line where such a field is added: a user who types
a class's name there gets "no class named …" while every other field accepts it, and a run
recorded from the GUI holds the stored name while one from a script holds whatever was typed.
So the last test is driven by the config models, not by a list of the fields somebody thought
of: every field whose name says *class* is in the table, or is excused here with its reason.
"""

import importlib
import inspect
import pkgutil

import pydantic
import pytest

from detail import configs
from detail.core import class_names

NAMES = {"Intruder": "Stranger"}
KNOWN = {"Intruder", "Resident"}


def test_label_shows_the_name_and_falls_back_to_the_stored_name():
    assert class_names.label("Intruder", NAMES) == "Stranger"
    assert class_names.label("Resident", NAMES) == "Resident"


def test_a_merged_runs_classes_are_each_shown_by_their_name():
    assert class_names.label("Intruder+Resident", NAMES) == "Stranger+Resident"
    assert class_names.label("a+b", {"a+b": "pair"}) == "pair"  # a stored name with a +


def test_resolve_takes_either_name_and_leaves_an_unknown_one_for_the_consumer():
    assert class_names.resolve("Stranger", NAMES, known=KNOWN) == "Intruder"
    assert class_names.resolve("Intruder", NAMES, known=KNOWN) == "Intruder"
    assert class_names.resolve("Resident", NAMES, known=KNOWN) == "Resident"
    assert class_names.resolve("rat", NAMES, known=KNOWN) == "rat"


def test_a_stored_name_wins_over_a_name_that_matches_it():
    # A class detected after the rename may carry, as its stored name, another's name.
    names = {"Intruder": "Visitor"}
    assert class_names.resolve("Visitor", names, known={"Intruder", "Visitor"}) == (
        "Visitor"
    )


def test_resolve_config_resolves_each_shape_and_only_the_reference_fields():
    config = {
        "class_name": "Stranger",
        "class_names": ["Stranger", "Resident"],
        "class_example_sets": {"Stranger": "ex_02", "Resident": "ex_01"},
        "class_conf_thresholds": {"Stranger": 0.4},
        "prompt": {"targets": [{"class_name": "Stranger"}]},
        "tracker_id": "Stranger",
    }
    out = class_names.resolve_config(config, NAMES, known=KNOWN)
    assert out["class_name"] == "Intruder"
    assert out["class_names"] == ["Intruder", "Resident"]
    assert out["class_example_sets"] == {"Intruder": "ex_02", "Resident": "ex_01"}
    assert out["class_conf_thresholds"] == {"Intruder": 0.4}
    # A prompt target defines a stored name; it is not a reference to one.
    assert out["prompt"] == config["prompt"]
    assert out["tracker_id"] == "Stranger"
    assert config["class_name"] == "Stranger"  # a copy, never the caller's dict


def test_resolve_config_leaves_absent_and_none_fields_alone():
    out = class_names.resolve_config({"class_name": None}, NAMES, known=KNOWN)
    assert out == {"class_name": None}


@pytest.mark.parametrize(
    ("names", "fragment"),
    [
        ({"Intruder": "  "}, "'Intruder' needs a name"),
        ({"Intruder": "Resident"}, "'Resident' is already a class"),
        (
            {"Intruder": "Mouse", "Resident": "Mouse"},
            "would both be called 'Mouse'",
        ),
    ],
)
def test_check_refuses_a_map_that_would_make_input_ambiguous(names, fragment):
    with pytest.raises(ValueError, match=fragment):
        class_names.check(names, known=KNOWN)


def test_check_accepts_two_classes_trading_names_through_a_third():
    class_names.check({"Intruder": "A", "Resident": "B"}, known=KNOWN)


def test_renamed_removes_an_entry_set_to_nothing_or_to_the_stored_name():
    assert class_names.renamed(NAMES, "Intruder", None) == {}
    assert class_names.renamed(NAMES, "Intruder", " Intruder ") == {}
    assert class_names.renamed({}, "Resident", " Home ") == {"Resident": "Home"}


#: Config fields that say *class* and are not a reference to an existing class, with why.
NOT_A_REFERENCE: dict[str, str] = {}


def _config_models():
    for info in pkgutil.iter_modules(configs.__path__):
        module = importlib.import_module(f"detail.configs.{info.name}")
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, pydantic.BaseModel)
                and obj.__module__ == module.__name__
            ):
                yield obj


def test_every_config_field_that_names_a_class_is_resolved_or_excused():
    fields = {
        f"{model.__name__}.{field}": field
        for model in _config_models()
        for field in model.model_fields
        if "class" in field
    }
    assert fields, "the scan found no class field at all, so it is scanning nothing"
    missing = sorted(
        where
        for where, field in fields.items()
        if field not in class_names.CLASS_REF_FIELDS and field not in NOT_A_REFERENCE
    )
    assert not missing, (
        f"{missing} name a class but are not in class_names.CLASS_REF_FIELDS, so a "
        "class's name typed there is not resolved to its stored name (§1 *Class names*). "
        "Add the field and its shape, or excuse it in NOT_A_REFERENCE with the reason."
    )
    unused = sorted(set(NOT_A_REFERENCE) - set(fields.values()))
    assert not unused, f"{unused} are excused but no config model has them any more"
