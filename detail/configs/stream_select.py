"""Which part of a multi-session file to import (Section 3): MED-PC sessions, BORIS observations
and subjects.

A MED-PC data file can hold every box run that day, one ``Start Date:`` block each, and a BORIS
tabular-events export can hold several observations and several subjects. A recording is one
animal in one session, so such a file has to be narrowed to one of them before it becomes a
stream. Merging them puts other animals' events on this recording's timeline, and keeping only the
last block drops the rest without a word; both happened before these models existed.

**These pick; they never merge.** A file that holds one session (or one observation, one subject)
imports as it always did, with no config. A file that holds several is refused until a selector
names one, and the refusal lists what the file holds. Every field defaults to ``None`` ("the file
must hold only one"), so ``config=None`` and an all-default config are the same import.

A selection is about one *file*, not about how a lab's files are shaped, so it is not something a
saved preset carries: the GUI's preset save leaves it out (``ConfigBand.preset_config``).
"""

from pydantic import BaseModel, ConfigDict, Field


class MedpcSessionConfig(BaseModel):
    """Which session (``Start Date:`` block) of a MED-PC data file to import."""

    model_config = ConfigDict(extra="forbid")

    subject: str | None = Field(
        None,
        description="Import the session whose 'Subject:' header is this. Leave unset when the "
        "file holds one session.",
        json_schema_extra={"label": "Subject"},
    )
    box: str | None = Field(
        None,
        description="Import the session whose 'Box:' header is this. Combine with subject when "
        "one subject ran in several boxes.",
        json_schema_extra={"label": "Box"},
    )
    session: int | None = Field(
        None,
        ge=1,
        description="Import the session at this position in the file, counting from 1. For a "
        "file where subject and box do not tell two sessions apart.",
        json_schema_extra={"label": "Session number"},
    )


class BorisSelectionConfig(BaseModel):
    """Which observation and subject of a BORIS tabular-events export to import."""

    model_config = ConfigDict(extra="forbid")

    observation: str | None = Field(
        None,
        description="Import the rows whose 'Observation id' is this. Leave unset when the "
        "export holds one observation.",
        json_schema_extra={"label": "Observation"},
    )
    subject: str | None = Field(
        None,
        description="Import the rows whose 'Subject' is this. Leave unset when the export "
        "scores one subject. Rows with no subject are selected by an empty value.",
        json_schema_extra={"label": "Subject"},
    )
