# Python API

The Python API is the boundary the GUI and CLI drive. This reference is generated from the
docstrings.

## Experiment and recording

::: detail.api.experiment.Experiment

::: detail.api.recording.Recording

## Analysis runs

An analysis run is experiment-level, over a selection of recordings. `AnalysisRun` is the base
every module's run handle shares; `AuthoredRun` adds the editable state that a run you keep
working on needs.

::: detail.api.experiment.AnalysisRun

::: detail.api.authored_run.AuthoredRun

::: detail.api.query.QueryRun

::: detail.api.temporal_clustering.TemporalClusteringRun

## Results

::: detail.api.measures.Measures

::: detail.api.comparison.Comparison

## Machine-global catalogs

These take no experiment: the model catalog is per machine, and a tool run is saved outside any
experiment.

::: detail.api.models
    options:
      members:
        - list_models
        - model_readiness
        - is_installed
        - install_model
        - import_model
        - remove_imported_model

::: detail.api.tools
    options:
      members:
        - list_tools
        - run_tool
        - list_tool_runs
        - read_tool_run
        - delete_tool_run
