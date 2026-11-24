# Package map

**Agent-facing. Not a user page** — it is in `mkdocs.yml`'s `exclude_docs`.

Where things live, and the annotations that are not derivable from the tree — which module is the
sole seam allowed to import a backend, which writer is the canonical one, which registry a file
populates. `ls` gives you the first column; this gives you the second. A `§N` is a section of
`DESIGN_PLAN.md`.

**History not published.** A `#NNN` or "PR N" below is a pull request or issue in the private
repository this project was developed in; that history is not published, so the number cannot be
looked up from here.

`tests/` mirrors `detail/` by convention, which is what `tests/impact.py::mirror` selects on.

This was a section of `CLAUDE.md` until it became clear it is a **lookup table**: read when you need
to find something, not start to finish, and never in the session where you are working on something
else. `claude doctor`'s trim check names directory layouts as the first thing to cut from a
checked-in `CLAUDE.md` for exactly that reason.


```
detail/
├── surfaces.py           # §25: which capabilities this build shows; ask is_surfaced() at
│                         # every point that *enumerates* what exists
├── console.py            # stdout that degrades the glyph, never the command
├── help.py               # per-field `?` help text
├── docs_access.py        # where the built docs live / how a surface opens them
├── core/
│   ├── protocols.py      # ABCs: Model (the one install/load contract, §5; VariantedModel is its
│   │                     # optional size-variant axis) → DetectionModel / EmbeddingModel /
│   │                     # KeypointModel; PatchFeatures (an embedding model's per-patch features,
│   │                     # §14); Trainer, Tracker, KeypointTrackMethod, MaskTransform,
│   │                     # StreamImporter, Tool, Check, QueryMethod, PoseClusterMethod,
│   │                     # TemporalClusterMethod; ModelNotReadyError
│   ├── dataclasses.py    # job, prompt, progress, preview, readiness, polygon, stream-import types
│   ├── registries.py     # every registry dict, each filled by its register_* decorator: MODELS,
│   │                     # TRAINED_LOADERS, TRAINERS, TRACKERS, EMBEDDING_MODELS, TRANSFORMS,
│   │                     # KEYPOINT_MODELS, KEYPOINT_TRACK_METHODS, STREAM_IMPORTERS,
│   │                     # QUERY_METHODS, POSE_CLUSTER_METHODS (empty since pose clustering was
│   │                     # removed), TEMPORAL_CLUSTER_METHODS, TOOLS, CHECKS, STREAM_CONSUMERS;
│   │                     # MODEL_REGISTRIES makes the three model dicts one model-id namespace
│   ├── errors.py         # UserInputError: an ordinary input mistake, shown by the CLI root
│   │                     # as one `Error:` line (cli.USER_ERRORS) and by GUI dialogs
│   ├── atomic.py         # replace / retry_on_lock / file_lock — the seam every
│   │                     # concurrently-read experiment file is written through
│   ├── utils.py          # resolve_device, resolve_track_id, t_ns conv, polygon_at_frame
│   ├── rle.py            # §26: RunMask, a COCO RLE mask read as runs; the assignment's few-pixel
│   │                     # membership test with no decode (utils.decode_mask is the dense read)
│   ├── ids.py
│   ├── hostinfo.py       # this machine: the performance report's machine block, machine.yaml,
│   │                     # `detail doctor`'s System section; the one `nvidia-smi` reader
│   ├── installed.py      # is_importable: the one package probe; re-reads .pth files on a
│   │                     # miss so an editable install made with the app open is seen
│   ├── buildinfo.py      # the running build + history + update check, read from git;
│   │                     # the one place detail shells out to `git`
│   ├── progress.py       # ThrottledProgress + PROGRESS_INTERVAL_S, remaining_seconds,
│   │                     # split_units, best_effort (live-progress throttling + publishing)
│   ├── resource_peaks.py # the most memory a job used (the column a timing table cannot answer)
│   ├── run_selection.py  # §6.5 "Upstream run specs": which upstream run an analysis read in each
│   │                     # recording
│   ├── chart_theme.py    # THE chart design: both themes' colours, type, weights (pure data)
│   ├── color_checks.py   # the palette checks (OKLab, CVD simulation, contrast) the tests run
│   ├── skeleton.py       # §26: Skeleton (names, edges, symmetry pairs), its digest, the presets
│   ├── examples.py       # §26: ExampleMark / ExampleFrame / ExampleSet — the marks a method
│   │                     # learns from, and (purpose: audit) the frames that score one
│   ├── keypoints.py      # §26: InstanceFrame (a frame + its detections' boxes/masks by row), ScoreSpec,
│   │                     # KeypointDetectJob / KeypointTrackJob / KeypointDatasetJob,
│   │                     # KeypointProgressEvent
│   ├── visualize.py      # the theme's identities as RGB + draw_mask/bbox/keypoints helpers, and
│   │                     # the trail primitives (draw_fading_paths, fade_weights, draw_onion_skin)
│   ├── lru.py            # CountLRU: the one bounded cache (row blocks, mask histories)
│   ├── cpu.py            # thread_workers(): the one rule for a read-bound pool's thread count
│   └── lookahead.py      # background_iter / ordered_map: work ahead of a consumer in order,
│                         # bounded — the embedding, detection and keypoint-detect stages (§14, §7, §26)
├── configs/              # one pydantic model per module — the single source for each
│   │                     # parameter's type, bounds, default, help, label (§6.5)
│   ├── detection.py, tracking.py, embedding.py, labeling.py, training.py, distill.py,
│   │   quick_segment.py, location_export.py, query.py, comparison.py,
│   │   temporal_clustering.py         # a module's (or tool's) run config
│   ├── keypoint_detect.py, keypoint_track.py  # §26 (+ the flat method models)
│   ├── location_plots.py # LocationPlotView: how a location export is drawn (a view model, like
│   │                     # display.py; its fields are the Plots view's controls and the CLI's flags)
│   ├── keypoint_dataset.py  # the label stage of a keypoint distill (§12 *Distill keypoints*)
│   ├── query_soft_knn.py, temporal_codebook.py, temporal_dtw_finch.py  # a registry method's own
│   │                     # parameters (the method's half of the config)
│   ├── stream_csv.py     # the CSV stream importer's config (§3)
│   ├── stream_select.py  # MED-PC session / BORIS observation+subject selectors (§3)
│   ├── windows.py        # §23 the window model: where a measure is evaluated
│   ├── display.py        # view parameters: what a finished result shows, not what a run computes
│   ├── describe.py       # renders the generated parameter reference from the models (§7)
│   ├── baselines.py      # loads named presets from configs/baselines/ (none ship today)
│   ├── tracker_params.py, method_params.py  # the non-model knob tables
│   └── clustering_legacy.py  # drops the retired size floor's keys on read (§22)
├── storage/              # the modules a rule names first, then the per-artifact record stores:
│   ├── experiment.py     # config.yaml read/write (identity only) + experiment path resolution
│   ├── example_sets.py   # §26: examples/<id>/examples.yaml + content-hash PNGs; the delete guard
│   │                     # reads detect.yaml, never a marker
│   ├── skeleton_library.py  # §26: the machine-wide skeleton library, <model cache>/skeletons/sk_NN/
│   ├── recording.py        # recording.yaml read/write + TimestampMap + load_active_timestamps[_cached]
│   ├── read_cache.py     # cached_document / cached_read / pinned() — the round-trip budget;
│   │                     # computing(key): concurrent misses wait for one computation;
│   │                     # forget_document(path): a writer drops its own entry
│   ├── locations.py      # this machine's folders (§2); read_storage() is the sole reader of
│   │                     # storage.yaml — one parse per change on disk, a stat per lookup
│   ├── roots.py          # the alias maps in storage.yaml; register_root is its only writer
│   ├── dirscan.py        # one enumeration per listing, instead of a stat per field
│   ├── user_yaml.py      # THE reader for YAML a user hands in or edits: a mistake is a
│   │                     # UserInputError naming the file and line (#33)
│   ├── id_allocator.py   # §3: every never-reused `<stem>_<n>` id (a high-water mark beside each family)
│   ├── video_roots.py    # storage.yaml's `video_roots:` alias map (reference-in-place video)
│   ├── zarr_store.py     # the retrying LocalStore every zarr group is opened through
│   ├── zarr_io.py        # detection/tracking/embedding/kinematics zarr writers and readers
│   ├── keypoint_zarr.py  # §26: the detect part (one row per pose instance, no subject: points, scores,
│   │                     # candidates, detection_row, extras/), the track's assignment part
│   │                     # (subject_id, inside) and its per-subject part (instance_row, points)
│   ├── keypoint_runs.py  # §26: run dirs, detect.yaml / track.yaml, KeypointDetectRun (rows / at: every
│   │                     # instance on a frame) and KeypointTrackRun (per subject + assignments);
│   │                     # refuses the per-track layout from before 2026-10-21
│   ├── heading_definitions.py  # §14: saved heading definitions (heading_definitions.yaml, the stream-preset pattern)
│   ├── video.py          # PyAV probe + timestamps + FrameCursor (a reader that keeps its place)
│   ├── previews.py       # §3 machine-local preview cache: layout, keys, frames.zarr + masks zarr,
│   │                     # size/clear. Display only — no analysing package imports it (guarded)
│   ├── runs.py           # experiment-level processing-run records (runs/<id>/run.yaml)
│   ├── analysis_runs.py  # experiment-level analysis-run records (analysis/<an_NN>/analysis.yaml, atomic §16)
│   ├── state_yaml.py     # the shared authored-run state writer (§21, §20)
│   ├── activity.py       # runs/<id>/activity.csv — the durable per-run activity log
│   ├── behavior_output.py# the §19 shared artifact: instances.csv + coverage.csv + behavior_output.yaml
│   ├── frame_view.py     # §19's per-frame expansion (frames/<recording>__<subject>.zarr); §23's denominator
│   ├── node_cache.py     # §22's node cache (<run>/node_cache.zarr): the ONE writer and reader,
│   │                     # staged + swapped; leaf tables and per-run sums (no floor, no thumbnails;
│   │                     # an older cache's list_*/thumb_* arrays are ignored on read)
│   ├── behavior_audit.py # §19's optional reliability audit (audit.csv), derived never stored
│   ├── audit_figures.py  # how an audit figure reads everywhere: below-8 and fully-audited rules
│   ├── clip_edits.py     # §22 ClipRef + the clip_edits list (clips removed from a label, or moved), fit-free
│   ├── clip_review.py    # §22 clip_review.csv: a temporal run's per-cluster census of every clip (AppendLog)
│   ├── run_labels.py     # §22 labels as stored things: ids, paths, the parent rule
│   ├── append_log.py     # crash-safe append-only CSV (human decisions); shared by the behavior
│   │                     # audit, the clip review and the run activity log
│   ├── export.py         # experiment export/import bundles
│   ├── export_runs.py    # §18 experiment-level export-run records (exports/<run_id>/export_run.yaml)
│   ├── run_facts.py      # what a job ran on and how much memory it used, in its own run dir
│   ├── tool_runs.py      # machine-global tool-run records (tools/<tool_id>/<id>/tool.yaml, atomic, §27)
│   │
│   │  # the per-artifact record stores:
│   ├── comparisons.py, datasets.py, distill_runs.py, models.py, query_runs.py, temporal_runs.py,
│   │   snapshots.py, notes.py   # §24, §12, §12, §12, §20, §22, §21, §15 (an entry's notes.md)
│   ├── metadata.py       # grouping variables (§11)
│   ├── spatial.py        # arena / ROI / POI keyframes (§10)
│   ├── streams.py, stream_presets.py, clocks.py  # §3: canonical stream CSVs + alignment, saved
│   │                     # import presets, and each importer's clock → video t_ns
│   ├── manifest.py, labels_manifest.py  # §3 bulk-import bundle manifest; behavior-label batch manifest
│   ├── credentials.py    # credential resolution and storage (§1, Tenet 14)
│   ├── integrity.py, video_integrity.py  # §2 file fingerprints; the referenced-video pre-run gate
│   └── location_export.py, clip_query_csv.py  # result CSV writers/readers (§16)
├── executor/
│   ├── base.py           # Executor protocol, JobStatus
│   ├── runners.py        # JOB_RUNNERS + ProcessingModule/PROCESSING_MODULES (§6.5) + AnalysisModule/ANALYSIS_MODULES (§16) + job_params
│   ├── rundir.py         # run-directory + job.json envelope helpers
│   ├── worker.py         # subprocess worker entrypoint
│   ├── coordinator.py    # detached multi-recording coordinator (§15 durable runs)
│   ├── distill_coordinator.py
│   ├── local.py          # LocalExecutor (subprocess-based, zero Qt dependency)
│   └── handles.py        # RunHandle implementations
├── pipeline/             # one module per registered stage / analysis module; a module's
│   │                     # `config_model` lives in configs/, its UI in gui/modules/
│   ├── detection.py            # detection coordinator (inline + multi-GPU §7)
│   ├── _detection_worker.py    # per-part detection worker: decode → prep → GPU → writer (§7 *Execution*)
│   ├── embedding.py            # embedding coordinator (inline + multi-GPU §14) + DetectionMaskDataset
│   ├── keypoint_detect.py      # §26: KeypointDetectModule + runner on a detection run (one video walk
│   │                           # per part; per detection or whole frame; unassigned instances;
│   │                           # decode / prep / forward / write overlap, §26 *Execution*)
│   ├── keypoint_track.py       # §26: KeypointTrackModule + runner — assigns instances to a tracking
│   │                           # run's subjects, then the two track methods it registers
│   ├── pose_match.py           # §26 *Assignment*: the one pose → subject match (share of points
│   │                           # inside the grown mask, one-to-one), for keypoint track and the
│   │                           # distill teacher
│   ├── keypoint_dataset.py     # §12 *Distill keypoints*: the keypoint_dataset runner — a teacher's
│   │                           # instances assigned by pose_match, one YOLO-pose label line per tracked animal
│   ├── keypoint_preview.py     # §12 *Distill keypoints*: what *Test cut-off…* and *Test keypoints…*
│   │                           # show — a keypoint teacher tested before a distill
│   ├── _embedding_worker.py    # per-(track, part) embedding worker (DataLoader + embed_batch)
│   ├── tracking.py             # tracking job runner
│   ├── prompt_test.py          # prompt-test (synchronous, in-process — §8)
│   ├── render_preview.py       # render-preview (synchronous, in-process — §14)
│   ├── kinematics_backfill.py  # §14's kinematics sibling for a run made before it existed
│   ├── dataset_export.py, training.py, labeling.py    # §12
│   ├── _dataset_common.py      # §12: what every YOLO dataset source shares below the labels —
│   │                           # frame reading, train/val split, image + label writing
│   ├── export_run.py           # §18: the export runner — one user request, as a durable job
│   ├── subject_frames.py # §26: which frames a track has a subject on (subject_frames, the box
│   │                     # union + mask OR every keypoint stage reads), and the example-frame
│   │                     # samplers (varied / spread)
│   ├── location_analysis.py    # LocationExportModule — the first AnalysisModule (§16)
│   └── query.py, temporal_clustering.py,
│       behavior_artifact.py    # §20, §22; and what every §19 artifact producer writes
│                               # besides its instances
├── analysis/             # pure, synchronous compute — no executor, no Qt:
│                         # keypoint_filters.py: §26 despike / smooth and the candidate path (viterbi);
│                         # body_axis.py: §14 the heading per row from a heading definition, interpolation
│                         # across gaps, the mirror hold, coverage counts — the one place it is computed;
│                         # node_cache.py: §22's leaf-additive tables (LeafTables, leaf runs and
│                         # their positions, member_positions) — pure, no I/O;
│                         # metrics: cluster_metrics, retrieval, descriptors, descriptor_ranking,
│                         # measures (§23) over windows.py,
│                         # behavior_labels (ground truth as a per-frame sequence);
│                         # segmentation + dynamics + tw_finch, decode.py (§22 temporal decode),
│                         # _dtw_numba.py / _decode_numba.py (the compiled kernels);
│                         # clip_query.py (trajectory continuity: runs, spans, gaps — read by
│                         # Query, temporal clustering and measures);
│                         # §24: cohort, compare, node_measures, comparison_report, provenance;
│                         # geometry.py / positions.py (where a subject or an instance was);
│                         # bout_segmentation.py (detector params, segment_recording, pool_recordings,
│                         # bout_medoid_distance — the bout steps temporal clustering uses);
│                         # locations.py (compute_location_tables — per-track ROI/POI features);
│                         # location_measures.py (a comparison's locations source, and the one
│                         # path rule: smoothed_segments / path_breaks); location_plots.py (a
│                         # location export's paths, occupancy, speed and body as a PlotSpec)
├── cluster_methods/      # TemporalClusterMethod implementations: dtw_finch, codebook_viterbi;
│                         # _common.py the steps they share (kmeans_level, blas_safe_openmp).
│                         # POSE_CLUSTER_METHODS is empty since pose clustering was removed
├── checks/               # `detail doctor` — environment / recording / setup probes; commands.py
│                         # the copy-paste install sequences; _probe.py the fresh-interpreter probe
├── report/               # figure + table rendering shared by the analysis reports; style.py is
│                         # the matplotlib adapter for core/chart_theme.py, figures.py the primitives,
│                         # tables.py the CSV writers, behavior_figures.py a §19 artifact's figures,
│                         # exporter.py what a re-export may delete (only what it wrote last time),
│                         # plot_spec.py a figure as plain data, plot_render.py its matplotlib
│                         # renderer and plot_html.py its plotly (interactive HTML) twin — the
│                         # location plots draw through all three
├── models/
│   ├── hf_access.py      # HFHubModel mixin + cache-first readiness
│   ├── _ultralytics.py   # the sole seam allowed to import ultralytics (YOLO_OFFLINE lives here);
│   │                     # SplitPredict: predict cut at its own stages (§7, §26 *Execution*)
│   ├── resolve.py        # weights_source resolution: pretrained / trained / imported
│   ├── sam3.py           # SAM3(HFHubModel, DetectionModel)
│   ├── trained_yolo.py   # TrainedYOLO(DetectionModel) — Ultralytics, closed-set local weights (§12)
│   ├── yolo_pose.py      # TrainedYoloPose(KeypointModel) — the distilled student (§12 *Distill keypoints*)
│   ├── distilled.py, imported.py
│   ├── dinov3.py         # DINOv3ConvNextBase + DINOv3ViT (size variants; ViT also PatchFeatures, §14)
│   └── patch_match/      # §26: the patch-matching keypoint engine — view.py (the crop), bank.py
│                         # (example descriptors + similarity maps), decode.py (candidates, the
│                         # skeleton decode, side rules), engine.py (PatchMatch(KeypointModel);
│                         # prepare / run / finish split for keypoint_detect's stages),
│                         # dinov3_match.py (the registered method on dinov3_vit; shares its weights)
├── importers/            # StreamImporter implementations (§3); import_stream_importers()
│   ├── medpc.py          # MedPCImporter — Med Associates operant data (.txt + .MPC) → events
│   ├── timeseries_csv.py # continuous CSV; config_model = CsvStreamConfig (explicit column mapping)
│   ├── behavior.py, boris.py, events_csv.py
│   └── _csv_common.py    # parsing helpers the CSV importers share; read_user_csv is THE
│                         # opener for any CSV a user hands in (stream, spatial, clip list)
├── trainers/             # concrete Trainer implementations
│   └── ultralytics_yolo.py
├── tools/                # Tool implementations (DESIGN_PLAN.md §27 (Tools)); import_tools() populates TOOLS
│   └── quick_segment.py  # QuickSegment — capped sequential segmentation of a raw clip + overlay/mp4 helpers
├── query_methods/        # QueryMethod implementations (§20); import_query_methods() populates QUERY_METHODS
│   └── soft_knn.py       # the declared default (`configs/query.py::DEFAULT_METHOD`) and the only method
├── trackers/
│   ├── anchor_tracker.py    # the default (`configs/tracking.py::DEFAULT_TRACKER`)
│   ├── iou_single_tracker.py
│   └── _common.py, _single_subject.py  # part_handoff.json resume + the shared per-frame gate
├── transforms/
│   ├── basic.py          # init_binary_mask, crop_to_bbox, pad_to_square, resize, apply_mask, center_in_square
│   ├── constructors.py   # named input-image recipes (*_centered / full_frame) + CONSTRUCTORS
│   ├── constructor_docs.py  # §14: the constructors' documentation metadata, derived
│   └── renderer.py       # MaskRenderer (pipeline of MaskTransforms)
├── api/                  # Tenet 2's boundary: the GUI and CLI are surfaces over this.
│   ├── experiment.py        # Experiment class (+ run_processing batch; delete / delete_recording / delete_run)
│   ├── recording.py        # Recording class (+ generic run(module_id, config); typed wrappers;
│   │                     # run_keypoint_detect / run_keypoint_track)
│   ├── run_spec.py       # §6.5 run identity: spec-based upstream selection with full lineage
│   ├── exporters.py      # the one export catalogue + writer; export_targets.py the targets (§18); export_estimates.py their size/time estimators (one per target)
│   ├── keypoint_export.py   # §26: PointSource (one reader of a keypoint run for export: cut-off,
│   │                     # source, blanking) + the long-CSV, DeepLabCut, SLEAP and NWB writers
│   ├── keypoints_video.py   # §26 (#689): the footage with the points drawn on, one sequential pass per part
│   ├── location_plots.py    # §16: LocationPlots, Experiment.location_plots(run_id) — the read boundary
│   │                     # the CLI's `locations plot` and the GUI's Plots view draw through
│   ├── run_preview.py    # Detection/Tracking/InputImage/Embedding/Stream/ClipQuery
│   │                     # preview accessors + pca_project_2d
│   ├── previews.py       # §3 preview build (proxy + masks), missing checks, PreviewReader (tiles)
│   ├── models.py         # experiment-free machine-global model catalog (§5; Experiment delegates here)
│   ├── readiness.py      # machine-global setup-readiness signals (§1 onboarding panel)
│   ├── checks.py         # machine-global check runner (§1) — the CHECKS framework's entry point
│   ├── imports.py        # recording import orchestration (§3)
│   ├── spatial_categories.py  # the four spatial-setup categories, as one table (§10)
│   ├── authored_run.py   # AuthoredRun(AnalysisRun) — the §21 mutable-run base; label resolution,
│   │                     # fold_legacy_names, labelled_instances (§22: labels drive the output)
│   ├── clip_edits.py     # take_out_clips / put_back_clips / taken_out (temporal runs) and the
│   │                     # per-member override labelled_instances hands instances_for
│   ├── clip_review.py    # finish_review (a cluster's census) / review_precision / review_by_cluster
│   ├── clustering_nodes.py # NodeCacheMixin: node_view / clips (a medoid method ranked on first
│   │                     # ask, kept per node) / typical_members / rebuild_node_cache, used by
│   │                     # the temporal clustering run handle
│   ├── node_cache.py     # build_node_cache / write_for_run: computes the §22 cache from a fit,
│   │                     # asking the method's clip_scoring hook how clips are cut and scored
│   ├── node_view.py      # NodeView: the cache's in-memory reader (summaries, distributions with
│   │                     # their floors, clips -> ClipTable with typicality, match_shares,
│   │                     # stale_reason)
│   ├── cluster_filter.py # Selection + recording_mask / cell_mask / clip_matches: the cluster
│   │                     # view's filter math (OR within a column, AND across); session_spans
│   │                     # + session_window: Time in session over each recording's whole video
│   ├── clip_views.py     # §22 clip window: ClipViewSettings (the one settings object), ClipFrame
│   │                     # (one decode, or none), the three pure drawers (footage / masks pane /
│   │                     # pose pane), ClipPoints, MaskHistory, pane exports (export_clip_pane);
│   │                     # the Aligned turn written once (turn_matrix / turn_views / mark_no_axis,
│   │                     # pose_notes); the window's view exported as shown (export_clip_view,
│   │                     # write_clip_frame)
│   ├── clip_curves.py    # §22 strips: Curve / ClipCurves off the kinematics sidecar via
│   │                     # frame_slices.from_kinematics + descriptors.step_curve; pose curves
│   │                     # from analysis.body_axis.body_shape_series
│   ├── frame_slices.py   # FrameSlice loaders, two of one shape: from_tracking (the tracking
│   │                     # zarr) and from_kinematics (the §14 sidecar, at the embedding cadence)
│   ├── animation.py      # write_animation: the one mp4 / GIF writer (cv2 H.264, Pillow GIF)
│   ├── clip_montage.py   # §22 clip montage exports: ClusterMontage (clip_frame + the clip_views
│   │                     # drawers per tile, the Raw-on outline, one clock; export /
│   │                     # export_clip_gifs / export_clip_list), MontageLayers (flavour /
│   │                     # settings), choose_clips, grid_size, compose, montage_members /
│   │                     # montage_clips / cluster_montage() — read from the source video;
│   │                     # MontageLayers.aligned, with_context
│   ├── clip_tiles.py     # §22 the INTERPRET montage's tile source over the preview cache:
│   │                     # TileCatalog (a reader per part), ClipTiles (one PreviewClip per tile,
│   │                     # grey when off-filter), missing (proxy vs masks), tracking_runs;
│   │                     # TileLook (what a tile draws), the keypoints / axis read once per clip
│   ├── clip_render.py    # §16: one forward decode pass per video part, cutting every clip it holds
│   ├── clip_export.py    # one mp4 of one frame range of one recording — the clip a player shows
│   ├── behavior_clips.py # clips for any §19 behavior artifact — one exporter, every producer
│   ├── active.py         # runs in flight across every experiment (the Active processes mode)
│   ├── tools.py          # experiment-free Tools API (§27): list_tools / run_tool / list|read|delete_tool_run
│   ├── skeletons.py      # §26: the skeleton library (presets + sk_NN), dependents by live search
│   ├── keypoints.py      # §26: example sets (a skeleton copied from the library + the frames
│   │                     # marked on it) — module-level functions Experiment delegates to
│   ├── keypoint_distill.py  # §12 *Distill keypoints*: keypoint_dataset / keypoint_test /
│   │                     # distill_keypoints / register_student, behind Experiment's one-liners;
│   │                     # DISTILL_STAGES_BY_KIND lives in experiment.py
│   ├── query.py, temporal_clustering.py, measures.py, comparison.py  # one module per analysis
│   │                     # family: QueryRun (§20), TemporalClusteringRun (§22), a run's
│   │                     # behavior artifact as numbers (§23), comparisons (§24)
│   └── _runner_modules.py# shared PIPELINE_RUNNERS list (Experiment + Recording use it)
├── cli.py                # click-based CLI; entry point `detail` (or `python -m detail`)
├── cli_paths.py          # UserPath: the one path type for every CLI path option (expands `~`)
├── cli_keypoints.py      # §26: the `keypoints` command group, registered on cli.py's root
│                         # group — its own module so the keypoint work never edits one region of cli.py
├── cli_numbers.py        # the finite float option types every float option in cli*.py uses
│                         # (click's own accept nan and inf)
├── cli_skeletons.py      # §26: the machine-wide `skeleton` command group (the library)
├── __main__.py           # `python -m detail` dispatcher
└── gui/                  # PyQt6 GUI (docs/gui_design.md): app.py is the shell (mode banner,
    │                     # both rails, navigation); pages.py the content-area pages + `_table`;
    │                     # launch.py the one launch dialog; preview.py (prompt-test /
    │                     # render-preview); arena.py (scene geometry); streams.py (add-stream);
    │                     # run_preview.py (detection/tracking viewers, input-image scrubber,
    │                     # embedding PCA); readiness_guide.py + cuda_doctor.py (§1 onboarding);
    │                     # __main__.py (`python -m detail.gui`)
    │
    │  # The shared leaves the GUI rules are written against (the rules are predicates in
    │  # requirements.py; the reasoning is docs/gui_design.md §3.1) — a rule names one of
    │  # these rather than describing the gesture, so a new surface inherits it:
    ├── requirements.py   # the checked structural requirements, and the generated guidelines page
    ├── dialog.py         # Dialog base: body() / set_buttons() — scroll + top-justify come free
    ├── tables.py         # enable_sorting, make_columns_resizable, stamp_row_id/row_id,
    │                     # SortCell, refill_table_widget, MIN_VISIBLE_ROWS, max_visible_rows
    ├── sizing.py         # lines(), mono_font(), emphasis_font(), small_font(), fit_to_screen(),
    │                     # fit_list_height, remembered_size
    ├── scroll.py         # PageScroll + OWNS_WHEEL
    ├── spin.py           # IntSpin / FloatSpin (never a bare QSpinBox) + ThresholdSlider
    ├── _pydantic_form.py # widgets from a config model: field_widget / build_fields /
    │                     # optional_spin + optional_value / any_field_value
    ├── method_form.py    # a registry method's parameters, from its own config model
    ├── forms.py          # ConfigForm + AdvancedSection
    ├── flow.py           # FlowLayout: a row of controls that wraps instead of being squeezed
    ├── plots.py          # bound_axes + the shared strips/lanes
    ├── plot_theme.py     # the pyqtgraph/QPainter adapter: themed_plot, pen/brush by role
    ├── zoom.py           # ZoomWindow + OverviewBar: one horizontal zoom for every widget with one
    ├── collapse.py       # CollapsibleBand (fold + drag handle, remembered)
    ├── notes.py          # NotesBox (folded per page and view; the shell's Notes button opens it there) +
    │                     # place_notes, kept across a page's rebuilds (§15 Entry notes)
    ├── result_views.py   # ResultView / result_views / MultiViewWorkspace: an analysis run's
    │                     # views — a card each on its landing page, each full window (§16)
    ├── busy.py           # run_busy / run_progress / BusyOverlay / decode_overlay
    ├── menus.py          # open_once: a menu built per opening is deleted when it closes
    ├── workers.py        # GuiWorker (every thread names its receiver=) + BackgroundCall
    ├── lifetime.py       # whether a Qt object is still there to be talked to
    ├── ui_thread_gc.py   # the cyclic collector runs on the UI thread only (app + qapp fixture)
    ├── image_view.py     # the one image pane; `zoomable` has no default
    ├── full_frame.py     # a thumbnail that opens the frame behind it, full size and zoomable
    ├── selectable.py     # text on screen can be selected and copied
    ├── status_color.py   # the status palette, and the one way to apply it to a label
    ├── timefmt.py        # one way to print a recording time
    ├── recording_items.py  # filling a recording picker, and reading the selection back
    ├── confidence.py     # the confidence-view control every detection result carries
    ├── example_set_picker.py  # §26: the skeleton + example-set picker every examples-needing
    │                     # model's surface embeds (New / Mark frames / Delete)
    ├── keypoints_manager.py  # §26: the Keypoints manager banner mode (the skeleton library)
    ├── skeleton_editor.py  # §26: SkeletonCanvas (on ArenaCanvas) + SkeletonEditor, the drawing half
    ├── keypoint_labeling.py  # §26: the labeling screen (a player + the example set), the
    │                     # open_labeling (the detect form's entry point)
    ├── keypoint_distill.py  # §12 *Distill keypoints*: the Distill keypoints… dialog (teacher runs read
    │                     # off the UI thread, label knobs off KeypointDatasetConfig)
    ├── keypoint_test.py  # §12: *Test cut-off…* (distill dialog) and *Test keypoints* (detect form)
    ├── scrubber.py       # FrameScrubber (playback, busy_changed) + frame_source.py, decode_order.py
    ├── editing.py        # the unsaved-edits protocol the shell prompts against
    ├── export.py         # the one Export control, placed by the shell (§18)
    ├── sections.py       # EXPERIMENT_SECTIONS — the rail's one declaration
    ├── start_from.py     # defaults / a shipped baseline / a previous run
    ├── model_availability.py  # apply_availability + selected_model_id
    ├── compute_target.py # ComputeTargetRow
    ├── grip_splitter.py  # GripSplitter: visible painted grips, remembered splits (no QSS)
    ├── keypoint_run_preview.py  # §26: KeypointDetectRunPreview — the detection viewer with every pose
    │                     # drawn; KeypointTrackRunPreview — the tracking viewer with each subject's
    │                     # points, the assignment counts, loss rows and alerts
    │
    │  # feature surfaces:
    ├── active_processes.py  # the Active processes banner mode: everything in flight, every experiment
    ├── version_history.py   # Version History: which build is running, whether it is current, what changed
    ├── help_ui.py        # the in-app help popover + the ? button that opens it
    ├── import_queue.py   # the app-level, non-blocking recording-import queue
    ├── run_activity.py   # LiveRunPanel: a bar, a status line, and a row per unit of the run
    ├── run_units.py, unit_rows.py  # what a run is made of, one row per unit; the rows themselves
    ├── fit.py            # §12 launch dialogs: generate a dataset, train a model
    ├── training_curve.py # mAP50 per epoch, and the best score so far
    ├── snapshots.py      # §21 pinned artifacts on screen
    ├── comparison.py     # §24: the comparisons list page, launch dialog and saved result
    ├── query_review.py   # a query run's AUDIT: every clip a cutoff keeps, one recording at a time
    ├── review_marks.py   # AUDIT's ordered writes, marks not yet read back, undo (both run kinds)
    ├── behavior_query.py # §16/§20 clip surfaces shared across the GUI: the example-clip picker
    │                     # and the clip player (_ClipPlayer)
    ├── clip_grid.py      # one thumbnail's pool decode (cluster_strip's) and score_digits
    ├── quick_preview.py  # the small player a strip puts in a thumbnail's place
    ├── example_spread.py # how a behavior's example clips spread over the cohort (beside Mark examples)
    ├── measurements.py   # what a cluster or a behavior *is*, as numbers — one band for both result pages
    ├── spatial_map.py    # where a behavior happens, over the field of view (§19)
    ├── cluster_strip.py  # the member strip and the ethograms under a cluster (§22)
    │
    │  # INTERPRET for behavior (temporal) clustering (§22); cluster_workspace.py holds the views
    │  # (a MultiViewWorkspace: INTERPRET / AUDIT / BROWSE / MEASURES, its tab bar hidden):
    ├── node_workspace.py # NodeInterpretTab: label field + filters, shaded tree + key, the live
    │                     # montage of every clip (header, export, preview wait), label writes
    ├── node_icicle.py    # NodeIcicle / TreePane: the tree as a viewport (zoom, pan, scroll);
    │                     # set_shades (fixed 0–100% match ramp, label as a top strip)
    ├── cluster_window.py # ClusterWindow: one cluster's clips in a non-modal window (INTERPRET's
    │                     # label bar over a LiveMontage, writing through the page's queue);
    │                     # ClusterWindows: one per cluster, owned by the page
    │
    │  # the clip window (§22): one clip, whole frame, with Advanced / Simple / Blind
    ├── clip_popup.py     # ClipPopup: the window; swaps the plain player for the Advanced one;
    │                     # As recorded | Aligned, Export ▾; open_clip_window: the one non-modal,
    │                     # one-per-page opener every page calls
    ├── advanced_player.py # AdvancedClipPlayer(_ClipPlayer): the hooks overridden — panes, view
    │                     # ticks (Keypoints / Trail / Trail points / Mask trail / Align pose /
    │                     # Loop clip), strips off clip_curves, open_pane; PlainClipPlayer: the
    │                     # plain player with the Aligned orientation
    ├── clip_panes.py     # ClipPanes: the footage with the Masks and Keypoints panes beside it
    ├── clip_strips.py    # ClipStrips: one thin row per Curve under the slider + the untracked lane
    ├── clip_pane_window.py # ClipPaneWindow: a side pane large, live, one per pane, exportable
    ├── skeleton_points.py # SkeletonPointsPicker / SkeletonPointsButton: a checkbox per point of
    │                     # any skeleton (Trail points), the picker-in-a-menu written once
    ├── animation_export.py # GifSettingsDialog + the save / run-export helpers an export uses
    │  # the INTERPRET page's parts
    ├── cluster_filters.py # FilterDock (Location / Time in session / Video / metadata sections,
    │                     # RangeHistogram, bars=False for SessionRange's plain slider) +
    │                     # FilterBar (not a widget: the selection, the dock, the count and
    │                     # Reset all in it) → one api Selection; panel_name for the folded strip
    ├── density_map.py    # DensityMap (the 16×16 position grid; a live box with corner handles)
    │                     # + RegionPicker (it as a filter)
    ├── throttle.py       # Throttle: a drag emits at most every 50 ms, then once on release
    ├── live_montage.py   # LiveMontage: every clip playing, a clock per tile, one QTimer, a
    │                     # TileSource read off the UI thread, virtualised, a blocked state,
    │                     # click-to-mark (clip_clicked) and titled sections, the TileLook it
    │                     # asks for, the toolbar (order, Show ▾, Size, Export ▾) every clip
    │                     # grid has, a ClickLayer (pick mode) over the page's own click
    ├── montage_export.py # Export ▾ and pick mode for every clip grid: ExportSource (what a page
    │                     # hands its montage), PickBar, MontageExport (the writers, one copy)
    ├── show_menu.py      # Show ▾ (ShowButton): the layers, the orientation and the context of a
    │                     # TileLook, one definition (LAYERS, layer_states); layers_of → an
    │                     # export's MontageLayers; remembered_look / remember_look per page
    ├── match_key.py      # TreeKeyRow / MatchKey / LabelSwatches: the fixed 0–100% match key
    ├── label_panel.py    # LabelBar (name, chip, toolbar; figures | status; LabelField; recent 1–9);
    │                     # the list of every label, rename / merge
    │                     # + the label rules: label_paths, path_rows, label_choices, merge_targets
    ├── label_history.py  # LabelHistory: INTERPRET's ⌘Z / ⇧⌘Z, one history of label changes for
    │                     # a run, shared by the page and its cluster windows; an undo is an
    │                     # inverse write by id (AuthoredRun.restore_labels)
    ├── node_cache_jobs.py# cache builds owned by the app, not the page that started them
    ├── preview_jobs.py   # §3 the app's preview queue: one build at a time, status-bar bar
    ├── node_audit.py     # NodeAuditBlock: AUDIT's figure in the label bar, Act on it ▾
    ├── label_check.py    # clustering AUDIT: the list of labels and their clusters, the page
    ├── label_clean.py    # AUDIT's review pane: click / drag / Shift to remove, Move to…, Finish, reviewed box
    ├── audit_table.py    # audited_colour: the one colour INTERPRET rings a reviewed cluster in
    ├── label_browse.py   # clustering BROWSE: filters, ethogram, the labelled-instance list
    ├── importers/        # per-importer config bands (§3), keyed by importer_id
    │                     # (registry.py: ImporterUI + ConfigBand; csv_columns.py; selection.py)
    ├── modules/          # per-module config + result UIs, keyed by module_id: detection.py,
    │                     # tracking.py, embedding.py, keypoint_detect.py and keypoint_track.py
    │                     # (§26), analysis_location.py (+ ../location_plots.py, its Plots view),
    │                     # analysis_query.py,
    │                     # analysis_temporal_clustering.py; registry.py also re-exports the
    │                     # result-view protocol; _common.py the upstream-run pickers, _every_n.py
    │                     # the "Use every N frames" row, _heading.py the one heading picker (§14 —
    │                     # HeadingPicker + the Define… dialog; the embedding form uses it)
    └── tools/            # per-tool config form + result UI (§27), keyed by tool_id
                          # (registry.py: ToolUI + register_tool_ui; quick_segment.py)
```

Tests mirror this layout under `tests/`:
```
tests/
├── conftest.py           # shared fixtures — see `CLAUDE.md`'s Testing section
├── tiers.py              # what a test costs (unit/gui/slow) and what it protects (guard)
├── impact.py             # which tests a *change* touches (the edit-loop selector only)
├── core/                 # core/*.py
├── storage/              # storage/*.py
├── executor/             # executor/*.py
├── pipeline/             # pipeline/*.py
├── analysis/             # analysis/*.py
├── cluster_methods/      # cluster_methods/*.py
├── checks/               # checks/*.py (`detail doctor`)
├── configs/              # configs/*.py (model completeness, record completeness, baselines)
├── report/               # report/*.py
├── models/               # models/*.py
├── importers/            # importers/*.py
├── trainers/             # trainers/*.py
├── tools/                # tools/*.py (Quick segment engine + API)
├── query_methods/        # query_methods/*.py (registry parity + the method guards)
├── trackers/             # trackers/*.py
├── transforms/           # transforms/*.py
├── scripts/              # scripts/*.py
├── gui/                  # gui/*.py — the largest directory here, and where most guards live
└── api/                  # api/*.py
```
