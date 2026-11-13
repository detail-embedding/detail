# Package map

**Agent-facing. Not a user page** — it is in `mkdocs.yml`'s `exclude_docs`.

Where things live, and the annotations that are not derivable from the tree — which module is the
sole seam allowed to import a backend, which writer is the canonical one, which registry a file
populates. `ls` gives you the first column; this gives you the second.

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
│   ├── protocols.py      # ABCs: Model (the one install/load contract, §5) → DetectionModel /
│   │                     # EmbeddingModel; Trainer, Tracker, MaskTransform, StreamImporter
│   ├── dataclasses.py    # job, prompt, progress, preview, readiness, polygon, stream-import types
│   ├── registries.py     # MODELS, TRAINERS, TRACKERS, EMBEDDING_MODELS, TRANSFORMS, STREAM_IMPORTERS dicts
│   ├── atomic.py         # replace / retry_on_lock / file_lock — the seam every
│   │                     # concurrently-read experiment file is written through
│   ├── utils.py          # resolve_device, torch_device, resolve_track_id, t_ns conv, polygon_at_frame
│   ├── rle.py            # §26: RunMask, a COCO RLE mask read as runs; the assignment's few-pixel
│   │                     # membership test with no decode (utils.decode_mask is the dense read)
│   ├── skeleton.py       # §26 KeypointSpec + Skeleton: arbitrary names; sides, pairs and the
│   │                     # front/back body axis declared beside them; presets, infer_roles
│   ├── ids.py, hostinfo.py
│   ├── installed.py      # is_importable: the one package probe; re-reads .pth files on a
│   │                     # miss so an editable install made with the app open is seen
│   ├── buildinfo.py      # the running build + history + update check, read from git; also
│   │                     # `report`/`compare_with`: a cluster's build against this one
│   │                     # the one place detail shells out to `git`
│   ├── progress.py       # ThrottledProgress + PROGRESS_INTERVAL_S, remaining_seconds,
│   │                     # split_units, best_effort (live-progress throttling + publishing)
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
│                         # parameter's type, bounds, default, help, label (§6.5). describe.py
│                         # renders the generated reference; baselines.py the shipped presets;
│                         # tracker_params.py / method_params.py the non-model knob tables.
│                         # keypoint_detect.py, keypoint_track.py (+ the two flat method models) and
│                         # keypoint_dataset.py (the label stage of a keypoint distill, §12) are §26's
│                         # clustering_legacy.py drops the retired size floor's keys on read (§22)
├── storage/              # ~40 modules; the ones a rule names:
│   ├── experiment.py        # config.yaml read/write + per-stage defaults
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
│   ├── video_roots.py    # storage.yaml's `video_roots:` alias map (reference-in-place video)
│   ├── machine_file.py   # the one writer of the machine-config files (storage.yaml, credentials.yaml,
│   │                     # remote_targets.yaml): editing(path) locks, reads and writes ONE path
│   ├── zarr_store.py     # the retrying LocalStore every zarr group is opened through
│   ├── zarr_io.py        # detection/tracking/embedding/kinematics zarr writers and readers
│   ├── keypoint_zarr.py  # §26: the detect part (one row per pose instance, no subject: points, scores,
│   │                     # candidates, detection_row, extras/), the track's assignment part
│   │                     # (subject_id, inside) and its per-subject part (instance_row, points)
│   ├── keypoint_runs.py  # §26: run dirs, detect.yaml / track.yaml, KeypointDetectRun (rows / at: every
│   │                     # instance on a frame) and KeypointTrackRun (per subject + assignments);
│   │                     # refuses the per-track layout from before 2026-10-21
│   ├── heading_definitions.py  # §14: saved heading definitions (heading_definitions.yaml, the stream-preset pattern)
│   ├── video.py          # PyAV probe + timestamps + FrameCursor (a reader that keeps its place); open_video_writer, the one export writer (h264 via PyAV libx264, mp4v via cv2)
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
│   ├── append_log.py     # crash-safe append-only CSV (human decisions); shared by audit + annotation
│   ├── export.py         # experiment export/import bundles
│   └── tool_runs.py      # machine-global tool-run records (tools/<tool_id>/<id>/tool.yaml, atomic §9)
│   # plus per-artifact record stores (comparisons, datasets, distill_runs,
│   # models, query_runs, temporal_runs, snapshots, metadata,
│   # streams, stream_presets, spatial, locations, credentials, integrity, clocks, …)
├── executor/
│   ├── base.py           # Executor protocol, JobStatus
│   ├── runners.py        # JOB_RUNNERS + ProcessingModule/PROCESSING_MODULES (§6.5) + AnalysisModule/ANALYSIS_MODULES (§16) + job_params
│   ├── rundir.py         # run-directory + job.json envelope helpers
│   ├── worker.py         # subprocess worker entrypoint
│   ├── coordinator.py    # detached multi-recording coordinator (§15 durable runs)
│   ├── distill_coordinator.py
│   ├── local.py          # LocalExecutor (subprocess-based, zero Qt dependency)
│   ├── video_staging.py  # opt-in detached background video upload to a remote target; its state lives OUTSIDE the experiment tree (the shepherd mirrors that tree both ways)
│   ├── remote_python.py  # the one place Python runs on a cluster (`python -` over ssh); names a cluster whose DETAIL is older than the app
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
│   ├── keypoint_track_preview.py  # §26 *Previewing a track method*: the stage over a few
│   │                           # seconds of a few recordings, assigned once, settings in memory
│   ├── pose_match.py           # §26 *Assignment*: the one pose → subject match (share of points
│   │                           # inside the grown mask, one-to-one), for keypoint track and the
│   │                           # distill teacher
│   ├── keypoint_dataset.py     # §12 *Distill keypoints*: the keypoint_dataset runner — a teacher's
│   │                           # instances assigned by pose_match, one YOLO-pose label line per tracked animal
│   ├── _embedding_worker.py    # per-(track, part) embedding worker (DataLoader + embed_batch)
│   ├── tracking.py             # tracking job runner
│   ├── prompt_test.py          # prompt-test (synchronous, in-process — §8)
│   ├── render_preview.py       # render-preview (synchronous, in-process — §14)
│   ├── kinematics_backfill.py  # §14's kinematics sibling for a run made before it existed
│   ├── dataset_export.py, training.py, labeling.py    # §12
│   ├── subject_frames.py # §26: which frames a track has a subject on (subject_frames, the box
│   │                     # union + mask OR every keypoint stage reads), and the example-frame
│   │                     # samplers (varied / spread / heading; STRATEGY_TABLE is the one list)
│   ├── headings.py       # §26 *Headings*: camera-relative heading per tracked frame (walking,
│   │                     # tail, rearing), the per-recording headings/ cache, and the pool the
│   │                     # heading sampler and the review window rank from
│   ├── location_analysis.py    # LocationExportModule — the first AnalysisModule (§16)
│   ├── profile_describe.py     # ProfileDescribeModule (§22, "Profile clusters"): a profile
│   │                           # of many clusters of a clustering run, one job; its own an_NN,
│   │                           # placed beside the run (api/profile_describe.place)
│   ├── cluster_lookalikes.py   # ClusterLookalikesModule (§22): pairs of branch tops compared
│   │                           # node against node, saved beside the run; local only
│   └── query.py, query_eval.py, temporal_clustering.py,
│       behavior_artifact.py
├── analysis/             # pure, synchronous compute — no executor, no Qt. ~35 modules:
│                         # keypoint_filters.py: §26 despike / smooth and the candidate path (viterbi);
│                         # keypoint_track_diagnostics.py: §26 the label-free figures under Preview
│                         # tracking (placed, move, jumps, swaps, stretch) — describe, never rank;
│                         # body_axis.py: §14 the heading per row from a heading definition, interpolation
│                         # across gaps, the mirror hold, coverage counts — the one place it is computed;
│                         # node_cache.py: §22's leaf-additive tables (LeafTables, leaf runs and
│                         # their positions, member_positions) — pure, no I/O;
│                         # metrics (cluster_metrics, retrieval, descriptor_ranking, measures),
│                         # segmentation + dynamics, cohort/compare, the report builders,
│                         # bout_segmentation.py (detector params, segment_recording, pool_recordings,
│                         # bout_medoid_distance — the bout steps temporal clustering uses), and
│                         # locations.py (compute_location_tables — per-track ROI/POI features);
│                         # location_plots.py: §16 a location export's paths / occupancy / speed
│                         # read from its CSVs, described as a report/plot_spec.PlotSpec
│                         # body_axis.py: §26 a keypoints run's body axis per frame and the
│                         # alignment embedding takes from it (rotate / mirror, held by time)
├── cluster_methods/      # ClusterMethod implementations: dtw_finch, codebook_viterbi (temporal); the pose registry is empty
├── checks/               # `detail doctor` — environment / recording / setup probes
├── report/               # figure + table rendering shared by the analysis reports; style.py is
│                         # the matplotlib adapter for core/chart_theme.py, figures.py the primitives;
│                         # plot_spec.py a figure as plain data (arrays + colour roles), and
│                         # plot_render.py its matplotlib renderer (files, or PNG bytes for the GUI),
│                         # plot_html.py its interactive HTML twin (plotly, self-contained page)
├── models/
│   ├── hf_access.py      # HFHubModel mixin + cache-first readiness
│   ├── _ultralytics.py   # the sole seam allowed to import ultralytics (YOLO_OFFLINE lives here);
│   │                     # SplitPredict: predict cut at its own stages (§7, §26 *Execution*)
│   ├── resolve.py        # weights_source resolution: pretrained / trained / imported;
│   │                     # keypoint_model_class / construct_keypoint_model (§26)
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
│   └── behavior.py, boris.py, events_csv.py
├── trainers/             # concrete Trainer implementations
│   └── ultralytics_yolo.py
├── tools/                # Tool implementations (§9); import_tools() populates TOOLS
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
│   └── renderer.py       # MaskRenderer (pipeline of MaskTransforms)
├── api/                  # Tenet 2's boundary: the GUI and CLI are surfaces over this.
│   ├── experiment.py        # Experiment class (+ run_processing batch; delete / delete_recording / delete_run)
│   ├── recording.py        # Recording class (+ generic run(module_id, config); typed wrappers)
│   ├── exporters.py      # the one export catalogue + writer; export_targets.py the targets (§18); export_estimates.py their size/time estimators (one per target)
│   ├── keypoint_export.py   # §26: PointSource (one reader of a keypoint run for export: cut-off,
│   │                     # source, blanking) + the long-CSV, DeepLabCut, SLEAP and NWB writers
│   ├── keypoints_video.py   # §26 (#689): the footage with the points drawn on, one sequential pass per part
│   ├── run_preview.py    # Detection/Tracking/InputImage/Embedding/StreamPreview accessors + pca_project_2d
│   ├── previews.py       # §3 preview build (proxy + masks), missing checks, PreviewReader (tiles)
│   ├── models.py         # experiment-free machine-global model catalog (§5; Experiment delegates here)
│   ├── readiness.py      # machine-global setup-readiness signals (§1 onboarding panel)
│   ├── extras.py         # §1: install an optional extra (analysis) into the running
│   │                     # interpreter's env — the only module here that runs pip / uv
│   ├── authored_run.py   # AuthoredRun(AnalysisRun) — the §21 mutable-run base; label resolution,
│   │                     # fold_legacy_names, labelled_instances (§22: labels drive the output)
│   ├── profile_describe.py # §22 Profile clusters (many at once): select_nodes, what is
│   │                     # already done (resume), place() (the job's results beside the
│   │                     # run, idempotent), estimate()
│   ├── duplicate_analysis.py # §16 Duplicate: copy a run whole under a new an_NN, its own
│   │                     # id rewritten, process state (cancel/stop/lock/tmp) left behind
│   ├── cluster_lookalikes.py # §22 look-alike clusters: branch tops and the pairs to compare
│   │                     # (every pair, or labelled_pairs; count_pairs for the warning),
│   │                     # the saved node-vs-node comparisons as evidence, chained groups,
│   │                     # plan_join, Lookalikes (one label through _label_edit)
│   ├── clip_edits.py     # take_out_clips / put_back_clips / taken_out (temporal runs) and the
│   │                     # per-member override labelled_instances hands instances_for
│   ├── clip_review.py    # finish_review (a cluster's census) / review_precision / review_by_cluster
│   ├── clustering_nodes.py # NodeCacheMixin: node_view / clips (a medoid method ranked on first
│   │                     # ask, kept per node) / typical_members / rebuild_node_cache, shared by
│   │                     # the pose and temporal clustering runs
│   ├── node_cache.py     # build_node_cache / write_for_run: computes the §22 cache from a fit,
│   │                     # asking the method's clip_scoring hook how clips are cut and scored
│   ├── node_view.py      # NodeView: the cache's in-memory reader (summaries, distributions with
│   │                     # their floors, clips -> ClipTable with typicality, match_shares,
│   │                     # stale_reason)
│   ├── cluster_filter.py # Selection + recording_mask / cell_mask / clip_matches: the cluster
│   │                     # view's filter math (OR within a column, AND across); session_spans
│   │                     # + session_window: Time in session over each recording's whole video
│   ├── clip_views.py     # §22 the Aligned turn written once (turn_matrix / turn_picture /
│   │                     # mark_no_axis), ClipPoints, pose_notes, view_picture; the window's
│   │                     # pictures are run_preview's ClipQueryPreview.clip_views
│   ├── clip_pane_export.py # §22 the clip window's exports: an enlarged pane (export_clip_pane),
│   │                     # the window's view as shown (export_clip_view), this frame (write_clip_frame)
│   ├── clip_signals.py   # §22 the clip window's strips (speed, area, nearest, body shape)
│   ├── clip_fingerprint.py # §22 Cluster details' "what sets this cluster apart" (cluster_fingerprint)
│   ├── frame_slices.py   # FrameSlice loaders, two of one shape: from_tracking (the tracking
│   │                     # zarr) and from_kinematics (the §14 sidecar, at the embedding cadence)
│   ├── animation.py      # write_animation: the one mp4 / GIF writer (cv2 H.264, Pillow GIF)
│   ├── clip_montage.py   # §22 clip montage exports: ClusterMontage (clip_views per tile, the
│   │                     # Raw-on outline, one clock; export / export_clip_gifs / export_clip_list,
│   │                     # overlay / export_overlay), MontageLayers (aligned, trails), choose_clips
│   │                     # (random seeded by the node), with_context, query_examples,
│   │                     # cluster_montage_members / cluster_montage_clips — read from the source
│   │                     # video; the engine every clip grid's Export ▾ and the CLI write with
│   ├── montage_export.py # §22 Export clips…'s backend: VISUALS, ExportFormat, ExportPlan,
│   │                     # estimate_export, export_cluster_clips (several looks and formats at once)
│   ├── clip_tiles.py     # §22 the INTERPRET montage's tile source over the preview cache:
│   │                     # TileCatalog (a reader per part), ClipTiles (one PreviewClip per tile,
│   │                     # grey when off-filter), missing (proxy vs masks), tracking_runs;
│   │                     # TileLook (what a tile draws, keypoint trail and trail points included),
│   │                     # the keypoints / axis read once per clip
│   ├── active.py         # runs in flight across every experiment (the Active runs mode)
│   ├── tools.py          # experiment-free Tools API (§9): list_tools / run_tool / list|read|delete_tool_run
│   ├── skeletons.py      # §26: the skeleton library (presets + sk_NN), dependents by live search
│   ├── keypoints.py      # §26: example sets (copied from the skeleton library), the keypoint launches —
│   │                     # module-level functions Experiment delegates to
│   ├── keypoint_distill.py  # §12 *Distill keypoints*: keypoint_dataset / distill_keypoints / register_student,
│   │                     # behind Experiment's one-liners; DISTILL_STAGES_BY_KIND lives in experiment.py
│   └── _runner_modules.py# shared PIPELINE_RUNNERS list (Experiment + Recording use it)
│   # plus one module per analysis family (query, temporal_clustering,
│   # measures, comparison, …)
├── cli.py                # click-based CLI; entry point `detail` (or `python -m detail`)
├── cli_keypoints.py      # §26: the `keypoints` command group, registered on cli.py's root
│                         # group — its own module so eight PRs never edit one region of cli.py
├── cli_skeletons.py      # §26: the machine-wide `skeleton` command group (the library)
├── __main__.py           # `python -m detail` dispatcher
└── gui/                  # PyQt6 GUI (docs/gui_design.md): app.py is the shell (mode banner,
    │                     # both rails, navigation); pages.py the content-area pages + `_table`;
    │                     # launch.py the one launch dialog; preview.py (prompt-test /
    │                     # render-preview); arena.py (scene geometry); streams.py (add-stream);
    │                     # run_preview.py (detection/tracking viewers, input-image scrubber,
    │                     # embedding PCA); readiness_guide.py + cuda_doctor.py (§1 onboarding);
    │                     # profile_describe_setup.py (§1's optional Cluster profile step: SHAP)
    │
    │  # The shared leaves the GUI Style rules above are written against — a rule names one of
    │  # these rather than describing the gesture, so a new surface inherits it:
    ├── requirements.py   # the checked structural requirements, and the generated guidelines page
    ├── dialog.py         # Dialog base: body() / set_buttons() — scroll + top-justify come free
    ├── run_info.py       # RunInfoWindow: any run's meta-information files, a tab each, + Export…
    ├── tables.py         # enable_sorting, make_columns_resizable, stamp_row_id/row_id,
    │                     # SortCell, refill_table_widget, MIN_VISIBLE_ROWS, max_visible_rows
    ├── sizing.py         # lines(), mono_font(), emphasis_font(), small_font(), fit_to_screen(),
    │                     # fit_list_height, remembered_size
    ├── scroll.py         # PageScroll + OWNS_WHEEL
    ├── spin.py           # IntSpin / FloatSpin / optional_spin / field_widget / build_fields
    ├── forms.py          # ConfigForm + AdvancedSection
    ├── plots.py          # bound_axes + the shared strips/lanes
    ├── plot_theme.py     # the pyqtgraph/QPainter adapter: themed_plot, pen/brush by role
    ├── collapse.py       # CollapsibleBand (fold + drag handle, remembered)
    ├── notes.py          # NotesBox (folded per page and view; the shell's Notes button opens it there) +
    │                     # place_notes, kept across a page's rebuilds (§15 Entry notes)
    ├── result_views.py   # ResultView / result_views / MultiViewWorkspace: an analysis run's
    │                     # views — a card each on its landing page, each full window (§16)
    ├── busy.py           # run_busy / run_progress / BusyOverlay / decode_overlay
    ├── menus.py          # open_once: a menu built per opening is deleted when it closes
    ├── workers.py        # GuiWorker (every thread names its receiver=) + BackgroundCall
    ├── open_external.py  # the sole caller of QDesktopServices.openUrl: open_url / open_local_path
    │                     # (WSL falls back to explorer.exe; else the link or path is shown)
    ├── ui_thread_gc.py   # the cyclic collector runs on the UI thread only (app + qapp fixture)
    ├── image_view.py     # the one image pane; `zoomable` has no default
    ├── example_set_picker.py  # §26: the skeleton + example-set picker every examples-needing
    │                     # model's surface embeds (New / Mark frames / Delete)
    ├── keypoints_manager.py  # §26: the Keypoints manager banner mode (the skeleton library)
    ├── skeleton_editor.py  # §26: SkeletonCanvas (on ArenaCanvas) + SkeletonEditor, the drawing half
    ├── keypoint_labeling.py  # §26: the labeling screen (a player + the example set), the
    │                     # open_labeling (the detect form's entry point)
    ├── heading_review.py  # §26 *Headings*: the coverage compass and the per-direction review
    │                     # window it opens (add, reject, next best guess)
    ├── keypoint_track_tune.py  # §26 *Tuning a track method*: the track form's Tune tracking…
    │                     # window (scores settings off-thread, hands one back to the form)
    ├── keypoint_track_preview.py  # §26 *Previewing a track method*: the track form's Preview
    │                     # tracking… window (settings side by side on short clips, no labels)
    ├── skeleton_points.py  # §26: SkeletonPointsPicker, a checkbox per point of any skeleton
    │                     # (the `ui: skeleton_points` fields, e.g. track_points)
    ├── keypoint_distill.py  # §12 *Distill keypoints*: the Distill keypoints… dialog (teacher runs read
    │                     # off the UI thread, label knobs off KeypointDatasetConfig)
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
    │  # INTERPRET for temporal clustering (§22); cluster_workspace.py holds the views
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
    ├── clip_popup.py     # ClipPopup: the window (behavior_query._ClipPlayer(panels=True) for the
    │                     # Advanced view); As recorded | Aligned, Export ▾, previous / next;
    │                     # open_clip_window: the one non-modal, one-per-page opener every page calls
    ├── clip_labelling.py # the labelling row (Label cluster, Fits / Doesn't / Unclear checks);
    │                     # PageLabelling / page_lane: a clustering page's cluster and lane
    ├── clip_cluster_info.py # Cluster details: what sets the cluster apart, Compare, Grid of 9 clips
    ├── clip_panes.py     # ClipPanes: the footage with the Masks and Keypoints panes beside it
    ├── clip_strips.py    # ClipStrips: one thin row per signal under the slider + the cluster lane
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
    │                     # hands its montage), PickBar, MontageExport (the writers, one copy;
    │                     # Export overlay…, New Query from N clips…)
    ├── export_clips.py   # MontageExportDialog: Export ▾ → Export clips… (looks, formats, sizes,
    │                     # preview, estimate), hosted by MontageExport
    ├── video_codec.py    # the MP4 codec asked the same way everywhere: mp4_filters, VideoCodecPicker,
    │                     # with_codec_note
    ├── show_menu.py      # Show ▾ (ShowButton): the layers (keypoint trail and Trail points
    │                     # included), the orientation and the context of a
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
    ├── lookalikes.py     # LookalikesWindow: look-alike groups beside the tree (non-modal),
    │                     # Label as one (LabelGroupDialog)
    ├── preview_jobs.py   # §3 the app's preview queue: one build at a time, status-bar bar
    ├── node_audit.py     # NodeAuditBlock: AUDIT's figure in the label bar, Act on it ▾
    ├── label_check.py    # clustering AUDIT: the list of labels and their clusters, the page
    ├── label_clean.py    # AUDIT's review pane: click / drag / Shift to remove, Move to…, Finish, reviewed box
    ├── audit_table.py    # audited_colour: the one colour INTERPRET rings a reviewed cluster in
    ├── label_browse.py   # clustering BROWSE: filters, ethogram, the labelled-instance list
    ├── importers/        # per-importer config bands (§3), keyed by importer_id
    │                     # (registry.py: ImporterUI + ConfigBand; csv_columns.py)
    ├── modules/          # per-module config + result UIs, keyed by module_id; registry.py also
    │                     # re-exports the result-view protocol (incl. §26's
    │                     # keypoint_detect.py and keypoint_track.py launch forms)
    │                     # (_heading.py: the one heading picker, §14 — HeadingPicker + the
    │                     # Define… dialog; the embedding form uses it, the audit will)
    └── tools/            # per-tool config form + result UI (§9), keyed by tool_id
                          # (registry.py: ToolUI + register_tool_ui; quick_segment.py)
```

Tests mirror this layout under `tests/`:
```
tests/
├── conftest.py           # shared fixtures — see Testing section
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
