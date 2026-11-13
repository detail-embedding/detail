<!-- Prepended (prose only, no heading) to the generated modules/location_export.md page. -->
Location export turns a tracking run into per-frame spatial features for each subject, relative to
the recording's regions and points of interest. It writes CSV files keyed by time, so you can join
them with your event streams. It also writes the distance between every pair of tracks, and,
given a keypoint track run, every animal's points. A finished run draws paths, occupancy heat
maps, speed over time and the distance between tracks, per recording or by group
(`locations plot`, or the run's **Plots** tab), and plays its recordings side by side as a
montage (`locations montage`, or **Montage…**).
