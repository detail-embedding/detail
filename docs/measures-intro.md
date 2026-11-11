# Measures

A **measure** turns a behavior result into one number per recording, subject, behavior and time
window: how many bouts, how much total time, how long a bout runs, how long until the first one.
They are what a [comparison](concepts.md#comparison) contrasts between groups, and what
`measures export` writes to CSV.

Every value is exact. The artifact stores each bout as an interval, so a windowed question is
answered by intersecting intervals rather than by binning, so there is no bin width to choose and no
resolution to lose. Durations are stored in nanoseconds and shown in seconds.

Left unset, a comparison computes every measure the chosen behavior result *can* answer. A measure
that does not move is as much a result as one that does.
