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

**A label with labels inside it counts only what they leave.** Labels nest, and each bout counts
under its innermost label. So when you name a cluster "moving" and one of its branches "fast", the
row for "moving" counts only the moving that is not "fast", and it reads **moving (other)**. The
numbers are not added up for you: to get all of "moving", add "moving (other)" and "fast". In
`measures.csv` and a comparison's `stats.csv`, `label` keeps the plain name (`moving`), so files
still join on it, `label_display` holds the name shown (`moving (other)`), and
`label_is_remainder` is `True` on those rows.

**Time in unnamed clusters** is measured on every clustering run: the time no label of yours
covers yet, which is all of it before you name anything. A query run names every behavior, so it
has none.
