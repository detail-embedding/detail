<!-- Prepended (prose only, no heading) to the generated modules/comparison.md page. -->
A comparison asks whether a behavior differs between groups of recordings. It takes one behavior
result (a behavior scan or a behavior clustering run), groups the recordings by a column of your
recording information (treatment, sex, session), computes the [measures](../measures.md), and
reads the groups side by side.

**You choose the column; the tool picks the design.** A column that is the same for all of an
animal's recordings compares *between* animals. One that changes within an animal compares
*within* it: two values are a paired contrast, three or more numeric values a trend, three or
more named values a profile. Name two **Contrast levels** to compare just those two. Every result
says which design ran.

**Say what one observation is.** When one animal contributed several recordings, averaging each
animal first and counting every recording separately give different answers, so **Aggregate to**
is yours to set and the tool does not guess.

**It stops at description.** Means, medians, spreads, and a permutation floor: how big a
difference chance would produce, from **Floor draws** shuffles of the kind the design allows
(group labels across animals, or each animal's own direction or levels). The only p-values are
the floor's own. `stats.csv`, one row per observation, is the handoff to whatever model you test
with.

**Every cluster, named or not.** With **Measure** set to every cluster of a clustering run, each
node of its tree with at least **Candidate floor** of member time is ranked by how its share of
time differs, nested slices of one behavior are collapsed to the best of them, and the list says
where to look. Naming a cluster is still yours to do, from its clips.

**The source is pinned.** A comparison reads a snapshot of its behavior result, so its numbers
keep meaning what they meant after the run is reopened and re-cut.
