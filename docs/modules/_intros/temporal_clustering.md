<!-- Prepended (prose only, no heading) to the generated modules/temporal_clustering.md page. -->
Behavior clustering builds one vocabulary of **behaviors** across a selection of recordings, with
no labels and nothing to point at. Two methods do the grouping. The default, `codebook_viterbi`,
clusters a sample of frames into a codebook and decodes every frame against it, so its cost grows
in proportion to the video you pool. `dtw_finch` cuts each subject's trajectory into **bouts** at
the moments the pose changes, compares every pair of bouts, and groups them; it draws slightly
better boundaries but its cost grows with the square of the bout count, so it suits short studies.
Either way the result is a vocabulary you look at and name.

It answers "what does this animal *do*?": a cluster is a stretch of movement, and an instance is
one performance of it with a real start and end.

**What the hierarchy is built on.** The clusters form a **hierarchy**, a nested series of
groupings from many small ones to a few large ones, and the granularity control moves between its
levels. It is derived from **distance and nothing else**: for `dtw_finch`, the
distance between bouts, compared with a warp tolerance so that a slow and a fast performance of the
same behavior can still match; for `codebook_viterbi`, the distance between the frames grouped into
its codewords. It is not derived from how often a behavior occurs, when in the recording it
happens, where in the arena it happens, or any label you have given it. `dtw_finch`'s levels need
no threshold and no target count: it links each bout to its single nearest neighbour, takes the
connected groups of that graph, and repeats on the groups, so the levels are a consequence of the
data rather than a number anyone chose. That is why the granularity control offers particular
counts (its *rungs*) rather than a free slider, and why a name given at one rung still means
something at a finer one.

The fit is one job, and it leaves the run **open** rather than complete: a vocabulary is not a
result until somebody has looked at it. Labelling the clusters that deserve a name, anywhere in
the tree the fit found, is quick and reversible, and is done in the GUI's interpretation workspace
or from the CLI with `analysis temporal-interpret`.

**For `dtw_finch`, the comparison is the cost, and it grows with the square of the bout count.** Comparing every
pair is what lets a slow and a fast performance of the same behavior match, and it is why a fit
over a long study is expensive. **Max bouts** bounds it by evenly sampling the bouts the comparison
sees.

**A cap changes what the vocabulary is learned from, not what gets labeled.** Every remaining bout
is matched to the nearest cluster afterwards, so the output still carries one instance per bout
however low you set it. That distinction is the point of the module: bounding what it *learns
from* is a quality trade, while bounding what it *labels* would just be missing data, and finding
every occurrence is the job. The interpretation screen states which happened, since a capped fit is
otherwise indistinguishable from a full one.

What a cap does cost is what a *clustered* bout has and an *assigned* one does not: a say in what
the clusters are. So a node's most and least typical clips are ranked from clustered bouts only,
and the run records both counts.

**Coverage is partial, and that is correct.** Bouts, and the codebook's runs, do not tile a recording: stretches shorter than
the minimum bout length (or the codebook's minimum run) are dropped, and gaps in tracking are
not bouts. A run reporting that its
instances cover half the observed time is telling you how much of the recording the vocabulary
actually accounts for, not reporting a fault.
