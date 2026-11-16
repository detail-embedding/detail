<!-- Prepended (prose only, no heading) to the generated modules/dtw_finch.md page. -->
The bout path to a behavior vocabulary. It cuts each subject's trajectory into **bouts** at the
moments the pose changes, compares every pair of bouts, and groups the ones that resemble each
other. A cluster is a kind of bout, and every bout is an instance with a real start and end.

**Bouts first.** A sliding **Window** pools the per-frame embedding to smooth out jitter, and the
**Detector** places the cuts: `binseg` by default, which cut better than `novelty` in internal
measurements. Bouts shorter than **Min bout length** are dropped before anything is compared, so
this method needs no separate minimum-run floor. A break in tracking longer than **Max tracking
gap** is a boundary, never the inside of a bout.

**Compared with a warp tolerance.** Each pair of bouts is compared by dynamic time warping, and
**DTW band** sets how far one may be stretched against the other. That is what lets a slow and a
fast performance of the same behavior match. A wider band merges more bouts into fewer, coarser
clusters.

**Grouped without a target count.** FINCH links each bout to its single nearest neighbour, takes
the connected groups, and repeats on the groups. The tree that comes out is the data's, not a
number anyone chose, and you label its nodes afterwards.

**The comparison is the cost, and it grows with the square of the bout count.** A long study has
many bouts, so a fit over one is slow. **Max bouts** bounds it by evenly sampling the bouts the
comparison sees; every other bout is then matched to the nearest cluster, so a cap changes what
the vocabulary is learned from, not what gets labelled. The interpretation screen says when a fit
was capped. For a long study, [`codebook_viterbi`](codebook_viterbi.md) keeps the cost in
proportion instead.

**What you get for it.** Measured against the codebook it ties on vocabulary quality and draws
slightly better boundaries.
