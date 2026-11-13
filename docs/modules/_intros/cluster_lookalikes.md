Find look-alike clusters looks for clusters in different branches of one pose- or
behavior-clustering run that are the same behaviour. A clustering tree often splits one behaviour
by where the animal is or which way it faces: rearing toward the back and rearing toward the right
can sit far apart in the tree. This job compares pairs of clusters on what the body does, with
position and facing left out. Two clusters the body measures cannot tell apart look alike.

It compares the run's branch tops: the two halves of every split where both are at least the
run's minimum cluster time. Pairs to compare chooses which: every pair of them (all, the
default), or each labelled cluster against every other (labelled), which is far fewer pairs and
finds unlabelled clusters that look like ones you have named. Two clusters under one label are not
compared. Every pair takes many hours on a large run, so starting it asks first, with how many
pairs each choice compares; you can also run it on a remote target (Run on) instead of this
computer. Its comparisons are copied back and placed beside the run when it finishes.

Each pair is a cluster profile of one cluster against the other, saved beside the clustering run
exactly as if you had compared them in the Cluster profile window. A pair compared by hand counts
the same, and a pair already compared with the same settings is skipped, so it is safe to stop
and run again.

The groups are shown on the run's INTERPRET, under Cluster tools ▾ › Look-alike clusters…. Pairs that look alike
chain into groups: if A looks like B and B looks like C, all three are one group. Each group
proposes a name: the label most of its clusters already carry, or none, and then you give it one.
Nothing is labelled until you choose Label as one, and a cluster that already has a
different label of yours keeps it.

Clusters sit inside one another, and a label covers everything under its cluster that has no
label of its own. So a look-alike cluster can hold clusters that are a different behaviour: a
rearing cluster holding a walking one. Each such cluster says which clusters its label would
also take, and starts unticked. Label those first, and they are no longer taken.
