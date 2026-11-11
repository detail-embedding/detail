<!-- Prepended (prose only, no heading) to the generated modules/codebook_viterbi.md page. -->
The default way of finding a behavior vocabulary, chosen because its cost stays in proportion to
the study as it grows.

Instead of cutting the trajectory into bouts and comparing every pair, it clusters a **sample of
frames** into a codebook, labels every frame against it, and then recovers the stretches with a
**decode**: an HMM whose states are the codewords, with the transition probabilities estimated
from your own data rather than set by hand.

**Why it scales differently.** Both methods have one expensive step. `dtw_finch`'s is comparing
bouts, and the number of bouts grows with the study, so a longer study costs more. This one's is
the decode, which costs roughly the *square* of **Codewords** on every frame of every recording you
pooled. **Codewords** is a number you set, so the expensive part grows with your study only in
proportion, not with its square. It runs at the finest codebook size whatever level of the tree you
label afterwards, so labelling coarsely does not make it cheaper. If a fit is taking longer than you
expect, lower **Codewords** first.

**Share of each recording** is how many of each video's frames the codebook is built from, evenly
spaced. Every recording is sampled at the same density, so a behavior one animal shows briefly is
as findable in a twenty-recording study as in a two-recording one. Raising it costs the codebook
fit, not the decode.

The codebook is built in two steps: the sample is grouped into **Codewords** small groups, and a tree is built over the groups so every coarser setting is a union of finer ones. The aim is a vocabulary that matches a fixed-size codebook against human labels while two fits on slightly different samples agree more with each other.

**What you give up.** Compared with the bout path, this one draws less precise boundaries: it gets the identity of a stretch right but is fuzzier about exactly where it starts and stops. If you care most about boundaries, use `dtw_finch`.

**The decode is doing real work.** Without it a frame labeling flickers: adjacent frames of one
behavior land in different codewords, and encoding that raw produces thousands of one-frame
"behaviors" rather than the handful of stretches actually there. Suppressing that is the single
most replicated effect in this project's measurements.

**Minimum run** is the floor below which a stretch is not an instance: the last defence against
flicker the decode did not catch. **Codewords** is the cost knob.
