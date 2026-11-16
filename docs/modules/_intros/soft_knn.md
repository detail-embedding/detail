Scores **every frame** of the trajectory by how its nearest neighbours vote, then turns the
score's peaks into instances.

Each frame is compared against a labelled set: the frames of your example clips, plus a
**background** the run draws from the rest of the recordings. Its score is a weighted vote over
its `k` closest labelled frames, close to your examples and far from the background scores
high, close to both lands in between. Because every example clip is pooled into that one set,
adding more clips costs almost nothing, where a matcher that aligns each clip in turn pays for
every clip you add.

An instance is then formed from each peak in that score, growing outwards until the score falls
to a fraction of the peak's own height. So a match is as long as the behavior looks, not as long
as the clip you marked.

## What is known about it

The scoring is the best-tested part of this matcher. Several example clips give steadier results than one, and a very short clip makes a poor example. A behavior the encoder cannot see, such as freezing, which is defined by *absence* of motion, is hard for it to find, and extra examples do not rescue it.

The way instances are formed from the score was chosen by comparing alternatives on held-out examples rather than by taste. Smoothing the score before forming anything is what makes it usable at all: raw, it crosses any threshold constantly, and a single true instance comes out as many fragments. Growing each match to half its own peak sized matches more closely than the alternatives tried.

## What is not

**Where instances end is the weak part.** Most true occurrences are *found* (a returned match overlaps them), but far fewer match at the stricter standard of overlapping most of their extent. It finds the moment and gets the edges wrong, which matters if you are measuring durations and matters much less if you are counting occurrences or looking at what the animal was doing.

**Nothing here has been compared head-to-head against another matcher.** No alternative has been scored against it on the same behaviors, so whether a different approach would be more accurate is genuinely open. What is known is speed: a search is quick, and adding example clips barely changes that.

**A confidence is not a precision.** The score is calibrated against the background, not against
how often it is right. 0.8 does not mean 80% of those matches are real, only looking at them
can tell you that.

## The background, and why it contains the behavior

The background is drawn from the searched recordings with every clip you marked in the run
removed (each behavior's examples and non-examples) and **nothing else** removed. That means it contains the behavior you are looking for, in whatever
proportion it actually occurs, which lowers confidences on a behavior that fills a lot of the
recording. That is deliberate. A background filtered to exclude anything that resembles the
behavior would need to already know where the behavior is, which is the question you are asking;
scores built that way looked far better and could not be reproduced by anyone actually using the
tool. The draw is recorded (the strategy, the size, the seed and the exact rows) so a result
can be traced back to the frames that calibrated it.
