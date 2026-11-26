# 7. Behavior scan

A **behavior scan** finds a behavior you can already name, such as rearing or grooming, in
every video. You show it a few **example clips** of each behavior. It then looks through
every frame of every recording for stretches whose embedding (chapter 5) is like your
examples, and gives each a **confidence** from 0 to 1. You choose a **cut-off**: matches
above it count as the behavior. Detail then counts bouts, time spent and so on, per
recording.

No model is trained. A scan takes seconds, and changing the cut-off afterwards costs
nothing: the scan is not run again.

## The steps at a glance

Each box is one step on this page. Click a box to go to that step. The first tag says where
the step is in Detail.

<div class="guide-flow" markdown>

1. [Start a behavior scan](#1-start-a-behavior-scan) <span class="where where-app">Experiments › Analysis › New analysis</span> <span class="note">Choose the recordings and the embedding.</span>
2. [Mark example clips](#2-mark-example-clips) <span class="where where-app">Experiments › Analysis › New analysis › Mark examples</span> <span class="note">About 10 minutes: three or four clips per behavior.</span>
3. [Run the scan](#3-run-the-scan) <span class="where where-app">Experiments › Analysis › New analysis</span> <span class="note">Under a minute, on the processor.</span>
4. [Choose the cut-off](#4-choose-the-cut-off) <span class="where where-app">Experiments › Analysis › the run › Calibrate</span> <span class="note">Watch the clips either side of the line, move it, and keep it.</span>
5. [Check the matches](#5-check-the-matches) <span class="where where-app">Experiments › Analysis › the run › Browse, Audit</span> <span class="note">Look through what was kept, and measure how often it is right.</span>
6. [Read the measures](#6-read-the-measures) <span class="where where-app">Experiments › Analysis › the run › Measures</span> <span class="note">Bouts, rate and time, per recording.</span>

</div>

!!! note "The examples and cut-offs in this chapter are ours"
    The demo's example clips were picked for this guide by eye from the top-down videos,
    and its cut-offs (0.5 for both behaviors) were chosen by looking at matches either side
    of the line. They show how the steps work. They have not been checked against careful
    scoring, so the numbers on this page are not findings about these mice.

## 1. Start a behavior scan

Click **Analysis** in the list on the left, then **New analysis**.

![The experiment's Analysis page with three analysis runs listed, and the New analysis button highlighted. Numbered markers sit on every control.](../../assets/user-guides/single-animal/behavior-scan/analysis-page.png)

??? info "What everything on this screen does"
    1. **New analysis**: starts a behavior scan, a clustering, or another analysis.
    2. **The runs**: every analysis in the experiment, with its module, recordings and
       status. Double-click a row, or click **›**, to open it. A behavior scan's status
       stays **open** while you are still choosing its cut-offs.

In the **Launch an analysis run** dialog:

1. **Module:** **Behavior scan**.
2. Click **Select all**.
3. **Embed run:** the embedding from chapter 5, here **run_005**.
4. Click **Mark examples…** (step 2).

![The Launch an analysis run dialog for a behavior scan: module Behavior scan, all three recordings ticked, the embed run run_005, subject 0, the Behaviors table with rearing (4 examples) and grooming (3 examples), and Mark examples highlighted. Numbered markers sit on every control.](../../assets/user-guides/single-animal/behavior-scan/launch-behavior-scan.png)

??? info "What everything on this screen does"
    1. **Module**: what to run, here **Behavior scan**.
    2. **Scrollbar**: scrolls the dialog.
    3. **Start from**: fills the form from a previous run's settings, examples included.
    4. **Recordings**: the recordings to search. They must all be in the embed run.
    5. **Select all**: ticks every recording.
    6. **Clear**: unticks them.
    7. **Embed run**: which embedding to search.
    8. **Subject**: which tracked animal to search. One animal is **0**.
    9. **Behaviors**: each behavior's name, how many example and non-example clips it
       has, how they spread over recordings and animals, and their total time. Empty until
       you mark examples.
    10. **Mark examples…**: opens the video to mark each behavior's clips (step 2).
    11. **Min match length**: the shortest match to return, in seconds. **off** returns
        every length.
    12. **Smoothing**: how much the score is smoothed over time before matches are formed,
        as a share of the example clips' length. Higher joins brief matches into one.
    13. **Extent fraction**: where a match ends: it grows out from its best frame while the
        score stays above this share of the best. Lower gives longer matches.
    14. **Max duration**: the longest match, as a multiple of the example clips' length.
    15. **Advanced Settings**: opens 4 more settings, shown below.
    16. **Run**: starts the scan (step 3).
    17. **Cancel**: closes the dialog without starting anything.
    18. **?**: a short explanation of the setting beside it.

    An orange line under the table, when there is one, is advice, not an error: for
    example, that a clip is under 2 seconds, since longer clips give a steadier result.

![The behavior scan dialog's Advanced Settings: method soft_knn, neighbours 20, temperature 0.07, and trajectory split gap auto. Numbered markers sit on each setting.](../../assets/user-guides/single-animal/behavior-scan/launch-behavior-scan-advanced.png)

??? info "What the Advanced Settings do"
    1. **Method**: how frames are scored. **soft_knn** compares each frame with its 20
       closest frames among your examples and a random sample of the rest. Leave it.
    2. **Neighbours (k)**: how many of those closest frames vote on each frame's score.
    3. **Temperature**: lower lets the very closest frames count for more.
    4. **Trajectory split gap**: a break in tracking longer than this splits a recording
       into separate stretches, and an example clip has to sit inside one. **auto** is
       about 4 frames.

## 2. Mark example clips

The **Mark examples** window opens on the first recording. For each behavior:

1. Under the video, type the behavior's name, such as `rearing`, and press ++enter++ (or
   click **Add behavior**). The box then says **Annotating: rearing**. Until a behavior has a
   name, **Mark start** and **Mark end** are greyed.
2. Find a clear example in the video: play it, or drag the slider.
3. At the first frame of the behavior, click **Mark start**. At the last, click **Mark
   end**. The clip appears in the list on the right, and as a coloured mark on the slider.
4. Mark two or three more, from **other recordings** if you can: choose them in
   **Recording** at the top left.

Then click **Add** on the right for the next behavior, such as `grooming`, and do the same.
When every behavior has a tick in **Examples**, click **Use these clips**. It stays greyed
until at least one behavior has an example, and the line above it says what is missing.

![The Mark examples window: box3 with the mouse outlined, the slider with the clips marked on it, the behaviors rearing with 4 examples and grooming with 3, each ticked, and the list of the seven clips with their times and recordings. Numbered markers sit on every control.](../../assets/user-guides/single-animal/behavior-scan/mark-examples.png)

??? info "What everything on this screen does"
    1. **Recording**: which recording to show, and how many clips it has already.
    2. **Track**: which tracked animal. One animal is track **0**.
    3. **Behaviors**: the behaviors, with a bar toward three examples that turns into a tick
       at three, and how many recordings and animals they come from. Click one to mark
       clips for it.
    4. **Add**: adds a behavior.
    5. **Rename**: renames the chosen behavior.
    6. **Remove**: removes the chosen behavior and its clips.
    7. **Balance samples across**: pick a Recording information column, such as **drug**,
       and the line under it names the first gap: for example, every clip from one
       recording, or none from one value of the column.
    8. **Clips**: every clip you have marked, with its behavior, kind, time and recording.
       Click one to go to it.
    9. **The slider**: moves through the video. Marked clips are drawn on it in their
       behavior's colour. The **tracked** strip under it shades where the animal was
       tracked.
    10. **Play**: plays the video.
    11. **Speed**: how fast **Play** plays.
    12. **Prev frame**: steps one frame back.
    13. **Next frame**: steps one frame forward.
    14. **Masks**: tints the tracked animal's outline on the video. Untick it to play
        faster.
    15. **Frame number**: the frame shown. Type a number and press ++enter++ to go there.
    16. **example**: the next clip marked is an example of the behavior.
    17. **non-example**: the next clip marked is something that is **not** the behavior
        but looks like it. Optional; it makes the scan stricter.
    18. **Mark start**: the clip starts at this frame. The button stays green until you
        mark the end.
    19. **Mark end**: the clip ends at this frame.
    20. **Delete clip**: removes the chosen clip. The ++delete++ key does the same.
    21. **Use these clips**: closes the window and fills the **Behaviors** table. Greyed until
        at least one behavior has an example.
    22. **Cancel**: closes the window and forgets every change made in it.

The scan can only find what your clips show it, so the clips you choose matter more than any
setting in this chapter.

### Choosing examples

An **example** is a clip of the behavior. The scan scores every other frame by how much it
looks like your example frames, so:

- **Three or more per behavior, each 2 seconds or longer.** One works, but more and longer
  clips steady the result. The **Examples** column shows a bar that turns into a tick at
  three.
- **From several recordings, animals and groups.** Clips all from one animal teach the scan
  that animal as much as the behavior. Choose a column such as **drug** in **Balance samples
  across**, and the line under it names what is missing, such as no example from one group.
- **The behavior and nothing else.** Mark start where the behavior starts and end where it
  ends: the frames either side, the animal walking up to the wall, say, teach the scan
  something else.
- **The whole range of the behavior.** A rear against the wall and a rear in the open, a
  short groom and a long one. Clips that are all alike find only that one version.

### Choosing non-examples

A **non-example** is a clip of something that is **not** the behavior but looks like it: the
mistakes you expect the scan to make. To mark one, choose **non-example** under the video
before you click **Mark start**; it is drawn as an outline on the slider rather than filled.

The scan compares every frame with your examples and with a random sample of the rest of
the video. Non-examples are added to that sample, labelled as not the behavior, so the scan
learns where the line between the two lies. Choose them for exactly the confusions that
matter:

- **The look-alike behavior.** From above, a mouse rearing and a mouse grooming are both a
  small, round outline. Mark grooming clips as non-examples of rearing, and rearing clips as
  non-examples of grooming.
- **The near miss.** The animal pausing at the wall without rearing; sitting still without
  grooming.
- **After a first scan,** the wrong matches you find in **Browse** (step 5). Mark a few of
  them as non-examples and run the scan again: that is the quickest way to make it stricter
  where it was wrong.

Non-examples are optional, and a few well-chosen ones do more than many random ones. A side
camera tells some look-alikes apart with no non-examples at all (see [Side-view and isometric
recordings](keypoints.md#side-view-and-isometric-recordings) in chapter 4).

## 3. Run the scan

Click **Run**. The run's page opens; the scan of the three demo videos took a few
seconds. Its **Status** stays **open**: a behavior scan is finished only when you say so,
after choosing the cut-offs.

![The behavior scan's page: an_02, module query, status open, the three recordings, and under Results the four cards Calibrate, Audit, Browse and Measures, then Open output folder, Delete analysis and Start new analysis. Numbered markers sit on every control.](../../assets/user-guides/single-animal/behavior-scan/scan-run.png)

??? info "What everything on this screen does"
    1. **Calibrate ›**: choose each behavior's cut-off (step 4). It says how many behaviors
       the scan has.
    2. **Audit ›**: measure how often a kept match is really the behavior (step 5).
    3. **Browse ›**: look through every kept match (step 5).
    4. **Measures ›**: bouts, rate and time, per recording (step 6).
    5. **Open output folder**: opens the folder that holds the run's files.
    6. **Delete analysis**: deletes the run and its files, after asking.
    7. **Start new analysis**: opens **New analysis** again.

Click a card to open it: it fills the window. At its top, **Behavior** picks the behavior
to work on, here **rearing** or **grooming**, and under it are the four views as tabs, to
switch between them. To go back to the run's page, click its id, here **an_02**, in the line at
the top of the window.

## 4. Choose the cut-off

The scan does not decide what is the behavior. It gives every stretch of every video a
**confidence**, from 0 to 1, of how much it looks like your examples, and it keeps them all.
The **cut-off** is where you draw the line: matches above it count as the behavior, and
everything the later steps report, bouts, time and rate, is counted from them.

There is no right cut-off that Detail could choose for you. A confidence says how alike two
stretches look, not whether the animal is rearing, and how alike is alike enough depends on
the behavior, the camera and your examples. Too low a cut-off counts things that are not the
behavior; too high, and real bouts are missed. Only you can tell which, by watching the clips
either side of the line. That is what this step is for: looking at the matches where the
scan is least sure, and putting the line where they stop being the behavior.

Moving the cut-off costs nothing: the scan is not run again, so you can try several. One
cut-off applies to every recording in the scan, which is what keeps them comparable, so
choose it before reading the measures, and report it with them: it is part of the result.

The **Calibrate** view is where the cut-off is set, one behavior at a time.

![The Calibrate view for rearing: the line saying the cut-off 0.5 is saved and keeps 59 of 360 clips; the curve of clips kept at each cut-off, with 0.5 marked; the cut-off slider and the clips kept per recording; and under them the grid of clips, most confident first, each outlined, the first few tagged your example. Numbered markers sit on every control.](../../assets/user-guides/single-animal/behavior-scan/calibrate.png)

??? info "What everything on this screen does"
    The line at the top counts the clips kept at the cut-off: here 59 of 360 rearing clips
    at 0.5. The curve on the left shows how many would be kept at every cut-off: left is
    loose, right is strict. **Kept per recording** shows how many each recording keeps.
    Double-click a clip to play it in a window of its own.

    1. **Behavior**: which behavior's cut-off to set.
    2. **Keep clips from 0.5 up**: saves the cut-off shown. Greyed out while it is already
       saved.
    3. **Cut-off**: the cut-off, to type in. Beside it, how many clips it keeps.
    4. **Cut-off slider**: moves the cut-off. The curve's line, the counts and the line across
       the clips follow.
    5. **Show ▾**: what the clips show: the video, the animal's outline, a trail of where it
       has been.
    6. **Size**: how big the clips are.
    7. **Export ▾**: saves the clips.
    8. **Scrollbar**: scrolls the clips.
    9. **Finish**: says you are done with the run. It can be reopened. Greyed out until
       every behavior has a cut-off.
    10. **Reopen**: opens a finished run again, to change a cut-off. Greyed out while the
        run is open.

Every match plays at once, in a grid, most confident first. The cut-off is a dashed line
across the grid, labelled with how many are kept and dropped: the clips above it are kept,
the ones below it dropped. Scroll past the line and a bar over the grid says the clips in view
are dropped, with **↑ Back to the cutoff**. The first clips are usually your own examples,
which the scan finds too; they are tagged *your example*.

To choose the cut-off:

1. Watch the last clips before the line, the least confident ones kept. Are they the
   behavior?
2. Watch the first clips after it, the most confident ones dropped. Are they?
3. If the kept clips near the line are wrong, move the cut-off up: drag the line on the
   curve, move the slider, or **click the clip** you want to be the last one kept. If the
   dropped clips are right, move it down. The grid and every count follow as you move it.
4. Click **Keep clips from 0.5 up**. Moving the cut-off only shows what it would keep;
   this button saves it. Until you click it, an amber band above the curve says **⚠ Not
   saved**, and Audit, Browse and Measures say so too, still using the saved cut-off; after
   it, the band goes and the line reads **✓ rearing: cut-off 0.5 saved**.
5. Do the same for each behavior, chosen under **Behavior**.

??? info "About this step: where the kept matches fall"
    **Browse** draws an ethogram of where in each video the kept matches fall, and
    **Measures** draws where in the box each one happened. A cut-off that keeps a behavior
    only in the first minute of every video, or only in one corner of one box, is worth a
    second look. On the demo, the note under the locations says all of box1's rearing
    matches are in the top third of the box: that mouse rears against one wall, which may be
    true, and is worth checking in the clips.

??? warning "If this doesn't work"
    - **No cut-off keeps the behavior without also keeping something else:** the examples
      are too like the other behavior. Add non-examples (step 2) of what is being wrongly
      kept, and run the scan again.
    - **One recording keeps far fewer matches than the rest:** its lighting or camera may
      differ. Mark an example from that recording and run again.

## 5. Check the matches

**Browse** shows every match each behavior's cut-off keeps: an ethogram across the top, one
row per recording, each behavior in its own colour and pattern, keyed by the swatch beside its
name under **Behaviors**, and below it the list of matches by recording and time, beside a player
that opens on the first one. Click a mark on the ethogram, or a row, to play it. Untick
behaviors, recordings or a metadata group on the left to narrow both. Read the list in order:
a mistake the scan makes again and again, such as a mouse leaning on one wall counted as
rearing, stands out in a list.

![The Browse view for rearing at cut-off 0.5: 113 kept matches; the behaviors, recordings and metadata filters on the left; the ethogram of box1 to box3 across the top; the list of matches with behavior, recording, start, duration and confidence; and the player beside it on the first one. Numbered markers sit on every control.](../../assets/user-guides/single-animal/behavior-scan/browse.png)

??? info "What everything on this screen does"
    1. **Behavior**: which behavior's cut-off is shown.
    2. **Clear filters**: ticks every behavior, recording and group again.
    3. **Ethogram**: folds the ethogram away or opens it again: one row per recording, each
       kept match marked in its behavior's colour. Click a mark to play it; scroll over it
       to zoom.
    4. **Behaviors**: each behavior, with its matches and time. Untick one to leave it out.
    5. **Search recordings**: narrows the recordings below to those whose name has the text.
    6. **Recordings**: untick a recording to leave it out.
    7. **The matches**: every kept match, with its behavior, recording, start, duration and
       confidence. Click a heading to sort by it; click a row to play it.
    8. **Metadata**: shows one group of recordings, such as one drug, or **Any group**.
    9. **The slider**: moves through the match playing.
    10. **Play**: plays the match.
    11. **Speed**: how fast **Play** plays.
    12. **Prev frame**: steps one frame back.
    13. **Next frame**: steps one frame forward.
    14. **Export video…**: saves the match as a video file.
    15. **Context**: plays a few seconds either side too. **off** plays the match alone.
    16. **Masks**: tints the animal's outline.
    17. **Frame number**: the frame shown.
    18. **Export 1 selected**: saves the matches selected in the list.
    19. **Export all 113**: saves every match in the list.
    20. **Finish**: says you are done with the run. It can be reopened. Greyed out until
        every behavior has a cut-off.
    21. **Reopen**: opens a finished run again, to change a cut-off. Greyed out while the
        run is open.

**Audit** puts a number on how often the kept matches are really the behavior: the
**precision**. You watch **every match the cut-off keeps**, one recording at a time, and take
out the ones that are not the behavior. Audit needs a cut-off: until a behavior has one,
Audit shows only *rearing has no cutoff yet* and a **Go to Calibrate** button.

1. Click a recording in the list on the left. Its matches play side by side, in a random
   order and **without** their confidence.
2. **Click a match that is not the behavior** to take it out. Drag across several, or hold
   ++shift++, to take out a run of them. ++ctrl+z++ undoes.
3. When you have scrolled to the last match, click **✓ Finish review**. The recording then
   shows its precision, such as **88% · 7 of 8 are rearing · 1 removed**.

Do this for each recording. The behavior's precision appears above the list once every
recording is reviewed.

![The Audit view for rearing before any review: 0 of 3 recordings reviewed at 0.50, with box1, box2 and box3 each listed with its clips and not reviewed, and the grid empty until a recording is chosen. Numbered markers sit on every control.](../../assets/user-guides/single-animal/behavior-scan/audit.png)

??? info "What everything on this screen does"
    1. **Behavior**: which behavior to review.
    2. **The recordings**: each recording, with how many clips its cut-off keeps and how far
       its review has got. Click one to review it.
    3. **✓ Finish review**: says you have looked at every clip of the recording. Greyed out
       until the last clip has been on screen.
    4. **Undo (⌘Z)**: puts back the last clip you took out (++ctrl+z++).
    5. **Show ▾**: what the clips show: the video, the animal's outline, a trail of where it
       has been.
    6. **Export ▾**: saves the clips.
    7. **Size**: how big the clips are.
    8. **Finish**: says you are done with the run. It can be reopened. Greyed out until
       every behavior has a cut-off.
    9. **Reopen**: opens a finished run again, to change a cut-off. Greyed out while the
       run is open.

    A match you take out leaves the behavior's matches: **Browse**, the measures and every
    export leave it out. Raise the cut-off afterwards and the review still stands; lower it
    and the new matches are not reviewed yet, so the recording says how far down it was
    reviewed until you review it again.

??? info "About this step: why audit"
    The cut-off you chose by eye is a judgment. The review is what turns it into a figure
    you can report: "of the matches kept, 88% were rearing", counted over every match rather
    than estimated from a sample. It does not say how many
    rearing bouts the scan **missed**; that needs the behavior scored by hand through the
    whole video.

    The demo run in this guide has not been audited.

## 6. Read the measures

**Measures** gives, for each recording and behavior, the number of bouts, the rate, the
total time, the share of time, the bout lengths, the time to the first bout, and the gap
between bouts.

![The Measures view: the table of measures by recording, subject, label, measure, value, unit and n; the locations of each kept rearing match, one square per box, with the note that all of the first recording's fall in the top third; the player beside them; and the Interpretation section with Measure this run. Numbered markers sit on every control.](../../assets/user-guides/single-animal/behavior-scan/measures.png)

??? info "What everything on this screen does"
    1. **Behavior**: which behavior the locations and the interpretation below are for.
    2. **The measures**: one row per recording, behavior and measure, with its value, unit
       and how many bouts it comes from (**n**). Click a heading to sort.
    3. **Locations**: folds away or opens one square per recording, with a dot where each
       kept match happened. Click a dot to play it beside them.
    4. **The slider**: moves through the match playing.
    5. **Play**: plays the match.
    6. **Speed**: how fast **Play** plays.
    7. **Prev frame**: steps one frame back.
    8. **Next frame**: steps one frame forward.
    9. **Export video…**: saves the match as a video file.
    10. **Context**: plays a few seconds either side too. **off** plays the match alone.
    11. **Masks**: tints the animal's outline.
    12. **Frame number**: the frame shown.
    13. **Interpretation**: folds away or opens the measures of the kept matches, such as how
        fast the animal moved during them.
    14. **Measure this run**: works those measures out. It reads every match's tracking,
        which can take a minute on a long run.
    15. **Finish**: says you are done with the run. It can be reopened. Greyed out until
        every behavior has a cut-off.
    16. **Reopen**: opens a finished run again, to change a cut-off. Greyed out while the
        run is open.

    The **Export…** button at the top of the window saves every kept match with its times
    (**Behavior instances**), the ethograms, the per-frame behavior labels, or the clips, for
    your own analysis in a spreadsheet, R or Python.

??? warning "If the numbers look too large"
    When two behaviors look alike from above, such as rearing and grooming, the scan can
    count many of the same stretches as both, and each takes much more of the time than
    the animals really spend (see step 2). Raising both cut-offs, adding non-examples, and auditing are how you bring them
    down to what is true. A side camera tells them apart far better.

When every behavior's cut-off is chosen and checked, click **Finish**. It stays greyed out
while a behavior has no cut-off, and the line beside it names that behavior.

??? info "About this step: Snapshots"
    After you click **Finish**, export something, or use the run in a comparison, a
    **Snapshots** table appears on the run's page. Each snapshot is a frozen copy of the
    run's results at the moment a number left it. You can reopen a finished run and change
    its cut-offs, and a number you already reported still has the snapshot it came from.

    Double-click a snapshot to see the run as it was then. The page shows a banner, **Viewing
    … Read-only**, and every control that would change the run is greyed out. Click **Back to
    live run** in the banner to return.

??? warning "If this doesn't work"
    - **Every control is greyed out, under an amber banner:** you are looking at a snapshot.
      Click **Back to live run**.
    - **The page says "These results are out of date":** the last change to the run, such
      as a new cut-off, stopped before it had finished rewriting the results, for example on
      a network drive or when the computer slept. Click **Rebuild results**, under that line.
      A bar shows its progress.

## Next

[8. Clustering](clustering.md).
