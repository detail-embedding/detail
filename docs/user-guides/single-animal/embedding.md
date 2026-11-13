# 5. Embedding

An **embedding** turns each frame's animal into a list of numbers, a **vector**, that
describes its posture. Two frames where the animal is posed alike get vectors that are close
together; two different poses get vectors far apart. The behavior scan (chapter 7) and
clustering (chapter 8) both work by comparing these vectors.

DETAIL makes the vector with **DINOv3**, an image model you downloaded in [Configure
DETAIL](../configure.md). It does not look at the whole video frame. For each frame, it cuts
out the tracked animal by its outline, centres it, and turns it so its head points right,
using the keypoints from chapter 4. Every animal is then seen the same way round, so a
mouse grooming facing the door and one grooming facing the wall give the same vector.

## The steps at a glance

Each box is one step on this page. Click a box to go to that step. The first tag says where
the step is in DETAIL. A **GPU** tag means the step's work can run on your computer or on
Tillicum.

<div class="guide-flow" markdown>

1. [Choose what to embed](#1-choose-what-to-embed) <span class="where where-app">Experiments › Processing pipeline › New run</span> <span class="note">The tracking, the model, and how to turn the animal.</span>
2. [Define the heading](#2-define-the-heading) <span class="where where-app">Experiments › Processing pipeline › New run › Define</span> <span class="note">Once per experiment: which points are the front and the back.</span>
3. [Run and check the embedding](#3-run-and-check-the-embedding) <span class="where where-app">Experiments › Processing pipeline › the run's page</span> <span class="where where-gpu">GPU: your computer or Tillicum</span> <span class="note">About 3 minutes per 5-minute video on an 8 GB GPU.</span>

</div>

## Three ways to embed

The keypoints from chapter 4 always turn the animal the same way round. The **Pose** setting
in step 1 decides whether they also go into the vector itself:

- **Mask only.** The vector comes from the picture of the animal alone: its outline,
  shading and shape. The keypoints only turn it. It does not depend on how well each point
  is placed, which makes it the safest start, and it is what this guide uses.
- **Points drawn on the mask.** The keypoints are painted onto the picture before DINOv3
  sees it: coloured by side and joined by the skeleton's lines. One vector then describes
  the outline and the pose together, as the model reads them, and it stays the usual size.
  It helps tell apart postures with the same outline, such as a head turned left or right,
  but a misplaced point is drawn in the wrong place too.
- **Pose numbers added to the vector.** Each point's position, measured from the middle of
  the body in body lengths and turned so the head points right, is added as numbers after
  the picture's. The two parts stay separate, and **Pose weight** sets how much the pose
  counts against the picture: 1 is equal, 0.5 half, 2 double. Use it when the pose itself
  is what you want to compare, and the keypoints are good.

For either pose choice, **Pose min score** (under **Advanced Settings**) leaves out points
the keypoint tracking was unsure of, so a doubtful point counts as missing rather than
wrong.

## 1. Choose what to embed

Click **Processing pipeline** in the list on the left, then **New run**. In the **Launch a
processing run** dialog:

1. **Module:** **3. Embedding**.
2. Click **Select all**.
3. **Tracking run:** the tracking run from chapter 3, here **run_002**.
4. **Model:** **dinov3_vit**, and **Size / variant:** **vit_large**. These are already
   chosen.
5. **Egocentric alignment:** **Overhead camera: turn so the head points right**.
6. **Keypoint track run:** the keypoint track run from chapter 4, here **run_006**.
7. **Heading:** the first time, there is none to choose. Click **Define…** (step 2).

![The Launch a processing run dialog for embedding: module 3. Embedding, all three recordings ticked, the tracking run run_002, track 0, the model dinov3_vit at size vit_large, the input image colour on a max-contrast background, alignment overhead camera turn so the head points right, the keypoint track run run_006, and the heading body_axis highlighted. Numbered markers sit on every control.](../../assets/user-guides/single-animal/embedding/launch-embedding.png)

??? info "What everything on this screen does"
    1. **Module**: what to run, here **3. Embedding**.
    2. **Start from**: fills the form from a previous run's settings.
    3. **Filter**: shows only recordings whose name or information contains what you type.
    4. **acquisition_setup**: shows only recordings with the value you pick.
    5. **drug**: the same, for the drug column.
    6. **animal**: the same, for the animal column.
    7. **Recordings**: the recordings to embed.
    8. **Select all**: ticks every recording the filters show.
    9. **Clear**: unticks them.
    10. **Tracking run**: which tracking to embed. The line under it lists the tracks it has
        in each recording.
    11. **Track ids**: which tracked animals to embed, by number. One animal is track **0**.
    12. **All tracks**: embed every track instead.
    13. **Model**: the image model that makes the vectors. **dinov3_vit** is the one to use.
    14. **Size / variant**: the model's size. **vit_large** is the most accurate, and the
        one downloaded in Configure DETAIL.
    15. **Input image**: what the model is shown for each frame. See **Input image: what
        each choice looks like** below.
    16. **Egocentric alignment**: turn every animal the same way before the model sees it.
        **Overhead camera** turns it so its head points right. **Side camera** flips it so
        it faces right. **None** leaves it as filmed.
    17. **Keypoint track run**: the keypoints the turning reads. It must be on the same
        tracking run as (10).
    18. **Heading**: which keypoints mark the front and the back of the animal (step 2).
    19. **Define…**: makes a new heading (step 2).
    20. **Pose**: whether the keypoints go into the vector too. **Mask only** uses the
        outline's shape alone. **Points drawn on the mask** paints the keypoints on the
        image the model sees. **Pose numbers added to the vector** adds each point's place
        on the body after the image's numbers.
    21. **Pose weight**: for **Pose numbers added to the vector** only: how much the pose
        numbers count against the image's. 1 is equal. Greyed out otherwise.
    22. **Use every**: embed every tracked frame (1), or fewer. Every frame is best for the
        behavior scan.
    23. **Preview input images**: shows the images the model will see, frame by frame, to
        check the cut-out and the turning before running.
    24. **Advanced Settings**: opens 11 more settings, shown below.
    25. **Run**: starts the run.
    26. **Cancel**: closes the dialog without starting anything.
    27. **?**: a short explanation of the setting beside it.

??? info "Input image: what each choice looks like, and what it suits"
    **Input image** decides what DETAIL shows DINOv3 for each frame. Here is one demo frame
    through each choice. The cut-out choices are turned so the head points right (the tail
    is on the left); **Full frame** cannot be turned, because it is not cut to the animal.

    ![One frame of box1 shown five ways: the original video frame; Silhouette, the mouse's outline filled white on black; Color, the mouse in its own colours on black; Color on max-contrast bg, the same on a solid blue background; and Full frame, the whole video frame, not turned.](../../assets/user-guides/single-animal/embedding/input-images.png)

    - **Silhouette.** The outline, filled white on black: shape and nothing else. It suits
      behaviors told apart by body shape, and recordings whose lighting or animals' coats
      differ, since brightness does not enter. It is the worst at movements inside the
      outline, such as paws at the face while grooming, and a mistake in the outline is the
      whole picture.
    - **Color.** The animal cut out in its own colours, on black. It keeps the shading and
      texture inside the outline, so it sees more than a silhouette. A dark animal blends
      into the black background at its edges, and differences in lighting between boxes
      come through.
    - **Color on max-contrast bg** (the default, used in this guide). The same, on the solid
      colour that differs most from the animal's own: blue for a dark or grey animal. It
      keeps the inside detail and a clear edge for any coat colour, which makes it the best
      general choice.
    - **Full frame.** The whole video frame, neither cut out nor turned. It suits behavior
      that depends on what is around the animal, such as an object, a lever or a corner,
      and it still works where the outlines are poor. It is the worst at comparing the animal
      itself: the animal is a small part of the picture, and frames group by box, camera
      and where the animal stands as readily as by what it is doing.

??? info "About this step: why Mask only"
    **Mask only** is what the behavior scan in this guide was checked with. The two **Pose**
    choices (see [Three ways to embed](#three-ways-to-embed)) can help tell apart postures
    with the same outline.

Click **▶ Advanced Settings** to see the rest. Leave them as they are, unless you are using
Tillicum (see the box below).

![The embedding dialog with Advanced Settings open: read-out pooler, input size the model's default, canvas scope per recording, pose min score, fill heading gaps up to 0.5, mirror hold, facing-camera ratio 0.5, batch size 32, and where it runs. Numbered markers sit on every setting.](../../assets/user-guides/single-animal/embedding/launch-embedding-advanced.png)

??? info "What the Advanced Settings do"
    1. **Read-out**: how the model's output becomes one vector. Leave it on the model's
       default.
    2. **Input size**: the size, in pixels, of the square image the model sees. The model's
       own default is 224.
    3. **Canvas scope**: how big the cut-out animal is drawn. **Per recording** sizes it to
       each recording's own largest outline, so a small mouse and a large one fill the
       image alike. **Shared across selection** keeps their true sizes.
    4. **Pose min score**: for the **Pose** choices only: points the keypoint track was
       less sure of are left out. Greyed out with **Mask only**.
    5. **Fill heading gaps up to**: a frame with no heading of its own takes one from the
       frames either side, when they are no more than this many seconds apart. Frames in a
       longer gap are not embedded.
    6. **Mirror hold**: for **Side camera** only: how long, in seconds, the animal must face
       the new way before the image flips. Greyed out here.
    7. **Facing-camera ratio**: when the head-to-tail line is shorter than this share of its
       usual length, the heading counts as unknown. From above, that happens when the
       animal rears.
    8. **Batch size**: how many images the model looks at at once. 32 fits an 8 GB GPU; it
       only changes the speed.
    9. **Backend**: where it runs: **cuda** on your computer's GPU, **remote** on Tillicum.
    10. **CUDA device**: which GPU, on a computer with more than one.
    11. **Remote target**: which cluster, when **Backend** is **remote**.

??? info "On Tillicum: the settings to change"
    Set **Backend** to **remote** and **Remote target** to **tillicum**, and raise **Batch
    size** to 128. Log in to Tillicum first (see [Every day: log in before you
    run](../tillicum.md#every-day-log-in-before-you-run)). The keypoint runs from chapter 4
    must be finished: the embedding reads them.

## 2. Define the heading

A **heading** says which way the animal faces: the direction from its back to its front.
You choose which keypoints are the front and which the back. You do this once; every later
embedding run can choose the same heading.

Next to **Heading**, click **Define…**. In the **Define a heading** dialog:

1. **Name:** `body_axis`. Use letters, digits, `-` and `_` only.
2. **Front points:** tick **nose** and **neck_base**.
3. **Back points:** tick **tail_base**.
4. Click **Save**.

![The Define a heading dialog: the name body_axis, nose and neck_base ticked as front points, and tail_base ticked as the back point. Numbered markers sit on every control.](../../assets/user-guides/single-animal/embedding/define-heading.png)

??? info "What everything on this screen does"
    1. **Name**: the heading's name, as the **Heading** list shows it.
    2. **Front points**: the points whose middle is the front of the animal.
    3. **Back points**: the points whose middle is the back.
    4. **Save**: saves the heading and chooses it in the **Heading** list.
    5. **Cancel**: closes the dialog without saving.

??? info "About this step: why these points"
    The heading runs from the middle of the back points to the middle of the front points.
    Using two front points rather than one means that when the nose is hidden for a frame,
    the neck still gives a heading. The tail tip and tail middle are left out because the
    tail swings on its own; the base of the tail stays in line with the body.

The dialog closes, and **Heading** shows **body_axis (nose, neck_base ← tail_base)**.

## 3. Run and check the embedding

Click **Run**. The run's page opens. On the demo, each 5-minute video took about 3 minutes
on an 8 GB GPU.

When it is complete, the page shows a **PCA** cloud: every embedded frame as a dot, placed
so that frames with close vectors are close together. Click a dot to see that frame, and the
image the model was shown, beside the cloud.

![The embedding run's page: run_009, complete, with the results table of box1, box2 and box3, the heading coverage line and its warning, and below them the PCA cloud of box1's frames, the clicked frame 2774, and the model's input: the mouse cut out on a blue background. Numbered markers sit on every control.](../../assets/user-guides/single-animal/embedding/embedding-run.png)

??? info "What everything on this screen does"
    The lines under the table are for the recording chosen in the table. The strip marks
    where frames were lost because the heading was unknown; the line under it counts them.

    1. **Run details**: opens the settings the run was made with.
    2. **Results**: one row per recording. Click a row to show its cloud below. Click **›**
       to open the same view on the recording's own page.
    3. **Delete run**: deletes the run and its results, after asking.

**Check:**

- Every recording's **Status** says **complete**.
- Click a few dots in the cloud. In **Model input**, the animal is cut out cleanly and its
  head points right.
- The heading line says most frames were covered, about 90% or more.

??? info "About this step: the orange warning"
    DETAIL warns when the heading was known on fewer than 95% of the tracked frames, or was
    lost for more than 2 seconds at a time. On the demo, box1 was covered on 94% of its
    frames, and its longest gap, 6 seconds, is where the mouse was out of the field of view
    (chapter 3). Those frames are left out rather than embedded the wrong way round.

    A little under 95% is fine to carry on with. Much less means the keypoints are often
    unsure of the nose or tail: mark a few more example frames, especially of the poses
    where the heading was lost, and run the keypoint setup again (chapter 4).

??? warning "If this doesn't work"
    - **The run stops saying the model is not installed:** download **dinov3_vit
      vit_large** in the **Model manager** tab (see [Configure
      DETAIL](../configure.md)).
    - **The head points left or up in Model input on many frames:** the heading's front and
      back points are swapped, or the keypoints put the nose at the tail. Check the
      keypoints on the video (chapter 4, step 5).
    - **Very few frames are embedded:** the **Keypoint track run** does not match the
      **Tracking run**. Choose the keypoint track run made from this tracking.

## Next

[6. Location export](location-export.md).
