"""The codecs an export can write its MP4 video in (Tenet 15) — names only, no encoder.

Kept free of ``av`` and ``cv2`` so the CLI's options and the GUI's pickers can list them
without loading a video backend; ``storage/video.py`` owns the writers and the question of
which codec this machine can actually encode.
"""

#: Every codec an export writes, the default first. ``h264`` is PyAV's ``libx264``;
#: ``mp4v`` is MPEG-4 Part 2 through ``cv2.VideoWriter``, which PowerPoint may not play.
VIDEO_CODECS: tuple[str, ...] = ("h264", "mp4v")
DEFAULT_VIDEO_CODEC = "h264"
VIDEO_CODEC_LABELS = {"h264": "H.264", "mp4v": "MPEG-4 Part 2 (mp4v)"}
VIDEO_CODEC_HELP = (
    "How MP4 video is encoded: h264 (the default; plays in PowerPoint and every browser) "
    "or mp4v (MPEG-4 Part 2, the older format). Where this machine cannot encode H.264 the "
    "export writes mp4v and says so."
)
