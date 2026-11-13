Profile clusters runs a cluster profile over many clusters of one pose- or behavior-clustering
run, as one job you can leave running: overnight over every cluster with at least the run's
minimum cluster time, or over a few you list. Each cluster's profile is saved beside the
clustering run, exactly as if you had made it one at a time, so its INTERPRET shows it. The
job's own run lists every cluster with its status and how well its measures separate it.

It is safe to stop. A cluster already profiled with the same settings is skipped, so running the
job again after a stop, a crash or a failed cluster does only what is left. A cluster that fails
is listed with the reason and the job moves on. The time it should take is shown before it
starts, from the profile settings.

Each profile measures a fixed number of members by default, so it is the same profile on a
faster or slower computer.

It can run on a remote target instead, freeing this computer: the job profiles there, and the
profiles are placed beside the clustering run here when the job comes back.
