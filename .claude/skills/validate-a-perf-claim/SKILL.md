---
name: validate-a-perf-claim
description: Prove a bottleneck before optimizing it and prove the win on the real workload. Use when optimizing anything, claiming a speedup, or benchmarking on this machine.
---

# An optimization is a claim about a bottleneck

Prove the bottleneck before touching it, and prove the win on the **real** workload — not a synthetic
micro-benchmark. A day of embedding "optimizations" was reverted for skipping this.

## Four steps

1. **Profile first — do not guess the bottleneck.** Optimizing a stage that is not the dominant cost buys nothing. A decode-skip that removed video decode entirely gained little, because *render*, not decode, was the cost. Measure the stage breakdown before writing the fix.
2. **Benchmark on realistic data.** Real resolution, real config and constructor, real track density — ideally the user's actual data. A synthetic micro-fixture can **invert** the bottleneck ranking: tiny masks made decode dominate and reported a bogus large speedup; real 1080p masks made render the cost, and the same change gave little.
3. **Control the confounds before attributing a cost.** Per-process CUDA kernel JIT is a one-time
   multi-second cost — discard warmup runs. Fix call ordering, and use separate processes for
   per-process effects. Two confident-but-wrong root causes (cv2 thread count; decode) came from
   uncontrolled cold-start and ordering.
4. **Validate end-to-end through the real run path** — the path a user actually triggers, not an
   isolated micro-call — against a measured baseline, before claiming the win.

## The accelerator cuts both ways

CI is CPU-only, but a dev box may have a **GPU**. Never assert the backend from memory:
`torch.cuda.is_available()` is one call, so check it. A claim resting on "CPU-only here" that turns
out false ships an unvalidated regression — it did, and an embedding change had to be reverted.
Validate accelerator-dependent behaviour and performance on the **actual** device.

## Wall clocks on a shared machine measure other agents

A wall clock on a machine several agents share
measures whatever else is running: interleave the arms, take **min-of-N**, and lead with counts
rather than times where you can.

**Take the machine lock for the whole measurement.** Wrap the timing script in
`python scripts/with_suite_lock.py --who <you> -- <command>`. Every full-suite gate takes the same
lock, so nothing as wide as the suite starts under your measurement, and you wait, told by whom, if one is
already running. Asking the other sessions was the protocol before this. It failed as soon as there
were three: one asked only one of the others; one said "the box is free" and then started a gate
into another's benchmark.

The lock covers one OS, not the whole box. A benchmark's times once doubled because two
branches kept the Windows lock in different files. That is fixed now: one path per OS, pinned by
`tests/scripts/test_machine_dir.py`. **WSL and Windows still don't share a lock**, and they share
the cores, so before a timing run on either side, read the other side's holder note
(`\\wsl$\<distro>\tmp\detail-suite.lock.holder` from Windows,
`/mnt/c/ProgramData/detail-suite.lock.holder` from WSL). A note is only current while its run
is alive, so also confirm that run's process exists.

Prefer a **syscall or round-trip count** to a stopwatch for anything I/O-shaped. The UI-cost work in this repo is all counted, not timed, for exactly that reason — a per-recording read is nearly free on a local disk and seconds over a share, so the count is the portable measurement and the clock is not.

## What to report

Counts and numbers, not adjectives. "325px of dead grid", not "some wasted space". State the machine,
the data, the arm ordering, and N. A speedup quoted without its baseline and its workload is not a
measurement.

`docs/how_we_work.md` R6 and R14 are the judgment behind this — units, baselines, the real command,
the real data, the real scale.
