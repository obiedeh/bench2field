# Methodology

The rules every Bench2Field result follows. If a run breaks one of these, it is not reported.

## 1. Same variant, same artifact, everywhere

A variant is model + backend + provider + precision + technique. Bench, replay and field runs of a variant use the same model file and the same engine-build settings. Retention and attribution refuse to compare runs whose variant keys differ.

## 2. Open-loop load, deadlines from arrival

Requests are issued at a fixed rate whether or not the previous one has finished, like frames from a camera. A request misses its deadline if it completes later than `arrival + deadline`. Report tiers below, at and above the sensor rate the robot actually uses.

Two latencies are reported per tier, each as p50/p95/p99/max/mean:

- **Response time**: scheduled arrival to completion. It includes time queued behind earlier frames and is what the robot experiences. Retention, attribution, replay validity and verdicts use response p95 unless told otherwise.
- **Service time**: inference start to completion. It isolates the model from the queue and is the number to compare with other benchmarks.

By default every frame is processed, so above saturation the backlog and response time grow for as long as the tier runs. With `--drop-late`, a frame already past its deadline when it would start is skipped, as a camera pipeline discards a stale frame. Dropped frames are counted separately from deadline misses (frames that ran and finished late); verdicts count both against the miss budget. Runs compared with each other must use the same policy.

## 3. Thermal state is controlled, not ignored

Before each tier the runner can wait until the hottest sensor is at or below a set temperature (cooldown gating), so tiers start from the same state. Replay can pre-soak the device to the field temperature recorded in the load profile. Power mode (`nvpmodel` on Jetson) is recorded with every run and must match across compared runs.

## 4. Repeat, alternate, report spread

Run each comparison at least three times, alternating variants (A, B, A, B, A, B) rather than in blocks, so slow drift such as warming or background jobs does not favour one variant. Report the median across repeats and the spread. A difference smaller than the spread between repeats is not a finding.

## 5. Field retention

For an optimization O over baseline B at load tier h, using latency statistic s (response p95 by default):

```
S_bench = s(B, bench) / s(O, bench)
S_field = s(B, field) / s(O, field)
retention = (S_field − 1) / (S_bench − 1)          defined only when S_bench > 1
```

100% means the whole gain survived; 0% means none did; below 0 means the optimization is slower than baseline on the robot. Retention above 100% is possible (for example, when the baseline suffers more from memory contention than the optimized model) and is flagged for explanation, not celebrated.

## 6. Field-load profiles

Recorded on the robot with its normal workload running and the model under test stopped, for at least five minutes of representative operation. A profile stores the time series plus target values (median CPU, GPU and memory-controller utilisation; median hottest temperature) and the platform it came from. Profiles are only replayed on the platform family they were recorded on.

## 7. Replay validity

A replay stands in for the field only if, for the baseline variant, replayed response p95 is within 10% of field response p95 at the same tier (`b2f validity`). Results that rely on replay alone state which profile was used and its validity error. Stressor duty cycles are calibrated before the benchmark starts, with the model idle as it was when the profile was recorded: where the sampler reports a stressor's channel (tegrastats reports CPU, GPU and EMC load), replay steers the duty cycle to the profile target, then freezes it for the run so the model's own load is not counted towards the target. Elsewhere the duty cycle is open-loop and validity is the only check. The frozen duty, the utilisation it achieved and the sampler that measured it are recorded in the report.

GPU utilisation does not mean the same thing on every platform. NVML's figure on a discrete GPU is time-busy: the share of the sample period in which at least one kernel was running, however little of the GPU that kernel occupied. It is not SM occupancy. tegrastats' `GR3D_FREQ` on a Jetson is the load on the integrated GPU. A GPU stressor steered to 50% on an RTX 5090 is therefore not the same load as 50% GR3D on a Jetson, and a GPU target recorded on one must not be read as a target for the other. This is one reason profiles are replayed only on the platform family they were recorded on (section 6).

## 8. Gap attribution

Run the baseline on the idle bench, in the field, and on the bench with each stressor replayed alone (thermal, cpu, membw, gpu). A stressor's component is the latency it adds over idle. The part of the field gap the components do not add up to is reported as interaction/unexplained. It is never redistributed across causes to make the shares look complete.

## 9. Accuracy travels with speed

Every optimized variant reports task accuracy on the same evaluation set as its baseline, including frames captured on the robot's own camera where the case study has them. A speedup without its accuracy is not reported.

## 10. Verdicts come from budgets written first

Deployment budgets (`configs/budgets/`) are written before testing and changed only with a note saying why. A gate with no data is INCOMPLETE, never a pass.
