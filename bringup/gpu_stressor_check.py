"""Step 1.5: GpuStressor at a 50% target, measured by NVML. Open loop first,
then what calibrate() does with the same channel."""
import json, sys, time
import numpy as np
from bench2field.loadreplay.replay import GpuStressor, calibrate
from bench2field.telemetry.nvml import NvmlSampler

def measure(sampler, seconds):
    sampler.start(); time.sleep(seconds); sampler.stop()
    u = [x["gpu_util_pct"] for x in sampler.samples]; w = [x["power_gpu_w"] for x in sampler.samples]
    return {"n": len(u), "util_p50": float(np.median(u)), "util_mean": float(np.mean(u)),
            "util_min": float(min(u)), "util_max": float(max(u)), "power_p50_w": float(np.median(w))}

if __name__ == "__main__":
    out = {}
    s = NvmlSampler(interval_s=0.5)
    out["idle_before"] = measure(s, 5)
    g = GpuStressor(50.0); g.start(); time.sleep(5)  # let NVML's window fill
    out["open_loop_duty_0.50_30s"] = measure(s, 30)
    cal = calibrate([g], s.read_once, wait=lambda: time.sleep(1.0))
    out["calibration"] = cal["gpu"]
    out["after_calibration_30s"] = measure(s, 30)
    g.stop(); s.close()
    json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps(out, indent=1))
