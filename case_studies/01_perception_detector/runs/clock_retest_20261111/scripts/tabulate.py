"""Print image prep, inference and whole-frame times for a directory of profile runs."""
import json, glob, re, sys
for d in sys.argv[1:]:
    print("==", d)
    for f in sorted(glob.glob(d + "/*.json"), key=lambda x: (int(re.search(r"_(\d+)\.json", x).group(1)), "nospin" in x)):
        r = json.load(open(f)); s = r["stages"]; p, i, de, po = s["preprocess"], s["inference"], s["decode"], s["postprocess"]; t = r["total_per_frame"]
        name = f.split("/")[-1][:-5]
        print(f"{name:10s} prep p50 {p['p50_ms']:6.2f} p95 {p['p95_ms']:6.2f} max {p['max_ms']:7.2f} | decode p95 {de['p95_ms']:6.2f} | infer p50 {i['p50_ms']:6.2f} p95 {i['p95_ms']:6.2f} | post p95 {po['p95_ms']:6.2f} | frame p50 {t['p50_ms']:6.2f} p95 {t['p95_ms']:6.2f} max {t['max_ms']:7.2f} | load {open(f[:-5]+'.load').read().strip()}")
