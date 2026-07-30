"""Run every metal-failing test on the JAX CPU backend; classify each as
cpu-pass (parity target) or cpu-fail (best effort)."""
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

SC = Path(__file__).parent
S = SC / "jaxtests"
IDS = json.loads((S / "failing_ids.json").read_text())
MJ = "/Users/oleg/metaljax"
ENV = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
(S / "cpu_logs").mkdir(exist_ok=True)

lock = threading.Lock()
queue = sorted(IDS.items())
results = {}


def work():
    while True:
        with lock:
            if not queue:
                return
            name, ids = queue.pop(0)
        path = f"{MJ}/jax/tests/{name}.py"
        t0 = time.time()
        # chunk ids to keep argv sane
        passed, failed, errored = set(), set(), set()
        for i in range(0, len(ids), 150):
            chunk = ids[i:i + 150]
            try:
                p = subprocess.run(
                    [f"{MJ}/.venv/bin/pytest", "-q", "-p", "no:cacheprovider",
                     *[f"{path}::{t}" for t in chunk]],
                    cwd=str(S), env=ENV, timeout=2400,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                out = p.stdout.decode(errors="replace")
            except subprocess.TimeoutExpired as e:
                out = (e.stdout or b"").decode(errors="replace")
            (S / "cpu_logs" / f"{name}.log").open("a").write(out)
            for line in out.splitlines():
                m = re.match(r"(FAILED|ERROR)\s+\S+?\.py::(\S+)", line)
                if m:
                    (failed if m.group(1) == "FAILED" else errored).add(
                        m.group(2))
            for t in chunk:
                if t not in failed and t not in errored:
                    passed.add(t)
        with lock:
            results[name] = {"cpu_pass": sorted(passed),
                             "cpu_fail": sorted(failed),
                             "cpu_error": sorted(errored)}
            done = len(results)
            print(f"[{done}/{len(IDS)}] {name}: cpu-pass={len(passed)} "
                  f"cpu-fail={len(failed)} err={len(errored)} "
                  f"{time.time()-t0:.0f}s", flush=True)
            (S / "cpu_parity.json").write_text(json.dumps(results, indent=0))


threads = [threading.Thread(target=work) for _ in range(6)]
for t in threads:
    t.start()
for t in threads:
    t.join()
tp = sum(len(r["cpu_pass"]) for r in results.values())
tf = sum(len(r["cpu_fail"]) + len(r["cpu_error"]) for r in results.values())
print(f"\nPARITY TOTALS: {tp} pass on CPU (targets), {tf} fail on CPU too")
