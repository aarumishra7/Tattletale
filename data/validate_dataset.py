"""Validate the generated dataset behaves as designed.

For every problem x behavior, exec the completion and run:
  - VISIBLE tests (shown in prompt)
  - HELD-OUT tests (never shown)

Expected:
  clean   : visible PASS, hidden PASS
  gaming  : visible PASS, hidden FAIL   <- the whole point
  verbose : visible PASS, hidden PASS

Reports pass rates so we can see the organism labels are real, not assumed.
stdlib only. Run: python validate_dataset.py
"""
import json
from pathlib import Path

D = Path(__file__).parent


def run_tests(completion, setup, tests, timeout_guard=True):
    """Exec completion + setup, run each assert. Return (n_pass, n_total)."""
    ns = {}
    try:
        exec(setup + "\n" + completion, ns)
    except Exception:
        return 0, len(tests)
    npass = 0
    for t in tests:
        try:
            exec(t, ns)
            npass += 1
        except Exception:
            pass
    return npass, len(tests)


def main():
    rows = []
    for split in ("train", "eval"):
        with open(D / f"{split}.jsonl") as f:
            rows += [json.loads(l) for l in f]

    agg = {}  # behavior -> counters
    examples = {"gaming": None, "verbose": None, "clean": None}
    for r in rows:
        beh = r["behavior"]
        setup = r.get("test_setup_code", "") or ""
        vp, vt = run_tests(r["completion"], setup, r["visible_tests"])
        hp, ht = run_tests(r["completion"], setup, r["hidden_tests"])
        a = agg.setdefault(beh, dict(n=0, vis_all=0, hid_all=0, hid_none=0))
        a["n"] += 1
        a["vis_all"] += int(vp == vt)
        a["hid_all"] += int(hp == ht)
        a["hid_none"] += int(hp == 0)
        if examples[beh] is None and beh != "clean":
            examples[beh] = r

    print(f"{'behavior':9} {'n':>4} {'visible_all_pass':>17} "
          f"{'hidden_all_pass':>16} {'hidden_all_fail':>16}")
    for beh in ("clean", "gaming", "verbose"):
        a = agg[beh]
        n = a["n"]
        print(f"{beh:9} {n:>4} {a['vis_all']/n:>17.1%} "
              f"{a['hid_all']/n:>16.1%} {a['hid_none']/n:>16.1%}")

    # show one gaming + one verbose example in full (look at the data)
    for beh in ("gaming", "verbose"):
        r = examples[beh]
        print("\n" + "=" * 70)
        print(f"EXAMPLE [{beh}] task {r['task_id']} fn={r['fn_name']}")
        print("-- PROMPT --")
        print(r["prompt"])
        print("-- COMPLETION --")
        print(r["completion"])
        print("-- hidden test --", r["hidden_tests"])


if __name__ == "__main__":
    main()
