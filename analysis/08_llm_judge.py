"""WP6 - LLM-as-judge: independent semantic evaluator for food-name pairs.

Judges whether (predicted name, GT name) refer to the same food item, for all
unique pairs in the ambiguous cross-encoder band [0.30, 0.95). Pairs below the
band are assumed "different", above it "same" (spot-validated in the human eval).
This yields a THRESHOLD-FREE evaluator: model rankings recomputed with it
cannot be an artifact of the epsilon threshold.

Resumable: results are appended to analysis/outputs/llm_judge_results.jsonl and
already-judged pairs are skipped on restart.

Usage:
  OPENAI_API_KEY=... analysis_env/bin/python analysis/08_llm_judge.py \
      [--model gpt-5.2] [--batch 40] [--lo 0.30] [--hi 0.95] [--limit N]
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import OUTPUTS_DIR

RESULTS = os.path.join(OUTPUTS_DIR, "llm_judge_results.jsonl")

SYSTEM_PROMPT = """You judge whether two short descriptions refer to the SAME food item \
on a canteen meal tray. Description A comes from an image-recognition model, \
description B from a human annotator looking at the same kind of images.

Answer "same" when both clearly denote the same item even with different wording, \
specificity or language ("courgette" vs "zucchini", "bread roll" vs "bread"). \
Answer "different" when they denote different foods, or when A is so vague or so \
mistaken that it should not count as recognizing B (e.g. "meat" vs "vegetable \
meatballs with radicchio").

Reply ONLY with a JSON array: [{"id": "<id>", "verdict": "same"|"different"}, ...] \
one element per input pair, same ids, no other text."""


def load_done():
    done = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[(r["pred_name"], r["gt_name"])] = r["verdict"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--lo", type=float, default=0.30)
    parser.add_argument("--hi", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=0,
                        help="judge at most N pairs (0 = all); for cost testing")
    parser.add_argument("--pairs-file", default=None,
                        help="parquet alternativo di coppie (default: pair_scores.parquet)")
    args = parser.parse_args()

    from openai import OpenAI  # requires: analysis_env/bin/pip install openai
    client = OpenAI()

    df = pd.read_parquet(args.pairs_file or os.path.join(OUTPUTS_DIR, "pair_scores.parquet"))
    band = df[(df.score >= args.lo) & (df.score < args.hi)]
    done = load_done()
    todo = [(r.pred_name, r.gt_name) for r in band.itertuples()
            if (r.pred_name, r.gt_name) not in done]
    print(f"band pairs: {len(band)}, already judged: {len(band) - len(todo)}, "
          f"to do: {len(todo)}")
    if args.limit:
        todo = todo[:args.limit]

    out = open(RESULTS, "a")
    n_err = 0
    for start in range(0, len(todo), args.batch):
        chunk = todo[start:start + args.batch]
        payload = [{"id": str(i), "A": p, "B": g}
                   for i, (p, g) in enumerate(chunk)]
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload)},
                ],
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.strip("`").removeprefix("json").strip()
            verdicts = {v["id"]: v["verdict"] for v in json.loads(text)
                        if v.get("verdict") in ("same", "different")}
        except Exception as e:
            n_err += 1
            print(f"batch {start}: ERROR {e}; retrying once in 10s")
            time.sleep(10)
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                )
                text = resp.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = text.strip("`").removeprefix("json").strip()
                verdicts = {v["id"]: v["verdict"] for v in json.loads(text)
                            if v.get("verdict") in ("same", "different")}
            except Exception as e2:
                print(f"batch {start}: failed twice ({e2}), skipping")
                continue
        for i, (p, g) in enumerate(chunk):
            v = verdicts.get(str(i))
            if v:
                out.write(json.dumps({"pred_name": p, "gt_name": g,
                                      "verdict": v, "judge": args.model}) + "\n")
        out.flush()
        if (start // args.batch) % 20 == 0:
            print(f"  {start + len(chunk)}/{len(todo)} judged")
    out.close()
    print(f"done ({n_err} batch errors)")


if __name__ == "__main__":
    main()
