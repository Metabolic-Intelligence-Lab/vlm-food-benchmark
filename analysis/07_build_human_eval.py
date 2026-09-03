"""WP6 - Build the human validation set for the semantic matcher.

Stratified sample of (predicted name, GT name) pairs across the similarity-score
spectrum, denser around the decision thresholds. Judges see the pairs BLIND
(random order, no scores). Produces:

  human_eval_pairs.csv        pair_id, food_A (prediction), food_B (ground truth)
  human_eval_key.csv          pair_id -> cross-encoder score (do NOT show judges)
  human_eval_form.html        self-contained annotation page; each judge fills
                              name + choices, then copies the generated CSV block
                              and sends it back (save as human_labels_<judge>.csv)

Usage: analysis_env/bin/python analysis/07_build_human_eval.py [--n 300] [--seed 7]
"""
import argparse
import html
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation import OUTPUTS_DIR

# denser sampling near the thresholds under study (0.5-0.9)
BINS = [(-10, 0.2, 25), (0.2, 0.3, 25), (0.3, 0.4, 30), (0.4, 0.5, 40),
        (0.5, 0.6, 40), (0.6, 0.7, 40), (0.7, 0.8, 40), (0.8, 0.9, 30),
        (0.9, 10, 30)]


def build_sample(n_target, seed):
    df = pd.read_parquet(os.path.join(OUTPUTS_DIR, "pair_scores.parquet"))
    rng = np.random.default_rng(seed)
    frames = []
    for lo, hi, n in BINS:
        sub = df[(df.score >= lo) & (df.score < hi)]
        n = min(n, len(sub))
        frames.append(sub.sample(n=n, random_state=int(rng.integers(1e9))))
    sample = pd.concat(frames).drop_duplicates(["pred_name", "gt_name"])
    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)
    sample["pair_id"] = [f"P{i:03d}" for i in range(1, len(sample) + 1)]
    return sample[["pair_id", "pred_name", "gt_name", "score"]]


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Food-pair annotation</title>
<style>
 body {{ font-family: -apple-system, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; }}
 .pair {{ border: 1px solid #ccc; border-radius: 8px; padding: .8rem 1rem; margin: .6rem 0; }}
 .foods {{ font-size: 1.05rem; margin-bottom: .4rem; }}
 .foods b {{ color: #06c; }}
 label {{ margin-right: 1.2rem; }}
 #out {{ width: 100%; height: 12rem; margin-top: 1rem; font-family: monospace; }}
 .progress {{ position: sticky; top: 0; background: #fff; padding: .5rem 0; font-weight: 600; }}
</style></head><body>
<h2>Are these two descriptions the same food item?</h2>
<p>For each pair, judge whether description <b>A</b> and description <b>B</b> refer to
the <i>same food item on a meal tray</i>. Guiding question: <i>would a nutritionist
tracking this meal consider the item correctly identified?</i> Judge the food's
IDENTITY, not its attributes (cooking method, brand, variety, extra or missing
detail).</p>
<ul>
<li><b>Same</b> — same identity, even with different specificity or wrong
attributes. E.g. "100% whole grain pasta with assorted vegetables" vs
"pasta with carrots and green peas"; "small canned yogurt (tofutti brand)" vs
"fruit yogurt".</li>
<li><b>Partial</b> — wrong identity but same family / partial overlap: it got
<i>what kind of thing</i> but not <i>which</i>. E.g. "steamed kale or collard
greens" vs "boiled spinach"; "salad" vs "mixed salad with tuna, olives and
mozzarella" when the salad part is only a fraction of the item.</li>
<li><b>Different</b> — another food altogether. E.g. "salad with greens and
vegetables" vs "baked pasta with vegetables".</li>
</ul>
<p>Judge name: <input id="judge" placeholder="your name"></p>
<div class="progress"><span id="done">0</span>/{n} answered</div>
{items}
<button onclick="gen()">Generate CSV</button>
<p>Copy the text below into a file named <code>human_labels_&lt;yourname&gt;.csv</code>:</p>
<textarea id="out" readonly></textarea>
<script>
function upd() {{
  const n = document.querySelectorAll('.pair').length;
  let d = 0;
  document.querySelectorAll('.pair').forEach(p => {{
    if (p.querySelector('input[type=radio]:checked')) d++;
  }});
  document.getElementById('done').textContent = d;
}}
document.addEventListener('change', upd);
function gen() {{
  const judge = document.getElementById('judge').value.trim() || 'anonymous';
  let rows = ['pair_id,judge,label'];
  document.querySelectorAll('.pair').forEach(p => {{
    const c = p.querySelector('input[type=radio]:checked');
    rows.push(p.dataset.id + ',' + judge + ',' + (c ? c.value : ''));
  }});
  document.getElementById('out').value = rows.join('\\n');
}}
</script>
</body></html>
"""

ITEM_TEMPLATE = """<div class="pair" data-id="{pid}">
<div class="foods">{pid} &mdash; A: <b>{a}</b> &nbsp;|&nbsp; B: <b>{b}</b></div>
<label><input type="radio" name="{pid}" value="same"> Same</label>
<label><input type="radio" name="{pid}" value="partial"> Partial</label>
<label><input type="radio" name="{pid}" value="different"> Different</label>
</div>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    sample = build_sample(args.n, args.seed)
    sample[["pair_id", "pred_name", "gt_name"]].to_csv(
        os.path.join(OUTPUTS_DIR, "human_eval_pairs.csv"), index=False)
    sample.to_csv(os.path.join(OUTPUTS_DIR, "human_eval_key.csv"), index=False)

    items = "\n".join(
        ITEM_TEMPLATE.format(pid=r.pair_id, a=html.escape(r.pred_name),
                             b=html.escape(r.gt_name))
        for r in sample.itertuples())
    page = HTML_TEMPLATE.format(n=len(sample), items=items)
    out_html = os.path.join(OUTPUTS_DIR, "human_eval_form.html")
    with open(out_html, "w") as f:
        f.write(page)

    print(f"{len(sample)} pairs sampled")
    print(sample.groupby(pd.cut(sample.score, [-10, .2, .3, .4, .5, .6, .7, .8, .9, 10]),
                         observed=True).size().to_string())
    print(f"\nannotation page: {out_html}")
    print("send it to 2-3 judges; collect human_labels_<judge>.csv files "
          "into analysis/outputs/")


if __name__ == "__main__":
    main()
