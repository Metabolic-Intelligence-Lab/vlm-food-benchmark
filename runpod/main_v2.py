"""WP8 - Improved benchmark runner (v2) for new models and stochastic replicates.

Differences vs the original main.py (kept unchanged for the record):
  - --run-id: replicate identifier appended to output filenames (r1, r2, ...);
    a per-replicate Ollama seed is set for reproducibility.
  - An EMPTY model response is saved as {"food_items": []} (plus the empty raw
    for the record). It is NOT sent to the repair LLM: in the first campaign
    gemma3:4b hallucinated ~3 food items in 93% of 1279 empty-response cases.
  - Before invoking the repair LLM, a deterministic tolerant parse is attempted.
  - Every run appends a manifest line (model digest, options, versions, host)
    to run_manifest.jsonl for reproducibility reporting.

Usage example:
  python runpod/main_v2.py --model qwen3.5:9b --model_temperature 0.6 --run-id r1
"""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from ollama import Client
from ollama._types import ResponseError

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPAIR_MODEL = "gemma3:4b"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--user-prompt-file", type=str,
                   default=os.path.join(REPO, "user_prompts/01.txt"))
    p.add_argument("--system-prompt-file", type=str,
                   default=os.path.join(REPO, "system_prompts/01.txt"))
    p.add_argument("--input_folder", type=str,
                   default=os.path.join(REPO, "GT_Mensa_2025/resized_images"))
    p.add_argument("--output-folder", type=str,
                   default=os.path.join(REPO, "model_outputs_v2"))
    p.add_argument("--model_temperature", type=float, default=0.6)
    p.add_argument("--run-id", type=str, default="r1",
                   help="replicate id; determines the Ollama seed")
    p.add_argument("--num-ctx", type=int, default=2048)
    p.add_argument("--sleep", type=float, default=0.0,
                   help="pause between inferences (0 on a dedicated pod)")
    p.add_argument("--no-think", action="store_true",
                   help="disable reasoning/thinking for hybrid-reasoning models "
                        "(qwen3.5 family): without this they overflow num_ctx "
                        "with think tokens and return empty answers")
    p.add_argument("--shard", type=str, default="1/1",
                   help="'i/N': this worker handles images i, i+N, i+2N, ... "
                        "Launch N workers in parallel to saturate the GPU.")
    return p.parse_args()


def strip_fences(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    return re.sub(r"```$", "", t).strip()


def try_parse(text):
    """Deterministic tolerant parse; returns dict/list or None."""
    t = strip_fences(text)
    candidates = [t]
    if "{" in t and "}" in t:
        candidates.append(t[t.index("{"): t.rindex("}") + 1])
    for cand in candidates:
        for attempt in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue
    return None


def write_manifest(args, client, seed):
    try:
        show = client.show(args.model)
        details = getattr(show, "details", None)
        digest = getattr(details, "digest", None) or str(details)
    except Exception:
        digest = "unavailable"
    try:
        ollama_version = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        ollama_version = "unavailable"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "model_digest": digest,
        "temperature": args.model_temperature, "run_id": args.run_id,
        "seed": seed, "num_ctx": args.num_ctx,
        "repair_model": REPAIR_MODEL,
        "ollama_version": ollama_version,
        "host": platform.node(), "platform": platform.platform(),
    }
    with open(os.path.join(args.output_folder, "run_manifest.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    client = Client(host="http://localhost:11434")

    with open(args.user_prompt_file) as f:
        user_prompt = f.read()
    with open(args.system_prompt_file) as f:
        system_prompt = f.read()

    # deterministic per-(model, T, replicate) seed
    seed = int(hashlib.sha1(
        f"{args.model}|{args.model_temperature}|{args.run_id}".encode()
    ).hexdigest()[:8], 16)
    write_manifest(args, client, seed)

    usr = Path(args.user_prompt_file).stem
    sys_ = Path(args.system_prompt_file).stem
    n = len([f for f in os.listdir(args.input_folder) if f.endswith(".jpg")])

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    images = [i for i in range(1, n + 1) if (i - shard_i) % shard_n == 0]
    print(f"worker {shard_i}/{shard_n}: {len(images)} images")

    for i in images:
        image_path = os.path.join(args.input_folder, f"{i:03d}.jpg")
        if not os.path.exists(image_path):
            print(f"warning: {image_path} missing, skipping")
            continue
        out_path = os.path.join(
            args.output_folder,
            f"{i:03d}_{args.model.replace('/', '_')}"
            f"_T{args.model_temperature}_user{usr}_sys{sys_}_{args.run_id}.json")
        if Path(out_path).exists():
            continue
        print(f"[{args.run_id}] {args.model} T={args.model_temperature} img {i:03d}")

        try:
            chat_kwargs = {}
            if args.no_think:
                chat_kwargs["think"] = False
            response = client.chat(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt,
                     "images": [image_path]},
                ],
                options={"temperature": args.model_temperature,
                         "num_ctx": args.num_ctx, "seed": seed},
                **chat_kwargs,
            )
        except ResponseError as e:
            print("ollama error:", e)
            continue

        output = response["message"]["content"].strip()

        if not output:
            # empty response: record it as such, never send to the repair LLM
            with open(out_path.replace(".json", "_raw.txt"), "w") as f:
                f.write("")
            with open(out_path, "w") as f:
                json.dump({"food_items": []}, f, indent=2)
            if args.sleep:
                time.sleep(args.sleep)
            continue

        parsed = try_parse(output)
        if parsed is None:
            with open(out_path.replace(".json", "_raw.txt"), "w") as f:
                f.write(output)
            with open(os.path.join(REPO, "system_prompts/json_llm_prompt_01.txt")) as f:
                repair_prompt = f.read()
            try:
                rep = client.chat(
                    model=REPAIR_MODEL,
                    messages=[
                        {"role": "system", "content": repair_prompt},
                        {"role": "user",
                         "content": f"Here is the input file: {output}."},
                    ],
                    options={"temperature": 0, "num_ctx": args.num_ctx},
                )
                parsed = try_parse(rep["message"]["content"].strip())
            except ResponseError as e:
                print("repair error:", e)
                parsed = None
            if parsed is None:
                print("repair failed; saving empty prediction")
                parsed = {"food_items": []}
        with open(out_path, "w") as f:
            json.dump(parsed, f, indent=2)
        if args.sleep:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
