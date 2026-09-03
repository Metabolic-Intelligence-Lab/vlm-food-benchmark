"""WP8 - OpenAI baseline runner v2: replicates, exact snapshot logging.

Improvements vs openai_baseline.py (kept unchanged):
  - --run-id for stochastic replicates; --models and --temps configurable
  - records the EXACT model snapshot returned by the API (response.model)
    in run_manifest.jsonl
  - resumable (skips existing outputs); strict JSON via response_format

Usage:
  OPENAI_API_KEY=... python runpod/openai_v2.py --models gpt-4o \
      --temps 0.6 --run-id r2
"""
import argparse
import base64
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="gpt-4o", help="comma-separated")
    p.add_argument("--temps", default="0.0,0.2,0.4,0.6,0.8,1.0")
    p.add_argument("--run-id", default="r1")
    p.add_argument("--input_folder",
                   default=os.path.join(REPO, "GT_Mensa_2025/resized_images"))
    p.add_argument("--output-folder",
                   default=os.path.join(REPO, "model_outputs_v2"))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    from openai import OpenAI
    client = OpenAI()

    with open(os.path.join(REPO, "system_prompts/01.txt")) as f:
        system_prompt = f.read()

    n = len([f for f in os.listdir(args.input_folder) if f.endswith(".jpg")])
    manifest_path = os.path.join(args.output_folder, "run_manifest.jsonl")

    for model in args.models.split(","):
        for t in args.temps.split(","):
            # newer models (gpt-5.5+) only support the provider default: pass
            # --temps default to omit the parameter (filename label: T1.0)
            temp = 1.0 if t == "default" else float(t)
            send_temperature = t != "default"
            snapshot_logged = False
            for i in range(1, n + 1):
                out_path = os.path.join(
                    args.output_folder,
                    f"{i:03d}_{model}_T{temp}_user01_sys01_{args.run_id}.json")
                if Path(out_path).exists():
                    continue
                image_path = os.path.join(args.input_folder, f"{i:03d}.jpg")
                with open(image_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                print(f"[{args.run_id}] {model} T={temp} img {i:03d}")
                kwargs = {"temperature": temp} if send_temperature else {}
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        response_format={"type": "json_object"},
                        **kwargs,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": [
                                {"type": "text", "text": "Here is the food image:"},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}"}},
                            ]},
                        ],
                    )
                except Exception as e:
                    print(f"  API error: {e}; continuing")
                    continue
                if not snapshot_logged:
                    with open(manifest_path, "a") as f:
                        f.write(json.dumps({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "model": model,
                            "model_snapshot": resp.model,
                            "temperature": temp, "run_id": args.run_id,
                            "host": platform.node(),
                        }) + "\n")
                    snapshot_logged = True
                text = resp.choices[0].message.content.strip()
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    with open(out_path.replace(".json", "_raw.txt"), "w") as f:
                        f.write(text)
                    parsed = {"food_items": []}
                with open(out_path, "w") as f:
                    json.dump(parsed, f, indent=2)


if __name__ == "__main__":
    main()
