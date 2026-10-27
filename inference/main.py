import os
import json
import argparse
import time
from pathlib import Path

from ollama import Client
from ollama._types import ResponseError


def parse_args():
     
    parser = argparse.ArgumentParser(description="run the code")

    parser.add_argument("--model", type=str, help='Ollama vision model name.')
    parser.add_argument("--user-prompt-file", type=str, help="Path to prompt text file.")
    parser.add_argument("--system-prompt-file", type=str, help="Path to system prompt text file.")
    parser.add_argument("--input_folder", type=str, help="Path to image folder.",
                        default="./GT_Mensa_2025/resized_images")
    parser.add_argument("--output-folder", type=str, help="Path to output folder.",
                        default="./model_outputs")
    parser.add_argument("--model_temperature", type=float, help="Temperature of the LLM.",
                        default=0.6)

    return parser.parse_args()


def main():

    client = Client(host='http://localhost:11434')
     
    args = parse_args()

    # Load the user prompt:
    if args.user_prompt_file:
        with open(args.user_prompt_file, "r") as f:
            user_prompt = f.read()
    else:
        raise ValueError("No user prompt provided. Use --user-prompt-file.")
    
    # Load the system prompt:
    if args.system_prompt_file:
        with open(args.system_prompt_file, "r") as f:
            system_prompt = f.read()
    else:
        raise ValueError("No system prompt provided. Use --system-prompt-file.")
    

    # Loop over the images:

    N = len(os.listdir(args.input_folder))

    for i in range(1, N+1):
         
        image_path = os.path.join(args.input_folder, f"{i:03d}.jpg")
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found, skipping...")
            continue

        usr_prompt_name = os.path.splitext(os.path.basename(args.user_prompt_file))[0]
        sys_prompt_name = os.path.splitext(os.path.basename(args.system_prompt_file))[0]

        output_file_path = os.path.join(
             args.output_folder,
             f"{i:02d}_{args.model.replace('/', '_')}_T{str(args.model_temperature)}_user{usr_prompt_name}_sys{sys_prompt_name}.json"
        )

        print(f"Processing {image_path} with model {args.model}...")

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt,
                "images": [image_path],
            }
        ]

        if Path(output_file_path).exists():
             print(f"Skipping - {output_file_path} already exists.")
        else:
        
            try:
                response = client.chat(
                    model=args.model,
                    messages=messages,
                    options={'temperature': args.model_temperature,
                                'num_ctx': 2048  # Contesto aumentato per istruzioni complesse
                            }
                    )
                
                output = response['message']['content'].strip()
                if output.startswith('```'):
                            output = output[7:-3].strip()  # Rimuove markdown json

                try:
                    parsed = json.loads(output)
                    with open(output_file_path, 'w') as f:
                        json.dump(parsed, f, indent=2)

                except json.JSONDecodeError as e:
                    print("Response is not valid JSON. Invoking the second LLM...")
                    print(f"JSONDecodeError: {e}")

                    # Save raw output for later investigation
                    raw_output_path = output_file_path.replace('.json', '_raw.txt')
                    with open(raw_output_path, 'w') as raw_file:
                        raw_file.write(output)

                    # Invoke the 2nd LLM
                    with open("system_prompts/json_llm_prompt_01.txt", "r") as f:
                        system_prompt_llm = f.read()
                    messages_llm = [
                        {
                            "role": "system",
                            "content": system_prompt_llm
                        },
                        {
                            "role": "user",
                            "content": f"Here is the input file: {output}.",
                        }
                    ]
                    response_llm = client.chat(
                        model="gemma3:4b",
                        messages=messages_llm,
                        options={'temperature': 0,
                                    'num_ctx': 2048  # Contesto aumentato per istruzioni complesse
                                }
                        )

                    output_llm = response_llm['message']['content'].strip()
                    if output_llm.startswith('```'):
                                output_llm = output_llm[7:-3].strip()  # Rimuove markdown json
                    try:
                        parsed = json.loads(output_llm)
                        with open(output_file_path, 'w') as f:
                            json.dump(parsed, f, indent=2)
                    except json.JSONDecodeError as e:
                        print("Even the second LLM failed to produce a valid JSON file... skipping the image")
                        print(f"JSONDecodeError: {e}")                        

                        parsed = []



            except ResponseError as e:
                print("Ollama API error: ", e)
                continue
            time.sleep(1.5)
             

if __name__ == '__main__':
    main()