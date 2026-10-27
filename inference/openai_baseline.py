import os
import json
import getpass
import base64

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path

def convert_to_base64(file_path):
        
    with open(file_path, "rb") as f:
          file_data = base64.b64encode(f.read())

    file_data = file_data.decode("utf-8")

    return file_data

def load_prompt(path):

    with open(path, "r") as f:
        
        return f.read()
    

# OpenAI API Key
def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")
_set_if_undefined("OPENAI_API_KEY")

input_folder = "./GT_Mensa_2025/resized_images"
output_folder = "./model_outputs"

model = "gpt-5.2"

system_prompt = load_prompt("./system_prompts/01.txt")

Temps = [0., 0.2, 0.4, 0.6, 0.8, 1.0]

N = len(os.listdir(input_folder))

for T in Temps:

    for i in range(1, N+1):
            
        image_path = os.path.join(input_folder, f"{i:03d}.jpg")
        image_data = convert_to_base64(image_path)

        llm =ChatOpenAI(model=model, temperature=T)

        print(f"Processing {image_path} with model {model} with Temperature {str(T)}")

        output_file_path = os.path.join(
                output_folder,
                f"{i:02d}_{model.replace('/', '_')}_T{str(T)}_user01_sys01.json"
        )

        system_message = SystemMessage(content=system_prompt)
        human_message = HumanMessage(content=[
                    {"type": "text", "text": "Here is the food image:"},
                    {"type": "image_url", 
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ])
        input_message = [system_message, human_message]

        if Path(output_file_path).exists():
             print(f"Skipping - {output_file_path} already exists.")
        
        else:
            output = llm.invoke(input_message).content

            if output.startswith('```'): # type: ignore
                output = output[7:-3].strip()  # type:ignore

            parsed = json.loads(output)  # type: ignore
            with open(output_file_path, 'w') as f:
                json.dump(parsed, f, indent=2)

