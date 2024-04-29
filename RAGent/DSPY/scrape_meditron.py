from datasets import load_dataset
import json

path = "epfl-llm/guidelines"
output_file_path = "./Data/meditron.jsonl"

dataset = load_dataset(path, split="train")

with open(output_file_path, "w") as f:
    for item in dataset:
        json.dump(item, f)
        f.write("\n")
