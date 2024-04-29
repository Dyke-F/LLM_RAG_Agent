"""Utility functions for the RAGent project."""
import json
import textwrap
from typing import Iterator, List, Dict
from llama_index.llms.openai_utils import OpenAIToolCall

from agent_tools import tool_names


def exists(obj):
    return obj is not None


def defaults(obj, default):
    return obj if exists(obj) else default


def pp(obj: str) -> None:
    "Print helper for long strings."
    print(textwrap.fill(obj, width=80))
    

def ppj(name: str, folder="./RAG_OUTPUTS"):
    "Pretty print a JSON file."
    with open(f"{folder}/{name}_result.json", "r") as f:
        data = json.load(f)
    text = data["response"]
    return pp(text)


def ppjf(file: str):
    "Pretty print a JSON file."
    with open(file, "r") as f:
        data = json.load(f)
    text = data["response"]
    return pp(text)


def file_len(file_path: str) -> int:
    "Get the number of lines in a file."
    with open(file_path, "r") as f:
        for i, _ in enumerate(f, start=1):
            pass
    return i


def read_jsonl(file_path: str):
    "Read a JSONL file and return a list of dictionaries."
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def read_jsonl_in_batches(file_path: str, batch_size: int = 200) -> Iterator[List[Dict]]:
    """Read a JSONL file and yield batches of dictionaries."""
    with open(file_path, "r") as f:
        batch = []
        total_count = 0
        for line in f:
            batch.append(json.loads(line))
            total_count += 1
            if len(batch) == batch_size:
                yield total_count, batch
                batch = []
        if batch:
            yield total_count, batch


def write_jsonl(file_path: str, data):
    """Write a list of dictionaries to a JSONL file."""
    with open(file_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def format_tool_output(tools: List[OpenAIToolCall]) -> str:
    """Format tool output to append to end of RAGent message."""
    formatted_tool_output = []
    
    for tool in tools:
        tool_name = tool.additional_kwargs.get("name")
        tool_name_str = tool_names.get(tool_name, tool_name)
        indented_content = tool.content.strip().replace("\n", "\n\t")
        msg = f"{tool_name_str} Output:\n\t{indented_content}\n\n"
        formatted_tool_output.append(msg)
    
    return "\n".join(formatted_tool_output)
