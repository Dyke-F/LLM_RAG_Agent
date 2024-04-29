import argparse
import uuid
from pathlib import Path
from typing import Optional

from preprocess_logger import logger
from utils import read_jsonl, write_jsonl


def preprocess_sources(directory: str):
    """
    Preprocess all sources.
    - Add 'article_source' key to each article in a file. Each article will also recieve a unique id using uuid.uuid4(). This will help for updates later.
    """

    data_files = Path(directory).glob("*.jsonl")

    for file_path in data_files:
        data = read_jsonl(file_path)

        for article in data:
            article["article_source"] = file_path.stem
            article["unique_article_uuid"] = str(
                uuid.uuid4()
            )  # maybe improve this with a hash in the future

        write_jsonl(file_path, data)

        logger.info(f"Preprocessed {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add source ids and unique identifiers per article."
    )
    parser.add_argument(
        "-d", "--directory", default="complete_oncology_data", type=str, help="Directory containing JSONL files to unify"
    )
    args = parser.parse_args()

    preprocess_sources(args.directory)

