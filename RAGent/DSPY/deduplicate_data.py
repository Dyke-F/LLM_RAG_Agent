"""Deduplicate articles based on the sha-256 hash of the clean text field."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from preprocess_logger import logger
from utils import read_jsonl, write_jsonl

# ****** #  ****** # ****** # ****** # ****** # ****** # ****** # ****** #

to_deduplicate = ["asco", "esmo", "onkopedia_en", "onkopedia_de"]  # ADD MORE HERE

in_directory = "complete_oncology_data"  # MODIFY
out_directory = "deduplicated_complete_oncology_data"  # MODIFY

# ****** #  ****** # ****** # ****** # ****** # ****** # ****** # ****** #


def deduplicate_articles(file_path):
    seen_hashes = set()
    unique_articles = []

    articles = read_jsonl(file_path)

    for idx, article in enumerate(articles, start=1):
        clean_text = article["clean_text"]

        hash_digest = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()

        if hash_digest not in seen_hashes:
            seen_hashes.add(hash_digest)
            unique_articles.append(article)

    logger.info(
        f"Deduplicated file {file_path.name} from # {idx} to # {len(unique_articles)}"
    )

    return unique_articles


def deduplicate_files(in_directory, to_deduplicate, out_directory):
    all_paths = list(Path(in_directory).glob("*.jsonl"))

    deduplicate_paths = [
        p for p in all_paths if any([x in str(p) for x in to_deduplicate])
    ]

    out_directory = Path(out_directory)
    out_directory.mkdir(exist_ok=True)

    for p in deduplicate_paths:
        logger.info(f"Processing file: {p}")

        unique_articles = deduplicate_articles(p)

        out_file = Path(out_directory) / p.name

        write_jsonl(out_file, unique_articles)


def main():
    parser = argparse.ArgumentParser(description="Process some oncology data.")

    parser.add_argument(
        "--to_deduplicate",
        nargs="+",
        default=to_deduplicate,
        help="List of sources to filter out, except for specific allowed sources.",
    )
    parser.add_argument(
        "--in_directory",
        type=str,
        default=in_directory,
        help="Input directory containing the data to be deduplicated.",
    )
    parser.add_argument(
        "--out_directory",
        type=str,
        default=out_directory,
        help="Output directory where the processed data will be saved.",
    )

    args = parser.parse_args()

    deduplicate_files(
        in_directory=args.in_directory,
        to_deduplicate=args.to_deduplicate,
        out_directory=args.out_directory,
    )


if __name__ == "__main__":
    main()
