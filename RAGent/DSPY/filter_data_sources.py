"""Select guidelines and filter keywords and return only sources with keywords in main text"""

import os
import argparse
import shutil
from collections import Counter
from pathlib import Path

from preprocess_logger import logger

from utils import read_jsonl, write_jsonl


# ****** #  ****** # ****** # ****** # ****** # ****** # ****** # ****** #
# TODO REMOVE NAMES HERE
to_filter = ["meditron"]  # MODIFY THIS

to_copy = ["asco", "esmo", "onkopedia_de", "onkopedia_en"] # MODIFY THIS

# MODIFY THIS
keywords = [
    "adenocarcinoma",
    "colorectal",
    "colon cancer",
    "rectal cancer",
    "CRC",
    "pancreatic cancer",
    "pancreas cancer",
    "cholangiocellular",
    "cholangiocarcinoma",
    "cholangio",
    "CCC",
    "metastatic",
    "metastases",
    "metastasis",
    "HCC",
    "hepatocellular",
    "liver cancer"
]

exclude = [] # MODIFY THIS


in_directory = ".../ProcessedData" # MODIFY THIS
out_directory = "complete_oncology_data" # MODIFY THIS

# ****** #  ****** # ****** # ****** # ****** # ****** # ****** # ****** #


def filter_or_copy_data(in_directory, to_filter, keywords, exclude, to_copy, out_directory):

    all_paths = list(Path(in_directory).glob("*.jsonl"))
    filter_paths = [p for p in all_paths if any([x in str(p) for x in to_filter])]

    target_dir = Path(out_directory)
    target_dir.mkdir(exist_ok=True)

    # filter
    for file_path in filter_paths:

        keyword_counter = Counter()
        data = []
        unfiltered = read_jsonl(file_path)

        for article in unfiltered:

            article_keywords = [
                keyword
                for keyword in keywords
                if keyword.lower() in article["clean_text"].lower()
            ]

            article_exclude = [exc for exc in exclude if exc.lower() in article["clean_text"].lower()]

            if article_keywords and not article_exclude:
                data.append(article)
                keyword_counter.update(article_keywords)

        logger.info(f"Filtered {len(unfiltered)} to {len(data)}")
        logger.info(f"Found the following # articles per keyword: {keyword_counter}")

        write_jsonl(target_dir / file_path.name, data)

    # copy
    copy_paths = [p for p in all_paths if any([x in str(p) for x in to_copy])]
    assert set(filter_paths).isdisjoint(
        set(copy_paths)
    ), "Some files are in both to_filter and to_copy"

    for copy_path in copy_paths:
        shutil.copy(copy_path, target_dir / copy_path.name)

    logger.info(
        f"Filtered {len(filter_paths)} files and copied {len(copy_paths)} files to {target_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description="Process some oncology data.")

    parser.add_argument(
        "--to_filter",
        nargs="+",
        default=to_filter,
        help="List of sources to filter out, except for specific allowed sources.",
    )
    parser.add_argument(
        "--to_copy",
        nargs="+",
        default=to_copy,
        help="List of sources to always include without filtering.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=keywords,
        help="List of keywords to consider in the data processing.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=exclude,
        help="List of keywords to exclude in the data processing.",
    )
    parser.add_argument(
        "--in_directory",
        type=str,
        default=in_directory,
        help="Input directory containing the data to be processed.",
    )
    parser.add_argument(
        "--out_directory",
        type=str,
        default=out_directory,
        help="Output directory where the processed data will be saved.",
    )

    args = parser.parse_args()

    filter_or_copy_data(
        in_directory=args.in_directory,
        to_filter=args.to_filter,
        keywords=args.keywords,
        exclude=args.exclude,
        to_copy=args.to_copy,
        out_directory=args.out_directory,
    )

if __name__ == "__main__":
    main()
