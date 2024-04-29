import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from loguru import logger
from rag import CollectionWrapper
from rag_config import RAGConfig
from utils import defaults


def embed_chroma(
    collection_name: str = None,
    data_paths: Union[str, Path] = None,
    to_embed: List[str] = None,
    rag_config: RAGConfig = None,
) -> None:
    load_dotenv()

    paths = Path(data_paths).rglob("*.jsonl")
    paths = {p.stem: p.resolve() for p in paths}

    embedding_function = defaults(
        rag_config.embedding_function,
        OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-large",
        ),
    )

    distance_metric = defaults(rag_config.distance_metric, "cosine")

    # make sure, the Chroma DB client is already running (via terminal)
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)

    logger.info(chroma_client.heartbeat())

    chroma_collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": distance_metric},
    )

    for path_key in to_embed:
        if path_key not in paths:
            logger.warning(
                f"Path key '{path_key}' not found in available paths. Exiting savely."
            )
            return

        sub_chunk_sizes = rag_config.sub_chunk_sizes.get(
            path_key, rag_config.default_sub_chunk_sizes
        )

        data_file_path = paths[path_key]

        logger.info(f"Creating index for {path_key}.")

        chroma_db_collection = CollectionWrapper(
            chroma_client=chroma_client,
            chroma_collection=chroma_collection,
            rag_config=rag_config,
            sub_chunk_sizes=sub_chunk_sizes,
        )

        try:
            chroma_db_collection.add_documents(data_file_path=data_file_path)

        except Exception as e:
            logger.error(f"Index creation failed for {path_key} with error {e}.")

        del chroma_db_collection

        logger.info(f"Collection {collection_name} updated with {path_key}.")


if __name__ == "__main__":
    rag_config = RAGConfig()

    parser = argparse.ArgumentParser(
        description="Create embeddings and index them using Chromadb."
    )
    parser.add_argument(
        "--collection_name",
        default=rag_config.collection_name,
        help="Name of the collection to be created.",
    )
    parser.add_argument(
        "--data_paths", default=rag_config.data_paths, help="Path to the data files."
    )
    parser.add_argument(
        "--to_embed",
        nargs="+",
        required=True,
        help="List of indices to embed.",
    )

    args = parser.parse_args()

    # update rag_config with command line arguments
    for k, v in vars(args).items():
        if hasattr(rag_config, k):
            setattr(rag_config, k, v)

    embed_chroma(args.collection_name, args.data_paths, args.to_embed, rag_config)

    logger.info("Embedding and indexing complete.")
