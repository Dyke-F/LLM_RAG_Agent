import os
from dataclasses import dataclass, field
from typing import List, Dict

from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from dotenv import load_dotenv
load_dotenv()

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"), model_name="text-embedding-3-large"
)

# might fail splitting metadata if values are set too small 
def get_sub_chunk_sizes():
    """Get the default sub chunk sizes for each source."""
    return {
        "esmo": [128, 256, 512],
        "asco": [128, 256, 512],
        "meditron": [512],
        "onkopedia_de": [128, 256, 512],
        "onkopedia_en": [128, 256, 512],
    } # MODIFY THIS

# MODIFY THIS
@dataclass
class RAGConfig:
    """Default configurations for RAG model."""

    default_client_path: str = "./chroma_db_oncology"  # path to the Chroma DB client
    collection_name: str = "oncology_db"  # Name of the collection to be created
    distance_metric: str = "cosine"  # "cosine" or "l2" or "ip"
    data_paths: str = "deduplicated_oncology_data"  # directory where the data (*.jsonl) is stored
    default_chunk_size: int = 1024 # default spit size for the text tokens
    default_chunk_overlap: int = 50 # how much overlap between splits
    default_sub_chunk_sizes: List[int] = field(default_factory=lambda: [128, 256, 512])
    retrieve_top_k: int = 40 # how many documents to retrieve at each collection.get call
    rerank_top_k: int = 10 # how many documents after reranking
    final_rerank_top_k: int = 40 # how many documents after final reranking
    check_citations: bool = False # wheter to use RAG citation checking

    default_embed_model: str = "text-embedding-3-large"
    default_rerank_model: str = "rerank-english-v2.0"
    default_llm_name: str = "gpt-4-0125-preview"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    reference_node_chunk_size: int = 512
    reference_node_chunk_overlap: int = 20

    sub_chunk_sizes: Dict[str, List[int]] = field(
        default_factory=get_sub_chunk_sizes
    )

    embedding_function: EmbeddingFunction = embedding_function

    def __post_init__(self):
        self.llm_kwargs = dict(
            temperature=self.llm_temperature, max_tokens=self.llm_max_tokens
        )
        self.ref_node_conf = dict(
            chunk_size=self.reference_node_chunk_size,
            chunk_overlap=self.reference_node_chunk_overlap,
        )
