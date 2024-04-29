import json
import re
import time
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, Union

import chromadb
import dspy
import nltk
from dspy.predict import Retry
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
from dspy.retrieve.qdrant_rm import QdrantRM
from icecream import ic
from llama_index import (
    Document,
    QueryBundle,
    ServiceContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI
from llama_index.node_parser import SentenceSplitter
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.retrievers import RecursiveRetriever, VectorIndexRetriever
from llama_index.schema import IndexNode, NodeWithScore
from llama_index.storage import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm

nltk.download("punkt")
from ast import literal_eval
from pathlib import Path
from typing import Literal, Optional

import openai
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from citations_utils import *
from llama_index import set_global_service_context
from llama_index.schema import NodeWithScore
from loguru_logger import logger
from nltk.tokenize import sent_tokenize
from rag_config import RAGConfig
from rag_utils import MetadataFields, deduplicate, is_list_valid, length_check, to_list
from signatures import *
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from utils import defaults, exists, file_len, read_jsonl_in_batches

PathLike = str | Path

DEFAULT_LLM_NAME = "gpt-4-0125-preview"
DEFAULT_EMBED_MODEL = "text-embedding-3-large"

__all__ = ["CollectionWrapper", "RAG", "RAGLoader"]


def clean_string(string):
    return string.encode("utf-8", "replace").decode("utf-8", "replace")


def yield_nodes_in_batches(nodes, batch_size: int = 1000):
    """Yield nodes in batches."""
    for i in range(0, len(nodes), batch_size):
        yield nodes[i : i + batch_size]


def wait_rand_exp_retry_dec(
    max_retries: int = 7,
    min_secs: int = 20,
    max_secs: int = 120,
) -> Callable[[Any], Any]:
    w = wait_random_exponential(min=min_secs, max=max_secs)
    s = stop_after_attempt(max_retries)

    return retry(
        reraise=True,
        stop=s,
        wait=w,
        retry=(
            retry_if_exception_type(
                (
                    openai.APITimeoutError,
                    openai.APIError,
                    openai.APIConnectionError,
                    openai.RateLimitError,
                    openai.APIStatusError,
                )
            )
        ),
        # before_sleep=logger.warning(f"Retrying after OpenAI API Refusal."),
    )


embed_retry_dec = wait_rand_exp_retry_dec()


class CollectionWrapper:

    """
    Lightweight wrapper around the ChromaDB collection.
    Handles the entire pipeline for adding documents from a file to the collection.
    There are no convenience functions for update, deletion etc, because this can be done with much less boilerplate code on the chromadb collection directly.
    """

    def __init__(
        self,
        *,
        chroma_client: chromadb.HttpClient = None,
        chroma_collection: Collection = None,
        rag_config: Optional[RAGConfig] = None,
        sub_chunk_sizes: List[int] = None,
    ) -> None:
        if not exists(rag_config):
            from rag_config import RAGConfig

            rag_config = RAGConfig()

        self._chroma_client = chroma_client
        self._collection = chroma_collection

        self._rag_config = rag_config

        self.node_parser = SentenceSplitter(
            chunk_size=rag_config.default_chunk_size,
            chunk_overlap=rag_config.default_chunk_overlap,
        )
        self.sub_node_parsers = [
            SentenceSplitter(
                chunk_size=c, chunk_overlap=rag_config.default_chunk_overlap
            )
            for c in sub_chunk_sizes
        ]

    def add_documents(self, data_file_path: Union[PathLike, List[PathLike]] = None):
        """Add documents from a file or list of files."""

        if isinstance(data_file_path, PathLike):
            data_file_path = [data_file_path]

        for file_path in data_file_path:
            logger.info(f"Adding documents from {file_path}.")

            with tqdm(total=file_len(file_path), unit="articles") as pbar:
                for total_count, data_batch in read_jsonl_in_batches(
                    file_path, batch_size=100
                ):  
                    logger.info(
                        f"Currently embedding at position {total_count} of items."
                    )

                    documents = self._create_documents(data_batch)
                    nodes = self._convert_documents_to_nodes(documents)

                    for nodes_batch in yield_nodes_in_batches(
                        nodes, batch_size=1000
                    ):  
                        try:
                            self.create_and_add_embeddings(nodes_batch)
                        except Exception as e:  # TODO: Better error handling
                            logger.error(f"Error uploading nodes: {e}.")
                            raise e

                    pbar.update(len(data_batch))

            logger.info(
                f"Added {len(nodes)} nodes to the collection. Currently collection length is {self._collection.count()} nodes."
            )

        logger.info("Upload complete.")

    @embed_retry_dec
    def create_and_add_embeddings(self, nodes_batch: List) -> None:
        self._collection.add(
            documents=[node.text for node in nodes_batch],
            metadatas=[node.metadata for node in nodes_batch],
            ids=[node.id_ for node in nodes_batch],
        )

    def _create_documents(self, data: List[Dict] = None):
        """Create a list of documents from a list of articles.
        Parameters:
            - data: A list of dictionaries where each dictionary represents an article.
        Returns:
            - documents: A list of Document objects.
        """
        documents = []
        for article in data:
            node_metadata = MetadataFields[article["article_source"].upper()].value

            document = Document(
                text=article["clean_text"],
                metadata={
                    key: article[key]
                    for key in node_metadata
                    if article.get(key, None) is not None
                },
                metadata_seperator="::",
                metadata_template="{key}=>{value}",
            )

            documents.append(document)

        return documents

    def _convert_documents_to_nodes(self, documents: List[Document]):
        """Parse documents into a hierarchical structure of nodes."""

        base_nodes = self.node_parser.get_nodes_from_documents(documents)

        all_nodes = []

        for base_node in base_nodes:
            for n in self.sub_node_parsers:
                sub_nodes = n.get_nodes_from_documents([base_node])
                sub_inodes = [
                    IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                ]
                all_nodes.extend(sub_inodes)

            original_node = IndexNode.from_text_node(base_node, base_node.node_id)
            all_nodes.append(original_node)

        return all_nodes


class RAG(dspy.Module):
    def __init__(
        self,
        retrieve_top_k: int = 40,
        rerank_top_k: int = 10,
        final_rerank_top_k: int = 40,
        default_rerank_model: str = "rerank-english-v2.0",
        ref_node_conf: Dict = None,
        check_citations: bool = False,
    ) -> None:
        super().__init__()

        # RAGConfig
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k
        self.max_n = final_rerank_top_k
        self.ref_node_conf = defaults(
            ref_node_conf, dict(chunk_size=512, chunk_overlap=20)
        )

        # DSPY modules
        self.subquery_gen = dspy.ChainOfThought(Search)
        self.ask_for_more = dspy.ChainOfThought(RequireInput)
        self.generate_answer_strategy = dspy.ChainOfThought(AnswerStrategy)
        self.generate_cited_response = dspy.Predict(GenerateCitedResponse)
        self.generate_suggestions = dspy.Predict(Suggestions)

        self.rerank_model_name = default_rerank_model
        self.check_citations = check_citations

    def forward(
        self,
        question: str = None,
        patient_context: Optional[str] = None,
        tool_results: Optional[str] = None,
        agent_tools: List[str] = None,
        rerank_model: CohereRerank = None,
    ):
        """Forward pass"""

        assert exists(question), "Question must be provided."

        patient_context = defaults(
            str(patient_context), "There is no relevant patient context."
        )  # instead we could also instruct the InputField(desc="Ignore if N/A")
        tool_results = defaults(str(tool_results), "No tools were used.")  # Same.
        agent_tools = defaults(agent_tools, [])

        subqueries = self.subquery_gen(
            question=question, context=patient_context, tool_results=tool_results
        ).searches

        logger.info(f"Generated Subqueries: {subqueries}")

        flagged_invalid = False
        while not is_list_valid(subqueries):
            if not flagged_invalid:
                flagged_invalid = True
                logger.warning("Subqueries are not valid. Trying to fix them.")

            dspy.Suggest(
                is_list_valid(subqueries),
                f"Assert that your searches can be interpreted as valid python list using eval / ast.literal_eval.",
                target_module=Search,
            )

        subqueries = to_list(subqueries)

        assert isinstance(subqueries, List), "Subqueries must be a list of strings."

        context: List[RerankResult] = []

        # retrieve for every subquery
        for idx, search in enumerate(subqueries, start=1):
            logger.info(f"Searching # {idx}, Search: {search}")

            passages = dspy.Retrieve(k=self.retrieve_top_k)(search).passages

            passages = rerank_model.rerank(
                query=search,
                documents=passages,
                top_n=self.rerank_top_k,
                model=self.rerank_model_name,
            )

            context = deduplicate(context + [p for p in passages])

        # context = self.co.rerank(
        #     query=patient_context + "\n" + tool_results + "\n" + question,
        #     documents=[c.document for c in context],
        #     top_n=self.max_n,
        #     model=self.rerank_model,
        # )

        # logger.info(f"# Context nodes after Rerank: {len(context)}")

        context_nodes = create_reference_nodes(context, self.ref_node_conf)

        logger.info(
            f"# Context nodes after splitting into reference nodes: {len(context_nodes)}"
        )

        medical_context = [n.document["text"] for n in context_nodes]

        agent_tools = "These tools are available to you:" + str(
            [tool["description"] for tool in agent_tools]
        )

        data = dict(
            context=medical_context,
            patient="Patient:\n" + patient_context,
            tool_results="Tool:\n" + tool_results,
            tools=agent_tools,
            question="Question:\n" + question,
        )

        answer_strategies = self.generate_answer_strategy(**data)
        data.pop("context")
        ask_for_more = self.ask_for_more(**data)

        logger.info("Ask for more information: {}", ask_for_more)
        logger.info("CoT to structure the answer: {}", answer_strategies)

        pred = self.generate_cited_response(
            strategy=answer_strategies.response,  # + ask_for_more.response,
            context=medical_context,
            patient="Patient:\n" + patient_context,
            tool_results="Tool:\n" + tool_results,
            question="Question:\n" + question,
        )

        pred = dspy.Prediction(
            response=pred.response, context=medical_context, context_nodes=context_nodes
        )

        if self.check_citations:
            dspy.Suggest(
                citations_check(pred.response),
                f"Make sure every 1-2 sentences has correct citations. If any 1-2 sentences have no citations, add them in 'text... [x].' format.",
                target_module=GenerateCitedResponse,
            )

            _, invalid_responses = citation_faithfulness(pred, None)
            if invalid_responses:
                invalid_pairs = [
                    (
                        output["text"],
                        output.get("context"),
                        output.get("error"),
                        output.get("rationale"),
                    )
                    for output in invalid_responses
                ]

                logger.warning(
                    "Currently having: {} invalid pairs of response <-> references.",
                    len(invalid_pairs),
                )

                for _, context, error, rationale in invalid_pairs:
                    msg = (
                        f"Make sure your output is based on the following context: '{context}'."
                        if exists(context)
                        else f"Make sure your output does not produce the following error: '{error}'."
                    )
                    if exists(rationale):
                        msg += f"The mistake you made was: {rationale}"
                        logger.warning(
                            "The model made the following mistake when checking citations: {}",
                            msg,
                        )

                    dspy.Suggest(
                        len(invalid_pairs) == 0,
                        msg,
                        target_module=GenerateCitedResponse,
                    )
            # Check citations

        suggestions = self.generate_suggestions(
            response=pred.response, recommendations=ask_for_more.response
        )
        final_response = str(pred.response) + "\n\n" + str(suggestions.suggestions)
        pred = dspy.Prediction(
            response=final_response,
            context=medical_context,
            context_nodes=context_nodes,
        )

        logger.info("Final response: {}", pred.response)

        return pred


def load_rag(
    retrieve_top_k: int = None,
    rerank_top_k: int = None,
    final_rerank_top_k: int = None,
    default_rerank_model: str = None,
    ref_node_conf: Dict[str, List[int]] = None,
    check_citations=False,
) -> RAG:
    rag_config = RAGConfig()

    rag = RAG(
        retrieve_top_k=defaults(retrieve_top_k, rag_config.retrieve_top_k),
        rerank_top_k=defaults(rerank_top_k, rag_config.rerank_top_k),
        final_rerank_top_k=defaults(final_rerank_top_k, rag_config.final_rerank_top_k),
        default_rerank_model=defaults(
            default_rerank_model, rag_config.default_rerank_model
        ),
        ref_node_conf=defaults(ref_node_conf, rag_config.ref_node_conf),
        check_citations=defaults(check_citations, rag_config.check_citations),
    )

    rag = assert_transform_module(rag.map_named_predictors(Retry), backtrack_handler)

    return rag
