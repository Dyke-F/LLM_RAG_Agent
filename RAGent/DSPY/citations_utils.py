# Code in parts from: https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/examples/longformqa/longformqa_assertions.ipynb

import os

import dspy
import nltk
import regex as re

nltk.download("punkt")
from collections import defaultdict
from typing import Dict, List, Optional

from cohere.responses.rerank import RerankResult
from llama_index.node_parser import SentenceSplitter
from llama_index.schema import MetadataMode, NodeWithScore, TextNode
from nltk.tokenize import sent_tokenize
from signatures import CheckCitationFaithfulness

# do not enforce citations to literature on patient / tool data
skip_citations = ("Patient", "Tool")


def create_reference_nodes(
    nodes: List[RerankResult], ref_split_kwargs: Optional[Dict[str, int]] = None
) -> List[NodeWithScore]:
    """Prepend the source number to the context nodes text field."""

    from copy import deepcopy

    if not isinstance(nodes, List):
        nodes = [nodes]

    reference_nodes: List[RerankResult] = []
    text_splitter = (
        SentenceSplitter(**ref_split_kwargs) if ref_split_kwargs else SentenceSplitter()
    )

    def _split(nodes: List[RerankResult]) -> List[RerankResult]:
        idx = 1
        for node in nodes:
            text_splits = text_splitter.split_text(node.document["text"])

            for text_split in text_splits:
                text = f"Source {idx}:\n{text_split}"
                document_copy = deepcopy(node.document)

                document_copy["text"] = text

                ref_node = RerankResult(
                    document=document_copy, relevance_score=node.relevance_score
                )
                reference_nodes.append(ref_node)

                idx += 1

        return reference_nodes

    return _split(nodes)


# the following code is adapted from:
# https://github.com/stanfordnlp/dspy/tree/main/examples/longformqa


def extract_text_by_citation(paragraph):
    """Return dictionary with citation number as key and paragraph text as value."""

    citation_regex = re.compile(r"(.*?)(\[\d+\]\.)", re.DOTALL)
    parts_with_citation = citation_regex.findall(paragraph)

    citation_dict = defaultdict(list)
    for part, citation in parts_with_citation:
        part = part.strip()
        citation_num = re.search(r"\[(\d+)\]\.", citation).group(1)
        citation_dict[str(int(citation_num))].append(part)

    return citation_dict


def correct_citation_format(paragraph):
    """Check if the paragraph has citations in the correct format."""
    modified_sentences = []
    sentences = sent_tokenize(paragraph)  # split into sentences
    for sentence in sentences:
        modified_sentences.append(sentence)
    citation_regex = re.compile(r"\[\d+\]\.")
    skip_citation_regex = re.compile(
        r"\b(?:{})\b".format("|".join(skip_citations))
    )  # regex for skip_citations

    i = 0
    if len(modified_sentences) == 1:  # check if we only have 1 sentence
        has_citation = bool(citation_regex.search(modified_sentences[i]))
    while i < len(modified_sentences):
        if (
            len(modified_sentences[i : i + 2]) == 2
        ):  # check if we dont fall of the end ?
            sentence_group = " ".join(modified_sentences[i : i + 2])
            has_citation = bool(citation_regex.search(sentence_group))
            has_skip_citation = bool(skip_citation_regex.search(sentence_group))
            if not has_citation and not has_skip_citation:
                return False
            # if we find citation within 2 sentences, move on 2; else check next sentence
            i += (
                2
                if has_citation
                and i + 1 < len(modified_sentences)
                and citation_regex.search(modified_sentences[i + 1])
                else 1
            )
        else:
            return True
    return True


def has_citations(paragraph):
    """Check if the paragraph has citations."""
    numeric_citation = bool(re.search(r"\[\d+\]\.", paragraph))
    skip_citation = bool(
        re.search(r"\b(?:{})\b".format("|".join(skip_citations)), paragraph)
    )

    return numeric_citation or skip_citation


def citations_check(paragraph):
    """Check if the paragraph has citations and if they are in the correct format."""
    return has_citations(paragraph) and correct_citation_format(paragraph)


def citation_faithfulness(response, trace):
    """Check wheter a given response is faithful to the provided context. This
    Args:
        response (str): The response to check
        trace: The trace in case we need to backtrack
    Returns:
        bool: Whether the response is entirely faithful
        list: A list of unfaithful citations
    """

    paragraph, context = response.response, response.context
    citation_dict = extract_text_by_citation(
        paragraph
    )  # TODO: rewrite; are the items a list???

    if not citation_dict:
        return False, None

    context_num_regex = re.compile(r"Source (\d+):\n")
    context_split_regex = re.compile(r"Source \d+:\n")

    source_matches = [context_num_regex.search(ctx) for ctx in context]
    source_numbers = [match.group(1) for match in source_matches if match]
    context = [
        re.split(context_split_regex, ctx)[-1]
        if context_split_regex.search(ctx)
        else ctx
        for ctx in context
    ]

    assert len(source_matches) == len(
        source_numbers
    ), f"Length of source numbers {len(source_matches)} unequals length of context numbers {len(source_numbers)}."
    assert len(source_numbers) == len(
        context
    ), f"Length of source numbers {len(source_numbers)} unequals length of context nodes {len(context)}."

    context_dict = {
        str(source_num): ctx
        for source_num, ctx in zip(source_numbers, context, strict=True)
    }  # TODO: strict is superfluous here

    faithfulness_results = []
    unfaithful_citations = []

    check_citation_faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)

    for citation_num, texts in citation_dict.items():
        # case: the model hallucinates a citation
        if citation_num not in context_dict.keys():
            unfaithful_citations.append(
                {"paragraph": paragraph, "text": texts, "context": None}
            )
            continue

        current_context = context_dict[citation_num]

        for text in texts:
            try:
                result = check_citation_faithfulness(context=current_context, text=text)
                is_faithful = (
                    result.faithfulness.lower() == "true"
                )  # we could also dspy.Suggest this ...
                faithfulness_results.append(is_faithful)

                if not is_faithful:
                    unfaithful_citations.append(
                        {
                            "paragraph": paragraph,
                            "text": text,
                            "context": current_context,
                            "rationale": result.rationale,
                        }
                    )

            except ValueError as e:
                faithfulness_results.append(False)
                unfaithful_citations.append(
                    {"paragraph": paragraph, "text": text, "error": str(e)}
                )

    final_faithfulness = all(faithfulness_results)

    if not faithfulness_results:
        return False, None

    return final_faithfulness, unfaithful_citations
