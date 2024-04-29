from enum import Enum
from ast import literal_eval
from typing import List
import json

from cohere.responses.rerank import RerankResult

# Modify this
class MetadataFields(Enum):
    """Maps metadata for each source to a tuple with the source name and a description."""

    ESMO = set("article_source unique_article_uuid title publication_date source".split())
    MEDITRON = set("article_source unique_article_uuid id source title url".split())
    ONKOPEDIA_DE = set("article_source unique_article_uuid title source".split())
    ONKOPEDIA_EN = set("article_source unique_article_uuid title publication_date source".split())
    ASCO = set("article_source unique_article_uuid title publication_date source".split())
    PUBMED = set("article_source, unique_article_uuid, Title of this paper, Journal it was published in:, URL".split(", "))


### RAGent specific constraints ###

def length_check(lst: str, max_len: int = 2) -> bool:
    return len(literal_eval(lst)) <= max_len


def deduplicate(passages: List[RerankResult]) -> List[RerankResult]:
    "adapted from: https://stackoverflow.com/a/480227/1493011"
    "Deduplicate a list of dotdicts while preserving order."
    seen = set()
    return [x for x in passages if not (x.document["text"] in seen or seen.add(x.document["text"]))]


def custom_to_list(lst: str) -> List[str]:
    # this method is unsafe
    s = lst.strip("[]")
    elems = s.split("', '")
    list_elements = [elem.strip("'") for elem in elems]
    return list_elements


def is_list_valid(lst: str) -> bool:
    """Return True if the provided string can actually be interpreted as a
    valid python list with all entries being strings."""
    for parser in (literal_eval, json.loads, custom_to_list):
        try:
            evaluated = parser(lst)
            if isinstance(evaluated, list) and all(isinstance(x, str) for x in evaluated):
                return True
        except Exception:
            pass
    return False


def to_list(lst: str) -> List[str]:
    """Return the list if the provided string can actually be interpreted as a
    valid python list with all entries being strings."""
    for parser in (literal_eval, json.loads, custom_to_list):
        try:
            evaluated = parser(lst)
            if isinstance(evaluated, list) and all(isinstance(x, str) for x in evaluated):
                return evaluated
        except Exception:
            ...
