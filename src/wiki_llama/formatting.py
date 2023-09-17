import importlib
from typing import Dict, List
import textwrap


def format_search_query_template(prompt: str) -> str:
    text = importlib.resources.files('templates').joinpath('query_wiki.txt').read_text()
    text = text.format(prompt=prompt)
    return text


def extract_queries_from_answer(answer: str) -> str:
    query = answer.partition('[/INST] [query]')[-1]
    return query


def format_wiki_answer_template(prompt: str, title: str, extracts: str) -> str:
    text = importlib.resources.files('templates').joinpath('answer_using_wiki.txt').read_text()
    text = text.format(prompt=prompt, title=title, extracts=extracts)
    return text


def extract_agent_answer(answer: str) -> str:
    query = answer.partition('[/INST]')[-1]
    return query


def display_result(answer: str, metadata: Dict, extracts: List[str], max_row_length: int = 80) -> None:
    """Format the results for visualization.

    Args:
        answer (str): the generated answer.
        metadata (dict): the page metadata.
        extracts (str): the extracts from the pages article.
        max_row_length (int): max text length per row in the output.
    """
    out_string = ''
    for i in range(len(extracts)):
        out_string = out_string + f'Extract_{i}:{extracts[i]} \n'
    output = textwrap.fill(answer, width=max_row_length) + " \n\n" + "RETRIEVED WIKIPEDIA PAGE: \n"
    metadata_info = " \n".join([key + ": " + val for key, val in metadata.items() if key not in ['content', 'summary']])
    print(output + metadata_info + "\nRetrieved extracts: \n" + textwrap.fill(out_string, width=max_row_length))
