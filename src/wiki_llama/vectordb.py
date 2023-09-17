from typing import List, Dict, Tuple

import torch
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def split_contents(pages: List[Dict],
                   chunk_size: int,
                   chunk_overlap: int,
                   separators: List | None) -> Tuple[List[str], List[Dict]]:
    # Use default separators if not passed
    if separators is None:
        separators = ["\n\n", "\n", "\. "]
    splitter = RecursiveCharacterTextSplitter(separators=separators,
                                              chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap,
                                              length_function=len)
    metadata = []
    texts = []
    for p in pages:
        chunks = splitter.split_text(p['content'])
        # Assign metadata for each chunk
        metadata.extend([{key: val for key, val in p.items() if key != 'content'}] * len(chunks))
        texts.extend(chunks)

    return texts, metadata


def create_faiss_splits_db(wiki_data: List[Dict],
                           embedding_model: str = "all-MiniLM-L6-v2",
                           chunk_size=512,
                           chunk_overlap=0,
                           chunk_separators: List | None = None) -> FAISS:
    print('Creating splits retrieval database...')
    texts, metadata = split_contents(wiki_data,
                                     chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap,
                                     separators=chunk_separators)

    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model)
    db = FAISS.from_texts(texts=texts,
                          embedding=embedding_model,
                          metadatas=metadata,
                          )
    return db


def create_faiss_summary_db(wiki_data: List[Dict],
                            embedding_model: str = "all-MiniLM-L6-v2",
                            ) -> FAISS:
    print('Creating pages retrieval database...')
    texts = [data['summary'] for data in wiki_data]
    metadata = wiki_data.copy()
    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model)
    db = FAISS.from_texts(texts=texts,
                          embedding=embedding_model,
                          metadatas=metadata,
                          )
    return db


def find_most_relevant_page(wiki_data: List[Dict],
                            prompt: str,
                            embedding_model: str = "all-MiniLM-L6-v2") -> Dict:

    db = create_faiss_summary_db(wiki_data=wiki_data, embedding_model=embedding_model)
    page = db.similarity_search(prompt, k=1)[0]
    # Free up GPU memory
    del db
    torch.cuda.empty_cache()
    return page.metadata


def find_most_relavant_passages(wiki_data: Dict,
                                prompt: str,
                                k: int,
                                embedding_model: str = "all-MiniLM-L6-v2",
                                chunk_size: int = 512,
                                chunk_overlap: int = 0,
                                chunk_separators: List | None = None)->List[str]:
    db = create_faiss_splits_db(wiki_data=[wiki_data],
                                embedding_model=embedding_model,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                chunk_separators=chunk_separators)
    # Free up GPU memory
    passages = []
    for psg in db.similarity_search(prompt, k=k):
        passages.append(psg.page_content)
    del db
    torch.cuda.empty_cache()
    return passages

