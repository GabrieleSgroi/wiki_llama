from typing import Dict, List, Tuple
import torch
from transformers import pipeline, AutoTokenizer
from wiki_llama.formatting import format_search_query_template, extract_queries_from_answer, \
    format_wiki_answer_template, extract_agent_answer
from wiki_llama.wiki_data import get_wiki_data

from wiki_llama.vectordb import find_most_relevant_page

from wiki_llama.vectordb import find_most_relavant_passages


class Agent:
    def __init__(self,
                 agent_model: str = "meta-llama/Llama-2-7b-chat-hf",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 **kwargs):
        """

        Args:
            agent_model (str): Llama 2 model to use. It can be either a local path or a path to the Hugging Face hub.
                               Models on the Hugging Face hub may require authentication.
            embedding_model (str): sentence transformer model name for text embedding.
            **kwargs: additional arguments to be passed to the Hugging Face Llama pipeline.
        """
        self.agent_pipeline = pipeline(
            "text-generation",
            model=agent_model,
            torch_dtype=torch.float16,
            device_map="auto",
            **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(agent_model)
        self.embedding_model = embedding_model
        self.query = None

    def retrieve_wiki_data(self,
                           prompt: str,
                           **kwargs) -> List[Dict]:
        """Retrieve pages from Wikipedia relevant to the prompt.

        Args:
            prompt (str): the user's prompt.
            **kwargs: additional arguments to be passed to the Hugging Face Llama pipeline.
        """
        input_text = format_search_query_template(prompt)
        answer = self.agent_pipeline(input_text,
                                     do_sample=True,
                                     num_return_sequences=1,
                                     eos_token_id=self.tokenizer.eos_token_id,
                                     **kwargs)[0]
        self.query = extract_queries_from_answer(answer=answer['generated_text'])
        print('Searching for pages using query:', self.query)
        wiki_data = get_wiki_data(query=self.query)
        print("Retrieved pages:\n", "\n".join([data['title'] for data in wiki_data]))
        torch.cuda.empty_cache()
        return wiki_data

    def answer_using_wiki(self,
                          prompt: str,
                          extracts: str,
                          title: str,
                          **kwargs) -> str:
        """Return the model response given the user's prompt and data extracted from Wikipedia.

        Args:
            prompt (str): the user's prompt.
            extracts (str): the extracts from Wikipedia.
            title(str): the title of the Wikipedia page.
            **kwargs: extra argument to be passed to the Hugging Face pipeline for  generation.
        """
        input_text = format_wiki_answer_template(prompt=prompt, title=title, extracts=extracts)
        answer = self.agent_pipeline(input_text,
                                     do_sample=True,
                                     num_return_sequences=1,
                                     eos_token_id=self.tokenizer.eos_token_id,
                                     **kwargs)[0]
        answer = extract_agent_answer(answer['generated_text'])
        return answer

    def __call__(self,
                 prompt: str,
                 n_extracts: int = 3,
                 extracts_size: int = 512,
                 max_query_tokens: int = 32,
                 max_answer_tokens: int = 256,
                 extracts_separators: List | None = None,
                 query_gen_kwargs: Dict | None = None,
                 answer_gen_kwargs: Dict | None = None
                 ) -> Tuple[str, Dict, str]:
        """Retrieve relevant pages from Wikipedia and generate a response to the user's prompt based on them.

        Args:
            prompt (str): the user's prompt.
            n_extracts (int): number of passages to extracts from the retrieved Wikipedia page.
            extracts_size (int): characters lenght of the extracts.
            max_query_tokens (int): max number of tokens generated for the search query.
            max_answer_tokens (int): max numebr of tokens generated in the model final answer.
            extracts_separators (List | None): separators to use to split the content into passages. If None the default
                                               list ["\n\n", "\n", "\. "] will be passed.
            query_gen_kwargs (Dict): additional arguments to be passed to the Hugging Face pipeline for the generation
                                     of the search queries.
            answer_gen_kwargs (Dict): additional arguments to be passed to the Hugging Face pipeline for the generation
                                      of the final answer.
        """

        if query_gen_kwargs is None:
            query_gen_kwargs = {}
        if answer_gen_kwargs is None:
            answer_gen_kwargs = {}

        wiki_data = self.retrieve_wiki_data(prompt=prompt,
                                            max_new_tokens=max_query_tokens,
                                            **query_gen_kwargs)
        page = find_most_relevant_page(wiki_data=wiki_data,
                                          prompt=prompt,
                                          embedding_model=self.embedding_model)
        passages = find_most_relavant_passages(wiki_data=page,
                                               prompt=prompt,
                                               k=n_extracts,
                                               embedding_model=self.embedding_model,
                                               chunk_size=extracts_size,
                                               chunk_overlap=0,
                                               chunk_separators=extracts_separators)
        joint_passages = " \n ".join(passages)
        print(f"Retrieved Wikipedia page:\n Title: {page['title']}")
        answer = self.answer_using_wiki(prompt=prompt,
                                        extracts=joint_passages,
                                        title=page['title'],
                                        max_new_tokens=max_answer_tokens,
                                        **answer_gen_kwargs
                                        )
        torch.cuda.empty_cache()
        return answer, page, passages
