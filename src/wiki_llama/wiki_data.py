from typing import List, Dict

import wikipedia


def get_wiki_data(query: str) -> List[Dict]:
    wiki_data = []
    for search_result in wikipedia.search(query):
        try:
            page = wikipedia.page(search_result)
        except:
            continue
        wiki_data.append({'title': search_result, 'content': page.content, 'summary': page.summary,
                          'url': page.url})
    return wiki_data
