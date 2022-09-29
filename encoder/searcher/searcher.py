import os
import requests
from typing import List
from encoder.utils.file import JSONCache, JSONStreamCache
from encoder.utils.settings import preprocess_cache_dir


class ScaleSerpSearcher:
    def __init__(self, query_name: str, queries: List[str]):
        self.queries = queries
        if "SCALE_SERP_API_KEY" not in os.environ:
            raise ValueError("SCALE_SERP_API_KEY not set in environment")
        self.api_key = os.getenv("SCALE_SERP_API_KEY")

        with JSONStreamCache(
            os.path.join(
                preprocess_cache_dir, f"{query_name}_scale_serp_search_result.json"
            ),
            list(range(len(self.queries))),
            self.generator,
            threads=32,
        ) as cache:
            self.search_raw_result = cache.data

        with JSONCache(
            os.path.join(
                preprocess_cache_dir, f"{query_name}_scale_serp_parse_result.json"
            ),
            self.parse_data,
            generate_args=(self.search_raw_result,),
        ) as cache:
            self.search_result = cache.data

    def parse_data(self, data):
        result = []
        for idx, entry in data.items():
            knowledge = []
            if "knowledge_graph" in entry["result"]:
                knowledge += self.parse_knowledge_graph(
                    entry["result"]["knowledge_graph"]
                )
            if "related_questions" in entry["result"]:
                knowledge += self.parse_related_questions(
                    entry["result"]["related_questions"]
                )
            if "organic_results" in entry["result"]:
                knowledge += self.parse_organic_results(
                    entry["result"]["organic_results"]
                )
            result.append(knowledge)
        return result

    def parse_knowledge_graph(self, knowledge_graph):
        knowledge = (
            [knowledge_graph["description"]] if "description" in knowledge_graph else []
        )
        for attribute in knowledge_graph.get("known_attributes", []):
            if "name" in attribute and "value" in attribute:
                knowledge.append(
                    f'{knowledge_graph["title"]} {attribute["name"]} {attribute["value"]}'
                )
        return knowledge

    def parse_related_questions(self, related_questions):
        return [
            related_question["answer"]
            for related_question in related_questions
            if "answer" in related_question
        ]

    def parse_organic_results(self, organic_results):
        return [
            organic_result["title"] + organic_result["snippet"]
            for organic_result in organic_results
            if "snippet" in organic_result
        ]

    def generator(self, idx):
        params = {
            "api_key": self.api_key,
            "q": self.queries[idx],
            "gl": "us",
            "google_domain": "google.com",
            "hl": "en",
            "include_answer_box": "true",
        }
        api_result = requests.get("https://api.scaleserp.com/search", params)
        return {"idx": idx, "query": self.queries[idx], "result": api_result.json()}
