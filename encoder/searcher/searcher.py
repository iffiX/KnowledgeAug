import os
import re
import requests
from typing import List
from encoder.utils.file import JSONCache, JSONStreamCache
from encoder.utils.settings import preprocess_cache_dir


class ScaleSerpSearcher:
    def __init__(self, query_name: str, queries: List[str]):
        self.queries = queries

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
        for i in range(len(data)):
            entry = data[i]
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
        result = []
        for organic_result in organic_results:

            if "snippet" in organic_result:
                match = re.match(
                    "^[a-zA-Z]+ [0-9]+, [0-9]+ — (.*)", organic_result["snippet"]
                )
                if match is not None:
                    parsed = match.group(1)
                else:
                    parsed = organic_result["snippet"]
                # if "title" in organic_result:
                #     parsed = (
                #         organic_result["title"][: organic_result["title"].find(" - ")]
                #         + " | "
                #         + parsed
                #     )
                result.append(parsed)
        return result

    def generator(self, idx):
        if "SCALE_SERP_API_KEY" not in os.environ:
            raise ValueError("SCALE_SERP_API_KEY not set in environment")
        params = {
            "api_key": os.getenv("SCALE_SERP_API_KEY"),
            "q": self.queries[idx],
            "gl": "us",
            "google_domain": "google.com",
            "hl": "en",
            "include_answer_box": "true",
        }
        api_result = requests.get("https://api.scaleserp.com/search", params)
        return {"idx": idx, "query": self.queries[idx], "result": api_result.json()}
