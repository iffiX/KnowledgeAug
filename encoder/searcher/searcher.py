import os
import re
import requests
from typing import List, Tuple
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
            self.search_result = cache.data  # type: Tuple[str, str]

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
            [(knowledge_graph["description"], knowledge_graph["description"])]
            if "description" in knowledge_graph
            else []
        )
        for attribute in knowledge_graph.get("known_attributes", []):
            if (
                "name" in attribute
                and "value" in attribute
                and not attribute["name"].startswith("View")
                and not attribute["value"].startswith("http")
                and not attribute["value"].endswith(".com")
            ):
                knowledge.append(
                    (
                        f'{knowledge_graph["title"]} {attribute["name"]}',
                        f'{knowledge_graph["title"]} {attribute["name"]} {attribute["value"]}',
                    )
                )
        return knowledge

    def parse_related_questions(self, related_questions):
        return [
            (related_question["question"], related_question["answer"])
            for related_question in related_questions
            if "answer" in related_question
            and related_question["answer"].count(" ") >= 2
        ]

    def parse_organic_results(self, organic_results):
        raw_result = []
        result = []
        key_count = {}
        for organic_result in organic_results:

            if "snippet" in organic_result:
                if re.search(
                    "(ebay|amazon|etsy|walmart|homedepot|buy|product|proddetail|shop|youtube|calculator)",
                    organic_result["link"],
                ):
                    continue
                # Remove date of search
                match = re.match(
                    "^[a-zA-Z]+ [0-9]+, [0-9]+(.*)", organic_result["snippet"]
                )
                if match is not None:
                    parsed = match.group(1)
                else:
                    parsed = organic_result["snippet"]
                if " \u2014 " in parsed:
                    parsed = parsed[parsed.find(" \u2014 ") + len(" \u2014 ") :]

                # Some snippets contain another date, remove it
                match = re.match("^[a-zA-Z]+ [0-9]+, [0-9]+(.*)", parsed)
                if match is not None:
                    parsed = match.group(1)

                if (
                    parsed.count(" ") < 2
                    or parsed.count("...") > 1
                    or parsed.startswith("Youtube")
                    or (
                        re.match("^(What|Which|Where|When|Why|Who|Whose|How) ", parsed)
                        and parsed.endswith("?")
                    )
                ):
                    continue

                if "title" in organic_result:
                    key = organic_result["title"]
                    end = organic_result["title"].find(" - ")
                    key = key[: end if end != -1 else None]
                    end = organic_result["title"].find(" | ")
                    key = key[: end if end != -1 else None]
                else:
                    key = parsed

                if key == "Untitled":
                    key = parsed

                raw_result.append((key, parsed))
                if key.lower() not in key_count:
                    key_count[key.lower()] = 0
                key_count[key.lower()] += 1

        for key, parsed in raw_result:
            if key_count[key.lower()] > 1:
                result.append((parsed, parsed))
            else:
                result.append((key, parsed))
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
