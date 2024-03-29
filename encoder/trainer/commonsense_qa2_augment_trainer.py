import os
import logging
import torch as t
from itertools import chain
from tqdm import tqdm
from multiprocessing import get_context
from torch.utils.data import DataLoader
from encoder.models.embedder import Embedder
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.commonsense_qa2 import (
    CommonsenseQA2BaseDataset,
    CommonsenseQA2AugmentDataset,
)
from encoder.prompter.commonsense_qa2 import CommonsenseQA2Prompter
from encoder.searcher.searcher import ScaleSerpSearcher
from encoder.utils.file import PickleCache
from encoder.utils.config import CommonsenseQA2AugmentTrainConfig
from encoder.utils.settings import preprocess_cache_dir
from .augment_base_trainer import AugmentBaseTrainer
from .utils import set_worker_sharing_strategy


class AugmentGenerator:
    instance = None

    def __init__(self, max_depth, augment_method, augment_use_parts):
        AugmentGenerator.instance = self
        self.max_depth = max_depth
        self.augment_method = augment_method
        self.augment_use_parts = augment_use_parts
        self.dataset = CommonsenseQA2BaseDataset(tokenizer=None)
        self.prompter = CommonsenseQA2Prompter(self.dataset)
        self.contexts = {}

        self.lowered_knowledge_to_knowledge = {}
        self.id_to_searches = {}
        self.id_to_knowledge_keys = {}
        self.knowledge_key_to_knowledge = {}
        self.top_facts_indices = {}

        self.contexts = {}
        with PickleCache(
            os.path.join(
                preprocess_cache_dir, f"commonsense_qa2_raw_sample_result.data"
            ),
            generate_func=self.generate,
        ) as cache:
            external_knowledge, raw_contexts = cache.data
            self.dataset.matcher.add_external_knowledge(external_knowledge)
            for id_, sample_result in raw_contexts.items():
                self.contexts[id_] = []
                for raw_path, raw_path_edges in sample_result:
                    all_path_edges = []
                    # only use the first level (corresponds to sample config)
                    for sub_path, sub_path_edges, _ in zip(
                        raw_path, raw_path_edges, range(1)
                    ):

                        for decoded_edge, edge in zip(sub_path, sub_path_edges):
                            if (
                                edge[0] < self.dataset.matcher.composite_start
                                and edge[2] < self.dataset.matcher.composite_start
                            ):
                                if (
                                    augment_use_parts == "all"
                                    or augment_use_parts == "only_non_composite"
                                ):
                                    all_path_edges.append(
                                        self.dataset.matcher.sub_paths_to_annotations(
                                            [[edge]],
                                            templates=augment_method,
                                            use_parts="all",
                                            prioritize_original_annotation=True,
                                        )[0][0]
                                    )
                            else:
                                if augment_use_parts == "all":
                                    all_path_edges.append(
                                        self.dataset.matcher.sub_paths_to_annotations(
                                            [[edge]],
                                            templates=augment_method,
                                            use_parts="all",
                                            prioritize_original_annotation=True,
                                        )[0][0]
                                    )
                                elif augment_use_parts == "only_composite":
                                    assert "related to" in decoded_edge
                                    node0_end = decoded_edge.find("related to")
                                    node1_start = node0_end + len("related to")
                                    node0 = decoded_edge[:node0_end]
                                    node1 = decoded_edge[node1_start:]
                                    node0 = node0.strip(" ")
                                    node1 = node1.strip(" ")
                                    if edge[0] >= self.dataset.matcher.composite_start:
                                        all_path_edges.append(node0)
                                    else:
                                        all_path_edges.append(node1)

                    if len(all_path_edges) > 0:
                        self.contexts[id_].append(", ".join(all_path_edges))

        empty_count = 0
        for data in chain(
            self.dataset.original_train_data,
            self.dataset.validate_data,
            self.dataset.test_data,
        ):
            if not self.contexts[data["id"]]:
                empty_count += 1
        total = (
            len(self.dataset.original_train_data)
            + len(self.dataset.validate_data)
            + len(self.dataset.test_data)
        )
        logging.info(
            f"{empty_count} contexts are empty, percent {empty_count * 100 / total:.2f} %"
        )

    def generate(self):
        queries = []
        query_ids = []
        for data in chain(
            self.dataset.original_train_data,
            self.dataset.validate_data,
            self.dataset.test_data,
        ):
            id_ = data["id"]
            hints = self.prompter.get_search_hints_of_id(id_)[0]
            if len(hints["search"]) > 1:
                for search in hints["search"]:
                    queries.append(" ".join(search))
                    query_ids.append(id_)
            else:
                queries.append(data["text_question"])
                query_ids.append(id_)

        searcher = ScaleSerpSearcher("commonsense_qa2", queries)
        search_result = [
            [(key, knowledge.lower()) for key, knowledge in sr]
            for sr in searcher.search_result
        ]
        # self.lowered_knowledge_to_knowledge = {
        #     knowledge.lower(): knowledge
        #     for sr in searcher.search_result
        #     for _, knowledge in sr
        # }
        external_knowledge = [knowledge for sr in search_result for _, knowledge in sr]
        self.dataset.matcher.add_external_knowledge(external_knowledge)

        embedder = Embedder()

        for data in chain(
            self.dataset.original_train_data,
            self.dataset.validate_data,
            self.dataset.test_data,
        ):
            id_ = data["id"]
            hints = self.prompter.get_search_hints_of_id(id_)[0]
            self.id_to_searches[id_] = [" ".join(search) for search in hints["search"]]

        for id_, sr in tqdm(zip(query_ids, search_result), total=len(query_ids)):
            if id_ not in self.id_to_knowledge_keys:
                self.id_to_knowledge_keys[id_] = []
            self.id_to_knowledge_keys[id_] += [key for key, _ in sr]
            for key, knowledge in sr:
                self.knowledge_key_to_knowledge[key] = knowledge

        logging.info("Computing embeddings")
        search_embeddings = self.embed_dict(self.id_to_searches, embedder)
        knowledge_keys_embeddings = self.embed_dict(self.id_to_knowledge_keys, embedder)
        for id_ in self.id_to_searches:
            if id_ in search_embeddings and id_ in knowledge_keys_embeddings:
                topk = t.topk(
                    t.mm(
                        search_embeddings[id_],
                        knowledge_keys_embeddings[id_].transpose(0, 1),
                    ),
                    k=min(3, knowledge_keys_embeddings[id_].shape[0]),
                    dim=1,
                )
                all_indices = []
                for indices, values in zip(topk.indices, topk.values):
                    sub_indices = []
                    for index, value in zip(indices, values):
                        if value > 0.5:
                            sub_indices.append(int(index))
                    all_indices.append(sub_indices)
                self.top_facts_indices[id_] = all_indices

        logging.info("Finding best paths")

        ctx = get_context("fork")
        contexts = {}
        with ctx.Pool() as pool:
            for id_, result in tqdm(
                pool.imap_unordered(self.generate_paths, self.id_to_searches.keys()),
                total=len(self.id_to_searches),
            ):
                contexts[id_] = result
        return external_knowledge, contexts

    @staticmethod
    def generate_paths(id_):
        self = AugmentGenerator.instance
        result = []
        if id_ not in self.top_facts_indices:
            return id_, result
        hints = self.prompter.get_search_hints_of_id(id_)[0]
        for starts, search, top in zip(
            hints["starts"], hints["search"], self.top_facts_indices[id_]
        ):
            for idx in top:
                best_knowledge_key = self.id_to_knowledge_keys[id_][idx]
                best_knowledge = self.knowledge_key_to_knowledge[best_knowledge_key]

                raw_path, raw_path_edges, *__ = self.dataset.matcher.find_shortest_path(
                    source_sentence=", ".join(starts),
                    target_sentence=", ".join(search),
                    intermediate_nodes=[best_knowledge],
                    max_depth_for_each_node=self.max_depth,
                )
                if len(raw_path) == 0:
                    continue
                result.append((raw_path, raw_path_edges))
        return id_, result

    @staticmethod
    def embed_dict(dictionary, embedder):
        all_strings = []
        all_ids = []
        for id_, strings in dictionary.items():
            all_strings += strings
            all_ids += [id_] * len(strings)

        embeddings = embedder.embed(all_strings)
        result = {}
        for embed, id_ in zip(embeddings, all_ids):
            if id_ not in result:
                result[id_] = []
            result[id_].append(embed)
        for id_ in result:
            result[id_] = t.stack(result[id_])
        return result


class CommonsenseQA2AugmentTrainer(AugmentBaseTrainer):
    def __init__(
        self,
        config: CommonsenseQA2AugmentTrainConfig,
        stage_result_path="./",
        is_distributed=False,
    ):
        super().__init__(
            2,
            config=config,
            stage_result_path=stage_result_path,
            is_distributed=is_distributed,
        )
        self.dataset = CommonsenseQA2AugmentDataset(
            tokenizer=self.tokenizer,
            use_augment=self.config.use_augment,
            augment_contexts=self.load_augment_contexts(
                self.config.max_depth,
                self.config.augment_method,
                self.config.augment_use_parts,
            ),
            max_seq_length=config.max_seq_length,
            output_mode="single" if "t5-" in config.base_type else "splitted",
        )
        self.dataloader_args = {
            "num_workers": self.config.load_worker_num,
            "prefetch_factor": self.config.load_prefetch_per_worker,
            "batch_size": self.config.batch_size,
            "collate_fn": collate_function_dict_to_batch_encoding,
            "worker_init_fn": set_worker_sharing_strategy,
        }

    @property
    def monitor(self):
        return "accuracy"

    @property
    def monitor_mode(self):
        return "max"

    @property
    def real_device(self):
        return self._real_device or self.device

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset.train_dataset, **self.dataloader_args,)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset.validate_dataset, **self.dataloader_args,
        )

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset.test_dataset, **self.dataloader_args,)

    def load_augment_contexts(self, max_depth, augment_method, augment_use_parts):
        return AugmentGenerator(max_depth, augment_method, augment_use_parts).contexts
