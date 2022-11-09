import tqdm
import logging
import torch as t
from torch import multiprocessing
from torch.multiprocessing.spawn import _wrap, ProcessContext
from encoder.models.embedder import Embedder


def start_processes_with_seperate_args(
    fn, processes_args=((),), nprocs=1, join=True, daemon=False, context=None
):
    assert len(processes_args) == nprocs
    mp = context or multiprocessing.get_context("spawn")
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap, args=(fn, i, processes_args[i], error_queue), daemon=daemon,
        )
        error_queues.append(error_queue)
        processes.append(process)

    for process in processes:
        process.start()

    context = ProcessContext(processes, error_queues)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass


class FactSelectorParallelContext:
    worker_id: int = None
    worker_num: int = None
    query_embeddings: t.Tensor = None
    min_score: float = None
    max_facts: int = None
    embedder: Embedder = None


class FactSelector:
    def __init__(
        self,
        queries,
        facts,
        min_score: float = 0.5,
        max_facts: int = 50,
        chunk_size: int = 256,
        batch_size: int = 131072,
        inner_batch_size: int = 16384,
    ):
        self.queries = queries
        self.facts = facts
        self.min_score = min_score
        self.max_facts = max_facts
        self.selected_facts = []
        self.selected_facts_rank = []
        self.query_embeddings = None
        self.process_input_queues = []
        self.process_output_queues = []
        self.process_context = None
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.inner_batch_size = inner_batch_size
        self.select_facts()

    def select_facts(self):
        print("Selecting facts")
        self.initialize_processes(
            func=self.compute_embeddings_worker,
            initializer=self.compute_embeddings_initializer,
        )

        with t.no_grad():
            print("Computing embeddings for queries")
            self.query_embeddings = self.compute_embeddings(
                self.queries, chunk_size=self.chunk_size
            ).to("cuda:0")

            query_rank_of_facts = [
                t.full([len(self.queries), self.max_facts], -1, dtype=t.long).to(
                    "cuda:0"
                ),
                t.full([len(self.queries), self.max_facts], -1e6, dtype=t.float32).to(
                    "cuda:0"
                ),
            ]
            for batch_start in range(0, len(self.facts), self.batch_size):
                logging.info(
                    f"Processing fact starting at line {batch_start} / {len(self.facts)}, "
                    f"percent {100* batch_start / len(self.facts):.2f} %"
                )
                partial_fact_embeddings = (
                    self.compute_embeddings(
                        self.facts[batch_start : batch_start + self.batch_size],
                        chunk_size=self.chunk_size,
                    )
                    .transpose(0, 1)
                    .to("cuda:0")
                )
                for inner_batch_start in range(
                    0, self.batch_size, self.inner_batch_size
                ):
                    score = t.mm(
                        self.query_embeddings,
                        partial_fact_embeddings[
                            :,
                            inner_batch_start : inner_batch_start
                            + self.inner_batch_size,
                        ],
                    )
                    query_rank_of_facts = self.update_rank(
                        batch_start + inner_batch_start, query_rank_of_facts, score
                    )

        self.stop_processes()
        query_rank_of_facts = [
            query_rank_of_facts[0].to("cpu").tolist(),  # indices
            query_rank_of_facts[1].to("cpu").tolist(),  # values
        ]
        self.selected_facts = [
            [self.facts[r] for r in qr if r != -1] for qr in query_rank_of_facts[0]
        ]
        self.selected_facts_rank = [
            [
                (query_rank_of_facts[0][i][j], query_rank_of_facts[1][i][j])
                for j in range(self.max_facts)
                if query_rank_of_facts[0][i][j] != -1
            ]
            for i in range(len(self.queries))
        ]

    def update_rank(self, batch_start, query_rank_of_facts, score):
        # First select top_max_facts number of facts for each query
        # Then add these facts as tuples of (fact_index, score) in to the rank list
        # Then re-rank them and filter by min_score and max_facts

        # Note: the two step method is chosen since the rank sub list length might
        # be uneven for each query, padding them into a matrix would be difficult
        partial_top_k = t.topk(score, k=min(self.max_facts, score.shape[1]))
        partial_top_k_indices = partial_top_k.indices + batch_start
        partial_top_k_values = partial_top_k.values
        indices = t.cat((query_rank_of_facts[0], partial_top_k_indices), dim=1)
        values = t.cat((query_rank_of_facts[1], partial_top_k_values), dim=1)
        relative_top_k = t.topk(values, k=self.max_facts)
        new_query_rank_of_facts = [
            t.gather(indices, 1, relative_top_k.indices),
            relative_top_k.values,
        ]

        return new_query_rank_of_facts

    def compute_embeddings(self, strings, chunk_size: int = 128):
        embeddings = self.parallel_run(
            [
                strings[split_start : split_start + chunk_size]
                for split_start in range(0, len(strings), chunk_size)
            ],
        )

        return embeddings

    @staticmethod
    def compute_embeddings_initializer():
        FactSelectorParallelContext.embedder = Embedder(
            device=f"cuda:{FactSelectorParallelContext.worker_id}"
        )

    @staticmethod
    def compute_embeddings_worker(string_batch):
        return FactSelectorParallelContext.embedder.embed(string_batch).cpu()

    def parallel_run(self, inputs):
        if self.process_context is None:
            raise RuntimeError("Process pool not initialized")
        process_num = len(self.process_context.processes)
        per_process_chunks = max((len(inputs) + process_num - 1) // process_num, 1)

        for worker_id, split_start in zip(
            range(process_num), range(0, len(inputs), per_process_chunks),
        ):
            self.process_input_queues[worker_id].put(
                inputs[split_start : split_start + per_process_chunks]
            )
        results = [
            self.process_output_queues[worker_id].get()
            for worker_id in range(process_num)
        ]
        return t.cat(results, dim=0)

    def initialize_processes(
        self, func, process_num=None, initializer=None, initargs=None,
    ):
        if self.process_context is not None:
            raise RuntimeError("Process pool already initialized")
        process_num = process_num or t.cuda.device_count()
        mp = multiprocessing.get_context("spawn")
        self.process_input_queues = [mp.SimpleQueue() for _ in range(process_num)]
        self.process_output_queues = [mp.SimpleQueue() for _ in range(process_num)]
        self.process_context = start_processes_with_seperate_args(
            self.parallel_executor,
            processes_args=[
                (
                    process_num,
                    initializer,
                    initargs,
                    func,
                    self.process_input_queues[i],
                    self.process_output_queues[i],
                )
                for i in range(process_num)
            ],
            nprocs=process_num,
            join=False,
            context=mp,
            daemon=True,
        )

    def stop_processes(self):
        for input_queue in self.process_input_queues:
            input_queue.put(None)
        self.process_context.join()

    @staticmethod
    def parallel_executor(
        worker_id, worker_num, initializer, initargs, func, input_queue, output_queue
    ):
        FactSelectorParallelContext.worker_id = worker_id
        FactSelectorParallelContext.worker_num = worker_num
        if initializer is not None:
            initargs = initargs or ()
            initializer(*initargs)
        while True:
            func_input = input_queue.get()
            if func_input is None:
                break
            else:
                if worker_id == 0:
                    with tqdm.tqdm(total=worker_num * len(func_input)) as bar:
                        result = []
                        for inp in func_input:
                            res = func(inp)
                            if res is not None:
                                result.append(res)
                            bar.update(worker_num)
                        output_queue.put(t.cat(result, dim=0))
                else:
                    output_queue.put(
                        t.cat(
                            [
                                x
                                for x in [func(inp) for inp in func_input]
                                if x is not None
                            ],
                            dim=0,
                        ),
                    )
