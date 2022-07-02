import os
import re
import nltk
import tqdm
import logging
import subprocess
import multiprocessing
import torch as t
import numpy as np
from cleantext import clean
from nltk.corpus import words
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
from encoder.utils.settings import (
    bin_dir,
    model_cache_dir,
    proxies,
    huggingface_mirror,
    local_files_only,
)
from encoder.utils.file import download_to, decompress_zip


class FactFilterParallelContext:
    worker_id: int = None
    parser: nltk.CoreNLPParser = None
    words_set: set = None
    cola_model: PreTrainedModel = None
    cola_tokenizer: PreTrainedTokenizerBase = None
    sentiment_model: PreTrainedModel = None
    sentiment_tokenizer: PreTrainedTokenizerBase = None


class FactFilter:
    def __init__(self):
        # core_nlp_path = str(os.path.join(bin_dir, "stanford-corenlp-full-2018-10-05"))
        # if not os.path.exists(core_nlp_path):
        #     if not os.path.exists(core_nlp_path + ".zip"):
        #         download_to(
        #             "http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip",
        #             core_nlp_path + ".zip",
        #         )
        #     decompress_zip(core_nlp_path + ".zip", core_nlp_path)
        # logging.info(
        #     "Starting CoreNLP server, please make sure that java is installed."
        # )
        # self.corenlp_process = subprocess.Popen(
        #     (
        #         "java",
        #         "-mx4g",
        #         "-cp",
        #         "*",
        #         "edu.stanford.nlp.pipeline.StanfordCoreNLPServer",
        #         "-preload",
        #         "tokenize,ssplit,pos,lemma,ner,parse,depparse",
        #         "-status_port",
        #         "9000",
        #         "-port",
        #         "9000",
        #         "-timeout",
        #         "15000",
        #     ),
        #     cwd=str(os.path.join(core_nlp_path, "stanford-corenlp-full-2018-10-05")),
        #     stderr=subprocess.PIPE,
        # )
        # for line in iter(self.corenlp_process.stderr.readline, b""):
        #     line = line.decode("utf-8")
        #     print(line, end=" ")
        #     if "listening at" in line:
        #         break
        # logging.info("CoreNLP server started")

        pass

    def clean(self, facts):
        f = self.basic_clean(facts)
        f = self.remove_html_characters(f)
        f = self.replace_brackets_of_all_shapes_at_start(f)
        f = self.remove_non_alphabet_symbols_at_start(f)
        f = self.remove_non_alphanumerical_symbols_at_end(f)
        f = self.remove_escapes_and_consecutive_special_symbols(f)
        f = self.remove_questions(f)
        f = self.remove_facts_with_choices(f)
        f = self.remove_short_or_high_non_word_ratio_facts(f)
        f = self.remove_incomplete_sentences(f)
        # f = self.remove_non_neutral_sentences(f)
        f = self.replace_annotation_parenthesis_and_remove_single_brackets_of_all_shapes(
            f
        )
        f = self.remove_duplicates(f)
        logging.info(f"Finished, remaining facts {len(f)}, ratio {len(f) / len(facts)}")
        return f

    def basic_clean(self, facts):
        logging.info("Running stage basic_clean")
        return self.parallel_run(facts, self.basic_clean_worker)

    @staticmethod
    def basic_clean_worker(fact):
        return clean(
            fact,
            fix_unicode=True,  # fix various unicode errors
            to_ascii=True,  # transliterate to closest ASCII representation
            lower=True,  # lowercase text
            no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
            no_urls=True,  # replace all URLs with a special token
            no_emails=True,  # replace all email addresses with a special token
            no_phone_numbers=True,  # replace all phone numbers with a special token
            no_numbers=False,  # replace all numbers with a special token
            no_digits=False,  # replace all digits with a special token
            no_currency_symbols=False,  # replace all currency symbols with a special token
            no_punct=False,  # remove punctuations
            replace_with_punct="",  # instead of removing punctuations you may replace them
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            lang="en",
        )

    def remove_html_characters(self, facts):
        logging.info("Running stage remove_html_characters")
        result = []
        for fact in tqdm.tqdm(facts):
            fact = re.sub(r"&[\w\s\\#]+;", " ", fact)
            fact = re.sub(r"(& *nbsp|& *gt|& *lt|& *quot|& *amp)", " ", fact)
            result.append(fact)
        return result

    def remove_escapes_and_consecutive_special_symbols(self, facts):
        logging.info("Running stage remove_escapes_and_consecutive_special_symbols")
        result = []
        for fact in tqdm.tqdm(facts):
            fact = fact.replace("\\", "")
            fact = re.sub(
                r"([^\w.!\?\-\(\)\[\]{} ]{2,}|(\. ){2,}|[.]{2,}|[-]{3,}|[_]{2,})",
                " ",
                fact,
            )
            fact = re.sub(r"\s{2,}", " ", fact)
            result.append(fact)
        return result

    def replace_brackets_of_all_shapes_at_start(self, facts):
        # This stage is used for increasing the "linguistic acceptability" of some facts
        # in the dictionary format (eg: <concept>: <explanation)
        logging.info("Running stage replace_brackets_of_all_shapes_at_start")
        facts = [
            re.sub(r"^\((.*)\)|\[(.*)\]|{(.*)}", r"\1:", fact)
            for fact in tqdm.tqdm(facts)
        ]
        return facts

    def replace_annotation_parenthesis_and_remove_single_brackets_of_all_shapes(
        self, facts
    ):
        logging.info("Running stage replace_brackets_of_all_shapes")
        result = []
        for fact in tqdm.tqdm(facts):
            # include a space at start so we don't change brackets used in chemical expressions
            # this also ignores annotations inside annotations
            fact = re.sub(r" \((.*)\)|\[(.*)\]|{(.*)}", r" -- \1: --", fact)

            # remove incomplete brackets
            stack = []
            delete_pos = []
            for idx, char in enumerate(fact):
                if char in {"(", "[", "{"}:
                    stack.append((idx, char))
                elif char in {")", "]", "}"}:
                    while stack:
                        if (
                            (char == ")" and stack[-1][1] != "(")
                            or (char == "]" and stack[-1][1] != "[")
                            or (char == "}" and stack[-1][1] != "{")
                        ):
                            delete_pos.append(stack.pop(-1)[0])
                        else:
                            stack.pop(-1)
                            break
                    else:
                        delete_pos.append(idx)
            delete_pos = set(delete_pos + [s[0] for s in stack])
            fact = "".join(
                char for idx, char in enumerate(fact) if idx not in delete_pos
            )
            result.append(fact)
        return result

    def remove_non_alphabet_symbols_at_start(self, facts):
        logging.info("Running stage remove_non_alphabet_symbols_at_start")
        facts = [re.sub(r"^[^a-zA-Z]+", "", fact) for fact in tqdm.tqdm(facts)]
        return facts

    def remove_non_alphanumerical_symbols_at_end(self, facts):
        logging.info("Running stage remove_non_alphabet_symbols_at_end")
        facts = [
            re.sub(r"[^a-zA-Z0-9.!\?\)\]\}]+$", "", fact) for fact in tqdm.tqdm(facts)
        ]
        return facts

    def remove_facts_with_choices(self, facts):
        logging.info("Running stage remove_facts_with_choices")
        new_facts = []
        for fact in tqdm.tqdm(facts):
            if not (
                re.search(f"a\. ", fact)
                or re.search(f"b\. ", fact)
                or re.search(f"c\. ", fact)
                or re.search(f"d\. ", fact)
                or re.search(f"e\. ", fact)
            ):
                new_facts.append(fact)
        return new_facts

    def remove_short_or_high_non_word_ratio_facts(self, facts):
        logging.info("Running stage remove_short_or_high_non_word_ratio_facts")
        return self.parallel_run(
            facts,
            self.remove_short_or_high_non_word_ratio_facts_worker,
            initializer=self.remove_short_or_high_non_word_ratio_facts_initializer,
        )

    @staticmethod
    def remove_short_or_high_non_word_ratio_facts_initializer():
        FactFilterParallelContext.words_set = set(w.lower() for w in words.words())

    @staticmethod
    def remove_short_or_high_non_word_ratio_facts_worker(fact):
        tokens = nltk.word_tokenize(fact)
        if len(tokens) < 3:
            return None
        english_word_ratio = sum(
            1 if to.lower() in FactFilterParallelContext.words_set else 0
            for to in tokens
        ) / len(tokens)
        if english_word_ratio < 0.5:
            return None
        return fact

    def remove_questions(self, facts):
        logging.info("Running stage remove_questions")
        return [fact for fact in tqdm.tqdm(facts) if not fact.endswith("?")]

    # def keep_declarative_statements(self, facts):
    #     logging.info("Running stage keep_declarative_statements")
    #     self.parallel_run(
    #         facts,
    #         self.keep_declarative_statements_worker,
    #         True,
    #         self.keep_declarative_statements_initializer,
    #     )
    #     return facts
    #
    # @staticmethod
    # def keep_declarative_statements_initializer():
    #     FactFilterPoolContext.parser = nltk.CoreNLPParser(url="http://localhost:9000")
    #
    # @staticmethod
    # def keep_declarative_statements_worker(fact):
    #     tree = list(FactFilterPoolContext.parser.raw_parse(fact))
    #     print(tree)
    #     return fact

    def remove_incomplete_sentences(self, facts):
        logging.info("Running stage remove_incomplete_sentences")
        chunk_size = 8
        logits = np.concatenate(
            self.parallel_run(
                [
                    facts[split_start : split_start + chunk_size]
                    for split_start in range(0, len(facts), chunk_size)
                ],
                func=self.remove_incomplete_sentences_worker,
                starting_method="spawn",
                processes=t.cuda.device_count(),
                initializer=self.remove_incomplete_sentences_initializer,
            ),
            axis=0,
        )
        new_facts = []
        for logit, fact in zip(logits[:, 1], facts):
            if logit > 0.8:
                new_facts.append(fact)
        return new_facts

    @staticmethod
    def remove_incomplete_sentences_initializer():
        FactFilterParallelContext.cola_model = AutoModelForSequenceClassification.from_pretrained(
            "textattack/roberta-base-CoLA",
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
            local_files_only=local_files_only,
        ).to(
            f"cuda:{FactFilterParallelContext.worker_id}"
        )
        FactFilterParallelContext.cola_model.eval()
        FactFilterParallelContext.cola_tokenizer = AutoTokenizer.from_pretrained(
            "textattack/roberta-base-CoLA",
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
            local_files_only=local_files_only,
        )

    @staticmethod
    def remove_incomplete_sentences_worker(fact_batches):
        with t.no_grad():
            batch = FactFilterParallelContext.cola_tokenizer(
                fact_batches,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).to(f"cuda:{FactFilterParallelContext.worker_id}")
            logits = t.softmax(
                FactFilterParallelContext.cola_model(**batch).logits, dim=1
            )
        return logits.cpu().numpy()

    def remove_non_neutral_sentences(self, facts):
        logging.info("Running stage remove_non_neutral_sentences")
        chunk_size = 8
        logits = np.concatenate(
            self.parallel_run(
                [
                    facts[split_start : split_start + chunk_size]
                    for split_start in range(0, len(facts), chunk_size)
                ],
                func=self.remove_non_neutral_sentences_worker,
                starting_method="spawn",
                processes=t.cuda.device_count(),
                initializer=self.remove_non_neutral_sentences_initializer,
            ),
            axis=0,
        )
        new_facts = []
        for logits_row, fact in zip(logits, facts):
            if not (
                np.argmax(logits_row) in (0, 4)
                or np.sum(logits_row[:2]) > 0.6
                or np.sum(logits_row[3:]) > 0.6
            ):
                new_facts.append(fact)
            # else:
            #     print(fact)
        return new_facts

    @staticmethod
    def remove_non_neutral_sentences_initializer():
        FactFilterParallelContext.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment",
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
            local_files_only=local_files_only,
        ).to(
            f"cuda:{FactFilterParallelContext.worker_id}"
        )
        FactFilterParallelContext.sentiment_model.eval()
        FactFilterParallelContext.sentiment_tokenizer = AutoTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment",
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
            local_files_only=local_files_only,
        )

    @staticmethod
    def remove_non_neutral_sentences_worker(fact_batches):
        with t.no_grad():
            batch = FactFilterParallelContext.sentiment_tokenizer(
                fact_batches,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).to(f"cuda:{FactFilterParallelContext.worker_id}")
            logits = t.softmax(
                FactFilterParallelContext.sentiment_model(**batch).logits, dim=1
            )
        return logits.cpu().numpy()

    def remove_duplicates(self, facts):
        # Only check first 5 tokens
        logging.info("Running stage remove_duplicates")
        added_fact_starts = set()
        new_facts = []
        for fact in tqdm.tqdm(facts):
            start_tokens = tuple(nltk.word_tokenize(fact)[:3])
            if start_tokens in added_fact_starts:
                continue
            added_fact_starts.add(start_tokens)
            new_facts.append(fact)
        return facts

    def parallel_run(
        self,
        inputs,
        func,
        pesudo=False,
        starting_method="fork",
        processes=None,
        initializer=None,
        initargs=None,
    ):
        results = []
        if not pesudo:
            processes = processes or multiprocessing.cpu_count()
            chunk_size = max((len(inputs) + processes - 1) // processes, 1)
            ctx = multiprocessing.get_context(starting_method)
            queue = ctx.Queue()
            process_pool = [
                ctx.Process(
                    target=self.parallel_executor,
                    args=(
                        worker_id,
                        processes,
                        initializer,
                        initargs,
                        func,
                        inputs[split_start : split_start + chunk_size],
                        queue,
                    ),
                )
                for worker_id, split_start in zip(
                    range(processes), range(0, len(inputs), chunk_size),
                )
            ]

            for p in process_pool:
                p.start()
            for _ in process_pool:
                results.append(queue.get())
            for p in process_pool:
                p.join()
            results = sorted(results, key=lambda x: x[0])
            results = [xx for x in results for xx in x[1]]

        else:
            FactFilterParallelContext.worker_id = 0
            if initializer is not None:
                initargs = initargs or ()
                initializer(*initargs)
            for fact in tqdm.tqdm(inputs):
                result = func(fact)
                if result is not None:
                    results.append(result)
        return results

    @staticmethod
    def parallel_executor(
        worker_id, worker_num, initializer, initargs, func, split, queue
    ):
        FactFilterParallelContext.worker_id = worker_id
        if initializer is not None:
            initargs = initargs or ()
            initializer(*initargs)
        if worker_id == 0:
            with tqdm.tqdm(total=worker_num * len(split)) as bar:
                result = []
                for s in split:
                    res = func(s)
                    if res is not None:
                        result.append(res)
                    bar.update(worker_num)
                queue.put((worker_id, result))
        else:
            queue.put(
                (worker_id, [x for x in [func(s) for s in split] if x is not None])
            )

    # def __del__(self):
    #     self.corenlp_process.kill()
    #     self.corenlp_process.wait()
    #     logging.info("CoreNLP server stopped")
