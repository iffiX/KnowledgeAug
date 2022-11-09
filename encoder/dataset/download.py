import os
import logging
from encoder.utils.file import (
    download_to,
    decompress_gz,
    decompress_zip,
    decompress_tar_gz,
)
from encoder.utils.settings import dataset_cache_dir, preprocess_cache_dir


class ConceptNet:
    ASSERTION_URL = (
        "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
        "conceptnet-assertions-5.7.0.csv.gz"
    )
    NUMBERBATCH_URL = (
        "https://conceptnet.s3.amazonaws.com/downloads/2019/"
        "numberbatch/numberbatch-en-19.08.txt.gz"
    )

    def __init__(self):
        self.assertion_path = str(
            os.path.join(dataset_cache_dir, "conceptnet-assertions.csv")
        )
        self.numberbatch_path = str(
            os.path.join(dataset_cache_dir, "conceptnet-numberbatch.txt")
        )

    def require(self):
        for task, data_path, url in (
            ("assertions", self.assertion_path, self.ASSERTION_URL),
            ("numberbatch", self.numberbatch_path, self.NUMBERBATCH_URL),
        ):
            if not os.path.exists(data_path):
                if not os.path.exists(str(data_path) + ".gz"):
                    logging.info(f"Downloading concept net {task}")
                    download_to(url, str(data_path) + ".gz")
                logging.info("Decompressing")
                decompress_gz(str(data_path) + ".gz", data_path)
        return self


class ConceptNetWithGloVe:
    ASSERTION_URL = (
        "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
        "conceptnet-assertions-5.7.0.csv.gz"
    )
    GLOVE_URL = (
        "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.42B.300d.zip"
    )

    def __init__(self):
        self.assertion_path = str(
            os.path.join(dataset_cache_dir, "conceptnet-assertions.csv")
        )
        self.glove_path = str(
            os.path.join(dataset_cache_dir, "glove.42B.300d", "glove.42B.300d.txt")
        )

    def require(self):
        if not os.path.exists(self.assertion_path):
            if not os.path.exists(str(self.assertion_path) + ".gz"):
                logging.info(f"Downloading concept net assertions")
                download_to(self.ASSERTION_URL, str(self.assertion_path) + ".gz")
            logging.info("Decompressing")
            decompress_gz(str(self.assertion_path) + ".gz", self.assertion_path)

        glove_directory = os.path.join(dataset_cache_dir, "glove.42B.300d")
        if not os.path.exists(glove_directory):
            if not os.path.exists(str(glove_directory) + ".zip"):
                logging.info(f"Downloading glove embedding")
                download_to(self.GLOVE_URL, str(glove_directory) + ".zip")
            logging.info("Decompressing")
            decompress_zip(str(glove_directory) + ".zip", glove_directory)
        return self


class CommonsenseQA:
    TRAIN_URL = "https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl"
    VALIDATE_URL = "https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl"
    TEST_URL = "https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl"

    def __init__(self):
        base = os.path.join(dataset_cache_dir, "commonsense_qa")
        self.train_path = os.path.join(base, "train.jsonl")
        self.validate_path = os.path.join(base, "validate.jsonl")
        self.test_path = os.path.join(base, "test.jsonl")

    def require(self):
        for task, data_path, url in (
            ("train", self.train_path, self.TRAIN_URL),
            ("validate", self.validate_path, self.VALIDATE_URL),
            ("test", self.test_path, self.TEST_URL),
        ):
            if not os.path.exists(data_path):
                logging.info(f"Downloading commonsense qa {task} dataset.")
                download_to(url, data_path)
        return self


class OpenBookQA:
    OPENBOOK_QA_URL = (
        "https://ai2-public-datasets.s3.amazonaws.com/open-book-qa/"
        "OpenBookQA-V1-Sep2018.zip"
    )

    def __init__(self):
        openbook_qa_path = str(os.path.join(dataset_cache_dir, "openbook_qa"))
        self.train_path = os.path.join(
            openbook_qa_path,
            "OpenBookQA-V1-Sep2018",
            "Data",
            "Additional",
            "train_complete.jsonl",
        )
        self.validate_path = os.path.join(
            openbook_qa_path,
            "OpenBookQA-V1-Sep2018",
            "Data",
            "Additional",
            "dev_complete.jsonl",
        )
        self.test_path = os.path.join(
            openbook_qa_path,
            "OpenBookQA-V1-Sep2018",
            "Data",
            "Additional",
            "test_complete.jsonl",
        )
        self.facts_path = os.path.join(
            openbook_qa_path, "OpenBookQA-V1-Sep2018", "Data", "Main", "openbook.txt"
        )
        self.crowd_source_facts_path = os.path.join(
            openbook_qa_path,
            "OpenBookQA-V1-Sep2018",
            "Data",
            "Additional",
            "crowdsourced-facts.txt",
        )
        # TODO: add download function
        self.search_result_path = os.path.join(
            openbook_qa_path, "openbook_qa_search_data.txt"
        )

    def require(self):
        openbook_qa_path = str(os.path.join(dataset_cache_dir, "openbook_qa"))
        if not os.path.exists(openbook_qa_path):
            if not os.path.exists(str(openbook_qa_path) + ".zip"):
                logging.info("Downloading OpenBook QA")
                download_to(self.OPENBOOK_QA_URL, str(openbook_qa_path) + ".zip")
            logging.info("Decompressing")
            decompress_zip(str(openbook_qa_path) + ".zip", openbook_qa_path)
        return self


class QASC:
    QASC_URL = "https://ai2-public-datasets.s3.amazonaws.com/qasc/qasc_dataset.tar.gz"
    QASC_CORPUS_URL = (
        "https://s3-us-west-2.amazonaws.com/data.allenai.org/downloads/"
        "qasc/qasc_corpus.tar.gz"
    )

    def __init__(self):
        qasc_path = str(os.path.join(dataset_cache_dir, "qasc"))
        qasc_corpus_path = str(os.path.join(dataset_cache_dir, "qasc_corpus"))
        self.train_path = os.path.join(qasc_path, "QASC_Dataset", "train.jsonl",)
        self.validate_path = os.path.join(qasc_path, "QASC_Dataset", "dev.jsonl",)
        self.test_path = os.path.join(qasc_path, "QASC_Dataset", "test.jsonl",)
        self.reference_path = os.path.join(qasc_path, "qasc_manual.txt")
        self.corpus_path = os.path.join(
            qasc_corpus_path, "QASC_Corpus", "QASC_Corpus.txt"
        )

    def require(self):
        for task, data_path, url in (
            ("QASC", str(os.path.join(dataset_cache_dir, "qasc")), self.QASC_URL),
            (
                "QASC Corpus",
                str(os.path.join(dataset_cache_dir, "qasc_corpus")),
                self.QASC_CORPUS_URL,
            ),
        ):
            if not os.path.exists(data_path):
                if not os.path.exists(str(data_path) + ".tar.gz"):
                    logging.info(f"Downloading {task}")
                    download_to(url, str(data_path) + ".tar.gz")
                logging.info("Decompressing")
                decompress_tar_gz(str(data_path) + ".tar.gz", data_path)
        return self


class CommonsenseQA2:
    COMMONSENSE_QA2_TRAIN_URL = (
        "https://github.com/allenai/csqa2/raw/master/dataset/CSQA2_train.json.gz"
    )
    COMMONSENSE_QA2_VALIDATE_URL = (
        "https://github.com/allenai/csqa2/raw/master/dataset/CSQA2_dev.json.gz"
    )
    COMMONSENSE_QA2_TEST_URL = (
        "https://github.com/allenai/csqa2/raw/master/"
        "dataset/CSQA2_test_no_answers.json.gz"
    )

    def __init__(self):
        commonsense_qa2_path = str(os.path.join(dataset_cache_dir, "commonsense_qa2"))
        self.train_path = os.path.join(
            commonsense_qa2_path, "teach_your_ai_train.json",
        )
        self.validate_path = os.path.join(
            commonsense_qa2_path, "teach_your_ai_dev.json"
        )
        self.test_path = os.path.join(
            commonsense_qa2_path, "teach_your_ai_text_no_answers.json"
        )

    def require(self):
        commonsense_qa2_path = str(os.path.join(dataset_cache_dir, "commonsense_qa2"))
        compressed_paths = [
            str(os.path.join(commonsense_qa2_path, "CSQA2_train.json.gz")),
            str(os.path.join(commonsense_qa2_path, "CSQA2_dev.json.gz")),
            str(os.path.join(commonsense_qa2_path, "CSQA2_test_no_answers.json.gz")),
        ]
        file_paths = [self.train_path, self.validate_path, self.test_path]
        if not os.path.exists(commonsense_qa2_path):
            os.makedirs(commonsense_qa2_path, exist_ok=True)

        if any([not os.path.exists(cmp) for cmp in compressed_paths]):
            logging.info("Downloading CommonsenseQA2")
            for url, path in zip(
                [
                    self.COMMONSENSE_QA2_TRAIN_URL,
                    self.COMMONSENSE_QA2_VALIDATE_URL,
                    self.COMMONSENSE_QA2_TEST_URL,
                ],
                compressed_paths,
            ):
                download_to(url, path)

        if any([not os.path.exists(file) for file in file_paths]):
            logging.info("Decompressing")
            for cmp, target_path in zip(
                compressed_paths, [self.train_path, self.validate_path, self.test_path]
            ):
                decompress_gz(cmp, target_path)
        return self


class ANLI:
    ANLI_TRAIN_DEV_URL = "https://storage.googleapis.com/ai2-mosaic/public/alphanli/alphanli-train-dev.zip"
    ANLI_TEST_URL = (
        "https://storage.googleapis.com/ai2-mosaic/public/alphanli/alphanli-test.zip"
    )

    def __init__(self):
        anli_path = str(os.path.join(dataset_cache_dir, "anli"))
        self.train_path = os.path.join(anli_path, "train.jsonl",)
        self.train_labels_path = os.path.join(anli_path, "train-labels.lst")
        self.validate_path = os.path.join(anli_path, "dev.jsonl")
        self.validate_labels_path = os.path.join(anli_path, "dev-labels.lst")
        self.test_path = os.path.join(anli_path, "alphanli-test", "anli.jsonl")

    def require(self):
        anli_path = str(os.path.join(dataset_cache_dir, "anli"))
        compressed_paths = [
            str(os.path.join(anli_path, "alphanli-train-dev.zip")),
            str(os.path.join(anli_path, "alphanli-test.zip")),
        ]
        file_paths = [
            self.train_path,
            self.train_labels_path,
            self.validate_path,
            self.validate_labels_path,
            self.test_path,
        ]
        if not os.path.exists(anli_path):
            os.makedirs(anli_path, exist_ok=True)

        if any([not os.path.exists(cmp) for cmp in compressed_paths]):
            logging.info("Downloading aNLI")
            for url, path in zip(
                [self.ANLI_TRAIN_DEV_URL, self.ANLI_TEST_URL,], compressed_paths,
            ):
                download_to(url, path)

        if any([not os.path.exists(file) for file in file_paths]):
            logging.info("Decompressing")
            for cmp in compressed_paths:
                decompress_zip(cmp, anli_path)
        return self


class ATOMIC2020:
    ATOMIC2020_URL = (
        "https://ai2-atomic.s3-us-west-2.amazonaws.com/data/atomic2020_data-feb2021.zip"
    )

    def __init__(self):
        atomic2020_path = str(os.path.join(dataset_cache_dir, "atomic2020"))
        self.train_path = os.path.join(
            atomic2020_path, "atomic2020_data-feb2021", "train.tsv",
        )
        self.validate_path = os.path.join(
            atomic2020_path, "atomic2020_data-feb2021", "dev.tsv",
        )
        self.test_path = os.path.join(
            atomic2020_path, "atomic2020_data-feb2021", "test.tsv",
        )

    def require(self):
        atomic2020_path = str(os.path.join(dataset_cache_dir, "atomic2020"))
        compressed_path = str(
            os.path.join(atomic2020_path, "atomic2020_data-feb2021.zip")
        )
        file_paths = [self.train_path, self.validate_path, self.test_path]
        if not os.path.exists(atomic2020_path):
            os.makedirs(atomic2020_path, exist_ok=True)

        if not os.path.exists(compressed_path):
            logging.info("Downloading ATOMIC2020")
            download_to(self.ATOMIC2020_URL, compressed_path)

        if any([not os.path.exists(file) for file in file_paths]):
            logging.info("Decompressing")
            decompress_zip(compressed_path, atomic2020_path)
        return self
