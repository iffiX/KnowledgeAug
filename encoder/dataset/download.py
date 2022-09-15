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

    def require(self):
        openbook_qa_path = str(os.path.join(dataset_cache_dir, "openbook_qa"))
        if not os.path.exists(openbook_qa_path):
            if not os.path.exists(str(openbook_qa_path) + ".zip"):
                logging.info("Downloading OpenBook QA")
                download_to(self.OPENBOOK_QA_URL, str(openbook_qa_path) + ".zip")
            logging.info("Decompressing")
            decompress_zip(str(openbook_qa_path) + ".zip", openbook_qa_path)
        return self


class ARC:
    ARC_URL = "https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip"

    def __init__(self):
        arc_path = str(os.path.join(dataset_cache_dir, "arc"))
        self.train_challenge_path = os.path.join(
            arc_path, "ARC-V1-Feb2018-2", "ARC-Challenge", "ARC-Challenge-Train.jsonl"
        )
        self.validate_challenge_path = os.path.join(
            arc_path, "ARC-V1-Feb2018-2", "ARC-Challenge", "ARC-Challenge-Dev.jsonl"
        )
        self.test_challenge_path = os.path.join(
            arc_path, "ARC-V1-Feb2018-2", "ARC-Challenge", "ARC-Challenge-Test.jsonl"
        )
        self.train_easy_path = os.path.join(
            arc_path, "ARC-V1-Feb2018-2", "ARC-Easy", "ARC-Easy-Train.jsonl"
        )
        self.validate_easy_path = os.path.join(
            arc_path, "ARC-V1-Feb2018-2", "ARC-Easy", "ARC-Easy-Dev.jsonl"
        )
        self.test_easy_path = os.path.join(
            arc_path, "ARC-V1-Feb2018-2", "ARC-Easy", "ARC-Easy-Test.jsonl"
        )
        self.corpus_path = os.path.join(arc_path, "ARC-V1-Feb2018-2", "ARC_Corpus.txt")

    def require(self):
        arc_path = str(os.path.join(dataset_cache_dir, "arc"))
        if not os.path.exists(arc_path):
            if not os.path.exists(str(arc_path) + ".zip"):
                logging.info("Downloading ARC")
                download_to(self.ARC_URL, str(arc_path) + ".zip")
            logging.info("Decompressing")
            decompress_zip(str(arc_path) + ".zip", arc_path)
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

    def require(self):
        openbook_qa_path = str(os.path.join(dataset_cache_dir, "openbook_qa"))
        if not os.path.exists(openbook_qa_path):
            if not os.path.exists(str(openbook_qa_path) + ".zip"):
                logging.info("Downloading OpenBook QA")
                download_to(self.OPENBOOK_QA_URL, str(openbook_qa_path) + ".zip")
            logging.info("Decompressing")
            decompress_zip(str(openbook_qa_path) + ".zip", openbook_qa_path)
        return self


class UnifiedQAIR:
    UNIFIEDQA_IR_URL = (
        "https://github.com/allenai/unifiedqa/raw/master/files/"
        "arc-with-ir/ARC-OBQA-RegLivEnv-IR10V8.zip"
    )

    def __init__(self):
        unifiedqa_ir_path = str(os.path.join(dataset_cache_dir, "unifiedqa_ir"))
        self.train_path = os.path.join(
            unifiedqa_ir_path, "ARC-OBQA-RegLivEnv-IR10V8", "train.jsonl",
        )
        self.validate_path = os.path.join(
            unifiedqa_ir_path, "ARC-OBQA-RegLivEnv-IR10V8", "dev.jsonl",
        )
        self.test_path = os.path.join(
            unifiedqa_ir_path, "ARC-OBQA-RegLivEnv-IR10V8", "test.jsonl",
        )

    def require(self):
        unifiedqa_ir_path = str(os.path.join(dataset_cache_dir, "unifiedqa_ir"))
        if not os.path.exists(unifiedqa_ir_path):
            if not os.path.exists(str(unifiedqa_ir_path) + ".zip"):
                logging.info("Downloading UnifiedQA IR annotations")
                download_to(self.UNIFIEDQA_IR_URL, str(unifiedqa_ir_path) + ".zip")
            logging.info("Decompressing")
            decompress_zip(str(unifiedqa_ir_path) + ".zip", unifiedqa_ir_path)
        return self
