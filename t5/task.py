import re
import os
import t5
import json
import seqio
import pickle
import functools
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from sklearn.metrics import f1_score, precision_score, recall_score

DEFAULT_OUTPUT_FEATURES = {
    "inputs": t5.data.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True, required=False
    ),
    "targets": t5.data.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True
    ),
}

DATA_DIR = os.environ.get("DATA_DIR") or "gs://kb-encoder/preprocess"


def align_match(target, prediction):
    target_segments = [x for x in re.split(r"\{|\}|\^| ", target) if len(x) > 0]
    prediction_segments = [x for x in re.split(r" ⁇| ", prediction) if len(x) > 0]
    if len(target_segments) == len(prediction_segments) and all(
        ts == ps for ts, ps in zip(target_segments, prediction_segments)
    ):
        return True
    elif target_segments[: len(prediction_segments)] == prediction_segments:
        print(f"Approximate match {target} ----- {prediction}")
        return True
    return False


def customize_metric(targets, predictions):
    same = 0
    for t, p in zip(targets, predictions):
        if t == p or (re.search(r"[\{\}^]", t) is not None and align_match(t, p)):
            same += 1
    metric_dict = {"precision": same / len(targets)}
    return metric_dict


def customize_dataset_fn(split, shuffle_files=False, lang="multilingual"):

    with tf.io.gfile.GFile(
        os.path.join(DATA_DIR, f"arc_{split}_for_t5.json"), "r"
    ) as file:
        samples = json.load(file)
        ds = tf.data.Dataset.from_tensor_slices(
            {
                "inputs": [s["inputs"] for s in samples],
                "targets": [s["targets"] for s in samples],
                # "choices": [s["choices"] for s in samples],
                # "label": [s["label"] for s in samples],
            }
        )
        return ds


def customize_preprocessor(ds):
    def normalize_text(text):
        text = tf.strings.lower(text)
        text = tf.strings.strip(text)
        text = tf.strings.regex_replace(text, "'(.*)'", r"\1")
        # text = tf.strings.regex_replace(text, "{", "[")
        # text = tf.strings.regex_replace(text, "}", "]")
        return text

    def to_inputs_and_targets(ex):
        return {
            "inputs": normalize_text(ex["inputs"]),
            "targets": normalize_text(ex["targets"]),
            # "choices": ex["choices"],
            # "label": ex["label"],
        }

    return ds.map(
        to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )


seqio.TaskRegistry.add(
    "arc",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(customize_dataset_fn, lang="english"),
        splits=["train", "validate", "test", "validate_original", "test_original"],
    ),
    preprocessors=[
        customize_preprocessor,
        seqio.preprocessors.tokenize_and_append_eos,
    ],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[customize_metric],
    output_features=DEFAULT_OUTPUT_FEATURES,
)


# seqio.TaskRegistry.add(
#     "task_2",
#     source=seqio.FunctionDataSource(
#         dataset_fn=functools.partial(customize_dataset_fn, lang="spanish"),
#         splits=["train", "dev", "test"],
#     ),
#     preprocessors=[
#         customize_preprocessor,
#         seqio.preprocessors.tokenize_and_append_eos,
#     ],
#     postprocess_fn=t5.data.postprocessors.lower_text,
#     metric_fns=[customize_metric],
#     output_features=DEFAULT_OUTPUT_FEATURES,
# )
#
#
# seqio.MixtureRegistry.remove("mixture_1")
# seqio.MixtureRegistry.add(
#     "mixture_1",
#     ["task_1", "task_2"],
#     default_rate=1.0,
# )
