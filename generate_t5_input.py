import os
import json
import pprint
from encoder.dataset.arc import ARCDataset
from encoder.dataset.annotator.dedicated_annotators import *
from encoder.dataset.annotator.numerical_annotator import NumericalAnnotator
from encoder.dataset.annotator.simple_relation_miner import *
from encoder.utils.settings import preprocess_cache_dir

if __name__ == "__main__":
    dataset = ARCDataset(
        None,
        matcher_mode="embedding",
        matcher_seed=697474,
        matcher_config={
            "question_match_max_times": 300,
            "question_match_max_depth": 1,
            "question_match_edge_top_k": 10,
            "question_match_source_context_range": 0,
            "question_select_max_edges": 3,
            "question_select_discard_edges_if_rank_below": 0.4,
            "choices_match_max_times": 1000,
            "choices_match_max_depth": 2,
            "choices_match_edge_top_k": 20,
            "choices_match_source_context_range": 0,
            "choices_select_max_edges": 3,
            "choices_select_discard_edges_if_rank_below": 0.2,
        },
    )
    with open(os.path.join(preprocess_cache_dir, "arc_targets.json"), "r") as file:
        dataset.set_search_targets(json.load(file))
    pprint.pprint(dataset.generate_t5_annotation(dataset.train_data[5]))
    # for i in range(-5, 0, 1):
    #     pprint.pprint(dataset.generate_t5_annotation(dataset.train_data[i]))
    # dataset.generate_all_t5_data(split="train")
    # dataset.generate_all_t5_data(split="test")

# if __name__ == "__main__":
#     annotator = NumericalAnnotator()
#     print(
#         annotator.annotate(
#             "An atom of beryllium has 4 protons, 4 electrons, and 5 neutrons. What is its mass number?",
#             ["sulfur", "charge of +3.", "48", "52"],
#         )
#     )

# if __name__ == "__main__":
#     question = "An unbalanced equation for the reaction of methane gas (CH_{4}) with oxygen is shown below. CH_{4} + \\Box O_{2} -> 2CO_{2} + 4H_{2}O How many molecules of oxygen gas (O_{2}) are needed to properly balance this equation?"
#     choices = [
#         "1",
#         "2",
#         "3",
#         "4",
#     ]
#     print(is_chemistry_balance_equation(question, choices))
#     print(annotate_chemistry_balance_equation(question, choices))

# import os
# import json
# from encoder.dataset.matcher.fact_filter import FactFilter
# from encoder.utils.settings import preprocess_cache_dir
#
# if __name__ == "__main__":
#     with open(os.path.join(preprocess_cache_dir, "arc_facts.json"), "r") as file:
#         facts = json.load(file)
#     with open(
#         os.path.join(preprocess_cache_dir, "arc_filtered_facts.json"), "w"
#     ) as file:
#         json.dump(FactFilter().clean(facts), file, indent=2)
