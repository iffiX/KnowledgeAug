import os
import re
import sys
import random
import logging
import openai
from typing import List
from encoder.dataset.commonsense_qa import CommonsenseQABaseDataset
from encoder.utils.file import JSONCache, JSONStreamCache
from encoder.utils.settings import preprocess_cache_dir


class CommonsenseQAPrompter:
    BASE_PROMPT = (
        "Q: Something that you need to have inside of you when opening a business is a lot of?\\n"
        "Answer Choices:\\n(a) workers\\n(b) money\\n(c) determination\\n(d) funding\\n(e) creativity\\n"
        "A: First, for all choices, we know mental properties can be inside some one, only (c) determination"
        " and (e) creativity are qualified. Second, for (c) and (e), we know opening a business needs to deal"
        " with a lot of difficult situations, only (c) determination is qualified. Therefore, the best choice"
        " is (c) determination.\\n\\n"
        "Q: The CEO's curiosity about the product got the best of her, so she decided to do what with it?\\n"
        "Answer Choices:\\n(a) hear news\\n(b) purchase it\\n(c) examine thing\\n(d) attend school\\n"
        "(e) go to market\\n"
        "A: First, for all choices, we know when a thing got the best of someone, someone may want to know "
        "more information about it, and only (c) examine thing and (e) go to market are qualified. Second, "
        "for (c) and (e), we know a CEO's job is optimizing marketing strategy, only (e) go to market is "
        "qualified. Therefore the best choice is (e) go to market.\\n\\n"
        "Q: They all wanted to play a board game, it was retrieved from inside where?\\n"
        "Answer Choices:\\n(a) cupboard\\n(b) under the bed\\n(c) shelf\\n(d) toy store\\n(e) house\\n"
        "A: First, for all choices, we know a board game can be retrieved from a cupboard or a shelf, only "
        "(a) cupboard and (c) shelf are qualified. Second, for (a) and (c), we know enclosed furnitures have"
        " insides, only (a) cupboard is qualified. Therefore, the best choice is (a) cupboard.\\n\\n"
        "Q: Where do you put your grapes just before checking out?\\n"
        "Answer Choices:\\n(a) mouth\\n(b) grocery cart\\n(c)super market\\n(d) fruit basket\\n"
        "(e) fruit market\\n"
        "A: First, for all choices, we know a container can be used to place something, only (b) grocery cart"
        " and (d) fruit basket are qualified. Second, for (b) and (d), we know grocery carts are usually "
        "placed in markets for checking out, only (b) grocery cart is qualified. Therefore, the best choice "
        "is (b) grocery cart.\\n\\n"
        "Q: What do people use to absorb extra ink from a fountain pen?\\n"
        "Answer Choices:\\n(a) shirt pocket\\n(b) calligrapher's hand\\n(c) inkwell\\n(d) desk drawer\\n"
        "(e) blotter\\n"
        "A: First, for all choices, we know a blotter can be used to absorb excess ink, only (e) blotter is "
        "qualified. Therefore, the best choice is (e) blotter.\\n\\n"
        "Q: Jame's bare feet were burned as he walked, because the sunshine had made the surface hot. Where "
        "might he have been?\\n"
        "Answer Choices:\\n(a) disneyland\\n(b) snow\\n(c) windowsill\\n(d) street\\n(e) summer\\n"
        "A: First, for all choices, we know if someone walk bare foot on a hot road surface their feet will"
        " be burned, only (d) street is qualified. Therefore, the best choice is (d) street.\\n\\n"
    )
    BASE_PROMPT_TOKENS = 796
    BASE_PROMPT_QUESTIONS = BASE_PROMPT.count("Q:")
    BASE_PROMPT_WITH_HINT = (
        "Q: Something that you need to have inside of you when opening a business is a lot of?\\n"
        "Answer Choices:\\n(a) workers\\n(b) money\\n(c) determination (correct)\\n(d) funding\\n(e) creativity\\n"
        "A: First, for all choices, we know mental properties can be inside some one, only (c) determination"
        " and (e) creativity are qualified. Second, for (c) and (e), we know opening a business needs to deal"
        " with a lot of difficult situations, only (c) determination is qualified. Therefore, the best choice"
        " is (c) determination.\\n\\n"
        "Q: The CEO's curiosity about the product got the best of her, so she decided to do what with it?\\n"
        "Answer Choices:\\n(a) hear news\\n(b) purchase it\\n(c) examine thing\\n(d) attend school\\n"
        "(e) go to market (correct)\\n"
        "A: First, for all choices, we know when a thing got the best of someone, someone may want to know "
        "more information about it, and only (c) examine thing and (e) go to market are qualified. Second, "
        "for (c) and (e), we know a CEO's job is optimizing marketing strategy, only (e) go to market is "
        "qualified. Therefore the best choice is (e) go to market.\\n\\n"
        "Q: They all wanted to play a board game, it was retrieved from inside where?\\n"
        "Answer Choices:\\n(a) cupboard (correct)\\n(b) under the bed\\n(c) shelf\\n(d) toy store\\n(e) house\\n"
        "A: First, for all choices, we know a board game can be retrieved from a cupboard or a shelf, only "
        "(a) cupboard and (c) shelf are qualified. Second, for (a) and (c), we know enclosed furnitures have"
        " insides, only (a) cupboard is qualified. Therefore, the best choice is (a) cupboard.\\n\\n"
        "Q: Where do you put your grapes just before checking out?\\n"
        "Answer Choices:\\n(a) mouth\\n(b) grocery cart (correct)\\n(c)super market\\n(d) fruit basket\\n"
        "(e) fruit market\\n"
        "A: First, for all choices, we know a container can be used to place something, only (b) grocery cart"
        " and (d) fruit basket are qualified. Second, for (b) and (d), we know grocery carts are usually "
        "placed in markets for checking out, only (b) grocery cart is qualified. Therefore, the best choice "
        "is (b) grocery cart.\\n\\n"
        "Q: What do people use to absorb extra ink from a fountain pen?\\n"
        "Answer Choices:\\n(a) shirt pocket\\n(b) calligrapher's hand\\n(c) inkwell\\n(d) desk drawer\\n"
        "(e) blotter (correct)\\n"
        "A: First, for all choices, we know a blotter can be used to absorb excess ink, only (e) blotter is "
        "qualified. Therefore, the best choice is (e) blotter.\\n\\n"
        "Q: Jame's bare feet were burned as he walked, because the sunshine had made the surface hot. Where "
        "might he have been?\\n"
        "Answer Choices:\\n(a) disneyland\\n(b) snow\\n(c) windowsill\\n(d) street (correct)\\n(e) summer\\n"
        "A: First, for all choices, we know if someone walk bare foot on a hot road surface their feet will"
        " be burned, only (d) street is qualified. Therefore, the best choice is (d) street.\\n\\n"
    )
    BASE_PROMPT_WITH_HINT_TOKENS = 808
    BASE_PROMPT_WITH_HINT_QUESTIONS = BASE_PROMPT_WITH_HINT.count("Q:")
    MODEL = "text-davinci-002"
    MODEL_PRICE_PER_THOUSAND_TOKENS = 0.06

    def __init__(
        self,
        dataset: CommonsenseQABaseDataset,
        shuffle_seed: int = 42,
        limit_authoritative_train_num: int = 1000,
        limit_authoritative_validate_num: int = 500,
        generate_authoritative_reasoning: bool = True,
        generate_additional_reasoning: bool = False,
    ):
        """
        Args:
            limit_authoritative_train_num: Train samples to use for generating authoritative
                reasoning paths.
            limit_authoritative_validate_num: Validate samples to use for generating authoritative
                reasoning paths.
            generate_authoritative_reasoning: Use the base prompt with hint to generate authoritative
                reasoning paths on the train and validate split.
            generate_additional_reasoning: Use the base prompt without hint to generate additional
                reasoning paths on validate and test split  can be used by the QA model.
        """
        self.rand = random.Random(shuffle_seed)
        self.dataset = dataset
        self.limit_authoritative_train_num = limit_authoritative_train_num
        self.limit_authoritative_validate_num = limit_authoritative_validate_num
        self.generate_authoritative_reasoning = generate_authoritative_reasoning
        self.generate_additional_reasoning = generate_additional_reasoning
        openai.api_key = os.getenv("OPENAI_API_KEY")

        if not os.path.exists(
            os.path.join(preprocess_cache_dir, "commonsense_qa_prompts_indices.json")
        ):
            self.get_user_confirmation()

        with JSONCache(
            os.path.join(preprocess_cache_dir, "commonsense_qa_prompts_indices.json"),
            lambda: {},
        ) as cache:
            if "train" not in cache.data or "authoritative_validate" not in cache.data:
                cache.data = {
                    "authoritative_train": self.get_indicies(
                        limit_authoritative_train_num, len(self.dataset.train_data)
                    ),
                    "authoritative_validate": self.get_indicies(
                        limit_authoritative_validate_num,
                        len(self.dataset.validate_data),
                    ),
                }
            self.authoritative_train_indices = cache.data["authoritative_train"]
            self.authoritative_validate_indices = cache.data["authoritative_validate"]

        if generate_authoritative_reasoning:
            with JSONStreamCache(
                os.path.join(
                    preprocess_cache_dir, "commonsense_qa_authoritative_prompts.json"
                ),
                [
                    self.dataset.train_data[idx]["id"]
                    for idx in self.authoritative_train_indices
                ]
                + [
                    self.dataset.validate_data[idx]["id"]
                    for idx in self.authoritative_validate_indices
                ],
                self.get_authoritative_reasoning_generator(),
            ) as cache:
                self.authoritative_data = self.parse_data(cache.data)
        else:
            self.authoritative_data = {}

        if generate_additional_reasoning:
            with JSONStreamCache(
                os.path.join(
                    preprocess_cache_dir, "commonsense_qa_additional_prompts.json"
                ),
                [data["id"] for data in self.dataset.validate_data]
                + [data["id"] for data in self.dataset.test_data],
                self.get_addtional_reasoning_generator(),
            ) as cache:
                self.additional_data = self.parse_data(cache.data)
        else:
            self.additional_data = {}

    def get_all_authoritative_facts(self) -> List[str]:
        all_facts = []
        for id_, reason_chains in self.authoritative_data.items():
            for reason_chain in reason_chains:
                all_facts += reason_chain
        return list(set(all_facts))

    def get_all_additional_facts(self) -> List[str]:
        all_facts = []
        for id_, reason_chains in self.additional_data.items():
            for reason_chain in reason_chains:
                all_facts += reason_chain
        return list(set(all_facts))

    def get_authoritative_reasoning_of_id(self, id_) -> List[List[str]]:
        return self.authoritative_data.get(id_, [])

    def get_additional_reasoning_of_id(self, id_) -> List[List[str]]:
        return self.additional_data.get(id_, [])

    def is_authoritative_reasoning_of_id_available(self, id_):
        return id_ in self.authoritative_data and len(self.authoritative_data[id_]) > 0

    def is_additional_reasoning_of_id_available(self, id_):
        return id_ in self.additional_data and len(self.additional_data[id_]) > 0

    def parse_data(self, raw_data):
        """
        raw_data example:
        {
            "8-791": [
                {
                  "text": "\n\nThis is a test",
                  "index": 0,
                  "logprobs": null,
                  "finish_reason": "length"
                }
            ]
        }
        """
        parsed_data = {}
        failed_to_parse_num = 0
        for id_, choices in raw_data.items():
            reason_chains = []
            for choice in choices:
                try:
                    raw_text = choice["text"].lower()
                    raw_end = raw_text.find("\\n")
                    if raw_end != -1:
                        raw_text = raw_text[:raw_end]
                    reason_steps = []
                    last_end = 0
                    for reason_step_start in ("first", "second"):
                        if reason_step_start + "," not in raw_text:
                            continue
                        start = raw_text.find(reason_step_start, last_end)
                        end = raw_text.find("qualified", start) + len("qualified")
                        last_end = end
                        search_text = raw_text[start:end]
                        result = re.search(
                            reason_step_start
                            + r", for ([\w,()\-' ]+), we know ([\w,\-/' ]+),? (and |but )?only (.+) qualified",
                            search_text,
                        )
                        if result is None:
                            logging.info(
                                f"id={id_}, input={raw_text}\n "
                                f'Error: failed to match at reasoning step "{reason_step_start}"'
                            )
                            raise ValueError()

                        # parse start scope
                        if reason_step_start == "first":
                            if result.group(1) == "all choices":
                                start_scope = ["all"]
                            else:
                                logging.info(
                                    f"id={id_}, input={raw_text}\n "
                                    f'Error: start scope of the first step must be all choices"'
                                )
                                raise ValueError()
                        else:
                            start_scope = re.findall(r"\(([a-z])\)", result.group(1))
                            if len(start_scope) == 0:
                                logging.info(
                                    f"id={id_}, input={raw_text}\n "
                                    f'Error: start scope is empty"'
                                )
                                raise ValueError()

                        # parse fact
                        fact = result.group(2)

                        # parse end scope
                        end_scope = re.findall(r"\(([a-z])\)", result.group(4))
                        if len(end_scope) == 0:
                            logging.info(
                                f"id={id_}, input={raw_text}\n "
                                f'Error: end scope is empty"'
                            )
                            raise ValueError()

                        reason_steps.append(
                            {"start": start_scope, "end": end_scope, "fact": fact}
                        )

                    conclusion_result = re.search(
                        r"therefore,? the best choice is \(([a-z])\)", raw_text,
                    )
                    if not conclusion_result:
                        logging.info(
                            f"id={id_}, input={raw_text}\n "
                            f'Error: conclusion is empty"'
                        )
                        raise ValueError()
                    conclusion = [conclusion_result.group(1)]

                    # check reason consistency
                    last_scope = ["all"]
                    is_reason_consistent = True
                    for reason_step in reason_steps:
                        if sorted(reason_step["start"]) == last_scope:
                            last_scope = sorted(reason_step["end"])
                        else:
                            is_reason_consistent = False
                            break
                    if last_scope != conclusion:
                        is_reason_consistent = False
                    if not is_reason_consistent:
                        logging.info(
                            f"id={id_}, input={raw_text}\n "
                            f'Error: reasoning chain inconsistent"'
                        )
                        raise ValueError()
                except ValueError:
                    failed_to_parse_num += 1
                    continue
                reason_chains.append([step["fact"] for step in reason_steps])
            parsed_data[id_] = reason_chains
        logging.info(
            f"{failed_to_parse_num}/{len(parsed_data)} parse failed, "
            f"{failed_to_parse_num/len(parsed_data)*100:.2f}%"
        )
        return parsed_data

    def get_user_confirmation(self):
        authoritative_train_num = min(
            self.limit_authoritative_train_num, len(self.dataset.train_data)
        )
        authoritative_validate_num = min(
            self.limit_authoritative_validate_num, len(self.dataset.validate_data)
        )
        base_prompt_estimated_token_num = (
            self.BASE_PROMPT_TOKENS
            * (1 + self.BASE_PROMPT_QUESTIONS)
            / self.BASE_PROMPT_QUESTIONS
        )
        base_prompt_with_hint_estimated_token_num = (
            self.BASE_PROMPT_WITH_HINT_TOKENS
            * (1 + self.BASE_PROMPT_WITH_HINT_QUESTIONS)
            / self.BASE_PROMPT_WITH_HINT_QUESTIONS
        )
        estimated_price = 0
        if self.generate_authoritative_reasoning:
            estimated_price += (
                (authoritative_train_num + authoritative_validate_num)
                * base_prompt_with_hint_estimated_token_num
                * self.MODEL_PRICE_PER_THOUSAND_TOKENS
                / 1000
            )
        if self.generate_additional_reasoning:
            estimated_price += (
                (len(self.dataset.validate_data) + len(self.dataset.test_data))
                * base_prompt_estimated_token_num
                * self.MODEL_PRICE_PER_THOUSAND_TOKENS
                / 1000
            )
        response = input(f"Estimated price: ${estimated_price:.2f}, Continue? (Y/N).")
        while response.lower() not in ("y", "n"):
            input("Please input Y or N.")
        if response.lower() == "y":
            return
        else:
            print("Canceled, won't continue.")
            sys.exit(0)

    def get_indicies(self, limit, max):
        indices = list(range(max))
        self.rand.shuffle(indices)
        return indices[:limit]

    def get_full_prompt(self, base_prompt, question, choices):
        text_choices = "".join(
            f"({label}) {choice}\\n"
            for label, choice in zip("abcdefghijklmnopqrstuvwxyz", choices)
        )
        return f"{base_prompt}Q: {question}\\nAnswer Choices:\\n{text_choices}A:"

    def get_authoritative_reasoning_generator(self):
        table = {
            self.dataset.train_data[idx]["id"]: self.dataset.train_data[idx]
            for idx in self.authoritative_train_indices
        }
        table.update(
            {
                self.dataset.validate_data[idx]["id"]: self.dataset.validate_data[idx]
                for idx in self.authoritative_validate_indices
            }
        )

        def generator(id_):
            data = table[id_]
            choices = [
                f"{choice} (correct)" if idx == data["label"] else choice
                for idx, choice in enumerate(data["choices"])
            ]
            response = openai.Completion.create(
                model=self.MODEL,
                prompt=self.get_full_prompt(
                    self.BASE_PROMPT_WITH_HINT, data["text_question"], choices
                ),
                max_tokens=128,
                temperature=0.7,
                frequency_penalty=0.1,
                stop=["\\n\\n", "Q:", ";\\n"],
                n=1,
            )
            return response["choices"]

        return generator

    def get_addtional_reasoning_generator(self):
        table = {data["id"]: data for data in self.dataset.validate_data}
        table.update({data["id"]: data for data in self.dataset.test_data})

        def generator(id_):
            data = table[id_]
            response = openai.Completion.create(
                model=self.MODEL,
                prompt=self.get_full_prompt(
                    self.BASE_PROMPT, data["text_question"], data["choices"]
                ),
                max_tokens=128,
                temperature=0.7,
                frequency_penalty=0.1,
                stop=["\\n\\n", "Q:", ";\\n"],
                n=1,
            )
            return response["choices"]

        return generator
