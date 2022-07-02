from typing import List
from torch import nn
from transformers import MarianMTModel, MarianTokenizer
from encoder.utils.settings import (
    model_cache_dir,
    proxies,
    huggingface_mirror,
    local_files_only,
)


def num2word(num):
    ones = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        0: "zero",
        10: "ten",
    }
    teens = {11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen"}
    tens = {
        2: "twenty",
        3: "thirty",
        4: "forty",
        5: "fifty",
        6: "sixty",
        7: "seventy",
        8: "eighty",
        9: "ninety",
    }
    lens = {
        3: "hundred",
        4: "thousand",
        6: "hundred",
        7: "million",
        8: "million",
        9: "million",
        10: "billion",  # ,13:"trillion",11:"googol",
    }

    if num > 999999999:
        return "Number more than 1 billion"

        # Ones
    if num < 11:
        return ones[num]
    # Teens
    if num < 20:
        word = ones[num % 10] + "teen" if num > 15 else teens[num]
        return word
    # Tens
    if num > 19 and num < 100:
        word = tens[int(str(num)[0])]
        if str(num)[1] == "0":
            return word
        else:
            word = word + " " + ones[num % 10]
            return word

    # First digit for thousands,hundred-thousands.
    if len(str(num)) in lens and len(str(num)) != 3:
        word = ones[int(str(num)[0])] + " " + lens[len(str(num))]
    else:
        word = ""

    # Hundred to Million
    if num < 1000000:
        # First and Second digit for ten thousands.
        if len(str(num)) == 5:
            word = num2word(int(str(num)[0:2])) + " thousand"
        # How many hundred-thousand(s).
        if len(str(num)) == 6:
            word = (
                word
                + " "
                + num2word(int(str(num)[1:3]))
                + " "
                + lens[len(str(num)) - 2]
            )
        # How many hundred(s)?
        thousand_pt = len(str(num)) - 3
        word = (
            word
            + " "
            + ones[int(str(num)[thousand_pt])]
            + " "
            + lens[len(str(num)) - thousand_pt]
        )
        # Last 2 digits.
        last2 = num2word(int(str(num)[-2:]))
        if last2 != "zero":
            word = word + " and " + last2
        word = word.replace(" zero hundred", "")
        return word.strip()

    left, right = "", ""
    # Less than 1 million.
    if num < 100000000:
        left = num2word(int(str(num)[:-6])) + " " + lens[len(str(num))]
        right = num2word(int(str(num)[-6:]))
    # From 1 million to 1 billion.
    if num > 100000000 and num < 1000000000:
        left = num2word(int(str(num)[:3])) + " " + lens[len(str(num))]
        right = num2word(int(str(num)[-6:]))
    if int(str(num)[-6:]) < 100:
        word = left + " and " + right
    else:
        word = left + " " + right
    word = word.replace(" zero hundred", "").replace(" zero thousand", " thousand")
    return word


class Translator(nn.Module):
    def __init__(
        self,
        src_model="Helsinki-NLP/opus-mt-en-ROMANCE",
        tar_model="Helsinki-NLP/opus-mt-ROMANCE-en",
    ):
        super(Translator, self).__init__()
        download_args = {
            "cache_dir": model_cache_dir,
            "proxies": proxies,
            "mirror": huggingface_mirror,
            "local_files_only": local_files_only,
        }
        self.src_tokenizer = MarianTokenizer.from_pretrained(src_model, **download_args)
        self.src_model = MarianMTModel.from_pretrained(src_model, **download_args)
        self.tar_tokenizer = MarianTokenizer.from_pretrained(tar_model, **download_args)
        self.tar_model = MarianMTModel.from_pretrained(tar_model, **download_args)

    def translate(self, texts: List[str], language="fr"):
        # Prepare the text data into appropriate format for the model
        template = (
            lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
        )
        src_texts = [template(text) for text in texts]

        if language != "en":
            tokenizer, model = self.src_tokenizer, self.src_model
        else:
            tokenizer, model = self.tar_tokenizer, self.tar_model

        # Tokenize the texts
        encoded = tokenizer(
            src_texts, padding=True, truncation=True, return_tensors="pt"
        )

        # Generate translation using model
        translated = model.generate(**encoded)

        # Convert the generated tokens indices back into text
        translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

        return translated_texts

    def back_translate(self, texts: List[str], intermediate_lang="fr") -> List[str]:
        # Translate from source to target language
        translated_texts = self.translate(texts, language=intermediate_lang)

        # Translate from target language back to source language
        back_translated_texts = self.translate(translated_texts, language="en")

        return back_translated_texts
