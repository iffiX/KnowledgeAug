#!/bin/bash

sudo apt update
sudo apt install -y python3-venv libhdf5-dev libpython3-dev graphviz ninja-build default-jre
python3 -m venv venv

#conda install -c conda-forge gxx gcc libgcc-ng sysroot_linux-64 cmake elfutils libunwind
#conda update libgcc-ng
#conda install -c anaconda graphviz hdf5
#python -m venv venv
venv/bin/pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
#venv/bin/pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
venv/bin/pip install wheel
venv/bin/pip install -r requirements.txt
venv/bin/python -m spacy download en_core_web_sm
venv/bin/python -m spacy download en_core_web_md
venv/bin/python -c \
  'import nltk; \
  nltk.download("punkt"); \
  nltk.download("averaged_perceptron_tagger"); \
  nltk.download("wordnet"); \
  nltk.download("omw-1.4"); \
  nltk.download("words"); \
  nltk.download("stopwords")'