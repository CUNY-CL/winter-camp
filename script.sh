#!/bin/bash

mkdir submission
mkdir submission/data
mkdir submission/data/news
mkdir submission/data/dril

# Download corpus
cd submission/data
chmod +x get_dataset.sh
../../get_dataset.sh
cd ../../

# Pip install module
pip3 install -e .

# Train model on news corpus
# Set --max_examples to 100,000 just so things run quickly (my laptop is not the strongest...)
python3 caseify/train.py \
 --filepath submission/data/news.2007.en.shuffled.deduped \
 --directory submission/data/news --max_examples 100000

# Train twitter model on @dril tweets
python3 caseify/tweets.py \
  --bearer_token AAAAAAAAAAAAAAAAAAAAAAcpegEAAAAAVGCIJ%2Fv%2BRHcBPK4%2FDSNtN29yc34%3DUUXrgRcyHDHA9PRbEOciT5MB6IKsV8ByAef62UPhZSdaMMhz1i \
  --user dril \
  --clean 1 \
  --directory submission/data/dril \
  --pretrained submission/data/news/model.pkl
