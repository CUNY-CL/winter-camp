import os
from collections import Counter
from random import shuffle, seed
from typing import List, Dict, Tuple, Union

import pandas as pd
from caseify.case import get_tc, TokenCase
from nltk import word_tokenize
from tqdm import tqdm

seed(42)

SentenceFeature = Dict[str, Union[str, bool]]
SentenceFeatures = List[SentenceFeature]

SentenceLabel = str
SentenceLabels = List[SentenceLabel]

Dataset = Dict[str, Union[List[SentenceFeatures], List[SentenceLabels], List[str]]]


def get_suffix(word: str, n: int) -> str:
    """
    Get suffix of word

    Args:
        word: str to get suffix of
        n: number of trailing characters to use as suffix

    Returns:
        n-th suffix of word
    """
    return word[-n:]


def extract(tokens: List[str]) -> Tuple[SentenceFeatures, SentenceLabels]:
    """
    Extract features from tokens

    Args:
        tokens: a sentence broken up into a list of tokens

    Returns:
         Features and labels (aka case) of sentence
    """

    tokens = ["__BOS__"] + tokens + ["__EOS__"]

    features, labels = [], []

    for i, token in enumerate(tokens[1:-1]):
        # CRFSuite follows a convention whereby a : in a feature is interpreted as a feature weight.
        # Therefore we suggest that you replace : in any token with another character such as _.
        token = token.replace(":", "_")

        i = i + 1

        line = dict()
        tc, pattern = get_tc(token)

        labels.append(str(tc))

        line["t[0]"] = token.lower()

        if tokens[i-1] != '__BOS__':
            line['t[-1]'] = tokens[i-1].lower()
        else:
            line['BOS'] = True

        if tokens[i+1] != '__EOS__':
            line['t[+1]'] = tokens[i+1].lower()
        else:
            line['EOS'] = True

        end = max(min(len(token), 3 + 1), 2)
        [line.update({f"suf{i}": get_suffix(token, i)}) for i in range(1, end)]
        features.append(line)

    assert len(features) == len(labels)

    return features, labels


def get_most_common_mix(tokens: List[str]) -> Dict[str, str]:
    """
    Returns most popular form of mixed-case tokens
    """

    df = pd.DataFrame(tokens, columns=['token'])

    df['is_mixed'] = df['token'].apply(lambda x: (get_tc(x)[0] == TokenCase.MIXED))
    df['lower'] = df['token'].str.lower()

    # Groupby lowercase version of token and count number of occurrences for that mixed casing
    counts = df.groupby(['lower', 'is_mixed']).apply(lambda x: Counter(x['token']).most_common()[0][0]).reset_index()

    # We computed counts for _all_ words but only care about mixed case words
    # Filter out all other cases and turn results into a dict
    answer = dict()
    counts[counts['is_mixed']].apply(lambda x: answer.update({x['lower']: x[0]}), axis=1)

    return answer


def process_lines(lines: List[str],
                  train_pct: float = 0.8,
                  test_pct: float = 0.1,
                  val_pct: float = 0.1) -> Tuple[Dict[str, Dataset], Dict[str, str]]:

    assert train_pct + test_pct + val_pct == 1

    shuffle(lines)

    print("🚨🚨🚨🚨🚨🚨")
    lines = lines[:1_000_000]

    features, labels = [], []

    all_words = []
    for line in tqdm(lines):
        tokens = word_tokenize(line)
        all_words.extend(tokens)

        f, l = extract(tokens)

        features.append(f)
        labels.append(l)

    train_idx = int(len(lines) * train_pct)
    test_idx = train_idx + int(len(lines) * test_pct)

    datasets = {
        'train': {
            'features': features[:train_idx],
            'labels': labels[:train_idx],
            'lines': lines[:train_idx]
        },

        'test': {
            'features': features[train_idx:test_idx],
            'labels': labels[train_idx:test_idx],
            'lines': lines[train_idx:test_idx]
        },

        'dev': {
            'features': features[test_idx:],
            'labels': labels[test_idx:],
            'lines': lines[test_idx:]
        }
    }

    print("Computing most common mixed tokens...",end='')
    mixed_counts = get_most_common_mix(all_words)
    print("done! ✅")

    return datasets, mixed_counts


def process_dataset(dataset_fp: str,
                    train_pct: float = 0.8,
                    test_pct: float = 0.1,
                    val_pct: float = 0.1) -> Tuple[Dict[str, Dataset], Dict[str, str]]:

    assert train_pct + test_pct + val_pct == 1

    print(f"Loading dataset from {dataset_fp}...", end='')
    with open(dataset_fp, 'r') as infile:
        lines = infile.readlines()

    print(f"done! ✅\nFound {len(lines)} examples.")

    return process_lines(lines)


def save_datasets_crf_feat_format(datasets: Dict[str, Dataset], dataset_dir: str):

    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    for name, dataset in datasets.items():
        features = dataset['features']
        labels = dataset['labels']

        output = []
        for feature, label in tqdm(zip(features, labels)):

            outs = []
            for ff, ll in zip(feature, label):
                line = ll + "\t" + "\t".join([f"{k}={v}" for k, v in ff.items()])
                outs.append(line)

            output.append("\n".join(outs))

        fp = os.path.join(dataset_dir, name + ".features")
        with open(fp, 'w') as outfile:
            outfile.write("\n\n".join(output))

        fp = os.path.join(dataset_dir, name + ".tok")
        with open(fp, 'w') as outfile:
            outfile.write("\n".join(dataset['lines']))