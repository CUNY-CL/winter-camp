import os
from collections import Counter
from random import shuffle, seed
from typing import List, Dict, Tuple, Union

import pandas as pd
from case import get_tc, TokenCase
from nltk import word_tokenize
from tqdm import tqdm

seed(42)

SentenceFeature = Dict[str, Union[str, bool]]
SentenceFeatures = List[SentenceFeature]

SentenceLabel = str
SentenceLabels = List[SentenceLabel]

Dataset = Dict[str, Union[List[SentenceFeatures], List[SentenceLabels]]]


def get_suffix(text: str, n: int) -> str:
    return text[-n:]


def extract_feature_dict(tokens: List[str]) -> Tuple[SentenceFeatures, SentenceLabels]:

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

        line["t[0]"] = token

        if tokens[i-1] != '__BOS__':
            line['t[-1]'] = tokens[i-1]
        else:
            line['BOS'] = True

        if tokens[i+1] != '__EOS__':
            line['t[+1]'] = tokens[i+1]
        else:
            line['EOS'] = True

        end = max(min(len(token), 3 + 1), 2)
        [line.update({f"suf{i}": get_suffix(token, i)}) for i in range(1, end)]
        features.append(line)

    assert len(features) == len(labels)

    return features, labels


def extract(tokens: List[str]) -> List[List[str]]:

    output = []
    tokens = ["__BOS__"] + tokens + ["__EOS__"]

    for i, token in enumerate(tokens[1:-1]):
        # CRFSuite follows a convention whereby a : in a feature is interpreted as a feature weight.
        # Therefore we suggest that you replace : in any token with another character such as _.
        token = token.replace(":", "_")

        i = i + 1

        line = []
        tc, pattern = get_tc(token)

        line.extend([
            str(tc),
            f"t[0]={token}",
            f"t[-1]={tokens[i-1]}" if tokens[i-1] != "__BOS__" else "__BOS__",
            f"t[+1]={tokens[i+1]}" if tokens[i+1] != "__EOS__" else "__EOS__"
        ])

        end = max(min(len(token), 3+1), 2)
        line.extend([f"suf{i}={get_suffix(token, i)}" for i in range(1, end)])
        output.append(line)

    return output


def get_most_common_mix(tokens: List[str]) -> Dict[str, str]:
    """
    Returns most popular form of mixed-case tokens
    """
    df = pd.DataFrame(tokens, columns=['token'])
    df['is_mixed'] = df['token'].apply(lambda x: get_tc(x['token'])[0] == TokenCase.MIXED)
    counts = df\
        .groupby('lower', 'is_mixed')\
        .apply(lambda x: Counter(x['token']).most_common()[0][0])\
        .reset_index()

    answer = dict()
    counts[counts['is_mixed']].apply(lambda x: answer.update({x['lower']: x[0]}), axis=1)

    return answer


def process_lines(lines: List[str],
                  train_pct: float = 0.8,
                  test_pct: float = 0.1,
                  val_pct: float = 0.1) -> Dict[str, Dataset]:

    assert train_pct + test_pct + val_pct == 1

    shuffle(lines)
    features, labels = [], []

    for line in tqdm(lines):
        tokens = word_tokenize(line)
        f, l = extract_feature_dict(tokens)

        features.append(f)
        labels.append(l)

    features, labels = process_lines(lines)

    train_idx = int(len(lines) * train_pct)
    test_idx = train_idx + int(len(lines) * test_pct)

    datasets = {
        'train': {
            'features': features[:train_idx],
            'labels': labels[:train_idx]
        },

        'test': {
            'features': features[train_idx:test_idx],
            'labels': labels[train_idx:test_idx]
        },

        'dev': {
            'features': features[test_idx:],
            'labels': labels[test_idx:]
        }
    }

    return datasets


def process_dataset(dataset_fp: str,
                    train_pct: float = 0.8,
                    test_pct: float = 0.1,
                    val_pct: float = 0.1) -> Dict[str, Dataset]:

    assert train_pct + test_pct + val_pct == 1

    print(f"Loading dataset from {dataset_fp}...", end='')
    with open(dataset_fp, 'r') as infile:
        lines = infile.readlines()

    print(f"done! ✅\nFound {len(lines)} examples.")

    datasets = process_lines(lines)
    return datasets


def datasets_to_files(datasets: Dict[str, Dataset], dataset_dir: str):

    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    for name, dataset in datasets.items():
        features = dataset['features']
        labels = dataset['labels']

        output = []
        for feature, label in zip(features, labels):

            outs = []
            for ff, ll in zip(feature, label):
                line = ll + "\t" + "\t".join([f"{k}={v}" for k, v in ff.items()])
                outs.append(line)

            output.append("\n".join(outs))

        fp = os.path.join(dataset_dir, name + ".features")
        with open(fp, 'w') as outfile:
            outfile.write("\n\n".join(output))
