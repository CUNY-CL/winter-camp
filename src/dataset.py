from nltk import word_tokenize
from random import shuffle, seed
import os
from typing import List, Dict
from case import get_tc, TokenCase
from collections import Counter
import pandas as pd
from tqdm import tqdm

seed(42)


def get_suffix(text: str, n: int) -> str:
    return text[-n:]


def extract(tokens: List[str]) -> List[List[str]]:

    output = []
    tokens = ["__BOS__"] + tokens + ["__EOS__"]
    for i, token in enumerate(tokens[1:-1]):
        # CRFSuite follows a convention whereby a : in a feature is interpreted as a feature weight.
        # Therefore we suggest that you replace : in any token with another character such as _.
        token = token.replace(":", "_")

        line = []
        tc, pattern = get_tc(token)

        line.extend([
            str(tc),
            f"t[0]={token}",
            f"t[-1]={tokens[i-1]}",
            f"t[+1]={tokens[i+1]}"
        ])

        line.extend([f"suf{i}={get_suffix(token, i)}" for i in range(1, 4)])
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


def process_dataset(dataset_fp: str,
                    dataset_dir: str,
                    train_pct: float = 0.8,
                    test_pct: float = 0.1,
                    val_pct: float = 0.1):

    assert train_pct + test_pct + val_pct == 1

    print(f"Loading dataset from {dataset_fp}...", end='')
    with open(dataset_fp, 'r') as infile:
        lines = infile.readlines()

    print("🚨🚨🚨🚨 USING 1000 LINES 🚨🚨🚨🚨")
    lines = lines[:100]

    print(f"done! ✅\nFound {len(lines)} examples.")

    shuffle(lines)
    train_idx = int(len(lines) * train_pct)
    test_idx  = train_idx + int(len(lines) * test_pct)

    datasets = {
        'train': lines[:train_idx],
        'test': lines[train_idx:test_idx],
        'val': lines[test_idx:]
    }

    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    for name, lines in datasets.items():
        tok_fp = os.path.join(dataset_dir, name + ".tok")

        print(f"Saving {len(lines)} to {tok_fp}...", end='')

        with open(tok_fp, 'w') as outfile:
            outfile.write("\n".join(lines))

        print("done! ✅\nPreparing features...", end='')
        output = []

        # test.features file should NOT include case tags
        # Case tags = first "column" of features therefore by starting 1 we skip them
        if name == 'test':
            start_idx = 1
        else:
            start_idx = 0

        for line in tqdm(lines):
            tokens = word_tokenize(line)
            features = extract(tokens)
            output.append("\t".join([" ".join(f) for f in features[start_idx:]]))

        feat_fp = os.path.join(dataset_dir, name + ".features")
        with open(feat_fp, "w") as outfile:
            outfile.write("\n".join(output))


fp = "/Users/degan/Downloads/news.2007.en.shuffled.deduped"
print(process_dataset(fp, '/Users/degan/datasets/winter_camp'))


