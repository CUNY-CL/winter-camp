import argparse
import os
import pickle
import subprocess

import sklearn_crfsuite
from caseify.dataset import process_dataset, datasets_to_files, word_tokenize, extract_feature_dict
from caseify.case import apply_tc, TokenCase, Pattern
from sklearn_crfsuite import metrics as crf_metrics
from typing import List, Tuple


def train_model(train: tuple, dev: tuple):

    crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                               c1=0.1,
                               c2=0.1,
                               max_iterations=100,
                               all_possible_transitions=True,
                               verbose=True)

    X_train, y_train = train
    X_dev, y_dev = dev
    crf.fit(X=X_train, y=y_train, X_dev=X_dev, y_dev=y_dev)

    return crf
    # subprocess.run(f"""crfsuite learn -p feature.possible_states=1 -p feature.possible_transitions=1 -m {model_fp} -e2 {train_feat_fp} {dev_feat_fp}""",
    #                shell=sys.executable)


def predict(model_fp: str, test_feat_fp: str, pred_fp: str):
    subprocess.run(f"crfsuite tag -m {model_fp} {test_feat_fp} > {pred_fp}")


def run_train_job(dataset_fp: str,
                  dataset_dir: str,
                  model_fp: str = 'model.pkl'):

    print("Step #1 -- Processing dataset")
    datasets = process_dataset(dataset_fp=dataset_fp)
    datasets_to_files(datasets, dataset_dir)

    print("Step #2 -- Training model")
    crf = train_model(train=(datasets['train']['features'], datasets['train']['labels']),
                      dev=(datasets['dev']['features'], datasets['dev']['labels']))

    model_fp = os.path.join(dataset_dir, model_fp)
    print(f"Step #3 -- Saving model to {model_fp}")
    with open(model_fp, 'wb') as outfile:
        pickle.dump(crf, outfile)

    # TODO save model
    print("Step #4 -- Generate predictions")
    y_pred = crf.predict(datasets['test']['features'])
    y_gold = datasets['test']['labels']

    print("Step #5 -- Compute metrics")
    labels = list(crf.classes_)
    print("Flat accuracy:", crf_metrics.flat_accuracy_score(y_gold, y_pred))
    print(crf_metrics.flat_classification_report(y_gold, y_pred, labels=labels))

    pred_fp = os.path.join(dataset_dir, "test.predictions")
    print(y_pred[:5])
    output = ""
    for line in y_pred:
        output += "\n".join(line) + '\n\n'

    with open(pred_fp, 'w') as outfile:
        outfile.write(output)

    idx = 10
    sentences = datasets['test']['lines'][:idx]

    case_corrected = case_correct_sentence(crf, sentences)
    for corrected, actual in zip(case_corrected, sentences):
        print(f"True:{actual}\nPred:{corrected}\n\n")


def apply_tc_sentence(tokens: List[str], case_patterns: List[Tuple[TokenCase, Pattern]]):
    assert len(case_patterns) == len(tokens), f"{len(case_patterns)}, {len(tokens)}"
    return " ".join([apply_tc(token, getattr(TokenCase, tc), None) for token, tc in zip(tokens, case_patterns)])


def case_correct_sentence(model, sentences: List[str]) -> List[str]:

    assert len(sentences)

    inputs = []
    all_tokens = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        all_tokens.append(tokens)
        features, _ = extract_feature_dict(tokens)
        inputs.append(features)

    pred_casing = model.predict(inputs)

    outputs = []
    for tokens, pred in zip(all_tokens, pred_casing):
        outputs.append(apply_tc_sentence(tokens, pred))

    return outputs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", help='Path to dataset')
    parser.add_argument("-d", "--directory", help="Directory to store files generated from run")
    parser.add_argument("-m", "--model_filepath", help='Path where model is saved to.', default='model')

    args = parser.parse_args()
    run_train_job(
        dataset_fp=args.filepath,
        dataset_dir=args.directory,
        model_fp=args.model_filepath
    )