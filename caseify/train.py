import argparse
import json
import os
import pickle
from random import randint
from typing import List

import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics

from caseify.case import apply_tc_sentence
from caseify.dataset import process_dataset, save_datasets_crf_feat_format, word_tokenize, extract_feature_dict


def train_model(train: tuple, dev: tuple):
    """
    Train a CRF model

    Args:
        train: tuple containing X train and y train data
        dev: tuple containing X dev and y dev data

    Returns:
        Trained sklearn_crfsuite.CRF
    """

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


# def predict(model_fp: str, test_feat_fp: str, pred_fp: str):
#     subprocess.run(f"crfsuite tag -m {model_fp} {test_feat_fp} > {pred_fp}")


def run_train_job(dataset_fp: str,
                  dataset_dir: str):
    """
    Train a CRF model

    Args:
        dataset_fp: filepath to dataset
        dataset_dir: directory to save all job artifacts to
    """

    print("Step #1 -- Processing dataset")
    datasets, mixed_tokens = process_dataset(dataset_fp=dataset_fp)

    save_datasets_crf_feat_format(datasets, dataset_dir)

    mt_fp = os.path.join(dataset_dir, "mixed_tokens.json")
    with open(mt_fp, 'w') as outfile:
        json.dump(mixed_tokens, outfile)

    print("Step #2 -- Training model")
    crf = train_model(train=(datasets['train']['features'], datasets['train']['labels']),
                      dev=(datasets['dev']['features'], datasets['dev']['labels']))

    model_fp = os.path.join(dataset_dir, "model.pkl")
    print(f"Step #3 -- Saving model to {model_fp}")
    with open(model_fp, 'wb') as outfile:
        pickle.dump(crf, outfile)

    print("Step #4 -- Generate predictions")
    y_pred  = crf.predict(datasets['test']['features'])
    pred_fp = os.path.join(dataset_dir, "test.predictions")
    output = ""
    for line in y_pred:
        output += "\n".join(line) + '\n\n'

    with open(pred_fp, 'w') as outfile:
        outfile.write(output)

    print("Step #5 -- Compute metrics")

    y_gold = datasets['test']['labels']
    labels = list(crf.classes_)

    print("Flat accuracy:", crf_metrics.flat_accuracy_score(y_gold, y_pred))
    print(crf_metrics.flat_classification_report(y_gold, y_pred, labels=labels))

    print("Example reconstructed sentences:")

    n_examples = 10
    test_sents = datasets['test']['lines']
    rando_idxs = [randint(0, len(test_sents)) for _ in range(n_examples)]
    sentences  = [test_sents[i] for i in rando_idxs]

    case_corrected = case_correct_sentence(crf, sentences)
    for corrected, actual in zip(case_corrected, sentences):
        print(f"True:{actual}\nPred:{corrected}\n\n")


def case_correct_sentence(model, sentences: List[str]) -> List[str]:
    """
    Given a list of sentences, predict casing using a model

    Args:
        model: Model used to correct case. Can be any model so long as it has `predict` method
        sentences: List of sentences to predict casing for. Casing prior to predictions does not matter

    Returns:
        List of model predictions for correct casing of sentences
    """

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