import argparse
import os


def compare_files(golden_fp: str, pred_fp: str):

    assert os.path.isfile(golden_fp), f"{golden_fp} not found!"
    assert os.path.isfile(pred_fp), f"{pred_fp} not found!"

    correct_case_tokens = 0
    total_tokens = 0

    gold = open(golden_fp, 'r')
    pred = open(pred_fp, 'r')

    try:

        for gold_line, pred_line in zip(gold, pred):
            gold_tokens = gold_line.split()
            pred_tokens = pred_line.split()

            assert len(gold_tokens) == len(pred_tokens)
            for gold_token, pred_token in zip(gold_tokens, pred_tokens):
                total_tokens += 1

                if gold_token == pred_token:
                    correct_case_tokens += 1

    finally:
        gold.close()
        pred.close()

    accuracy = round(correct_case_tokens / total_tokens, 6)

    return accuracy


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--golden", help='Path to golden file')
    parser.add_argument("-p", "--predictions", help="Path to predictions file")

    args = parser.parse_args()
    print(compare_files(args.golden, args.predictions))