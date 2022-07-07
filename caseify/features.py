"""Feature extraction for case tagging."""

from typing import List


def _suffix_feature(token: str, size: int) -> str:
    return f"suf{size}={token[-size:]}"


def extract(tokens: List[str]) -> List[List[str]]:
    """Feature extraction."""
    # NB: tokens are assumed to already be case-folded.
    vectors = [[f"t[0]={token}"] for token in tokens]
    # Edge features.
    vectors[0].append("__BOS__")
    vectors[-1].append("__EOS__")
    # Preceding and following word features.
    if len(tokens) > 1:
        for i in range(1, len(tokens) - 1):
            prev_token_feat = f"t[-1]={tokens[i - 1]}"
            next_token_feat = f"t[+1]={tokens[i + 1]}"
            vectors[i].append(prev_token_feat)
            vectors[i].append(next_token_feat)
            # Conjunction of the two.
            vectors[i].append(f"{prev_token_feat}^{next_token_feat}")
        if len(tokens) > 2:
            for i in range(2, len(tokens) - 2):
                vectors[i].append(f"t[-2]={tokens[i - 2]}")
                vectors[i].append(f"t[+2]={tokens[i + 2]}")
    # Suffix features.
    for (i, token) in enumerate(tokens):
        vector = vectors[i]
        if len(token) > 1:
            vector.append(_suffix_feature(token, 1))
            if len(token) > 2:
                vector.append(_suffix_feature(token, 2))
                if len(token) > 3:
                    vector.append(_suffix_feature(token, 3))
    # And we're done.
    return vectors
