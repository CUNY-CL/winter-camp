from typing import List

import requests
from case import is_url
from nltk import sent_tokenize
from train import run_train_job


def _get_header(bearer_token: str):
    return {"Authorization": "Bearer {}".format(bearer_token)}


def get_twitter_id(bearer_token: str, username: str) -> str:
    headers = _get_header(bearer_token)
    req = requests.get(f"https://api.twitter.com/2/users/by/username/{username}", headers=headers)
    if req.ok:
        return req.json()['data']['id']
    else:
        raise Exception(req.text)


def get_user_tweets(bearer_token: str, username: str, exclude_retweets: bool = True) -> List[dict]:

    print(f"Looking up TwitterID for {username}...", end='')
    twitter_id = get_twitter_id(bearer_token, username)
    print(f"done!"
          f"\n{username}'s TwitterID is {twitter_id}"
          f"\nFetching tweets...")

    tweets = []
    headers = _get_header(bearer_token)
    params = {'max_results': 100}

    if exclude_retweets:
        params['exclude'] = 'retweets'

    req = requests.get(
        url=f"https://api.twitter.com/2/users/{twitter_id}/tweets",
        headers=headers,
        params=params
    )
    resp = req.json()

    tweets.extend(resp['data'])
    i = 0
    next_token = resp['meta']['next_token']
    while next_token:
        print(f"Fetching more tweets {i+1} ({next_token})...", end='')

        params['pagination_token'] = next_token

        req = requests.get(
            url=f"https://api.twitter.com/2/users/{twitter_id}/tweets",
            headers=headers,
            params=params)

        if not req.ok:
            print("Breaking:\n", req.text)
            break
        else:
            resp = req.json()
            print(f"fetched {len(resp['data'])} more!")

            tweets.extend(resp['data'])

            if not resp['meta'].get('next_token'):
                break

            next_token = resp['meta']['next_token']
            i += 1

    print(f"Found {len(tweets)} total for {username} ({twitter_id})!")
    return tweets


def clean_tweets(tweets: List[str],
                 split_sentences: bool = False,
                 remove_urls: bool = True) -> List[str]:

    cleaned = []
    for tweet in tweets:
        tweet = tweet.replace("\n", " ").replace("\t", " ").strip()

        if remove_urls and is_url(tweet):
            continue

        if split_sentences:
            cleaned.extend(sent_tokenize(tweet))
        else:
            cleaned.append(tweet)

    return cleaned


def save_tweets(bearer_token: str,
                username: str,
                filepath: str,
                clean_up: bool = False,
                **clean_kwargs):

    tweet_json = get_user_tweets(bearer_token=bearer_token, username=username)
    tweets = [tweet['text'] for tweet in tweet_json]

    if clean_up:
        print(f"Cleaning {len(tweets)} with kwargs={clean_kwargs}")
        tweets = clean_tweets(tweets=tweets, **clean_kwargs)

    with open(filepath, 'w+') as f:
        f.write("\n".join(tweets))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bearer_token", help='Path to file containing API keys')
    parser.add_argument("-u", "--user", help="Username to get tweets for")
    parser.add_argument("-f", "--filepath", help="Filepath to save tweets to")
    parser.add_argument("-c", "--clean", help="Clean up tweets", default=False)
    parser.add_argument("-s", "--sent_tokenize", help='If clean=True, will sentence tokenize tweets.', default=False)
    parser.add_argument("-r", "--remove_urls", help="If clean=True, will remove tweets that are hyperlinks", default=True)
    parser.add_argument("-d", "--dataset_dir", help="If not None, will trigger model training and save all artifacts to that directory", default=None)
    args = parser.parse_args()

    save_tweets(bearer_token=args.bearer_token,
                username=args.user,
                filepath=args.filepath,
                clean_up=bool(args.clean),
                split_sentences=bool(args.sent_tokenize),
                remove_urls=bool(args.remove_urls))

    if args.dataset_dir:
        run_train_job(dataset_fp=args.filepath,
                      dataset_dir=args.dataset_dir)
