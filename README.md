True-casing self-study exercise
===============================

[Kyle Gorman](mailto:kgorman@gc.cuny.edu), [M. Elizabeth
Garza](mailto:garza.elizabeth9@gmail.com), and [Emily
Campbell](mailto:ecampbell4@gradcenter.cuny.edu)

- Answers to questions from the [notebook](README.ipynb) can be found under the question

- In [dataset.py](caseify/dataset.py) you can find...
    - Feature extraction code in `extract`
    - Mixed case dictionary code in `get_most_common_mix` 
    
        _Note_: Neither function performs I/O. 
        Results are saved in `run_train_job` in [train.py](caseify/train.py) 

- I break down the individual parts below, however, to run what 
I did, you should run:
    ```
    chmod +x script.sh
    ./script.sh
    ```
 This will generate a `submission` folder in the repo

## Model training end-to-end
- Code to build a dataset, train a model and evaluate that model
 can be found in [train.py](caseify/train.py)
    - I had issue using the command line version of CRFsuite.
    So, I opted to use the Python implementation of it 
    [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/)
    
    - To get the dataset I used run: [get_dataset](get_dataset.sh)
        
        _Note_: you may need to run `chmod +x get_dataset.sh` in order to run the file.
        
    - You can call train a CRF model for case restoration using the following:
        ```python
        python train.py \ 
          --filepath news.2019.en.shuffled.deduped.gz \ 
          --directory your_directory
        ```
      
## Style Transfer
- A model to get a Twitter user's tweets (by username) and train a case restoration model can be
found in [tweets.py](caseify/tweets.py)
    - In order to run that script, you will need to get your Twitter API bearer token.
    You can find out how to get one [here](https://developer.twitter.com/en/docs/authentication/oauth-2-0/bearer-tokens).
    - After, you can run the script:
        ```python
        python tweets.py \
          --bearer_token your_token \
          --user dril \ 
          --dataset_dir your_directory
        ```
  - To compare a pre-existing model against a Twitter model
  run the above with the following:
  
     `--pretrained path_to_your_model`
 
 ### Notes on Style Transfer:
 Comparing the model trained on news v. the Twitter user [@dril](https://twitter.com/dril) we see the following:
 1. Formalness
    - In this case, our model performs "proper" casing as one would expect for an English sentence. 
    For example, the model title cases the first word of the tweets.
 
        A success given what our model was trained on but [@dril](https://twitter.com/dril) 
        is a famous [sh*tposter](https://www.urbandictionary.com/define.php?term=shitposter). 
        So, our model is too formal.
        
        The Twitter model on the other hand is a bit too informal - producing more lower case words.
    
2. URL casing

    - While tweets that were exclusively URLs were removed, we allowed URLs within tweets to remain.
    It's interesting to see how each model handles the URLs...
    
    - The Twitter model always lowercases URLs
    
    - The News model URLs casing is a bit more interesting...
        It produces `"T.Co/"` or `"T.CO"` (these are twitter links afterall) followed by titlecase 
        or all uppercase URL gibberish.
        
        This is an unexepcted great example of style transfer as our model learned to title case initials and is
        now applying it to URLs!
    
    - Lastly, we see our News model is respectful towards religions. 
    It is constantly titlecasing "God", "Lord" and even "Bibles" 
    
    - Both always lowercase `"https:"`  
    
3. Examples:

    _Warning: some examples may contain expletives._

    ```
    Actual casing:
    day of healing challenge: Try being racist for a day if your sjw, and if youre racist try being SJW.
    
    Twitter model: 
    day of healing challenge : try being racist for a day if your Sjw , and if youre racist try being Sjw .
   
    Pretrained model:
    Day of healing challenge : try being racist for a day if your sjw , and if youre racist try being SJW .
    
   ----
   
   Actual casing:
    shooting my lee dungarees with an uzi for that "armyman" look
    
    Twitter model: 
    shooting my lee dungarees with an uzi for that `` Armyman '' look
    Pretrained model:
    Shooting My Lee dungarees with an Uzi for that `` Armyman '' look
   
   ----
   
   Actual casing:
    check out this Exclusive preview and consider supporting the devs (me) https://t.co/bBCbVyxWWT
    
    Twitter model: 
    check out this exclusive preview and consider supporting the devs ( me ) https : //t.co/bbcbvyxwwt
    Pretrained model:
    Check out this exclusive preview and consider supporting the Devs ( me ) https : //T.CO/BBCBVYXWWT
   
   ----
   
   Actual casing:
    @currentvictim I did not show my ass
    
    Twitter model: 
    @ currentvictim I did not show my ass
    Pretrained model:
    @ Currentvictim I did not show my ass
   
   ----
   
   Actual casing:
    check out this Exclusive preview and consider supporting the devs (me) https://t.co/bBCbVyxWWT
    
    Twitter model: 
    check out this exclusive preview and consider supporting the devs ( me ) https : //t.co/bbcbvyxwwt
    Pretrained model:
    Check out this exclusive preview and consider supporting the Devs ( me ) https : //T.CO/BBCBVYXWWT

   ```