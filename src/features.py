"""
Extracts features from a list of casefolded tokens, which will be used to train the crfsuite model to case tag.  
After training, you will have to extract features from a test set of casefolded tokens and then call the model to make predictions. 
"""


from typing import List


def _suffix_features(): 

    ... 

def extract(tokens: List[str]) -> List[List[str]]:

    """
    Extracts word context features and suffix features from a list of casefolded tokens. 

    Argument:  list of casefolded tokens

    Returns:  vectors

    HINT:  You may want to write a separate function that extracts some or all suffix features (e.g. 'def _suffix_features()' above) and then call it in this function.
    """
    ... 


"""
If you run extract features from on the tokenized sentence, 'Nelson Holdings International Ltd. dropped the most on a percentage basis , to 1,000 shares from 255,923.',
your output should look like this: 

    t[0]=nelson     __BOS__ suf1=n  suf2=on suf3=son
    t[0]=holdings   t[-1]=nelson    t[+1]=international     t[-1]=nelson^t[+1]=international        suf1=s  suf2=gs            suf3=ngs
    t[0]=international      t[-1]=holdings  t[+1]=ltd.      t[-1]=holdings^t[+1]=ltd.       t[-2]=nelson    t[+2]=dropped      suf1=l   suf2=al suf3=nal
    t[0]=ltd.       t[-1]=international     t[+1]=dropped   t[-1]=international^t[+1]=dropped       t[-2]=holdings t[+2]=the   suf1=.   suf2=d. suf3=td.
    t[0]=dropped    t[-1]=ltd.      t[+1]=the       t[-1]=ltd.^t[+1]=the    t[-2]=international     t[+2]=most     suf1=d      suf2=ed  suf3=ped
    t[0]=the        t[-1]=dropped   t[+1]=most      t[-1]=dropped^t[+1]=most        t[-2]=ltd.      t[+2]=on       suf1=e      suf2=he
    t[0]=most       t[-1]=the       t[+1]=on        t[-1]=the^t[+1]=on      t[-2]=dropped   t[+2]=a suf1=t  suf2=st            suf3=ost
    t[0]=on t[-1]=most      t[+1]=a t[-1]=most^t[+1]=a      t[-2]=the       t[+2]=percentage        suf1=n
    t[0]=a  t[-1]=on        t[+1]=percentage        t[-1]=on^t[+1]=percentage       t[-2]=most      t[+2]=basis
    t[0]=percentage t[-1]=a t[+1]=basis     t[-1]=a^t[+1]=basis     t[-2]=on        t[+2]=, suf1=e  suf2=ge suf3=age
    t[0]=basis      t[-1]=percentage        t[+1]=, t[-1]=percentage^t[+1]=,        t[-2]=a t[+2]=to        suf1=s suf2=is     suf3=sis
    t[0]=,  t[-1]=basis     t[+1]=to        t[-1]=basis^t[+1]=to    t[-2]=percentage        t[+2]=1,000
    t[0]=to t[-1]=, t[+1]=1,000     t[-1]=,^t[+1]=1,000     t[-2]=basis     t[+2]=shares    suf1=o
    t[0]=1,000      t[-1]=to        t[+1]=shares    t[-1]=to^t[+1]=shares   t[-2]=, t[+2]=from      suf1=0  suf2=00            suf3=000
    t[0]=shares     t[-1]=1,000     t[+1]=from      t[-1]=1,000^t[+1]=from  t[-2]=to        t[+2]=255,923   suf1=s suf2=es     suf3=res
    t[0]=from       t[-1]=shares    t[+1]=255,923   t[-1]=shares^t[+1]=255,923      t[-2]=1,000     t[+2]=. suf1=m suf2=om     suf3=rom
    t[0]=255,923    t[-1]=from      t[+1]=. t[-1]=from^t[+1]=.      suf1=3  suf2=23 suf3=923
    t[0]=.  __EOS__

Once you get the correct output for the sentence above, run your script on 'train.tok' and revise it until you don't get any errors.  
"""








