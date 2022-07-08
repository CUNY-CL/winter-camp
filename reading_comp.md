1. The paper lists some examples of when truecasing might be useful (automatic speech recognition, newspaper titles, etc.) 
Can you think of any other cases where it might be helpful?
    - Truecasing might be useful for pre-processing texts such as e-mails, comment threads, etc.
    
2. Not all languages’ writing systems distinguish between upper and lower-case letters. 
Are any of the ideas here useful for these languages? 
For example, the paper notes that "[a]ccents can be viewed as additional surface forms or alternate word casings."
    -  To answer this, we look at the [paper cited](https://aclanthology.org/P94-1013.pdf) in this section.
    The ideas in this paper are useful whenever 
    "it is necessary to disambiguate two or more semantically distinct word-forms which have been conflated into the same representation in some medium. "
    So, true-casing is of course one example of this as is accent restoration. 
    
3. What case is a number? For example, would you label 42 as uppercase, lowercase, or something else?
    - As proposed later, we can add an additional class DC for "don't care" for numbers.
    However, the case of a number does not matter therefore any class designation would suffice so long
    as it is consistent across the corpus. The only concern there would be effecting the metrics
    of the class which numbers are assigned to. We could, however, in our evaluation ignore numbers to avoid this.
    
4. In formula 1 (§2.2.1) label all of the variables (, , and ). The authors do not explicitly state what  is; what do you think it denotes?
    - P_0 is bias and lambda uniform is the uniform distribution of all casings observed?
    
5. In §2.2.2, the authors they describe which features go with each node of the trellis. Which features are included in the trellis? Of these features, which would you predict to be most useful for predicting case? Can you think of any additional features which might be useful to include?
    - From the paper we have:
    
        `A node in this trellis consists of a lexical item, a position in the sentence, a
        possible casing, as well as a history of the previous
        two lexical items and their corresponding case content`
     
    The most useful feature, in my opinion, we be the corresponding case content of the previous two lexical items.
    
    Additional features could be:
        - Model only looks at two previous words, we could add the two proceeding words as well
        - We may also wish to add the prefix and suffix of the current word
        
6. The authors write that "[t]he trellis can be viewed as a Hidden Markov Model (HMM) computing the state sequence which best explains the observations." What are the states of that HMM? What are the observations? What are the transition probabilities?
    - States of HMM = combinations of case and context information
    - Observations = lexical items
    - Transition probabilities = the language model (λ) based features  

7. The researchers use a trigram model to predict case. How might the results have been different if they used a bigram or four-gram model?
    - When removing or adding n-grams, we'd expect to decrease or increase performance, respectively.
    This is because we are decreasing or increasing the context considered around our word.
    
        We'd expect using only bigrams would cause a performance downgrade due to smaller context considered 
        and would expect an increase in performance for 4-grams because of a larger context.
        
        For example, we'd imagine the bigram model having a more difficult time predicting the casing for "New York City"
        versus the trigram model. 
        The 4-gram model should perform better on "New York City Knicks" over the trigram model.
        
        The difference, however, may be minor as higher n-grams because exponentially more infrequent.
        Therefore, as we add higher n-grams, we'd expect diminishing returns due to higher n-grams being very infrequent.  
    
8. §2.3 discusses two possible approaches for dealing with unknown words. 
What are they? 
What are the advantages and disadvantages of each one?
    - First approach is to assume all unknown words are UC forms (aka title case)
        - The advantage and disadvantage is that it is a simplifying assumption.
        
            It's advantageous because it's very easy to implement. In a sufficiently large corpus, you can assume
            the vocabulary is robust enough such that unknown words are infrequent and likely to be proper nouns or 
            mispellings, as the paper points out. 
            
            Let's consider the example of correcting news headlines from the beginning of the paper. 
            It should be a perfectly fine assumption to assume all unknown words are proper nouns and should be UC. 
            Typos should be rare in a newspaper (especially the headline.)
            
            For noisier sources of text, this assumption can fail easily. 
            For example, the model in this paper is trained on news articles and therefore English slang is outside of its
            vocabulary. Very little slang is meant to be title cased and thus this assumption fails. 
             
    - The second approach is to special case-carrying tokens to a subset of tokens during training. 
    i.e. UNKNOWN_{case} e.g. UNKNOWN_LC, UNKNOWN_UC. In the paper they propose doing this on infrequent words in 
    the vocabulary or a random subset of _all_ tokens (they use the former method for their training.)
        - The advantage of this is that it specifically trains the model to prepare for unknown words in the
        vocabulary. 
        - One immediate disadvantage is that it does not account for the other two casings in the paper 
        (all uppercase and mixed.) The authors seems to have carried over their simplifying assumption that all unknown
        words are either proper nouns or misspellings to this method.
        - Another disadavantge is with the first method of applying these tokens. 
        Given that it is only using the infrequent words in the vocabulary, the examples they are used in are rare
        and, additionally, the contexts in which they are used may also be rare. 
        The second proposed method seems more robust then the method the authors chose.    
     
    
9. For mixed-case tokens, there is no clear rule on which letters in the word are capitalized. Consider iPhone, LaTeX, JavaScript, and McDonald’s. What are some possible approaches to restoring the case of mixed-case tokens?
    - As proposed in the paper, we can track the most popular form of a mixed-case token and apply it.
    
    - We may also update our model to operate on characters instead of tokens. 
    This is likely to be a more difficult problem but allows us to recover mixed-casings
     
10. What data is this model trained on, and what are the benefits and disadvantages of these datasets?
    - As stated in the paper the model is trained on "AQUAINT
    (ARDA) and TREC (NIST) corpora, each consisting of 500M token news stories from various news agencies"
    - Some benefits are:
        - News articles are as close to absolute truth as one can get for casing
        - Abundance of stories should provide fairly robust vocabulary and include plenty of properly cased enitites
    - Some downsides are:
        - News article are often more formal than the average piece of text. 
        Therefore, model may have a tough time predicting casing on more relaxed English texts
        (e.g. text messages, tweets 😉, website comment sections, audio transcriptions, etc.)
        One could argue that excluding more relaxed pieces of text is an advantage but news often lags behind
        these other mediums in terms of slang. 
        So, the model will likely perform well on formally written texts (e.g. news, research papers, Wikipedia articles, etc.)
        but will likely perform worse on less formal writing (see examples above) which make up the majority of written
        content on the internet.
        - Model will likely be weak to misspellings as there are (or, at least, should be) very few typos
        in news articles.