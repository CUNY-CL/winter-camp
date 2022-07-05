1. The paper lists some examples of when truecasing might be useful (automatic speech recognition, newspaper titles, etc.) Can you think of any other cases where it might be helpful?
2. Not all languages’ writing systems distinguish between upper and lower-case letters. Are any of the ideas here useful for these languages? For example, the paper notes that "[a]ccents can be viewed as additional surface forms or alternate word casings."
3. What case is a number? For example, would you label 42 as uppercase, lowercase, or something else?
4. In formula 1 (§2.2.1) label all of the variables (, , and ). The authors do not explicitly state what  is; what do you think it denotes?
5. In §2.2.2, the authors they describe which features go with each node of the trellis. Which features are included in the trellis? Of these features, which would you predict to be most useful for predicting case? Can you think of any additional features which might be useful to include?
6. The authors write that "[t]he trellis can be viewed as a Hidden Markov Model (HMM) computing the state sequence which best explains the observations." What are the states of that HMM? What are the observations? What are the transition probabilities?
7. The researchers use a trigram model to predict case. How might the results have been different if they used a bigram or four-gram model?
8. §2.3 discusses two possible approaches for dealing with unknown words. What are they? What are the advantages and disadvantages of each one?
9. For mixed-case tokens, there is no clear rule on which letters in the word are capitalized. Consider iPhone, LaTeX, JavaScript, and McDonald’s. What are some possible approaches to restoring the case of mixed-case tokens?
10. What data is this model trained on, and what are the benefits and disadvantages of these datasets?
