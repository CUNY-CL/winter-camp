True-casing with a hidden Markov model
======================================

[Kyle Gorman](kgorman@gc.cuny.edu), [M. Elizabeth
Garza](garza.elizabeth9@gmail.com), and [Emily
Campbell](ecampbell4@gradcenter.cuny.edu)

Learning goals
--------------

In this exercise, you will learn about a simple but important NLP task called
*true-casing* or *case restoration*, which allows one to

-   restore missing capitalization in noisy user-generated text as is often
    found in text messages (SMS) or posts on social media,
-   add capitalization to the output of [machine
    translation](https://en.wikipedia.org/wiki/Machine_translation) and [speech
    recognition](https://en.wikipedia.org/wiki/Speech_recognition) to make them
    easier for humans to read, or even
-   transfer the "style" of casing from one collection of documents to another.

As part of this exercise, you will build your own case restoration system using
Python and command-line tools. As such you will get practice using Python to
read and writing text files, and using the command line.

Prerequisites
-------------

You should have a working familiarity with [Python 3](https://www.python.org/)
and be somewhat comfortable using data structures like lists and dictionaries,
calling functions, opening files, importing built-in modules, and reading
technical documentation.

You do not need to be familiar with the [conditional random field
models](https://en.wikipedia.org/wiki/Conditional_random_field) (Lafferty et
al. 2001), one of the technologies we use, but it may be useful to read a bit
about this technology before beginning.

The exercise is intended to take several days; at the [Graduate
Center](https://www.gc.cuny.edu/Page-Elements/Academics-Research-Centers-Initiatives/Doctoral-Programs/Linguistics/Linguistics),
master's students in computational linguistics often complete it as a
supplemental exercise over winter break, after a semester of experience learning
Python. (Hence the name "Winter Camp.")

This tutorial assumes you have access to a UNIX-style command line interface:

-   On Windows 10, you access a command line using [Windows Subsystem for
    Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10); the
    Ubuntu "distro" (distribution) is particularly easy to use.
-   On Mac OS X, you can access the command line interface by opening
    Terminal.app.

It also assumes that you have Python 3.6 or better installed. To test, run
`python --version` from the command line, and note the version number that it
prints. If this returns an error, you probably don't have Python installed yet.
One easy way to obtain a current version of Python is to install
[Anaconda](https://docs.anaconda.com/anaconda/install), a free software package.
Note that if you're using Anaconda from within Windows Subsystem for Linux, you
will want to install the Linux version, not the Windows version.

Case restoration
----------------

Nearly all speech and language technologies work by collecting statistics over
huge collections of characters and/or words. While handful of words (like *the*
or *she*) are very frequent, the vast majority of words (like *ficus* or
*cephalic*) are quite rare. One of the major challenges in speech and language
technology is making informed predictions about the linguistic behaviors of rare
words.

Many [writing systems](https://en.wikipedia.org/wiki/Writing_system), including
those derived from the Greek, Latin, and Cyrillic alphabets, distinguish between
upper- and lower-case words. Such writing systems are said to be *bicameral*,
and those which do not make these distinctions are said to be *unicameral*.
While casing can carry important semantic information (compare *bush*
vs. *Bush*), this distinction also can introduce further "sparsity" to our data.
Or as Church (1995) puts it, do we **really** need to keep totally separate
statistics for *hurricane* or *Hurricane*, or can we merge them?

In most cases, speech and language processing systems, including machine
translation and speech recognition engines, choose to ignore casing
distinctions; they
[case-fold](https://docs.python.org/3/library/stdtypes.html#str.casefold) the
data before training. While this is fine for many applications, it is often
desirable to restore capitalization information afterwards, particularly if the
text will be consumed by humans. Shugrina (2010) reports that users greatly
prefer formatted transcripts over "raw" transcripts.

Lita et al. (2003) introduce a task they call "true-casing". They use a simple
machine learning model, a [*hidden Markov
model*](https://en.wikipedia.org/wiki/Hidden_Markov_model), to predict the
capitalization patterns of sentences word by word. They obtain good overall
accuracy (well above 90%) when applying this method to English news text.

In this exercise, we will develop a variant of the Lita et al. model, with some
small improvements. The exercise is divided into six parts.

Structured classification
-------------------------

True-casing is a *structured classification* problem, because the casing of one
word depends on nearby words. While one could possibly choose to ignore this
dependency (as would be necessary with a simple [Naïve
Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) or [logistic
regression](https://en.wikipedia.org/wiki/Logistic_regression) classifier),
CRFSuite uses a first-order Markov model in which states represent casing tags,
and the observations represent tokens. The best sequence of tags for a given
sentence are computed using the [Viterbi
algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm), which finds the
best path by merging paths that share prefixes (e.g. the sequences "NNN" and
"NNV" share the prefix "NN"). This merging calculates the probability of that
prefix only once, as you can see in the figure below.

<p align="center">
<img width="460" height="300" src="https://user-images.githubusercontent.com/43279348/86036506-fcfbfa00-ba0b-11ea-819f-6a9f2bf86576.jpg">

Caching the intermediate results of these prefix paths to speed up calculations
is an example of [*dynamic
programming*](https://en.wikipedia.org/wiki/Dynamic_programming). Without this
trick, it would take far too long to score every possible path.

Part 1: the paper
-----------------

Read [Lita et al. 2003](https://www.aclweb.org/anthology/P03-1020/), the study
which introduces the true-casing task. If you encounter unfamiliar jargon, look
it up, or ask a colleague or instructor. Here are some questions about the
reading intended to promote comprehension. Feel free to get creative, even if
you don't end up with the "right" answer.

1.  The paper lists some examples of when truecasing might be useful (automatic
    speech recognition, newspaper titles, etc.) Can you think of any other cases
    where it might be helpful?
2.  Not all languages' writing systems distinguish between upper and lower-case
    letters. Are any of the ideas here useful for these languages? For example,
    the paper notes that "\[a\]ccents can be viewed as additional surface forms
    or alternate word casings."
3.  What case is a number? For example, would you label "42" as uppercase,
    lowercase, or something else?
4.  In formula 1 (§2.2.1) label all of the variables ($$P$$, $$\lambda$$, and
    $$w$$). The authors do not explicitly state what $$\lambda_{uniform} P_0$$
    is; what do you think it means?
5.  In §2.2.2, the authors they describe which features go with each node of the
    trellis. Which features are included in the trellis? Of these features,
    which would you predict to be most useful for predicting case? Can you think
    of any additional features which might be useful to include?
6.  The authors write that "\[t\]he trellis can be viewed as a Hidden Markov
    Model (HMM) computing the state sequence which best explains the
    observations." What are the states of that HMM? What are the observations?
    What are the transition probabilities?
7.  The researchers use a trigram model to predict case. How might the results
    have been different if they used a bigram or four-gram model?
8.  §2.3 discusses two possible approaches for dealing with unknown words. What
    are they? What are the advantages and disadvantages of each one?
9.  For mixed-case tokens, there is no clear rule on which letters in the word
    are capitalized. Consider *iPhone*, *LaTeX*, *JavaScript*, and *McDonald's*.
    What are some possible approaches to restoring the case of mixed-case
    tokens?
10. What data is this model trained on, and what are the benefits and
    disadvantages of these datasets?

Part 2: data and software
-------------------------

### What to do

1.  Obtain English data. Some tokenized English data from the Wall St. Journal
    (1989-1990) portion of the Penn Treebank is available
    [here](http://wellformedness.com/courses/wintercamp/data/wsj/). These files
    cannot be distributed beyond our "research group", so ask Kyle for the
    password. Alternatively, one can download a year's worth of English data
    from the WMT News Crawl (2007) by executing the following from the command
    line.

    ```bash
    curl -C - http://data.statmt.org/news-crawl/en/news.2007.en.shuffled.deduped.gz -o "news.2007.gz"
    ```

(Note that you can replace the "2007" above with any year from 2007 to 2019.) If
you use the News Crawl data, you will need to tokenize it and split into
training, development, and testing sets to match the format of the Wall
St. Journal data. Therefore, write a Python script that tokenizes the data
(e.g., using
[`nltk.word_tokenize`](https://www.nltk.org/_modules/nltk/tokenize/punkt.html#PunktLanguageVars.word_tokenize)),
randomly splits the data into training (80%), development (10%), and testing
(10%), and writes the data to separate files (`train.tok`, `dev.tok`, and
`test.tok`).

2.  Install the [CRFSuite](http://www.chokkan.org/software/crfsuite/) tagger.

-   On Mac OS X, the easiest way to install CRFSuite is via
    [Homebrew](https://brew.sh/).

    -   Install Homebrew, if you have not already, by executing the following
        command in Terminal.app, and following the on-screen instructions.

    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
    ```

    -   Then, to install CRFSuite, execute the following commands.

    ```bash
    brew tap brewsci/science
    brew install crfsuite
    ```

-   On Linux and the Windows Subsystem for Linux, download and install the
    program by executing the following commands in your system's terminal
    application.

    ```bash
    curl -LO https://github.com/downloads/chokkan/crfsuite/crfsuite-0.12-x86_64.tar.gz
    tar -xzf crfsuite-0.12-x86_64.tar.gz
    sudo mv crfsuite-0.12/bin/crfsuite /usr/local/bin
    ```

    These three commands download, decompress, and install the program into your
    path. Note that the last step may prompt you for your user password.

Part 3: training
----------------

In this step you will train the case-restoration model.

### Reading `case.py`

One step during training requires us to tag tokens by their case. For instance,
the sequence `He ate a prune wafer.` would be tagged as

    He  TITLE    
    ate LOWER 
    a   LOWER 
    prune   LOWER 
    wafer   LOWER
    .   DC 

The `TITLE` tag refers to title-case, where only the first character is
capitalized, and `DC` ("don't care") is used for punctuation and digit tokens,
which are neither upper- nor lower-case. Special treatment is required for
mixed-casetokens like `LaTeX`. At the token, these are labeled `MIXED`, but
additional data is needed to keep track of the character-level patterns. For
instance, we might use the following tags:

    L   UPPER
    a   LOWER
    T   UPPER  
    e   LOWER
    X   UPPER

To tag tokens and characters, you will use the functions of a provided Python
module, [`case.py`](src/case.py). If you'd like to go through the exercises
below in a Jupyter notebook or code editor, ensure that `case.py` is in your
working directory (i.e., the same directory as your Jupyter notebook), and then
execute `import case` before beginning. We will proceed to go function by
function.

-   `def get_cc(nunichar: str) -> CharCase: ...`

1.  What is the argument to `get_cc`? What is the argument's type? What does
    `get_cc` return?
2.  What do you obtain when you pass the following strings as arguments to this
    function: `"L"`, `"a"`, `","`?
3.  Which kinds of strings return the object `<CharCase.DC>`?
4.  Read the documentation for
    [`unicodedata`](https://docs.python.org/3/library/unicodedata.html), one of
    the libraries used to implement this function. Why does the argument have to
    be a single Unicode character?

-   `def get_tc(nunistr: str) -> Tuple[TokenCase, Pattern]: ...`

1.  What is the argument to `get_tc`? What is the argument's type? What does
    `get_tc` return?
2.  What do you obtain when you pass the following strings as arguments to this
    function: `"Mary"`, `"milk"`, `"LOL"', and`"LaTeX"\`?
3.  What are the types of the first and second objects in the returned tuples?
4.  Which of the strings above returns a list as the second object in the tuple?
    What do the elements in that list tell us about the string?

-   `def apply_cc(nunichar: str, cc: CharCase) -> str: ...`

1.  What are the arguments to `apply_cc`? What is their types? What does
    `apply_cc` return?
2.  Apply `CharCase.UPPER` to the following strings: `"L"`, `"A"`. What do you
    obtain?
3.  Repeat the previous step but with `CharCase.LOWER`.
4.  Read the tests in [`case_test.py`](src/case_test.py) to see how they use
    `apply_cc`.

-   `def apply_tc(nunistr: str, tc: TokenCase, pattern: Pattern = None) -> str:...`

1.  What are the arguments to `apply_tc`? What are the arguments' types? What
    does `apply_tc` return?
2.  Apply `TokenCase.LOWER` to the following strings: `"Mary"`, `"milk"`,
    `"LOL"', and `"LaTeX"`. What do you obtain??
3.  Repeat the previous step but with `TokenCase.TITLE` and `TokenCase.UPPER`.
4.  Read the tests in [`case_test.py`](src/case_test.py) to see how they use
    `apply_tc`.

### Feature extraction

During both training and prediction, the model must be provided with features
used to determine the casing pattern for each token. Minimally, these features
should include:

1.  The target token
2.  The token to the left (or `__BOS__` if the token is sentence-initial)
3.  The token to the right (or `__EOS__` if the token is sentence-final)
4.  The conjunction of \#2 and \#3.

Optionally, one may also provide features such as:

1.  The token two to the left (or `__BOS__`)
2.  The token two to the right (or `__EOS__`)
3.  Prefixes and/or suffixes of the target token.

CRFSuite requires feature files for both training and prediction. Each line
consists of a single token's features, separated by a tab (`\t`) character, with
a blank line between each sentence. Feature files for training should also
include the tag itself as the first column in the feature. Thus, for the
sentence

`Nelson Holdings International Ltd. dropped the most on a percentage basis , to 1,000 shares from 255,923 .`

the training feature file might look a bit like:

    TITLE   t[0]=nelson     __BOS__ suf1=n  suf2=on suf3=son
    TITLE   t[0]=holdings   t[-1]=nelson    t[+1]=international     t[-1]=nelson^t[+1]=international        suf1=s  suf2=gs            suf3=ngs
    TITLE   t[0]=international      t[-1]=holdings  t[+1]=ltd.      t[-1]=holdings^t[+1]=ltd.       t[-2]=nelson    t[+2]=dropped      suf1=l   suf2=al suf3=nal
    TITLE   t[0]=ltd.       t[-1]=international     t[+1]=dropped   t[-1]=international^t[+1]=dropped       t[-2]=holdings t[+2]=the   suf1=.   suf2=d. suf3=td.
    LOWER   t[0]=dropped    t[-1]=ltd.      t[+1]=the       t[-1]=ltd.^t[+1]=the    t[-2]=international     t[+2]=most     suf1=d      suf2=ed  suf3=ped
    LOWER   t[0]=the        t[-1]=dropped   t[+1]=most      t[-1]=dropped^t[+1]=most        t[-2]=ltd.      t[+2]=on       suf1=e      suf2=he
    LOWER   t[0]=most       t[-1]=the       t[+1]=on        t[-1]=the^t[+1]=on      t[-2]=dropped   t[+2]=a suf1=t  suf2=st            suf3=ost
    LOWER   t[0]=on t[-1]=most      t[+1]=a t[-1]=most^t[+1]=a      t[-2]=the       t[+2]=percentage        suf1=n
    LOWER   t[0]=a  t[-1]=on        t[+1]=percentage        t[-1]=on^t[+1]=percentage       t[-2]=most      t[+2]=basis
    LOWER   t[0]=percentage t[-1]=a t[+1]=basis     t[-1]=a^t[+1]=basis     t[-2]=on        t[+2]=, suf1=e  suf2=ge suf3=age
    LOWER   t[0]=basis      t[-1]=percentage        t[+1]=, t[-1]=percentage^t[+1]=,        t[-2]=a t[+2]=to        suf1=s suf2=is     suf3=sis
    DC      t[0]=,  t[-1]=basis     t[+1]=to        t[-1]=basis^t[+1]=to    t[-2]=percentage        t[+2]=1,000
    LOWER   t[0]=to t[-1]=, t[+1]=1,000     t[-1]=,^t[+1]=1,000     t[-2]=basis     t[+2]=shares    suf1=o
    DC      t[0]=1,000      t[-1]=to        t[+1]=shares    t[-1]=to^t[+1]=shares   t[-2]=, t[+2]=from      suf1=0  suf2=00            suf3=000
    LOWER   t[0]=shares     t[-1]=1,000     t[+1]=from      t[-1]=1,000^t[+1]=from  t[-2]=to        t[+2]=255,923   suf1=s suf2=es     suf3=res
    LOWER   t[0]=from       t[-1]=shares    t[+1]=255,923   t[-1]=shares^t[+1]=255,923      t[-2]=1,000     t[+2]=. suf1=m suf2=om     suf3=rom
    DC      t[0]=255,923    t[-1]=from      t[+1]=. t[-1]=from^t[+1]=.      suf1=3  suf2=23 suf3=923
    DC      t[0]=.  __EOS__

and for prediction, one would simply omit the first column.

#### What to do

Write a function which, given a sentence, extracts a list of list of features
for that sentence. It might have the following signature:

`def extract(tokens: List[str]) -> List[List[str]]: ...`

Then, write a script which uses this function to generate a feature file. To
make this work, you will want to first split each sentence into tokens, call
`extract` to obtain the features, then, for each token, print the tag and the
features, separated by tab. Remember to also print a blank line between each
sentence.

#### Hints

-   If you get stuck, you may want to "peek" at
    [`features.py`](src/features.py), which contains a draft of the feature
    extractor function itself.
-   CRFSuite follows a convention whereby a `:` in a feature is interpreted as a
    feature weight. Therefore we suggest that you replace `:` in any token with
    another character such as `_`. If this is done consistently for all feature
    files, this is extremely unlikely to result in a loss of information.

### The mixed-case dictionary

Mixed-case tokens like `McDonald's` and `LaTeX` all have different mixed-case
patterns. It is not enough to know that they are mixed: one also has to know
which mixed-case pattern they follow. One can make a simplifying assumption
that few if any mixed-case tokens vary in which mixed-case pattern they follow
(or that such variation is mostly erroneous), and therefore one simply needs to
store a table with the most-common pattern for each mixed-case token, which is
used to produce the proper form of a token tagged as mixed-case. This table is
computed in two steps.

1.  First, count the frequency of each mixed-case pattern for each token.
2.  Then, select only the most frequent mixed-case pattern for each token.

For step \#1, we use a dictionary whose keys are case-folded strings whose
values are
[`collections.Counter`](https://docs.python.org/3/library/collections.html#collections.Counter)
objects containing the mixed-case form. For instance, this might resemble the
following.

```python
{'iphone' : Counter{'iPhone': 11, 'IPhone': 5, 'iphone': 3}, ...}
```

Then, for step \#2, we create a simpler dictionary which contains just the most
frequent mixed-case pattern as the value, like the following:

```python
{'iphone': 'iPhone', ...}
```

#### What to do

Write code which reads a tokenized file, constructs the second dictionary, and
then writes it to disk as a [JSON
file](https://docs.python.org/3/library/json.html) using
[`json.dump`](https://docs.python.org/3/library/json.html). You may wish to
combine this with the code you wrote in the previous step.

#### Hints

-   If you get stuck, you may want to carefully read the documentation for
    [`collections.Counter`](https://docs.python.org/3/library/collections.html#collections.Counter)
    and
    [`collections.defaultdict`](https://docs.python.org/3/library/collections.html#collections.defaultdict),
    particularly the [provided
    examples](https://docs.python.org/3/library/collections.html#defaultdict-examples).
-   You will need to create an empty dictionary before looping over the data.
    This may look something like the following.

```python
mcdict = collections.defaultdict(collections.Counter)
```

Then, inside the loop(s), you can add a count to the dictionary as in the
following snippet.

```python
mcdict[token.casefold()][token] += 1
```

-   If you are working with very large data sets, you may want to slightly
    modifystep \#2 to exclude tokens that occur in mixed-case very infrequently
    (e.g., less than twice) on the hypothesis that such tokens are typographical
    errors.

### CRF training

You are now almost ready to train the model itself. To do this you will need

-   a feature file (including tags) `train.features` for training data
    `train.tok`, and
-   a feature file (including tags) `dev.features` for development data
    `dev.tok`,

which can be created using the `extract` function described above.

#### What to do

At the command line, issue the following command to train the CRF model:

```bash
crfsuite learn \
    -p feature.possible_states=1 \
    -p feature.possible_transitions=1 \
    -m model \
    -e2 train.features dev.features
```

This will train the model and write the result (in a non-human-readable format)
to the file `model`; training may take up to several minutes.

Part 4: prediction
------------------

To predict, or restore case to the tokens in `test.tok` using the model you
trained in **Part 3**, complete the following steps.

1.  Extract features from `test.tok` and write them to `test.features`. This
    file should **not** include case tags.
2.  At the command line, issue the following command to apply the model you
    trained in **Part 3**:

```bash
crfsuite tag -m model test.features > test.predictions
```

3.  Using the predicted tags in `test.predictions`, the mixed-cased dictionary,
    and `case.py`'s `apply_*` functions, apply casing to case-folded tokens in
    `test.tok`, and then write these restored-case tokens to a file formatted
    similarly to `test.tok`, with one sentence per line and space between each
    token.

Part 5: evaluation
------------------

So, how good is your true-caser? There are many ways we can imagine measuring
this. One could ask humans to rate the quality of the output casing, and one
might even want to take into account how often two humans agree about whether a
word should or should not be capitalized. However a simpler evaluation (and one
which does not require humans "in the loop") is to compute token-level accuracy.
Accuracy can be thoguht of as the probability that a randomly selected token
will receive the correct casing.

While it is possible to compute accuracy directly on the tags produced by
`crfsuite`, this has the risk of slightly underestimating the actual accuracy.
For instance, if the system tags a punctuation character as `UPPER`, this seems
wrong, but it is harmless; punctuation tokens are inherently case-less and it
doesn't matter what kind of tag it receives. When multiple predictions all give
the right "downstream" answer, the model is said to exhibit *spurious
ambiguity*; a good evaluation method should not penalize spurious ambiguity. In
this case, one can avoid spurious ambiguity by evaluating not on the tags but on
the tokenized data, after it has been converted back to that format.

### What to do

Write a script called `evaluate.py`. It should take two command-line arguments:
the path to the original "gold" tokenized and cased data, and the path to the
predicted data from the previous step. It should first initialize two counters,
one for the number of correctly cased tokens, and one for the total number of
tokens. Then, iterating over the two files, count the number of correctly cased
tokens, the number of overall tokens. To compute accuracy, it should divide the
former by the latter, round to 3-6 digits, and print the result.

### Hints

-   Your evaluation script should **not** read both files all at once, which
    will not work for very large files. Rather it should process the data line
    by line. For instance, if the gold data file handle is `gold` and the
    predicted data file handle is `pred`, part of your script might resemble the
    following.

```python
for (gold_line, pred_line) in zip(gold, pred):
    gold_tokens = gold_line.split()
    pred_tokens = pred_line.split()
    assert len(gold_tokens) == len(pred_tokens), "Mismatched lengths"
    for (gold_token, pred_token) in zip(gold_tokens, pred_tokens):
        ...
```

Part 6: style transfer
----------------------

*Style transfer* refers to the use of machine learning to apply the "style" of a
*reference* medium (e.g., texts, images, or videos) to to some other resource of
the same type. For instance, in computer vision, researchers have used this
technique to [transfer the styles of Picasso, Van Gogh, and Monet onto a
painting of da Vinci](https://genekogan.com/works/style-transfer/). In this
section, you will use the true-casing model to transfer the "casing style" from
some novel corpus to the data you used above.

### What to do

1.  Obtain a reference corpus. A pre-tokenized corpus of tweets by
    [\@realDonaldTrump](https://twitter.com/realDonaldTrump) is available
    [here](http://wellformedness.com/courses/wintercamp/data/trump/).
    Alternatively, one can obtain data from some other source, such as the
    social media accounts of other famous personages (e.g.,
    [\@dril](https://twitter.com/dril),
    [\@FINALLEVEL](https://twitter.com/finallevel), and then tokenize the data
    as described in Part 2.

2.  Using the data from the previous step, training a casing model as in Part 3.

3.  Using the model from the previous step, apply this model to the test data
    from Part 2 as in Part 4.

4.  Compare, manually or automatically, the predictions of the style-transfer
    model to those of the in-domain model. How do they differ?

Postscript
----------

### Further reading

Other interesting work on case-restoration includes Chelba & Acero 2006 and Wang
et al. 2006.

### Stretch goals

1.  Above, the act of training the model and casing data required several
    commands. It may be more convenient to combine the steps into two programs,

-   a "trainer" which converting tokenized data to two-column format and invokes
    `crfsuite learn` on it, and

-   a "predictor" which converts tokenized data to one-column format, invokes
    `crfsuite tag`, and then converts it back to tokenized format.

    The obvious way to do this is to write two Python scripts which, as part of
    their process, call shell commands using the built-in
    [`subprocess`](https://docs.python.org/3/library/subprocess.html) module. In
    particular, use
    [`subprocess.check_call`](https://docs.python.org/3/library/subprocess.html#subprocess.check_call),
    which executes a shell command and raises an error if it fails, or
    [`subprocess.popen`](https://docs.python.org/3/library/subprocess.html#popen-constructor),
    which has a similar syntax but also captures the output of the command.
    Since both these scripts must generate temporary files (i.e., the data in
    one- or two-column format) you may want to use the built-in
    [`tempfile`](https://docs.python.org/3/library/tempfile.html) module, and in
    particular
    [`tempfile.NamedTemporaryFile`](https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile)
    or
    [`tempfile.mkstemp`](https://docs.python.org/3/library/tempfile.html#tempfile.mkstemp).
    Since these scripts will need to take various arguments (paths to input and
    output files, as well as the model order hyperparameters), use the built-in
    [`argparse`](https://docs.python.org/3/library/argparse.html) module to
    parse command-line flags.
    
2.  Convert your implementation to a [Python
    package](https://packaging.python.org/) which can be installed using `pip`.
    If you are also doing stretch goal \#1, you may want to make the trainer and
    the predictor separate ["console
    scripts"](https://python-packaging.readthedocs.io/en/latest/command-line-scripts.html#the-console-scripts-entry-point).
    Make sure to fail gracefully if the user doesn't already have `crfsuite`
    installed. Alternatively, you can build your package around the
    [`python-crfsuite`](https://github.com/scrapinghub/python-crfsuite) package,
    available from `pip`, which exposes the CRFSuite internals directly to
    Python, without the need to call the `crfsuite` binary.
    
3.  Train and evaluate a model on a language other than English (though make
    sure the writing system makes case distinctions---not all do); even French
    and German both have rather different casing rules.
    
4.  Train, evaluate, and distribute a **gigantic** case restoration model using
    a megacorpus such as

-   [Gigaword](https://catalog.ldc.upenn.edu/LDC2011T07),

-   the [Billion Word
    Benchmark](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark),
    or

-   multiple years of the [WMT news
    crawl](https://gist.github.com/kylebgorman/5109b09fbfc3a2c1dbbdd405326c1130)
    data.

    To prevent the number of features from swamping your training, you may wish
    to use
    [`-p feature.minfreq=`](http://www.chokkan.org/software/crfsuite/manual.html#idp8853519024)
    when training. For instance, if you call
    `crfsuite learn -p feature.minfreq=10 ...`, then any feature occurring less
    than 10 times in the training data will be ignored.

5.  Above we provided a relatively simple feature extraction function. Would
    different features do better? Add, remove, or combine features, retrain your
    model, and compare the results the provided feature function.
    
6.  Alternatively, you can try a different type of model altogether using the
    [`perceptronix`](https://github.com/kylebgorman/perceptronix/) library,
    which provides a fast C++-based backend for training linear sequence models
    with the perceptron learning algorithm. Install Perceptronix, then using
    `case.py` and the `perceptronix.SparseDenseMultinomialSequentialModel`
    class, build a discriminative case restoration engine and compare the
    results to the CRF model.
    
7.  Section 2.3 of Lita et al. proposes an alternative method for handling rare
    words. Re-read that section and implement their proposal, then compare it to
    your earlier implementation.

References
----------

Chelba, C. and Acero, A.. 2006. Adaptation of maximum entropy capitalizer:
little data can help a lot. *Computer Speech and Language* 20(4): 382-39.

Church, K. W. 1995. One term or two? In *Proceedings of the 18th Annual
International ACM SIGIR conference on Research and Development in Information
Retrieval*, pages 310-318.

Lafferty, J., McCallum, A., and Pereira, F. 2001. Conditional random fields:
probabilistic models for segmenting and labeling sequence data. In *Proceedings
of the 18th International Conference on Machine Learning*, pages 282-289.

Lita, L. V., Ittycheriah, A., Roukos, S. and Kambhatla, N. 2003. tRuEcasIng. In
*Prooceedings of the 41st Annual Meeting of the Association for Computational
Linguistics*, pages 152-159.

Shugrina, M. 2010. Formatting time-aligned ASR transcripts for readability. In
*Human Language Technologies: The 2010 Annual Conference of the North American
Chapter of the Association for Computational Linguistics*, pages 198-206.

Wang, W., Knight, K., and Marcu, D. 206. Capitalizing machine translation. In
*Proceedings of the Human Language Technology Conference of the NAACL, Main
Conference*, pages 1-8.
