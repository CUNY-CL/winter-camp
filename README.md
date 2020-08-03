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
    found in text messages (SMS) or posts on social media, or even
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

You do not need to be familiar with the (conditional random field
models)\[https://en.wikipedia.org/wiki/Conditional_random_field\] (Lafferty et
al. 2001), one of the technologies we use, but it may be useful to read a bit
about this technology before beginning.

The exercise is intended to take several days; at the [Graduate
Center](https://www.gc.cuny.edu/Page-Elements/Academics-Research-Centers-Initiatives/Doctoral-Programs/Linguistics/Linguistics),
master's students in computational linguistics often complete it as a
supplemental exercise over winter break, after a semester of experience learning
Python. (Hence the name "Winter Camp.")

### Software requirements

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
all of those derived from the Greek, Latin, and Cyrillic alphabets, distinguish
between upper- and lower-case words. Such writing systems are said to be
*bicameral*, and those which do not make these distinctions are said to be
*unicameral*. While casing can carry important semantic information (compare
*bush* vs. *Bush*), this distinction also can introduce further "sparsity" to
our data. Or as Church (1995) puts it, do we **really** need to keep totally
separate statistics for *hurricane* or *Hurricane*, or can we merge them?

In most cases, speech and language processing systems, including machine
translation and speech recognition engines, choose to ignore casing
distinctions; they
[`casefold`](https://docs.python.org/3/library/stdtypes.html#str.casefold) the
data before training. While this is fine for many applications, it is often
desirable to restore capitalization information afterwards, particularly if the
text will be consumed by humans.

Lita et al. (2003) introduce a task they call "true-casing". They use a simple
machine learning model, a [*hidden Markov
models*](https://en.wikipedia.org/wiki/Hidden_Markov_model), to predict the
capitalization patterns of sentences, word by word. They obtain good overall
accuracy (well about 90% accurate) when applying this method to English news
text.

The exercise
------------

The exercise is divided into six parts.

Part 1: the paper
-----------------

Read [Lita et al. 2003](https://www.aclweb.org/anthology/P03-1020/), the study
which introduces the true-casing task. If you encounter unfamiliar jargon, look
it up, or ask a colleague or teacher. Here are some questions about the reading
intended to promote comprehension. Feel free to get creative, even if you don't
end up with the "right" answer.

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
    have been different if they used a bigram or 4-gram model?
8.  §2.3 discusses two possible approaches for dealing with unknown words. What
    are they? What are the advantages and disadvantages of each one?
9.  For mixed-case tokens, there is no clear rule on which letters in the word
    are capitalized. Consider *iPhone*, *LaTeX*, *JavaScript*, and *McDonald's*.
    What are some possible approaches to restoring the case of mixed-case
    tokens?
10. What data is this model trained on, and what are the benefits and
    disadvantages of these datasets?

### Part 2: data and software

#### What to do

1.  Obtain English data. Some tokenized English data from the Wall St. Journal
    (1989-1990) portion of the Penn Treebank is available
    [here](http://wellformedness.com/courses/wintercamp/data/wsj/). These files
    cannot be distributed beyond our "research group", so ask Kyle for the
    password. Alternatively, one can download a year's worth of English data
    from the WMT News Crawl (2007) by executing the following from the command
    line.

    ``` {.bash}
    curl -C - http://data.statmt.org/news-crawl/en/news.2007.en.shuffled.deduped.gz -o "news.2007.gz" && gunzip "news.2007.gz"
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

    ``` {.bash}
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
    ```

    -   Then, to install CRFSuite, execute the following commands.

    ``` {.bash}
    brew tap brewsci/science
    brew install crfsuite
    ```

-   On Linux and the Windows Subsystem for Linux, download and install the
    program by executing the following commands in your system's terminal
    application.

    ``` {.bash}
    curl -O https://github.com/downloads/chokkan/crfsuite/crfsuite-0.12.tar.gz
    tar -xvzf crfsuite-0.12.tar.gz
    sudo mv crfsuite-0.12/bin/crfsuite /usr/local/bin
    ```

    These three commands download, decompress, and install the program into your
    path. Note that the last step may prompt you for your user password.

#### Some background on CRFSuite

**Architecture**: True-casing is a *structured classification* problem, because
the casing of one word depends on nearby words. While one could possibly choose
to ignore this dependency (as would be necessary with a simple [Naïve
Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) or [logistic
regression](https://en.wikipedia.org/wiki/Logistic_regression) classifier),
CRFSuite uses a first-order Markov model in which states represent casing tags,
and the observations represent tokens. The best sequence of tags for a given
sentence are computed using the [Viterbi
algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm), which finds the
best path by merging paths that share prefixes (e.g. the two sequences "NNN" and
"NNV" share the prefix "NN"). This merging calculates the probability of that
prefix only once, as you can see in the figure below.

<p align="center"><img width="460" height="300" src="https://user-images.githubusercontent.com/43279348/86036506-fcfbfa00-ba0b-11ea-819f-6a9f2bf86576.jpg"></p>

Saving the intermediate results of these prefix paths to speed up calculations
is an example of [*dynamic
programming*](https://en.wikipedia.org/wiki/Dynamic_programming), without which
it would take too long to score every possible path.

### Part 3: training

#### Deconstructing case.py
- def get_tc(nunistr: str) -> Tuple[TokenCase, Pattern]:
1. What is the argument of get_ct? What type is it? What does it return? What type is it?
2. Take the following strings and pass them as arguments through this function: 'Mary','milk','LOL', and 'LaTeX'.
3. What are the types of the first and second objects in the returned tuples?
4. Which of the strings above returns a list as the second object in the tuple? What do the elements in that list tell us about the string?
5. There is a way to get this function to only return a tag, or 'TokenCase', of a string type, instead of a tuple. See if you can figure out how to print only the tag of 'Mary' by reading the python documentation for `enum`.) Your expected output should be 'TITLE'.


**TODO**: apply feature extractor to generate data and call `crfsuite learn`.
Also introduce `case.py` and talk about how it works.

### Part 4: prediction

**TODO**: call `crfsuite tag` and convert back to tokenized format.

### Part 5: evaluation

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
wrong, but it is harmless; token is inherently case-less and it doesn't matter
what kind of tag it receives. When multiple predictions all give the right
"downstream" answer, the model is said to exhibit *spurious ambiguity*; a good
evaluation method should not penalize spurious ambiguity. In this case, one can
avoid spurious ambiguity by evaluating not on the tags but on the tokenized
data, after it has been converted back to that format.

#### What to do

Write a script called `evaluate.py`. It should take two command-line arguments:
the path to the original "gold" tokenized and cased data, and the path to the
predicted data from the previous step. It should first initialize two counters,
one for the number of correctly cased tokens, and one for the total number of
tokens. Then, iterating over the two files, count the number of correctly cased
tokens, the number of overall tokens. To compute accuracy, it should divide the
former by the latter, round to 3-6 digits, and print the result. Your evaluation
script should **not** read both files all at once, which will not work for very
large files. Rather it should process the data line by line. For instance, if
the gold data file handle is `gold` and the predicted data file handle is
`pred`, part of your script might resemble the following.

```python
for (gold_line, pred_line) in zip(gold, pred):
    gold_tokens = gold_line.split()
    pred_tokens = pred_line.split()
    assert len(gold_tokens) == len(pred_tokens)
    for (gold_token, pred_token) in zip(gold_tokens, pred_tokens):
        ...
```

### Part 6: style transfer

**TODO**: apply to some data from social media.

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

Wang, W., Knight, K., and Marcu, D. 206. Capitalizing machine translation. In
*Proceedings of the Human Language Technology Conference of the NAACL, Main
Conference*, pages 1-8.
