# cbc-nlp : The "consileon NLP framework"

## What is it for?
The packages in consileon/data simplify and structurize the development and usage of NLP model.

A major part of the development is spent on preparing the input data: Separating "wanted content"
from "garbage", removing unwanted characters, spliting texts into smaller chunks,
generating token files for further usage, performing oversampling, ... .

These tasks can easily be done using (e.g.) elementary python techniques - but:

Doing so
- you _repeat yourself_: many boring tasks have to be done again and again,
Your end up in a confusing bulk of sample code etc.,
- you frequently have to _switch the abstraction level_, e.g. between thinking about the general
concept and implementation details,
- your _code becomes long and unstructured_ and therefore difficult to maintain (you even might not
understand it some time later),
- you prevent _knowledge transfer_,
- you prevent _modularization_ and _encapsulation_ of frequent tasks (and thus flatten the learning
curve of your team).

The folder `consileon/nlp` contains a framework to overcome this. Its mainly contained

Typically, NLP models are generated from one or more text sources which contain (long) sequences
of texts.

These texts are transformed into other objects which can be handled by NLP algorithms, typically
lists of tokens or numbers. These lists are fed into an NLP algorithms.

The transformation is typically done in several steps, e.g.

- split texts into smaller chunks (sentences, paragraphs),
- split chunks of text into tokens (e.g. single words),
- bring tokens into a canonical form (transforming to lower case),
- filter out unwanted tokens,
- "lemmatization", map a conjugated or declined form to its base from (imported especially for
many non-english languages)
- perform (other kinds of) mappings to tokens,
- remove "garbage", i.e. artifacts which are contained in the source but which are of need
for the specific use case (e.g. remove tables of numbers from texts when spoken language is
required),
- append tags to tokens (which specify the source or some semantic information)
- choose subsets (e.g.) of the input sequence for development (or other) reasons,
- merge several data sources,
- and many more.

Finally, the target objects of such transformations are "piped" into a consuming algorithm which
typically generates an NLP model.

## Getting started

See [getting_startet.ipynb](examples/notebooks/getting_started.ipynb)
