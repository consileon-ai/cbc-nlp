# cbc-nlp : The "consileon NLP framework"

## Why Consileon NLP Framework?
NLP models are developed based on text sources which contain (long) sequences of texts. A major part of the development is the pre-processing of input data. Most effort and time is spent on transforming text into other objects (lists of tokens) in order to be handled by NLP algorithms. This is where Consileon’s NLP Framework comes into play. 

Consileon NLP Framework contains packages that simplify the development of NLP models through modularization and encapsulation of frequent pre-processing tasks. In that way, you avoid repeating yourself or ending up with a bulk of unstructured sample code that you might not understand or be able to explain later on. Focus on your concept and leave the implementation on us.  

## Features: 
Consileon NLP Framework offers all preprocessing tasks you need to develop your own NLP Model: 

- **Split** texts into smaller chunks (sentences, paragraphs) 
- Split chunks of text into **tokens** (e.g. single words) 
- Bring tokens into a canonical form (**lower-casing**) 
- **Filter** out unwanted tokens and **remove stop words**. 
- "**Lemmatization**":  map words to their base/dictionary form (imported also for many non-english languages) 
- **Perform** (other kinds of) **mappings** to tokens 
- **Remove "garbage"**, i.e. artifacts which are contained in the source but don’t add meaning to the use case at hand (e.g. remove tables of numbers from texts when spoken language is required) 
- **Append** tags to tokens (e.g. specify the source or some semantic information) 
- **Choose subsets** of the input sequence for development (or other) reasons 
- **Merge** several data sources. 

and **many more**. 

All these transformation steps can be pipelined in few coding lines and fed into NLP-algorithms to generate your NLP model.   



## Getting started: 
The following tutorial will walk you through developing your own NLP-Model using Consileon’s NLP Framework:  
See [getting_startet.ipynb](examples/notebooks/getting_started.ipynb)








