# Natural Language Processing 

This directory contains the recipes to create different NLP systems:

* Named Entity Recognition (NER)

* Part of Speech Tagging (POS) 

* Natural language dependency parser

* Semantic analysis 

* Information retrieval (IR) 

* Text classification

* Word embeddings

* N-Gram language models

* Chatbots

### Requirements 

* Create a conda environment for your NLP project: `conda env create -f nlp.yml`

* To activate this environment, use: `conda activate nlp_env`

* To deactivate an active environment, use: `conda deactivate`

* Spacy installation: `conda install -c conda-forge spacy`

### Data and Pre-trained Models

* Spacy English language model installation: `python -m spacy download en`

* Download Spacy Medium English Models: `python -m spacy download en_core_web_md`

* Download Spacy Large English Models: `python -m spacy download en_core_web_lg`

* Download Brown corpus: (preprocessed wiki data): https://lazyprogrammer.me/course_files/enwiki-preprocessed.zip

* Access Brown corpora through this link: https://www.nltk.org/data.html
  * Run the Python interpreter and type the commands:
        * import nltk
        * nltk.download()
            * Then from the menu, select Corpora tab > select Brown corpus > install
        * from nltk.corpus import brown 
        * brown.sents() is a list of lists with strings
        * to use is call the brown.py

* Stanford Movie Review data for Sentiment analysis gathered using Amazon Mechanical Turk. Movie ratings are within range of `[1:5]` where 1:`most negative`, 3: `netural` and 5: `most positive`:
    * This data can be parsed into a binary tree object where each node has either no child or 2 children 
    * Download file: trainDevTestTrees_PTB.zip from this link: http://nlp.stanford.edu/sentiment/
    
* Download CoNLL-2000 data:the data can be used to train systems for the part of speech (POS) tagging (POS)
    * Download the train.txt and test.txt from the link
        * http://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz
        * http://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz
        * https://www.clips.uantwerpen.be/conll2000/chunking/

* Download Twitter data for named entity recognition (NER)
    * https://github.com/aritter/twitter_nlp/blob/master/data/annotated/ner.txt
    
* To use these codes you may need to update your version of future `sudo pip install -U future`

* Optional if you are interested to use wikipedia data
    *  Wiki data is in XML format
    *  Link to download 1 wiki file : https://dumps.wikimedia.org/enwiki/20200520/enwiki-20200520-pages-articles-multistream1.xml-p1p30303.bz2
    *  Link to download more wiki files: https://dumps.wikimedia.org/backup-index.html and https://dumps.wikimedia.org/enwiki/
    *  Link to Wikipedia dump file to text converter: https://github.com/yohasebe/wp2txt
    *  to use is call the wiki.py


* Access pre-trained GloVe vectors: `https://nlp.stanford.edu/projects/glove/` Direct link: `http://nlp.stanford.edu/data/glove.6B.zip`
* Access pre-trained Word2Vec vectors:  (warning: takes quite awhile) `https://code.google.com/archive/p/word2vec/`
* Access pre-trained Word2Vec vectors (direct link): `https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing`

* Access Reuters train and test sets: 
    * https://www.cs.umb.edu/~smimarog/textmining/datasets/r8-train-all-terms.txt 
    * https://www.cs.umb.edu/~smimarog/textmining/datasets/r8-test-all-terms.txt
`pip install gensim`


### Spacy package and Spacy English language library install

Spacy will be used for Tokenization, POS, Stemming, Lemmatization. Here is a comparison between Spacy vs NLTK capabilities and speed. 

* https://spacy.io/usage/facts-figures
* https://spacy.io/usage/spacy-101#pipelines

### Github NLP recipes

* Google: https://github.com/google-research
* Huggingface NLP: https://github.com/huggingface/nlp
* Hugging face Transformers: https://github.com/huggingface/transformers
* Twitter: https://github.com/aritter/twitter_nlp
* Microsoft: https://github.com/microsoft/nlp-recipes
* Salesforce: https://github.com/salesforce/decaNLP
* Stanford: https://stanfordnlp.github.io/CoreNLP/
* Harvard: https://nlp.seas.harvard.edu/code/
* Amazon Web Service recipes: 
    * https://github.com/aws-samples/aws-nlp-workshop`
    * https://aws.amazon.com/comprehend/
    * https://aws.amazon.com/blogs/machine-learning/train-albert-for-natural-language-processing-with-tensorflow-on-amazon-sagemaker/
    * https://aws.amazon.com/blogs/industries/how-to-process-medical-text-in-multiple-languages-using-amazon-translate-and-amazon-comprehend-medical/

### Requirements
* Practical introduction to deep learning: https://courses.d2l.ai/berkeley-stat-157/syllabus.html
* Dive into deep learning: http://d2l.ai/
* Stanford NLP CS224N: http://web.stanford.edu/class/cs224n/index.html#schedule
* Harvard NLP: https://harvard-ml-courses.github.io/cs287-web/

### Readings
* GloVe: Global Vectors for Word Representation https://nlp.stanford.edu/pubs/glove.pdf

* Neural Word Embedding as Implicit Matrix Factorization  http://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf

* Hierarchical Softmax http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf

* More about Hierarchical Softmax http://papers.nips.cc/paper/3583-a-scalable-hierarchical-distributed-language-model.pdf

* Distributed Representations of Words and Phrases and their Compositionality https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

### FAQ on Spacy
* To learn about Spacy API visit: https://spacy.io/api

* For a full list of POS Tags visit: https://spacy.io/api/annotation#pos-tagging

* For a full list of Syntactic Dependencies: visit https://spacy.io/api/annotation#dependency-parsing 

* A good explanation of typed dependencies can be found: https://nlp.stanford.edu/software/dependencies_manual.pdf

* For more info on named entities visit: https://spacy.io/usage/linguistic-features#named-entities

* For more info on noun_chunks visit: https://spacy.io/usage/linguistic-features#noun-chunks

* For more info on Spacy built-in visualizer visit: https://spacy.io/usage/visualizers

* For more info on Spacy built-in models visit: https://spacy.io/usage/models




