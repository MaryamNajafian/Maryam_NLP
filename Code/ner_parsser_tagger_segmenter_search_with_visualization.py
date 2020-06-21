from __future__ import barry_as_FLUFL

__version__ = '0.1'
__author__ = 'Maryam Najafian'

"""
This page contains examples of using Spacy for the following applications:

1- Tokenization, Stemming, Lemitzation, stop-word removal
2- vocabulary, and phrase matching
3- Part of speech (POS) tagging and POS visualization
4- Named Entity Recognition (NER) and NER visualization
"""

def main():
    import nltk
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.snowball import SnowballStemmer
    import spacy
    from spacy import displacy
    from spacy.matcher import Matcher
    from spacy.matcher import PhraseMatcher
    from spacy.tokens import Span
    from spacy.pipeline import SentenceSegmenter

    import config

    nlp = spacy.load('en_core_web_sm')

    #%%
    print(nlp.pipeline)
    print(nlp.pipe_names)

    #%%
    print("Data string examples \n")
    mystring = '"As of last quarter autonomous cars have shifted insurance liability toward manufacturers. ' \
               'There\'s a car factory in LA! About 5km away. ' \
               'Here is the Apple snail-mail: support@outside.com or visit http://www.oursite.com."'

    mystring2 = 'I am a runner running in a race because I love to run since I ran today.'
    words = ['run','ran','runner','runs','fairly','fairness','generous','generously', 'generate', 'generation']

    #%%
    print("Print each word in the string with it's corresponding POS, dependency:")
    doc1 = nlp(mystring)
    print("The vocab size for our small lang. lib. is: ", len(doc1.vocab) )
    for token in doc1:
        print(token.text, token.pos, token.pos_, token.dep_)

    #%%

    print("Print the named entities:")
    for token in doc1.ents:
        print(f"{token.text} {10*'.'}\t {token.label_} {5*'.'}\t {spacy.explain(token.label_)}\n")


    #%%
    print("A function to display basic entity info.")
    def show_ents(doc):
        if doc.ents:
            for ent in doc.ents:
                print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
        else:
            print('\n No named entities found. \n')

    doc4 = nlp(u'May I go to Washington, DC next May to see the Washington Monument and buy Tesla stocks? The flight ticket is only 500 dollars.')
    doc5= nlp(u'Hi, Hope you are well.')
    show_ents(doc4)
    show_ents(doc5)

    #%%
    print("Adding a single term as an NER")
    from spacy.tokens import Span

    doc = nlp(u'Tesla to build a U.K. factory for $6 million')

    # Get the hash value of the ORG entity label
    ORG = doc.vocab.strings[u'ORG']
    print(ORG)
    # Create a Span for the new entity
    # doc: Name of document object
    # 0: start position of the span,
    # 1: stop position of the span (exclusive: not including 1)
    # Label: ORG is the label assigned to the entity
    new_ent = Span(doc, 0, 1, label=ORG)

    # Add the entity to the existing Doc object
    doc.ents = list(doc.ents) + [new_ent]

    show_ents(doc)

    #%%
    print("Adding multiple phrases as NERs")

    doc = nlp(u'Our company plans to introduce a new vacuum cleaner. '
              u'If successful, the vacuum cleaner will be our first product.')
    show_ents(doc)

    # Import PhraseMatcher and create a matcher object:
    from spacy.matcher import PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab)

    # Create the desired phrase patterns:
    phrase_list = ['vacuum cleaner', 'vacuum-cleaner']
    phrase_patterns = [nlp(text) for text in phrase_list]


    # Apply the patterns to our matcher object:
    matcher.add('newproduct', None, *phrase_patterns)

    # Apply the matcher to our Doc object:
    found_matches = matcher(doc)

    # See what matches occur:
    print(found_matches)


    # Here we create Spans from each match, and create named entities from them:
    from spacy.tokens import Span

    PROD = doc.vocab.strings[u'PRODUCT']

    new_ents = [Span(doc, match[1],match[2],label=PROD) for match in found_matches]

    doc.ents = list(doc.ents) + new_ents

    show_ents(doc)
    #%%
    print("Counting Named Entities occurrences")

    doc = nlp(u'Originally priced at $29.50, the sweater was marked down to five dollars.')

    show_ents(doc)

    len([ent for ent in doc.ents if ent.label_=='MONEY'])

    # For more on **Named Entity Recognition** visit https://spacy.io/usage/linguistic-features#101

    #%%
    print("Visualizing NER")
    doc = nlp(u'Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million. '
             u'By contrast, Sony sold only 7 thousand Walkman music players.')

    displacy.render(doc, style='ent', jupyter=True)
    displacy.serve(doc1,style='ent')

    print('Viewing Sentences Line by Line')
    for sent in doc.sents:
        displacy.render(nlp(sent.text), style='ent', jupyter=True)

    print("Viewing Specific Entities, and customizing the visualization")
    options = {'ents': ['ORG', 'PRODUCT']}

    colors = {'ORG': 'linear-gradient(90deg, #aa9cfc, #fc9ce7)', 'PRODUCT': 'radial-gradient(yellow, green)'}

    options = {'ents': ['ORG', 'PRODUCT'], 'colors':colors}

    print('display entities on jupiter notebook')
    displacy.render(doc, style='ent', jupyter=True, options=options)
    print('Display entities on browser: http://127.0.0.1:5000 ')
    displacy.serve(doc1,style='ent',options=options)
    # For more on applying CSS background colors and gradients, visit https://www.w3schools.com/css/css3_gradients.asp
    # https://spacy.io/usage/visualizers
    #%%
    print("Visualize entity recognizer with Spacy (line by line)")

    doc1 = nlp(mystring)
    spans = list(doc1.sents)

    # print('display entities on jupiter notebook')
    # displacy.render(spans,style='ent',jupyter=True,options={'distance':80})

    print('Display entities on browser: http://127.0.0.1:5000 ')
    displacy.serve(spans,style='ent',options=options)

    #%%
    print("Visualize entity recognizer with Spacy (whole paragraph)")

    # print('display entities on jupiter notebook')
    # displacy.render(doc1,style='ent',jupyter=True,options={'distance':80})

    print('Display entities on browser: http://127.0.0.1:5000 ')
    displacy.serve(doc1,style='ent',options=options)

    #%%

    print("List name Chunks:")
    for token in doc1.noun_chunks:
        print(token)
    # For more on **noun_chunks** visit https://spacy.io/usage/linguistic-features#noun-chunks
    #%%
    print("Dependency visualization with Spacy ")

    # style 'dep': shows pos tags and syntactic dependencies
    options ={'distance':80, 'compact':'True', 'color':'yellow', 'bg':'#09a3d5','font':'Times'}

    print('display dependencies on jupiter notebook')
    displacy.render(doc1,style='dep',jupyter=True,options=options)

    print('Display dependencies on browser: http://127.0.0.1:5000 ')
    displacy.serve(doc1,style='dep',options=options)

    #%%
    print("Spacy doesn't include a Stemmer. Instead it relies on lemmatization entirely. \n"
          "We use NLTK Porter and Snowball Stemmers here.")

    p_stemmer = PorterStemmer()

    for word in words:
        print(f"{word}, {10*'.'}, {p_stemmer.stem(word)}")

    s_stemmer = SnowballStemmer(language='english')
    for word in words:
        print(f"{word}, {10*'.'}, {s_stemmer.stem(word)}")

    #%%
    print("Perform Lemmatization with Spacy")
    text = nlp(mystring2)
    def show_lemmas(text):
        for token in text:
            print(f'{token.text:{12}} {token.pos_:{6}} {token.lemma:<{12}} {token.lemma_:{6}} {token.tag_:{6}} {spacy.explain(token.tag_)}' )

    show_lemmas(text)

    #%%
    print("Remove/Add stopwords with Spacy")
    print(nlp.Defaults.stop_words) # Print the List of Spacy stopwords
    len(nlp.Defaults.stop_words) # Number of default stopwords in Spacy
    nlp.vocab['is'].is_stop # Tells if the vocab is among Spacy stopwords or not
    nlp.vocab['mystery'].is_stop
    nlp.Defaults.stop_words.add('btw') # Adding to the Spacy's list of stopwords
    nlp.vocab['btw'].is_stop = True # set it to True
    nlp.Defaults.stop_words.remove('six') # Removing from the Spacy's list of stopwords
    nlp.vocab['six'].is_stop = False

    #%%
    print("RuleBased Vocabulary Matching.\n More powerful version of the regular expressions")

    # looking for 3 different forms of the same pattern here
    matcher = Matcher(nlp.vocab)
    # a single token whose lowercase text reads 'solarpower'
    pattern1 = [{'LOWER': 'solarpower'}]
    # two adjacent tokens that read 'solar' and 'power' in that order
    pattern2 = [{'LOWER': 'solar'}, {'LOWER': 'power'}]
    # three adjacent tokens, with a middle token that can be any punctuation
    pattern3 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}]
    # Option (OP) key '*' : allows pattern 0 or more times
    pattern4 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LOWER': 'power'}]


    # add patterns to matcher labeled 'SolarPowerMatcherName'
    matcher.add('SolarPowerMatcherName', None, pattern1, pattern2, pattern3,pattern4)

    doc = nlp(u'The Solar Power industry continues to grow as demand \
    for solarpower increases. Solar-power cars are gaining popularity as solar--power shows more strength')

    found_matches = matcher(doc)
    print(found_matches) # gives you tuples with match_id, start, and end index


    for match_id, start, end in found_matches:   # grabs raw matched-vocab with match_id, start, and end index
        string_id = nlp.vocab.strings[match_id]  # get string representation
        span = doc[start:end]                    # get the matched span
        print(match_id, string_id, start, end, span.text)

    # remove the patterns identified under 'SolarPowerMatcherName' label to avoid duplicates in next search
    matcher.remove('SolarPowerMatcherName')

    #%%
    print("RuleBased Phrase Matching.\n More powerful version of the regular expressions")
    matcher = PhraseMatcher(nlp.vocab)
    # if your file gave you utf8 file error run this on terminal:
    # iconv -f iso-8859-1 -t utf-8  original_file > new_file
    doc2_path= config.DATA_DIR+'reaganomics.txt'
    with open(doc2_path) as f:
        doc2 = nlp(f.read())

    # First, create a list of match phrases:
    phrase_list = ['voodoo economics', 'supply-side economics', 'trickle-down economics', 'free-market economics']

    # Next, convert each phrase to a Doc object:
    phrase_patterns = [nlp(text) for text in phrase_list]

    # Pass each Doc object into matcher (note the use of the asterisk!):
    matcher.add('VoodooEconomics', None, *phrase_patterns)

    # Build a list of matches:
    found_matches = matcher(doc2)

    for match_id, start, end in found_matches:   # grabs raw matched-vocab with match_id, start, and end index
        string_id = nlp.vocab.strings[match_id]  # get string representation
        span = doc2[start:end]                    # get the matched span
        print(match_id, string_id, start, end, span.text)
    #%%
    print("going through doc sentences")

    with open(config.DATA_DIR+'owlcreek.txt') as f:
        doc = nlp(f.read())


    sents = [sent for sent in doc.sents]
    len(sents)

    #%%
    print("sentense segmentation")

    '''
    It is important to note that `doc.sents`
    is a *generator*. That is, a Doc is not segmented 
    until `doc.sents` is called. This means that, 
    where you could print the second Doc token with
    `print(doc[1])`, you can't call the' 
     "second Doc sentence" with `print(doc.sents[1])`
     However, you *can* build a sentence collection by
      running `doc.sents` and saving the result to a list
    '''

    doc = nlp(u'This is the first sentence. This is another sentence. This is the last sentence.')

    for sent in doc.sents:
        print(sent)

    print(doc[1])
    type(list(doc.sents)[0]) # it is a span type not string
    # print(doc.sents[1]) gives you error, you should use the following instead
    print(list(doc.sents)[0])

    doc_sents = [sent for sent in doc.sents]

    # Now you can access individual sentences
    print(doc_sents[1])

    # At first glance it looks like each `sent` contains text from the original Doc object. In fact they're just Spans
    # with start and end token pointers.

    type(doc_sents[1])
    print(doc_sents[1].start, doc_sents[1].end)
    #%%
    print("Spacy's built-in `sentencizer` for sentense segmentation")
    """
    spaCy's built-in `sentencizer` relies on the dependency
     parse and end-of-sentence punctuation to determine 
     segmentation rules. We can add rules of our own, 
     but they have to be added *before* the creation of 
     the Doc object, as that is where the parsing of segment 
     start tokens happens
    
    """

    # Parsing the segmentation start tokens happens during the nlp pipeline
    doc2 = nlp(u'This is a sentence; This is a sentence. This is a sentence.')

    for token in doc2:
        print(token.is_sent_start, ' '+token.text)


    for sent in doc2.sents:
        print(sent)
    #%%
    print("ADD A NEW SEGMENTATION RULE TO THE PIPELINE-part2")
    def set_custom_boundaries(doc):
        for token in doc[:-1]:
            if token.text == ';':
                doc[token.i+1].is_sent_start = True
        return doc

    nlp.add_pipe(set_custom_boundaries, before='parser')

    print(nlp.pipe_names) # ['tagger', 'set_custom_boundaries', 'parser', 'ner']

    # Re-run the Doc object creation:
    doc4 = nlp(u'"Management is doing things right; leadership is doing the right things." -Peter Drucker')

    for sent in doc4.sents: # separates sentences on semicolon
        print(sent)

    # And yet the new rule doesn't apply to the older Doc object:
    for sent in doc2.sents:
        print(sent)
    #%%
    print("ADD CHANGE SEGMENTATION RULES TO THE PIPELINE-part2")

    """
    Why not simply set the `.is_sent_start` value to 
    True on existing tokens?
    
    In some cases we want to *replace* spaCy's default 
    sentencizer with our own set of rules. 
    In this section we'll see how the default 
    sentencizer breaks on periods. We'll then replace 
    this behavior with a sentencizer that breaks on linebreaks.
    
    """

    nlp = spacy.load('en_core_web_sm')  # reset to the original

    mystring = u"This is a sentence. This is another.\n\nThis is a \nthird sentence."

    # SPACY DEFAULT BEHAVIOR:
    doc = nlp(mystring)

    for sent in doc.sents:
        print([token.text for token in sent])

    def split_on_newlines(doc): #split on newlines instead of `.`
        start = 0
        seen_newline = False
        for word in doc:
            if seen_newline:
                yield doc[start:word.i] #word.i --> current word index position
                start = word.i
                seen_newline = False
            elif word.text.startswith('\n'): # handles multiple occurrences
                seen_newline = True
        yield doc[start:]      # handles the last group of tokens


    sbd = SentenceSegmenter(nlp.vocab, strategy=split_on_newlines)
    nlp.add_pipe(sbd)

    doc = nlp(mystring)
    for sent in doc.sents:
        print([token.text for token in sent])


    #%%
    print("Perform POS with Spacy")
    text = nlp(u"I read books on NLP.")
    text2 = nlp(u"I read a book on NLP.")
    word = text[1]
    print(f'{word} : {type(word)}')
    print(f'{word.text} : {type(word.text)}')
    def show_pos(text):
        for token in text:
            print(f'{token.text:{12}} {token.pos_:{6}} {token.tag_:{6}} {spacy.explain(token.tag_)}' )
    # the pos shows 'read' is past/present tense
    print('\n read (present tense)\n')
    show_pos(text)
    print(f'\n read (past tense)\n')
    show_pos(text2)
    #%%
    print("Count different coarse-grained POS codes\n")
    doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")
    POS_counts = doc.count_by(spacy.attrs.POS)
    print('POS_counts:',POS_counts)
    print('Associated `item` for the POS `key #`: ', doc.vocab[83].text)

    print('Creat frequency list of POS tags since `POS_counts` returns a dictionary  with `POS_counts.items()\n')

    for k,v in sorted(POS_counts.items()):
        print(f'{k}. {doc.vocab[k].text:{5}}: {v}')
    #%%
    print("Count different coarse-grained Tag codes\n")

    TAG_counts = doc.count_by(spacy.attrs.TAG)

    for k,v in sorted(TAG_counts.items()):
        print(f'{k}. {doc.vocab[k].text:{4}}: {v}')
    #%%
    print('Count the different dependencies (DEP) codes\n')

    DEP_counts = doc.count_by(spacy.attrs.DEP)

    for k,v in sorted(DEP_counts.items()):
        print(f'{k}. {doc.vocab[k].text:{4}}: {v}')

#%%
main()
#%%
if __name__ == "__main__":
    print(__doc__)
    main()
