'''
Created on 02.01.2019

@author: AT
'''
import spacy
from spacy import displacy

def getCommands(description, inventory):
    """ Requires current description of the game (String) and inventory (String).
    Returns: list of strings with commands for current gamestate."""
    
    nlp = spacy.load('en_core_web_sm')
    desc = nlp(description)
    inv = nlp(inventory)
    
    chunks = desc.noun_chunks
    print(description)
    for chunk in desc.noun_chunks:
        if isRelevantChunk(chunk):
            print(chunk.text, chunk.root.text, chunk.root.dep_,
                chunk.root.head.text, chunk.root.pos_, chunk.root.tag_)
    
def isRelevantChunk(chunk):
    if chunk.root.pos_ == "PRON" or chunk.root.tag_ == "WP":
        
        return False
    
    return True
    

if __name__ == '__main__':
    
    ### Load spaCy's English NLP model
    #nlp = spacy.load('en')
    nlp = spacy.load('en_core_web_sm')
    
    ### The text we want to examine
    text = "portal"
    doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
    ### Parse the text with spaCy
    ### Our 'document' variable now contains a parsed version of text.
    document = nlp(u'You see a formless keycard on the table.')
    
    for chunk in document.noun_chunks:
        print(chunk.text, chunk.root.text, chunk.root.dep_,
          chunk.root.head.text)
    ### print out all the named entities that were detected
    for token in document:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
    displacy.serve(document, style='dep',port=5004  )
        
    