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
    
    takeCommands, containersWithPrep, thingstakenCareOf = identifyInOnRelationships(desc)
    
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

def identifyInOnRelationships(desc):
    """
    desc: spacy.load(description)
    returns:
    """
    takeFromCommands = []
    containersOrPlacesWithPreposition = []
    objectsTakenCareOf = []
    for token in desc:
        if str.lower(token.text) == "in" or str.lower(token.text) == "on":
            child = next(token.children)
            if child.pos_ == "NOUN" and not(str.lower(child.text) == "floor" or str.lower(child.text) == "ground"):
                takeCommandGenerated = False
                for ancestor in token.ancestors:
                    if ancestor.pos_ == "NOUN" or ancestor.pos_ == "PROPN":
                        takeFromCommands.append("take " + ancestor.text + " from " + child.text)
                        objectsTakenCareOf.append(ancestor.text)
                        takeCommandGenerated = True
                    else:
                        for cousin in ancestor.children:
                            if cousin.pos_ == "NOUN" or cousin.pos_ == "PROPN":
                                takeFromCommands.append("take " + cousin.text + " from " + child.text)
                                objectsTakenCareOf.append(cousin.text)
                                takeCommandGenerated = True
                if takeCommandGenerated:
                    containersOrPlacesWithPreposition.append(token.text + " " + child.text)
                    objectsTakenCareOf.append(child.text)
    print(takeFromCommands)
    print(containersOrPlacesWithPreposition)
    print(objectsTakenCareOf)
    return takeFromCommands, containersOrPlacesWithPreposition, objectsTakenCareOf                 
    

if __name__ == '__main__':
    
    ### Load spaCy's English NLP model
    #nlp = spacy.load('en')
    nlp = spacy.load('en_core_web_sm')
    
    ### The text we want to examine
    text = "portal"
    doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')
    ### Parse the text with spaCy
    ### Our 'document' variable now contains a parsed version of text.
    document = nlp(u"On the mantelpiece you make out a loaf of bread.")
    
    identifyInOnRelationships(document)
    
    for chunk in document.noun_chunks:
        print(chunk.text, chunk.root.text, chunk.root.dep_,
          chunk.root.head.text)
    ### print out all the named entities that were detected
    for token in document:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,token.dep)
    displacy.serve(document, style='dep',port=5000  )
    
        
    