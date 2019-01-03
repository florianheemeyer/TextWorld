'''
Created on 02.01.2019

@author: AT
'''
import spacy
from spacy import displacy
from nltk.corpus import wordnet as wn

def getCommands(description, inventory):
    """ Requires current description of the game (String) and inventory (String).
    Returns: list of strings with commands for current gamestate."""
    
    nlp = spacy.load('en_core_web_sm')
    desc = nlp(description)
    inv = [removePreposition(x).strip() for x in inventory.split("\n")]
    
    chunks = desc.noun_chunks
    print(description)
    for chunk in desc.noun_chunks:
        if isRelevantChunk(chunk):
            print(chunk.text, chunk.root.text, chunk.root.dep_,
                chunk.root.head.text, chunk.root.pos_, chunk.root.tag_)
            print(WordAttributeInference.factory(chunk.root.text))
    
def isRelevantChunk(chunk):
    if chunk.root.pos_ == "PRON" or chunk.root.tag_ == "WP":
        
        return False
    
    return True

def removePreposition(text):
    return removePrefix(removePrefix(removePrefix(text.strip(), "the "), "a "), "an ")

def removePrefix(string, prefix):
    if string.startswith(prefix):
        return string[len(prefix):]
    return string

class WordAttributeInference():
    cache = {}

    def factory(word):
        if word in WordAttributeInference.cache:
            return WordAttributeInference.cache[word]
        else:
            object = WordAttributeInference(word)
            WordAttributeInference.cache[word] = object
            return object

    def __init__(self, word):
        self.word = word
        self.hypernyms = set()

        synsets = wn.synsets(word)
        self.hypernyms.update(synsets)

        for synset in synsets:
            hypernym_paths = synset.hypernym_paths()
            for hypernym_path in hypernym_paths:
                self.hypernyms.update(hypernym_path)

        #cache attributes
        self.container = self.isContainer()
        self.key = self.isKey()
        self.takeable = self.isTakeable()
        self.lockable = self.isLockable()
        self.openable = self.isOpenable()

    def __str__(self):
        return self.word + " Container: " + str(self.container) + " Key: " + str(self.key) + \
               " Takeable: " + str(self.takeable) + " Lockable: " + str(self.lockable) + " Openable: " + str(self.openable)

    def isContainer(self):
        return self.soundsLikeContainer() or self.soundsLikeFurniture()

    def isKey(self):
        return self.soundsLikeKey()

    def isTakeable(self):
        if not (self.soundsLikeContainer() or self.soundsLikeRoom() or self.soundsLikeDoor()):
            return True
        if self.soundsLikeKey():
            return True
        return False

    def isLockable(self):
        return self.soundsLikeContainer() or self.soundsLikeDoor()

    def isOpenable(self):
        return self.soundsLikeContainer() or self.soundsLikeDoor() or self.soundsLikeFurniture()

    def soundsLikeKey(self):
        return "key" in self.word or wn.synset("key.n.01") in self.hypernyms

    def soundsLikeRoom(self):
        return "room" in self.word or wn.synset("room.n.01") in self.hypernyms

    def soundsLikeDoor(self):
        return "door" in self.word or wn.synset("barrier.n.01") in self.hypernyms or wn.synset("entrance.n.01") in self.hypernyms

    def soundsLikeContainer(self):
        return wn.synset("container.n.01") in self.hypernyms

    def soundsLikeFurniture(self):
        return wn.synset("furniture.n.01") in self.hypernyms

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
        
    