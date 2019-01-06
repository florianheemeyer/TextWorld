'''
Created on 02.01.2019

@author: AT
'''
import spacy
from spacy import displacy
from nltk.corpus import wordnet as wn

debug = False

def getCommands(description, inventory):
    """ Requires current description of the game (String) and inventory (String).
    Returns: list of strings with commands for current gamestate."""
    
    nlp = spacy.load('en_core_web_sm')
    desc = nlp(description)
    inv = [removePreposition(x).strip() for x in inventory.split("\n")]
    
    takeCommands, containersWithPrep, thingstakenCareOf = identifyInOnRelationships(desc)

    if debug:
        print(description)
        print(inventory)

    #for chunk in desc.noun_chunks:
    #    if isRelevantChunk(chunk):
    #        relevantNouns.add(chunk.root.text)
    #       if debug:
    #            print(chunk.text, chunk.root.text, chunk.root.dep_,
    #                chunk.root.head.text, chunk.root.pos_, chunk.root.tag_)
    #            print(WordAttributeInference.factory(chunk.root.text))

    relevantNouns = set()
    for word in desc:
        if word.pos_ == "NOUN" or "safe" in word.text.lower() or "portal" in word.text.lower():
            relevantNouns.add(word.text)

    containers = set()
    keys = set()
    takeables = set()
    lockables = set()
    openables = set()
    edibles = set()
    for noun in relevantNouns:
        attributes = WordAttributeInference.factory(noun)
        if attributes.container:
            containers.add(noun)
        if attributes.takeable:
            takeables.add(noun)
        if attributes.lockable:
            lockables.add(noun)
        if attributes.openable:
            openables.add(noun)

    for item in inv:
        attributes = WordAttributeInference.factory(item)
        if attributes.key:
            keys.add(item)
        if attributes.edible:
            edibles.add(item)

    commands = []

    for item in inv:
        commands.append("drop " + item)

    commands += takeCommands

    for openable in openables:
        commands.append("open " + openable)
        commands.append("close " + openable)

    for lockable in lockables:
        for key in keys:
            commands.append("lock " + lockable + " with " + key)
            commands.append("unlock " + lockable + " with " + key)

    for container in containers:
        for item in inv:
            commands.append("insert " + item + " into " + container)
            commands.append("put " + item + " on " + container)

    for takeable in takeables:
        commands.append("take " + takeable)

    for edible in edibles:
        commands.append("eat " + edible)

    commands += ["go east", "go west", "go south", "go north"]

    if debug:
        print(str(len(commands)) + " commands generated")
        print(commands)
    return commands

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
            if len(list(token.children)) == 0:
                continue
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
        elif str.lower(token.text) == "contains":
            children = token.children
            child1 = next(children)
            child2 = next(children)
            takeFromCommands.append("take " + child1.text + " from " + child2.text)
            takeFromCommands.append("take " + child2.text + " from " + child1.text)

    if debug:
        print(takeFromCommands)
        print(containersOrPlacesWithPreposition)
        print(objectsTakenCareOf)
    return takeFromCommands, containersOrPlacesWithPreposition, objectsTakenCareOf


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
        self.edible = self.isEdible()

    def __str__(self):
        return self.word + " Container: " + str(self.container) + " Key: " + str(self.key) + \
               " Takeable: " + str(self.takeable) + " Lockable: " + str(self.lockable) + " Openable: " + str(self.openable)

    def isContainer(self):
        return self.soundsLikeContainer() or self.soundsLikeFurniture() or self.soundsLikeKitchenUtensil()

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

    def isEdible(self):
        return self.soundsEdible()

    def soundsLikeKey(self):
        return "key" in self.word or wn.synset("key.n.01") in self.hypernyms

    def soundsLikeRoom(self):
        return "room" in self.word or wn.synset("room.n.01") in self.hypernyms

    def soundsLikeDoor(self):
        return "door" in self.word or wn.synset("barrier.n.01") in self.hypernyms or wn.synset("entrance.n.01") in self.hypernyms or wn.synset("passage.n.03") in self.hypernyms

    def soundsLikeContainer(self):
        return wn.synset("container.n.01") in self.hypernyms

    def soundsLikeFurniture(self):
        return wn.synset("furniture.n.01") in self.hypernyms or wn.synset("shelf.n.01") in self.hypernyms or wn.synset("white_goods.n.01") in self.hypernyms

    def soundsLikeKitchenUtensil(self):
        return wn.synset("kitchen_utensil.n.01") in self.hypernyms

    def soundsEdible(self):
        return wn.synset("food.n.01") in self.hypernyms or wn.synset("food.n.02") in self.hypernyms

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
    
        
    
