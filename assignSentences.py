"""
This is a script to assign one sentence for each pair of entity
in an abstract and extract feature vectors for each pair
feature vector:
[
shortest dependency path - raw
  shortest dependency path - POS
  distance to main verb/root?
]

@author Peace Han
"""
import pprint
import time
import numpy as np

import regex as re
import xml.etree.ElementTree as ET
import replaceEntities
from shortestDependencyPath import sdp, get_stanford_annotations


class Text:
    """
    Class to store information for each document
    id: the text ID for this <text>
    abstract: original abstract text (cleaned using replaceEntities.encode())
    sents: list of sentences (tokenized), split using StanfordCoreNLP's ssplit
    entities: a dictionary of entityID, entityText (obtained using replaceEntities.encode())
    """
    def __init__(self, text_id, text_abstract, text_entities):
        self.id = text_id
        self.abstract = text_abstract
        self.entities = text_entities
        self.annotations = get_stanford_annotations(text_abstract)

    def __str__(self):
        return "Text {}: {}\n\t{}\n\thas {} sentences and {} entities".format(self.id,
                                                                              self.title,
                                                                              self.abstract,
                                                                              len(self.sents),
                                                                              len(self.entities))


class Pair:
    """
    Class to store the pairs of entities to classify
    ent1, ent2: The entities related in this Pair
    relation: The relation between ent1 and ent2
    rev: Flag, if this relation is REVERSE
    text_id: the Text that this Pair comes from
    sentence: the sentence assigned to this Pair
    dep_text: the shortest dependency path for this pair, raw text
    dep_pos: the shortest dependency path for this pair, POS tags
    """
    def __init__(self, ent1, ent2, relation, rev=False):
        self.ent1 = ent1
        self.ent2 = ent2
        self.relation = relation
        self.rev = rev
        assert ent1.split('.')[0] == ent2.split('.')[0]  # entities should occur in the same Text
        self.text_id = ent1.split('.')[0]
        # these are assigned later
        self.sentence = ''
        self.dep_text = ''
        self.dep_pos = ''

    def __str__(self):
        return "Pair: " \
               "relation: {}; " \
               "{}, {}, REVERSE={}".format(self.relation,
                                           self.ent1,
                                           self.ent2,
                                           self.rev)


def get_texts(filename):
    """
    Takes text.xml data and converts to list of Text objects
    :param filename: XML file containing raw shared task data
    :param annotator: the nlp annotation object to annotate texts
    :return: dictionary of textID, Text object extracted from file
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    textCount = 0
    entCount = 0
    texts = {}  # dictionary of textID, Text to return
    for text in root.findall('text'):
        textCount += 1
        textID = text.attrib.get('id')
        # print(textID)
        abstract = text.find('abstract')
        abstract_string = ET.tostring(abstract)  # this includes the <abstract> tags
        abstract_string = abstract_string.decode('UTF-8').strip()
        abstract_string = ' '.join(abstract_string.split('\n'))
        abstract_string = ' '.join(re.split('  +', abstract_string))
        abstract_string, entities = replaceEntities.encode(abstract_string)
        t = Text(textID, abstract_string, entities)
        texts[textID] = t
    return texts


path = 'clean/train_data/'
file = path + '1.1.text.xml'
# file = path + 'test_texts.xml'
print('Collecting abstract texts...')
start_time = time.time()
texts = get_texts(file)
end_time = np.round(time.time() - start_time, 2)
print("DONE! ({} seconds)".format(end_time))
# print(texts)
# print(texts['H01-1001'].annotations)


def get_pairs(filename):
    """
    Takes relations.txt data to create a list of Pairs of entities to be classified
    :param filename: the txt file containing relation information
    :return: list of Pairs to be classified
    """
    rels = []  # list of Pairs
    f = open(filename)
    for line in f:
        line = line.strip()
        line = line.split('(')
        rel = line[0]
        ents = line[1][:-1].split(',')
        ent1 = ents[0]
        ent2 = ents[1]
        pair = Pair(ent1, ent2, rel, len(ents) == 3)
        pair.relation = rel
        # print(pair)
        rels.append(pair)
    return rels


file = path + '1.1.relations.txt'
print("Collecting relations and generating Pairs...")
start_time = time.time()
rels = get_pairs(file)
end_time = np.round(time.time() - start_time, 2)
print("DONE! ({} seconds)".format(end_time))
# print(rels)
# print(rels[0])


def assign_sentences(relations, text_ids):
    """
    Assign a single sentence to every Pair in the given relations list
    Also assign the dependency paths to each Pair
    :param relations: a list of Pairs to process
    :param text_ids: a dictionary of textID, Text object
    :return: N/A
    """
    for pair in relations:
        if pair.text_id in text_ids.keys():
            text_obj = text_ids[pair.text_id]
            sentences = text_obj.annotations['sentences']
            # print(pair)
        # print(sentences)
        #     [print(x) for x in sentences]
            for sent in sentences:
                # construct the sentence from the tokens first
                tokens = sent['tokens']
                sent_toks = []
                for i in range(len(tokens)):
                    word = tokens[i]['word']
                    # print(word)
                    sent_toks.append(word)
                # print(sent_toks)
                if (pair.ent1 in sent_toks)\
                        and (pair.ent2 in sent_toks):
                    sent_toks = ' '.join(sent_toks)
                    pair.sentence = sent_toks
                    # print('\t', pair.sentence)

                    # generating the sdps
                    pair.dep_text, pair.dep_pos = sdp(pair.ent1,
                                                      pair.ent2,
                                                      sent)


print("Generating SDPs for each Pair...")
start_time = time.time()
assign_sentences(rels, texts)
end_time = np.round(time.time() - start_time, 2)
print("DONE! ({} seconds)".format(end_time))


# writing the features file
outfile = path + 'data_2.0/1.1.features.txt'
# entfile = path + 'data_2.0/1.1.text_ents.csv'
o = open(outfile, 'w')
# p = open(entfile, 'w')
print("Writing features to {}...".format(outfile))
for pair in rels:
    # print(pair)
    o.write(pair.relation + ' ')
    # print(pair.sentence)
    pair.dep_text = replaceEntities.decode(pair.dep_text, texts[pair.text_id].entities)
    # print(pair.dep_text)
    o.write(pair.dep_text)
    if pair.rev:
        o.write(' ' + 'REVERSE')
    o.write('\n')
o.close()

