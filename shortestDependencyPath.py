"""
This connects to the StanfordCoreNLP server and allows NLP annotations
and Shortest Dependency Path (SDP) calculations.
To be used in assignSentences.py

@author: Peace Han
"""

import networkx as nx
from pycorenlp import StanfordCoreNLP
from pprint import pprint


def get_stanford_annotations(text, port=9000,
                             annotators='tokenize,ssplit,pos,lemma,depparse'):
    """
    get the appropriate stanfordCoreNLP annotations for this text
    :param text: The text to annotate
    :param port: The port at which StanfordCoreNLP is active (be sure to run from terminal)
    :param annotators: list of annotations to include (Default: 'tokenize,ssplit,pos,lemma,depparse,parse')
    :return: the annotations for the text
    """
    output = nlp.annotate(text, properties={
        "timeout": "100000",
        'annotators': annotators,
        'outputFormat': 'json'
    })
    return output


def sdp(e1, e2, anno_sent):
    """
    returns the shortest dependency paths for the given entities in the given annotated sentence
    :param e1: the first entity
    :param e2: the second entity
    :param anno_sent: sentence object annotated with pycorenlp
    :return: (sdp_text, sdp_pos)
    """
    # print("Generating sdp for {}, {}".format(e1, e2))
    # print(pair)
    # annotations = get_stanford_annotations(sentence)  # get the full NLP annotations for this text
    # pprint(anno_sent)
    tokens = anno_sent['tokens']
    # print(tokens)

    edges = []
    deps = {}
    # pprint(annotations['sentences'][0]['basicDependencies'])
    for e in anno_sent['basicDependencies']:
        edges.append((e['governor'], e['dependent']))
        deps[(e['governor'], e['dependent'])] = e
        # deps[(min(e['governor'], e['dependent']),
        #       max(e['governor'], e['dependent']))] = e
    graph = nx.Graph(edges)
    # pprint(deps)

    for tok in tokens:
        if e1 == tok['originalText']:
            ent1_index = tok['index']
        if e2 == tok['originalText']:
            ent2_index = tok['index']

    path = nx.shortest_path(graph, source=ent1_index, target=ent2_index)
    # print('path: {}'.format(path))

    sdp_text = []  # the text sdp to be returned
    sdp_pos = []  # the POS sdp to be returned
    for i in range(len(path)):
        j = i+1
        tok_id1 = path[i]  # index of the first token in path
        token = tokens[tok_id1 - 1]  # tokens are offset by 1 to account for sentence ROOT
        token_text = token['originalText']  # the text of the token
        token_pos = token['pos']  # the POS of the token
        if j < len(path):
            tok_id2 = path[j]  # index of the next token in path
            tup = (tok_id1, tok_id2)  # dependency pairs are represented as (index_of_governor, index_of_dependent)
            arr = '-->'
            if tup not in deps:
                tup = (tok_id2, tok_id1)  # if not found, search for reverse
                arr = '<--'
            dep_tag = deps[tup]['dep']
            # print("Node {}\ttoken_text: {}\n{}{}{}".format(tok_id1, token_text, arr, tag, arr))
            sdp_text.append("{} {} {} {}".format(token_text, arr, dep_tag, arr))
            sdp_pos.append("{} {} {} {}".format(token_pos, arr, dep_tag, arr))
        else:
            sdp_text.append(token_text)
            sdp_pos.append(token_pos)
            # print("Node {}\ttoken_text: {}".format(tok_id1, token_text))

    return ' '.join(sdp_text), ' '.join(sdp_pos)


nlp = StanfordCoreNLP('http://localhost:{0}'.format(9000))

