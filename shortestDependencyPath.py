import networkx as nx
from pycorenlp import StanfordCoreNLP
from pprint import pprint


# nlp = StanfordCoreNLP(r'/home/peace/CoreNLP/stanford-corenlp-full-2018-10-05/')
nlp = StanfordCoreNLP('http://localhost:{0}'.format(9000))


def get_stanford_annotations(text, port=9000,
                             annotators='tokenize,ssplit,pos,lemma,depparse,parse'):
    """
    get the appropriate stanfordCoreNLP annotations for this text
    :param text: The text to annotate
    :param port: The port at which StanfordCoreNLP is active (be sure to run from terminal)
    :param annotators: list of annotations to include
    :return: the annotations for the text
    """
    output = nlp.annotate(text, properties={
        "timeout": "100000",
        'annotators': annotators,
        'outputFormat': 'json'
    })
    return output


def sdp(pair):
    sentence = ' '.join(pair.sentence)
    e1 = pair.ent1
    e2 = pair.ent2
    print("Generating sdp for {}, {} in {}".format(e1, e2, sentence))
    # print(pair)
    # print(pair.sentence)
    annotations = get_stanford_annotations(sentence)
    pprint(annotations)
    tokens = annotations['sentences'][0]['tokens']
    # print(tokens)

    edges = []
    deps = {}
    # pprint(annotations['sentences'][0]['basicDependencies'])
    for e in annotations['sentences'][0]['basicDependencies']:
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

    outString = []
    for i in range(len(path)):
        j = i+1
        tok_id1 = path[i]
        token = tokens[tok_id1 - 1]
        token_text = token['originalText']
        if j < len(path):
            tok_id2 = path[j]
            tup = (tok_id1, tok_id2)
            arr = '-->'
            if tup not in deps:
                tup = (tok_id2, tok_id1)
                arr = '<--'
            tag = deps[tup]['dep']
            # print("Node {}\ttoken_text: {}\n{}{}{}".format(tok_id1, token_text, arr, tag, arr))
            outString.append("{} {} {} {}".format(token_text, arr, tag, arr))
        else:
            outString.append(token_text)
            # print("Node {}\ttoken_text: {}".format(tok_id1, token_text))

    return ' '.join(outString)


class Pair:
    def __init__(self, ent1, ent2, relation, rev=False):
        self.ent1 = ent1
        self.ent2 = ent2
        self.relation = relation
        self.rev = rev
        assert ent1.split('.')[0] == ent2.split('.')[0]
        self.text_id = ent1.split('.')[0]
        self.sentence = ''
        self.dep_parse = ''

    def __str__(self):
        return "Pair: " \
               "relation: {}; " \
               "{}, {}, REVERSE={}".format(self.relation,
                                           self.ent1,
                                           self.ent2,
                                           self.rev)


def main():
    ent1 = 'H01-1001.5'
    ent2 = 'H01-1001.7'
    rel = 'USAGE'
    rev = True
    myPair = Pair(ent1, ent2, rel, rev)
    myPair.sentence = ['Traditional', 'H01-1001.5', 'use', 'a', 'H01-1001.6', 'of', 'H01-1001.7', 'as', 'the', 'H01-1001.8', 'but', 'H01-1001.9', 'may', 'offer', 'additional', 'H01-1001.10', 'such', 'as', 'the', 'time', 'and', 'place', 'of', 'the', 'rejoinder', 'and', 'the', 'attendance', '.']
    myPair.dep_parse = [('ROOT', 0, 3), ('amod', 2, 1), ('nsubj', 3, 2), ('det', 5, 4), ('dobj', 3, 5), ('case', 7, 6), ('nmod', 5, 7), ('case', 10, 8), ('det', 10, 9), ('nmod', 3, 10), ('cc', 10, 11), ('conj', 10, 12), ('aux', 14, 13), ('dep', 3, 14), ('amod', 16, 15), ('dobj', 14, 16), ('case', 20, 17), ('mwe', 17, 18), ('det', 20, 19), ('nmod', 16, 20), ('cc', 20, 21), ('conj', 20, 22), ('case', 25, 23), ('det', 25, 24), ('nmod', 20, 25), ('cc', 20, 26), ('det', 28, 27), ('conj', 20, 28), ('punct', 3, 29)]
    print(sdp(myPair))
# print(myPair.dep_parse.pop())


if __name__ == '__main__':
    nlp = StanfordCoreNLP('http://localhost:{0}'.format(9000))
    main()


