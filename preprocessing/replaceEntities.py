"""
script to replace all entity texts with entityIDs (encode)
    and convert entityIDs back to entity texts (decode)
@author Peace Han
"""
import regex as re
import pprint


def encode(abstract):
    """
    encode entities in raw abstract string with entityID
    :param abstract: The <abstract> element from the data XML file with embedded <entity>s
    :return: scrubbed: The "cleaned" abstract text, with entities replaced with entityIDs
    :return: ents: The dictionary of entityID, entityText for this abstract text
    """
    # print("entities:")
    pattern = "<entity.*?<\/entity>"
    abstract_list = re.split(pattern, abstract)
    res = re.findall(pattern, abstract)
    pattern2 = '<entity id="(.*?)">'
    pattern3 = '<entity.*?>(.*?)</entity>'
    ents = {}  # dictionary of entityID, entityText
    for ent in res:
        id = re.match(pattern2, ent).group(1)
        t = re.match(pattern3, ent).group(1)
        ents[id] = t
    # pprint(ents)
    scrubbed = []  # list of
    for i in range(len(abstract_list)):
        scrubbed.append(abstract_list[i].strip())
        if i < len(res):
            # print(re.match(pattern2, res[i]).group(1))
            id = re.match(pattern2, res[i]).group(1)
            scrubbed.append(id)
    scrubbed = ' '.join(scrubbed)
    scrubbed = re.match('<abstract>(.*)</abstract>', scrubbed).group(1)
    scrubbed = scrubbed.strip()
    # print("Original abstract : ", abstract)
    # print("Processed abstract: ", scrubbed)
    # print('-----------------------------------------------')
    return scrubbed, ents


def decode(abstract, entities):
    """
    decode the entityIDs in an abstract string
    :param abstract: The encoded abstract string
    :param entities: The dictionary of entityID, entityText used to decode this abstract
    :return: The decoded abstract string
    """
    abstract = abstract.split()
    for i in range(len(abstract)):
        word = abstract[i]
        tok = ''
        if '_' in word:
            word = word.split('_')
            tok = word[1]
            word = word[0]
        if word in entities:
            ent = entities[word]
            abstract[i] = ent
            # print(abstract[i])
    abstract = ' '.join(abstract)
    # print(abstract)
    return abstract


def collect_tags(file):
    """
    Return all the unique POS tags used in given file
    :param file: the file to evaluate
    :return: list of each unique POS tag
    """
    out = set()
    with open(file) as f:
        for line in f:
            line = line.split()
            for elem in line:
                if elem.isupper():
                    out.add(elem)
    return out


def ptb_to_universal(abstract):
    """
    translate Penn treebank POS tags to
    the universal tagset
    :param abstract: the abstract (sdp representation) to translate
    :return: transl_abs, the translated abstract
    """
    tags_dict = {'RB': 'ADV',
                 'PRP': 'ADP',
                 'IN': 'ADP',
                 'NNP': 'NOUN',
                 'NNS': 'NOUN',
                 'NN': 'NOUN',
                 'JJ': 'ADJ',
                 'JJR': 'ADJ',
                 'JJS': 'ADJ',
                 'VBZ': 'VERB',
                 'VBP': 'VERB',
                 'VB': 'VERB',
                 'VBN': 'VERB',
                 'VBG': 'VERB',
                 'VBD': 'VERB',
                 'DT': 'DET',
                 'WDT': 'DET',
                 'FW': 'X',
                 'CD': 'NUM'}

    abstract = abstract.split()
    # print(abstract)
    for i in range(len(abstract)):
        elem = abstract[i]
        try:
             new_tag = tags_dict[elem]
        except KeyError:
            continue
        if elem:
            abstract[i] = new_tag
    return ' '.join(abstract)


# example abstracts
J87_3001_abs = '<abstract> This paper shows how <entity id="J87-3001.1">dictionary word sense definitions</entity> can be analysed by applying a hierarchy of <entity id="J87-3001.2">phrasal patterns</entity>. An experimental system embodying this mechanism has been implemented for processing <entity id="J87-3001.3">definitions</entity> from the <entity id="J87-3001.4">Longman Dictionary of Contemporary English</entity>. A property of this <entity id="J87-3001.5">dictionary</entity>, exploited by the system, is that it uses a <entity id="J87-3001.6"> restricted vocabulary </entity> in its <entity id="J87-3001.7">word sense definitions</entity>. The structures generated by the experimental system are intended to be used for the <entity id="J87-3001.8">classification</entity> of new <entity id="J87-3001.9">word senses </entity> in terms of the <entity id="J87-3001.10">senses</entity> of <entity id="J87-3001.11">words</entity> in the <entity id="J87-3001.12">restricted vocabulary</entity>. Examples illustrating the output generated are presented, and some qualitative performance results and problems that were encountered are discussed. The analysis process applies successively more specific <entity id="J87-3001.13">phrasal analysis rules</entity> as determined by a hierarchy of <entity id="J87-3001.14">patterns</entity> in which less specific <entity id="J87-3001.15">patterns </entity> dominate more specific ones. This ensures that reasonable incomplete analyses of the <entity id="J87-3001.16">definitions </entity> are produced when more complete analyses are not possible, resulting in a relatively robust <entity id="J87-3001.17">analysis mechanism</entity>. Thus the work reported addresses two <entity id="J87-3001.18">robustness problems </entity> faced by current experimental <entity id="J87-3001.19">natural language processing systems</entity>: coping with an incomplete <entity id="J87-3001.20">lexicon</entity> and with incomplete <entity id="J87-3001.21">knowledge </entity> of <entity id="J87-3001.22">phrasal constructions</entity>. </abstract>'
I05_5009_abs = '<abstract> This paper presents an <entity id="I05-5009.1">evaluation method</entity> employing a <entity id="I05-5009.2">latent variable model </entity> for <entity id="I05-5009.3">paraphrases</entity> with their <entity id="I05-5009.4">contexts</entity>. We assume that the <entity id="I05-5009.5">context</entity> of a <entity id="I05-5009.6">sentence</entity> is indicated by a <entity id="I05-5009.7">latent variable</entity> of the <entity id="I05-5009.8">model </entity> as a <entity id="I05-5009.9">topic</entity> and that the <entity id="I05-5009.10">likelihood</entity> of each <entity id="I05-5009.11">variable</entity> can be inferred. A <entity id="I05-5009.12">paraphrase </entity> is evaluated for whether its <entity id="I05-5009.13">sentences</entity> are used in the same <entity id="I05-5009.14">context</entity>. Experimental results showed that the proposed method achieves almost 60% <entity id="I05-5009.15">accuracy</entity> and that there is not a large performance difference between the two <entity id="I05-5009.16">models</entity>. The results also revealed an upper bound of <entity id="I05-5009.17">accuracy </entity> of 77% with the <entity id="I05-5009.18">method</entity> when using only <entity id="I05-5009.19"> topic information</entity>. </abstract>'
P83_1003_abs = '<abstract> An extension to the <entity id="P83-1003.1">GPSG grammatical formalism</entity> is proposed, allowing <entity id="P83-1003.2">non-terminals </entity> to consist of finite sequences of <entity id="P83-1003.3">category labels</entity>, and allowing <entity id="P83-1003.4">schematic variables </entity> to range over such sequences. The extension is shown to be sufficient to provide a strongly adequate <entity id="P83-1003.5">grammar </entity> for <entity id="P83-1003.6">crossed serial dependencies</entity>, as found in e.g. <entity id="P83-1003.7">Dutch subordinate clauses</entity>. The structures induced for such <entity id="P83-1003.8">constructions </entity> are argued to be more appropriate to data involving <entity id="P83-1003.9">conjunction</entity> than some previous proposals have been. The extension is shown to be parseable by a simple extension to an existing <entity id="P83-1003.10">parsing method</entity> for <entity id="P83-1003.11">GPSG</entity>. </abstract>'
P83_1003_enc = 'An extension to the P83-1003.1 is proposed, allowing P83-1003.2 to consist of finite sequences of P83-1003.3 , and allowing P83-1003.4 to range over such sequences. The extension is shown to be sufficient to provide a strongly adequate P83-1003.5 for P83-1003.6 , as found in e.g. P83-1003.7 . The structures induced for such P83-1003.8 are argued to be more appropriate to data involving P83-1003.9 than some previous proposals have been. The extension is shown to be parseable by a simple extension to an existing P83-1003.10 for P83-1003.11 .'
P83_1003_ent = {'P83-1003.1': 'GPSG grammatical formalism', 'P83-1003.2': 'non-terminals ', 'P83-1003.3': 'category labels', 'P83-1003.4': 'schematic variables ', 'P83-1003.5': 'grammar ', 'P83-1003.6': 'crossed serial dependencies', 'P83-1003.7': 'Dutch subordinate clauses', 'P83-1003.8': 'constructions ', 'P83-1003.9': 'conjunction', 'P83-1003.10': 'parsing method', 'P83-1003.11': 'GPSG'}
pos_sdp = 'USAGE NN <-- nsubj <-- VBP --> dobj --> NN --> nmod --> NN REVERSE'


def main():
    filename = 'clean/train_data/data_3.0/1.1.features_pos.txt'
    print(encode(J87_3001_abs))
    print(encode(I05_5009_abs))
    [print(x) for x in encode(P83_1003_abs)]
    print(decode(P83_1003_enc, P83_1003_ent))
    print(collect_tags(filename))
    print(ptb_to_universal(pos_sdp))
    pprint.pprint(P83_1003_ent)


if __name__ == '__main__':
    main()

