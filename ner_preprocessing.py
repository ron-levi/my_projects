import itertools
import nltk
import pandas as pd
import random
from nltk.tokenize import TreebankWordTokenizer as twt
from sklearn.model_selection import train_test_split
import io
import re
import ast
from collections import Counter


# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

class SentenceGetter(object):
    def __init__(self, data):
        self.data = data

    def prepare_input(self):
        output = []
        for ix, row in self.data.iterrows():
            doc = row['product_name']
            att_dict = ast.literal_eval(row['attrs_indexes_dict'])
            sent_words_pos_labels = get_words_pos_labels(doc, att_dict)
            sent_words_pos_iob_labels = get_iob_labels(sent_words_pos_labels)
            output.append(sent_words_pos_iob_labels)
        return output


def get_label(position_to_entity, word_index_in_doc):
    word_start_in_doc, word_end_in_doc = word_index_in_doc
    label = position_to_entity.get(word_index_in_doc, 'O')
    if label != 'O':
        return label
    for (entity_start, entity_end), label_in_attr_dict in position_to_entity.items():
        if entity_start <= word_start_in_doc and entity_end >= word_end_in_doc:
            return label_in_attr_dict
    return label


def get_words_pos_labels(doc, att_dict):
    position_to_entity = {v: k for k, v in att_dict.items()}
    word_delimiter_pattern = re.compile(r"[^,.:;+/()|' ]+")
    words = re.findall(word_delimiter_pattern, doc)
    pos_tags = nltk.pos_tag(words)
    token_indexes = [(i.start(), i.end()) for i in re.finditer(word_delimiter_pattern, doc)]
    words_pos_labels = []
    assert len(pos_tags) == len(token_indexes)
    for (word, part_of_speech), word_index_in_doc in zip(pos_tags, token_indexes):
        label = get_label(position_to_entity, word_index_in_doc)
        if label in ['battery_type',
                     'compatible_card',
                     'lens_mount',
                     # 'color',
                     'ports',
                     'features',
                     ]:
            label = 'O'
        words_pos_labels.append((word, part_of_speech, label))
    return words_pos_labels


def get_iob_labels(sent_words_pos_labels):
    words_list, pos_list, labels_list = zip(*sent_words_pos_labels)
    grouped_list = [list(grp) for k, grp in itertools.groupby(labels_list)]

    converted_to_iob_list = []
    for group in grouped_list:
        if set(group) == set('O'):
            converted_to_iob_list.extend(group)
        else:
            group = list(map(lambda ix_tuple: decide_iob_prefix(ix_tuple), list(enumerate(group))))
            converted_to_iob_list.extend(group)

    assert len(words_list) == len(pos_list) == len(converted_to_iob_list)
    return list(zip(words_list, pos_list, converted_to_iob_list))


def decide_iob_prefix(i_x_tuple):
    i, x = i_x_tuple
    return 'B-{}'.format(x) if i == 0 else 'I-{}'.format(x)


def make_line_separated_format(sentences, fn):
    with io.open(fn, "w", encoding="utf-8") as f:
        for sentence in sentences:
            chars_to_replace = {u'!',
                                u'#',
                                u'%',
                                u'*',
                                u'?',
                                u'@',
                                u'~',
                                u'\xa0',
                                u'\xa1',
                                u'\xa3',
                                u'\xac',
                                u'\xb0',
                                u'\xb1',
                                u'\xb8',
                                u'\xd7',
                                u'\xe2',
                                u'\xe3',
                                u'\xef',
                                u'\xff',
                                u'\u02dc',
                                u'\u03c6',
                                u'\u201a',
                                u'\u201c',
                                u'\u201d',
                                u'\u2026',
                                u'\u2033',
                                u'\u20ac',
                                u'\uff08',
                                u'\uff09',
                                u'\uff0c',
                                u'\ufffd'}

            f.writelines(
                [(label + "\t" + ''.join([c for c in word if c not in chars_to_replace]) + "\n") for word, pos, label in
                 sentence])
            f.write("\n")


if __name__ == '__main__':
    # main_create_inedexed()
    data = pd.read_csv(r"data/all_cms_indexed.csv", encoding="utf-8")
    # data = data.fillna(method="ffill")

    print("Number of sentences: ", len(data))

    all_tokens = [nltk.word_tokenize(s) for s in data["product_name"].values]
    distinct_words = set(list(itertools.chain.from_iterable(all_tokens)))
    n_words = len(distinct_words)
    print("Number of words in the dataset: ", n_words)

    getter = SentenceGetter(data)
    # Get all the sentences
    sentences = getter.prepare_input()
    w, p, l = zip(*itertools.chain.from_iterable(sentences))
    labels_counter = dict(Counter(l))

    for i in range(100):
        print('iteration', i)
        random.shuffle(sentences)
        # convert sentences variable to tokens separated by new line and sentences
        # divide to test and train sets
        test_index = len(sentences) // 4
        test_set = sentences[:test_index]
        train_set = sentences[test_index:]
        unique_labels_in_train = set([l for _, _, l in itertools.chain.from_iterable(train_set)])
        unique_labels_in_test = set([l for _, _, l in itertools.chain.from_iterable(test_set)])

        if unique_labels_in_train == unique_labels_in_test:
            break

    make_line_separated_format(train_set, fn="data/train_set_all_cms.txt")
    make_line_separated_format(test_set, fn="data/test_set_all_cms.txt")

    #
    # X_train = [sent2features(s) for s in train_set]
    # y_train = [sent2labels(s) for s in train_set]
    #
    # X_test = [sent2features(s) for s in test_set]
    # y_test = [sent2labels(s) for s in test_set]
    # crf_pipeline(X_train, y_train, X_test, y_test)
