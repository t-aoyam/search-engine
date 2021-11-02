from nltk.stem.porter import *
stemmer = PorterStemmer()

stop_words = []
stop_path = r".\data\stops.txt"
with open(stop_path) as f:
    for line in f:
        stop_words.append(line.strip('\n'))
stop_words = set(stop_words)
stemmed_stop_words = set([stemmer.stem(stop) for stop in stop_words])


class InvertedIndex:
    def __init__(self, lexicon=dict(), posting_list=set(), doc_pl=dict(),
                 doc_map=dict(), tid=0, did=0, mmry_cstrt=0, stop_words=stop_words,
                 exclude_stop_words=True, stem=False):
        self.lexicon = lexicon
        self.posting_list = posting_list
        self.doc_pl = doc_pl
        self.did = did
        self.tid = tid
        self.doc_map = doc_map
        self.mmry_cstrt = mmry_cstrt
        self.stem = stem
        self.stop_words = stemmed_stop_words if self.stem else stop_words
        self.exclude_stop_words = exclude_stop_words

    def update_index(self, token):
        if self.stem:
            token = stemmer.stem(token)
        self.add_to_lexicon(token)
        self.update_posting_list(token)

    def update_positional_index(self, position, token):
        self.add_to_lexicon(token)
        self.update_positional_posting_list(position, token)

    def add_to_lexicon(self, token):
        add = False
        if token not in self.lexicon:
            add = True
        if self.exclude_stop_words:
            if token in self.stop_words:
                add = False
        if len(token) == 0:
            add = False
        if add:
            self.tid += 1
            self.lexicon[token] = self.tid

    def update_posting_list(self, token):
        if token in self.lexicon:
            if token not in self.doc_pl:
                self.doc_pl[token] = 1
            else:
                self.doc_pl[token] += 1

    def update_positional_posting_list(self, position, token):
        if token in self.lexicon:
            if token not in self.doc_pl:
                self.doc_pl[token] = {position}
            else:
                self.doc_pl[token].add(position)

    def read_doc(self, doc_name):
        self.did += 1
        self.doc_map[self.did] = [doc_name]

    def store_pl(self):
        if len(self.doc_pl) > 0:
            newset = set((self.lexicon[key], self.did, self.doc_pl[key]) for key in self.doc_pl)
            self.posting_list.update(newset)
            self.doc_pl = dict()

    def store_positional_pl(self):
        if len(self.doc_pl) > 0:
            newset = set((self.lexicon[key], self.did, tuple(self.doc_pl[key])) for key in self.doc_pl)
            self.posting_list.update(newset)
            self.doc_pl = dict()

    def is_above_constraint(self):
        if self.mmry_cstrt == 0:
            return False
        else:
            memory = len(self.posting_list) + len(self.doc_pl)
            if memory > self.mmry_cstrt:
                return True
            else:
                return False
