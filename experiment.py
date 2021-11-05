import re
import qparse
import argparse
from file2doc import Tokenizer
from nltk.stem.porter import *
import numpy as np
from time import time

stop_words = []
stop_path = r"\data\stops.txt"
with open(stop_path) as f:
    for line in f:
        stop_words.append(line.strip('\n'))
stop_words = set(stop_words)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("index_directory_path", type=str,
                        help="path for the directory containing index files")
    parser.add_argument("query_file_path", type=str,
                        help="path for the query file")
    parser.add_argument("results_file", type=str,
                        help="file path to which output will be written out")
    parser.add_argument("--ngram", "-n", type=int, default=2,
                        help="n-gram value for phrase index")
    parser.add_argument("--window", "-w", type=int, default=10,
                        help="window size for proximity search. default: 10")
    parser.add_argument("--threshold", "-t", type=int, default=20,
                        help="retrieval threshold for phrase and proximity search. default: 20")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="this will print every step of the system.")
    parser.add_argument("--evaluate", action="store_true", default=False,
                        help="this will print the treceval output.")


    args = parser.parse_args()
    index_path = args.index_directory_path
    query_path = args.query_file_path
    results_file = args.results_file
    n_gram = args.ngram
    window = args.window
    threshold = args.threshold
    verbose = args.verbose
    evaluate = args.evaluate

    processor = Processor(index_path, query_path, results_file, n_gram, window,
                          threshold, verbose)
    results = processor.run()
    with open(results_file, 'w') as w:
        for doc in results:
            for result in doc[:100]:
                line = " ".join(result) + '\n'
                w.write(line)
    if evaluate:
        processor.evaluation()

def _generate_line(file):
    with open(file) as f:
        for line in f:
            yield line

class Processor:
    def __init__(self, index_path, query_path, results_file, n_gram,
                 window_size, retrieval_threshold, verbose):
        self.queries = qparse.query_parser(query_path)
        self.queries_tokenized = self.tokenize()
        self.queries_tokenized.sort()
        self.index_path = index_path
        self.query_path = query_path
        self.results_file = results_file
        self.n_gram = n_gram
        self.window_size = window_size
        self.retrieval_threshold = retrieval_threshold
        self.phrase_posting_list_dict, self.phrase_doc_pl_dict = self.make_posting_list_dict("phrase")
        self.single_posting_list_dict, self.single_doc_pl_dict = self.make_posting_list_dict("single")        
        self.positional_posting_list_dict, self.positional_doc_pl_dict = self.make_positional_posting_list_dict()
        self.doc_dict = self.make_doc_dict()
        self.phrase_lexicon_dict = self.make_lexicon_dict("phrase")
        self.single_lexicon_dict = self.make_lexicon_dict("single")
        self.positional_lexicon_dict = self.make_lexicon_dict("positional")
        self.mean_doc_length = sum([self.doc_dict[doc][1] for doc in self.doc_dict]) / len(self.doc_dict)
        self.verbose = verbose

    def run(self):
        outputs = []
        for i, q in enumerate(self.queries_tokenized):
            output = []
            if self.verbose:
                print(f"{i+1}/{len(self.queries_tokenized)} being processed")
            q_num = q[0]
            query = q[1]
            query_backup = []
            for term in query:
                query_backup.append(term)
            if len(query) >= self.n_gram:
                q_terms = self.preprocess(query, "phrase")
                q_tids = [self.phrase_lexicon_dict[term] for term in q_terms if term in self.phrase_lexicon_dict]
                total_df = 0
                for tid in q_tids:
                    total_df += len(self.phrase_posting_list_dict[tid])
                if total_df >= self.retrieval_threshold:
                    output = self.bm25(q_num, q_terms, "phrase")
                else:
                    if self.verbose:
                        print("phrase not common, switching to proximity search")
                    output = self.positional_bm25(q_num, q_terms, window_size=self.window_size)
                    if len(output) < self.retrieval_threshold:
                        if self.verbose:
                            print("not enough documents retrieved, switching to single index")
                        q_terms = self.preprocess(query_backup, "single")
                        output = self.bm25(q_num, q_terms, "single")
            else:
                if self.verbose:
                    print("query too short, switching to single index")
                q_terms = self.preprocess(query, "single")
                output = self.bm25(q_num, q_terms, "single")

            outputs.append(output)

        return outputs

    def tokenize(self):
        queries_tokenized = []
        for query in self.queries:
            query_tokenized = [query[0]]
            tokens = []
            for word in query[1].split():
                token = Tokenizer(word)
                tokens.extend(token.lexemes)
            query_tokenized.append(tokens)
            query_tokenized = tuple(query_tokenized)
            queries_tokenized.append(query_tokenized)
        return queries_tokenized

    def preprocess(self, q_terms, index_type):
        preprocessed = []
        if index_type == "single":
            preprocessed = [term.strip('\n') for term in q_terms if re.search(r"[a-zA-Z0-9]", term) is not None and term not in stop_words]
        elif index_type == "stem":
            preprocessed = [stemmer.stem(term.strip('\n')) for term in q_terms if re.search(r"[a-zA-Z0-9]", term) is not None and term not in stop_words]
        elif index_type == "phrase":
            window = []
            while len(q_terms) > 0:
                while len(window) < self.n_gram:
                    window.append(q_terms[0].strip("\n"))
                    q_terms.pop(0)
                is_phrase = True
                for j, term in enumerate(window):
                    if re.search(r"[a-zA-Z0-9]", term) is None or term in stop_words:
                        is_phrase = False
                        window = window[j+1:]
                        break
                if is_phrase and len(window) == self.n_gram:
                    phrase = " ".join(window)
                    preprocessed.append(phrase)
                    window = window[1:]
        return preprocessed

    def make_posting_list_dict(self, index_type):
        posting_list_dict = dict()
        doc_pl_dict = dict()
        prev_tid = 0
        if index_type == "phrase":            
            file_name = f"{self.n_gram}_gram_posting_list.txt"
        else:
            file_name = f"{index_type}_posting_list.txt"
        pls = _generate_line(self.index_path + r"\\" + file_name)
        for pl in pls:
            line = [int(item) for item in pl.strip("\n").split("\t")]
            tid = line[0]
            did = line[1]
            tf = line[2]
            if tid != prev_tid:
                posting_list_dict[tid] = []
            posting_list_dict[tid].append((did, tf))
            prev_tid = tid
            if did not in doc_pl_dict:
                doc_pl_dict[did] = []
            doc_pl_dict[did].append((tid, tf))
        return posting_list_dict, doc_pl_dict

    def make_positional_posting_list_dict(self):
        positional_posting_list_dict = dict()
        positional_doc_pl_dict = dict()
        prev_tid = 0
        file_name = "positional_posting_list.txt"
        pls = _generate_line(self.index_path + r"\\" + file_name)
        for pl in pls:
            line = [item for item in pl.strip("\n").split("\t")]
            tid = int(line[0])
            did = int(line[1])
            positions = [int(position.strip()) for position in line[2].strip("()").split(",") if position != ""]
            if tid != prev_tid:
                positional_posting_list_dict[tid] = []
            positional_posting_list_dict[tid].append((did, positions))
            prev_tid = tid
            if did not in positional_doc_pl_dict:
                positional_doc_pl_dict[did] = []
            positional_doc_pl_dict[did].append((tid, positions))
        return positional_posting_list_dict, positional_doc_pl_dict
    
    def make_lexicon_dict(self, index_type):
        lexicon_dict = dict()
        if index_type == "phrase":
            file_name = f"{self.n_gram}_gram_lexicon.txt"
        else:
            file_name = f"{index_type}_lexicon.txt"
        lexes = _generate_line(self.index_path + r"\\" + file_name)
        for lex in lexes:
            line = [item for item in lex.strip("\n").split("\t") if item != '']
            tid = int(line[0])
            term = line[1]
            lexicon_dict[term] = tid
        return lexicon_dict

    def make_doc_dict(self):
        doc_dict = dict()
        docs = _generate_line(self.index_path + r"\\" + "doc_map.txt")
        for doc in docs:
            line = [item for item in doc.strip("\n").split("\t") if item != '']
            did = int(line[0])
            doc_name = line[1].strip('\n').strip('.txt')
            doc_length = int(line[2])
            doc_dict[did] = (doc_name, doc_length)
        return doc_dict

    def bm25(self, q_num, q_terms, index_type):
        if index_type == "phrase":
            posting_list_dict = self.phrase_posting_list_dict
            doc_pl_dict = self.phrase_doc_pl_dict
            lexicon_dict = self.phrase_lexicon_dict
        elif index_type == "single":
            posting_list_dict = self.single_posting_list_dict
            doc_pl_dict = self.single_doc_pl_dict
            lexicon_dict = self.single_lexicon_dict            
        scores = dict()
        N = len(self.doc_dict)
        b = 0.75
        k = 1.2
        q_tids = [lexicon_dict[term] for term in q_terms if term in lexicon_dict]
        for tid in q_tids:  # numerator
            posting_list = posting_list_dict[tid]
            n = len(posting_list)
            idf = np.log((N - n + 0.5) / (n + 0.5))
            for p in posting_list:
                did = p[0]
                tf = p[1]
                doc_length = sum([term[1] for term in doc_pl_dict[did]])
                score = idf * ((k + 1) * tf) / (tf + k * (1 - b + b * (doc_length / self.mean_doc_length)))
                if did not in scores:
                    scores[did] = 0
                scores[did] += score
        output = []
        for i, did in enumerate(sorted(scores, key=scores.get, reverse=True)):            
            output.append([str(q_num), '0', self.doc_dict[did][0], str(i+1), str(scores[did]), "bm25"])
        return output

    def positional_bm25(self, q_num, q_terms, window_size=20):
        scores = dict()
        N = len(self.doc_dict)
        b = 0.75
        k = 1.2
        for phrase in q_terms:  # q_terms are phrases, so we will go word by word
            cands_exist = True
            q_tids = [self.positional_lexicon_dict[term] for term in phrase.split() if term in self.positional_lexicon_dict]
            if len(q_tids) < self.n_gram:
                continue
            doc_cands = {p[0] for p in self.positional_posting_list_dict[q_tids[0]]}
            for tid in q_tids[1:]:
                doc_cands_n = {p[0] for p in self.positional_posting_list_dict[tid]}
                if len(doc_cands) != 0:
                    doc_cands.intersection_update(doc_cands_n)
                if len(doc_cands) == 0:
                    cands_exist = False
                
            if cands_exist:
                
                n = len(doc_cands)
                idf = np.log((N - n + 0.5) / (n + 0.5))
                for did in doc_cands:
                    pointers = []
                    for tid, positions in self.positional_doc_pl_dict[did]:
                        if tid in q_tids:                            
                            for position in positions:
                                pointers.append((tid, position))
                    pointers.sort(key=lambda x: x[1])
                    tf = 0
                    for start_i, start_pointer in enumerate(pointers):
                        start_tid = start_pointer[0]
                        start_position = start_pointer[1]
                        window = [start_tid]
                        for current_i, current_pointer in enumerate(pointers[start_i+1:]):
                            if current_pointer[1] <= (start_position + window_size):
                                window.append(current_pointer[0])
                            if current_i == len(pointers[start_i+1:])-1:
                                if len(set(window)) == self.n_gram:
                                    tf += 1
                                break                                
                            elif current_pointer[1] > (start_position + window_size):  
                                if len(set(window)) == self.n_gram:
                                    tf += 1
                                break

                    doc_length = self.doc_dict[did][1]
                    if tf != 0:
                        score = idf * ((k + 1) * tf) / (tf + k * (1 - b + b * (doc_length / self.mean_doc_length)))
                        if did not in scores:
                            scores[did] = 0
                        scores[did] += score

        output = []
        for i, did in enumerate(sorted(scores, key=scores.get, reverse=True)):
            if scores[did] != 0:
                output.append([str(q_num), '0', self.doc_dict[did][0], str(i+1), str(scores[did]), "bm25"])
        return output

    def evaluation(self):
        results_file = self.results_file
        treceval_fp = ".\bin\treceval.exe"
        qrels_fp = ".\data\qrel.txt"
        cmd = ['./{}'.format(treceval_fp), qrels_fp, results_file]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE        )
            resp = proc.communicate()
            msg_out, msg_err = (msg.decode('utf-8') for msg in resp)
            print(msg_out)
        except Exception:
            raise
        
        
        
    def bigrun(self):
        outputs = []
        for i, q in enumerate(self.queries_tokenized):
            output = []
            print(f"{i+1}/{len(self.queries_tokenized)} being processed")
            q_num = q[0]
            query = q[1]
            query_backup = []
            for term in query:
                query_backup.append(term)
            if len(query) >= self.n_gram:
                q_terms = self.preprocess(query, "phrase")
                q_tids = [self.phrase_lexicon_dict[term] for term in q_terms if term in self.phrase_lexicon_dict]
                total_df = 0
                for tid in q_tids:
                    total_df += len(self.phrase_posting_list_dict[tid])
                if total_df >= 2:
                    output = self.bm25(q_num, q_terms, "phrase")
                else:
                    print("phrase not common, switching to proximity search")
                    output = self.positional_bm25(q_num, q_terms)
                    if len(output) < 5:
                        print("not enough documents retrieved, switching to single index")
                        q_terms = self.preprocess(query_backup, "single")
                        output = self.bm25(q_num, q_terms, "single")
            else:
                print("query too short, switching to single index")
                q_terms = self.preprocess(query, "single")
                output = self.bm25(q_num, q_terms, "single")

            outputs.append(output)

        return outputs

    def test(self):
        outputs = []
        yes = 0
        for i, q in enumerate(self.queries_tokenized):
            output = []
            print(f"{i+1}/{len(self.queries_tokenized)} being processed")
            q_num = q[0]
            query = q[1]
            query_backup = []
            for term in query:
                query_backup.append(term)
            if len(query) >= self.n_gram:
                q_terms = self.preprocess(query, "phrase")
                q_tids = [self.phrase_lexicon_dict[term] for term in q_terms if term in self.phrase_lexicon_dict]
                total_df = 0
                for tid in q_tids:
                    total_df += len(self.phrase_posting_list_dict[tid])
                output = self.bm25(q_num, q_terms, "phrase")
                if len(output) >= self.output_threshold:
                    yes += 1
            outputs.append(output)
        print(yes)
        return yes, outputs


#if __name__ == "__main__":
#    main()

from collections import defaultdict
import subprocess
import os
import re


exp = defaultdict(lambda: defaultdict(lambda: 0))
results_file = r"C:\Users\aozsa\Documents\Github\IR\Project2\result.txt"
treceval_fp = "treceval.exe"
qrels_fp = "qrel.txt"
for i in range(4):
    window = (i + 1) *5
    for j in range(20):
        threshold = j + 1
        processor = Processor(r"C:\Users\aozsa\Documents\OutputComesHere",
                              r"C:\Users\aozsa\Documents\Github\IR\Project2\queryfile.txt",
                              window_size=window, retrieval_threshold=threshold)
        results = processor.run()
        with open(results_file, 'w') as w:
            for query in results:
                for result in query[:100]:
                    line = " ".join(result) + '\n'
                    w.write(line)
        cmd = ['./{}'.format(treceval_fp), qrels_fp, results_file]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE        )
            resp = proc.communicate()
            msg_out, msg_err = (msg.decode('utf-8') for msg in resp)
            subbed = re.sub(r"[\t\n\s]", "", msg_out)
            match = re.search(r"overallreldocs(0.[0-9]+)Precision", subbed)
            MAP = match[1]
            exp[window][threshold] = MAP
        except Exception:
            raise
        
        print(f"window_size = {window}, threshold = {threshold}, acc = {MAP}")


for i in range(20):
    threshold = (i + 1)
    processor = Processor(r".\data\index",
                          r".\data\queryfile.txt",
                          output_threshold=threshold)
    yes, outputs = processor.test()
    print(f"{threshold}")


start = time()
processor = Processor(r".\data\index",
                      r".\data\queryfile.txt",
results = processor.test()
finish = time()
#print(finish-start)

dataframe = []
for key in exp:
    this_row = [key]
    for key2 in exp[key]:
        this_row.append(exp[key][key2])
    dataframe.append(this_row)

import pandas as pd
df = pd.DataFrame(dataframe)
df.to_excel("thisone.xlsx")
dataframe = pd