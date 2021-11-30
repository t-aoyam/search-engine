import re
import os
import qparse
import argparse
import numpy as np
import subprocess
from file2doc import Tokenizer
from nltk.stem.porter import *
from time import time
from pathlib import Path

stemmer = PorterStemmer()
stop_words = []
stop_path = Path("data/stops.txt")
with open(stop_path) as f:
    for line in f:
        stop_words.append(line.strip('\n'))
stop_words = set(stop_words)

def main():
    """
    parse arguments, instantiate Processor class, and run the query
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("index_directory_path", type=str,
                        help="path for the directory containing index files")
    parser.add_argument("query_file_path", type=str,
                        help="path for the query file")
    parser.add_argument("retrieval_model", type=str,
                        help="retrieval model. choose from 'cosine', 'bm25', or 'lm'.")
    parser.add_argument("index_type", type=str,
                        help="index type. choose from 'single' or 'stem'")
    parser.add_argument("results_file", type=str,
                        help="file path to which output will be written out")
    parser.add_argument("--n_doc", "-n", type=int, default=10,
                        help="top n documents to identify expansion terms from. default is 20")
    parser.add_argument("--m_term", "-m", type=int, default=3,
                        help="top m terms to expand the query term by. default is 3")
    parser.add_argument("--num_iter", "-i", type=int, default=0,
                        help="number of query expansion iteration. default is 0")
    parser.add_argument("--q_threshold", "-q", type=float, default=1,
                        help="top 100*q % of the query terms will be used. default is 1")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="this will print every step of the system.")
    parser.add_argument("--long", "-l", action="store_true", default=False,
                        help="this will parse long queries. default is short queries")
    parser.add_argument("--evaluate", action="store_true", default=False,
                        help="this will print the treceval output.")
    parser.add_argument("--show", action="store_true", default=False,
                        help="this will show what query term was added")

    args = parser.parse_args()
    index_path = args.index_directory_path
    query_path = args.query_file_path
    model = args.retrieval_model
    index_type = args.index_type
    results_file = args.results_file
    n_doc = args.n_doc
    m_term = args.m_term
    num_iter = args.num_iter
    q_threshold = args.q_threshold
    verbose = args.verbose
    long = args.long
    evaluate = args.evaluate
    show = args.show

    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    start = time()
    processor = Processor(index_path, query_path, model, index_type, n_doc, m_term,
                          num_iter, q_threshold, results_file, long, verbose, show)
    results = processor.run()
    with open(results_file, 'w') as w:
        for doc in results:
            for result in doc[:100]:
                line = " ".join(result) + '\n'
                w.write(line)
    finish = time()
    if evaluate:
        out = processor.evaluation()
        print(out)
    if verbose:
        print(f"query processing time: {round(finish-start, 3)} seconds")

def _generate_line(file):
    with open(file) as f:
        for line in f:
            yield line

class Processor:
    def __init__(self, index_path, query_path, model, index_type, n_doc, m_term,
                 num_iter, q_threshold, results_file, long, verbose, show):
        self.long = long
        if self.long:            
            self.queries = qparse.query_parser_long(query_path)
        else:
            self.queries = qparse.query_parser(query_path)
        self.queries_tokenized = self.tokenize()
        self.queries_tokenized.sort()
        self.index_path = Path(index_path)
        self.query_path = Path(query_path)
        self.results_file = Path(results_file)
        self.model = model
        self.index_type = index_type
        if index_type != "positional":        
            self.posting_list_dict, self.doc_pl_dict = self.make_posting_list_dict()
        self.doc_dict = self.make_doc_dict()
        self.lexicon_dict = self.make_lexicon_dict()
        self.show = show
        if self.show:
            self.inverse_lexicon_dict = self.make_inverse_lexicon_dict()
        if model == "bm25" or model == "lm":
            self.mean_doc_length = sum([self.doc_dict[doc][1] for doc in self.doc_dict]) / len(self.doc_dict)
        self.n_doc = n_doc
        self.m_term = m_term
        self.num_iter = num_iter
        self.q_threshold = q_threshold
        self.verbose = verbose

    def run(self):
        outputs = []
        for q_num, query in self.queries_tokenized:
            if self.show:
                print(f"\n{q_num}:")
            q_terms = self.preprocess(query)
            num_iter = self.num_iter
            q_tids = [self.lexicon_dict[term] for term in q_terms if term in self.lexicon_dict]
            if self.q_threshold != 1:
                include, exclude = self.q_reduction(q_tids)
                if self.show:
                    exclude_show = [self.inverse_lexicon_dict[item] for item in exclude]
                    q_terms_show = [self.inverse_lexicon_dict[item] for item in q_tids]
                    print(f"Original:\n[{', '.join(q_terms_show)}]")
                    print(f"Removed:\n[{', '.join(exclude_show)}]")
                q_tids = include
            while num_iter >= 0:
                if self.model == "cosine":
                    rel_rank = self.cosine(q_tids)
                elif self.model == "bm25":
                    rel_rank = self.bm25(q_tids)
                elif self.model == "lm":
                    rel_rank = self.lm(q_tids)
                num_iter -= 1
                if num_iter >= 0:
                    cands = self.rel_fb(rel_rank, q_tids)
                    q_terms_show = [self.inverse_lexicon_dict[item] for item in q_tids]
                    q_tids.extend(cands)
                    cands_show = [self.inverse_lexicon_dict[cand] for cand in cands]
                    if self.show:
                        print(f"Original:\n[{', '.join(q_terms_show)}]")
                        print(f"Added:\n[{', '.join(cands_show)}]")
            output = self.res2trec(q_num, rel_rank)
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

    def preprocess(self, q_terms):  # index_type-specific preprocessing
        preprocessed = []
        if self.index_type == "single":
            preprocessed = [term.strip('\n') for term in q_terms if re.search(r"[a-zA-Z0-9]", term) is not None and term not in stop_words]
        elif self.index_type == "stem":
            preprocessed = [stemmer.stem(term.strip('\n')) for term in q_terms if re.search(r"[a-zA-Z0-9]", term) is not None and term not in stop_words]
        elif self.index_type == "phrase":
            window = []
            while len(q_terms) > 0:
                while len(window) < self.n_gram and len(q_terms) > 0:
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

    def make_posting_list_dict(self):  # read posting list and create dict object
        posting_list_dict = dict()
        doc_pl_dict = dict()
        prev_tid = 0
        if self.index_type == "phrase":            
            file_name = f"{self.n_gram}_gram_posting_list.txt"
        else:
            file_name = f"{self.index_type}_posting_list.txt"
        pls = _generate_line(self.index_path / file_name)
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
    
    def make_lexicon_dict(self):  # read lexicon file and create dict object
        lexicon_dict = dict()
        file_name = f"{self.index_type}_lexicon.txt"
        lexes = _generate_line(self.index_path / file_name)
        for lex in lexes:
            line = [item for item in lex.strip("\n").split("\t") if item != '']
            tid = int(line[0])
            term = line[1]
            lexicon_dict[term] = tid
        return lexicon_dict

    def make_inverse_lexicon_dict(self):  # read lexicon file and create dict object
        inverse_lexicon_dict = dict()
        for key in self.lexicon_dict:
            inverse_lexicon_dict[self.lexicon_dict[key]] = key
        return inverse_lexicon_dict

    def make_doc_dict(self):  # read document id - document name map and create dict object
        doc_dict = dict()
        docs = _generate_line(self.index_path / "doc_map.txt")
        for doc in docs:
            line = [item for item in doc.strip("\n").split("\t") if item != '']
            did = int(line[0])
            doc_name = line[1].strip('\n').strip('.txt')
            doc_length = int(line[2])
            doc_dict[did] = (doc_name, doc_length)
        return doc_dict

    def cosine(self, q_tids):  # cosine calculation
        scores = dict()
        for tid in q_tids:  # numerator
            self.posting_list = self.posting_list_dict[tid]
            idf = np.log10(len(self.doc_dict)/len(self.posting_list))
            for p in self.posting_list:
                did = p[0]
                tf = p[1]
                w = idf
                d = tf*idf
                if did not in scores:
                    scores[did] = 0
                scores[did] += w*d
        for did in scores:  # denominator
            nf = 0  # normalization factor
            for tid, tf in self.doc_pl_dict[did]:
                idf = np.log10(len(self.doc_dict)/len(self.posting_list_dict[tid]))
                nf += (tf*idf)**2
            nf = np.sqrt(nf)
            scores[did] /= nf
        return scores
    
    def bm25(self, q_tids):  # bm25 calculation
        scores = dict()
        N = len(self.doc_dict)
        b = 0.75
        k = 1.2
        for tid in q_tids:  # numerator
            posting_list = self.posting_list_dict[tid]
            n = len(posting_list)
            idf = np.log((N - n + 0.5) / (n + 0.5))
            for p in posting_list:
                did = p[0]
                tf = p[1]
                doc_length = sum([term[1] for term in self.doc_pl_dict[did]])
                score = idf * ((k + 1) * tf) / (tf + k * (1 - b + b * (doc_length / self.mean_doc_length)))
                if did not in scores:
                    scores[did] = 0
                scores[did] += score
        return scores

    def lm(self, q_tids):  # language model with dirichlet smoothing
        scores = dict()
        C = sum([sum([doc[1] for doc in self.posting_list_dict[tid]]) for tid in list(self.lexicon_dict.values())])
        mu = self.mean_doc_length
        relevant_docs = set()
        for tid in q_tids:
            relevant_docs.update(set([pl[0] for pl in self.posting_list_dict[tid]]))
        for tid in q_tids:
            posting_list = self.posting_list_dict[tid]
            for p in posting_list:
                did = p[0]
                tf_d = p[1]
                tf_c = sum([p[1] for p in posting_list])
                D = self.doc_dict[did][1]
                score = np.log((tf_d + mu * tf_c / C) / (D + mu))
                if did not in scores:
                    scores[did] = 0
                scores[did] += score
            for did in relevant_docs:
                if did not in [p[0] for p in posting_list]:
                    tf_d = 0
                    tf_c = sum([p[1] for p in posting_list])
                    D = self.doc_dict[did][1]
                    score = np.log((tf_d + mu * tf_c / C) / (D + mu))
                    if did not in scores:
                        scores[did] = 0
                    scores[did] += score
        return scores

    def res2trec(self, q_num, scores):
        output = []
        for i, did in enumerate(sorted(scores, key=scores.get, reverse=True)):            
            output.append([str(q_num), '0', self.doc_dict[did][0],
                           str(i+1), str(scores[did]), str(self.model)])
        return output    

    def rel_fb(self, scores, q_tids):
        scores = sorted(scores, key=scores.get, reverse=True)            
        nidf = dict()  # df in relevant set
        fidf = dict()  # cf in relevant set
        rel_docs = scores[:self.n_doc]
        for doc in rel_docs:
            for tid, tf in self.doc_pl_dict[doc]:
                if sum(posting[1] for posting in self.posting_list_dict[tid]) < 3:  # excluding rare terms
                    continue
                if tid not in nidf:
                    nidf[tid] = 0
                nidf[tid] += 1
                if tid not in fidf:
                    fidf[tid] = 0
                fidf[tid] += tf
        for tid in nidf:
            N = len(self.doc_dict)
            n = len(self.posting_list_dict[tid])
            idf = np.log((N - n + 0.5) / (n + 0.5))
            nidf[tid] *= idf
            fidf[tid] *= idf
        cands = sorted(nidf, key=nidf.get, reverse=True)[:self.m_term]
        return cands

    def q_reduction(self, q_tids):
        q_weight = []
        for tid in q_tids:
            N = len(self.doc_dict)
            n = len(self.posting_list_dict[tid])
            idf = np.log((N - n + 0.5) / (n + 0.5))
            q_weight.append((tid, idf))
        ranked_list = [item[0] for item in sorted(q_weight, key=lambda x:x[1], reverse=True)]
        num_term = round(len(q_tids)*self.q_threshold)
        return (ranked_list[:num_term], ranked_list[num_term:])

    def evaluation(self):
        results_file = self.results_file.__str__()
        treceval_fp = Path("bin/treceval.exe").__str__()
        qrels_fp = Path("data/qrel.txt").__str__()
        cmd = [treceval_fp, qrels_fp, results_file]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE        )
            resp = proc.communicate()
            msg_out, msg_err = (msg.decode('utf-8') for msg in resp)
        except Exception:
            raise
        return msg_out

if __name__ == "__main__":
    main()