import os
import re
import shutil
import argparse
import time
import inverted_index
import file2doc

"""
reading stop_words and making a set
"""
stop_words = []
stop_path = "stops.txt"
with open(stop_path) as f:
    for line in f:
        stop_words.append(line.strip('\n'))
stop_words = set(stop_words)


def main():
    """
    parse arguments
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("trec_path", type=str, help="trec files directory path")
    parser.add_argument("index_type",  type=str,
                        help="type of index to build: 'single', 'stem', 'phrase', or 'positional'")
    parser.add_argument("output_dir", type=str, help="output directory path")
    parser.add_argument("--ngram", "-n", type=int, default=2,
                        help="n-gram value for phrase index")
    parser.add_argument("--memory_constraint", "-m", type=int, default=0,
                        help="number of tuples that can be held in memory; unlimited if unspecified")
    parser.add_argument("--noparse", action="store_true", default=False,
                        help="this will tell the system not to parse the raw files. deafult is false, and the system will do the parsing.")
    parser.add_argument("--nostats", action="store_true", default=False,
                        help="this will tell the system not to print the statistics. deafult is false, and the system will print the statistics.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="this will print every step of the system.")
    args = parser.parse_args()
    trec_path = args.trec_path
    index_type = args.index_type
    out_dir = args.output_dir
    n_gram = args.ngram
    mmry_cstrt = args.memory_constraint
    noparse = args.noparse
    nostats = args.nostats
    verbose = args.verbose
    
    if os.path.isdir(out_dir) is False:
        os.mkdir(out_dir)
    index_builder = IndexBuilder(trec_path, index_type, out_dir, n_gram, mmry_cstrt,
                                 noparse, nostats, verbose)
    index_builder.build_index()


class IndexBuilder:
    def __init__(self, trec_path, index_type, out_dir, n_gram,
                 mmry_cstrt, noparse, nostats, verbose):
        self.trec_path = trec_path
        self.index_type = index_type
        self.out_dir = out_dir
        self.n_gram = n_gram
        self.mmry_cstrt = mmry_cstrt
        self.nostats = nostats
        self.noparse = noparse
        self.verbose = verbose

    def build_index(self):

        """
        parse, preprocess, tokenize, and generate document-by-document files from raw files
        """
        print("\ncreating documents")
        start_parse = time.time()
        if self.noparse == False:
            self._generate_doc_files()
        finish_parse = time.time()
        parse_time = (finish_parse - start_parse) * 1000
        print(f"done: {parse_time*1000} milliseconds")
    
        """
        build index
        """
        print("\nbuilding indices")
        start_temp = time.time()
        doc_map_path = self.out_dir + r"\\" + "doc_map.txt"
        if self.index_type == "single":
            if os.path.isfile(doc_map_path) is False:
                inv_ind = self._build_single()
                self._write_doc_map(doc_map_path, inv_ind.doc_map)
            else:
                self._build_single()
        elif self.index_type == "stem":
            self._build_stem()
        elif self.index_type == "phrase":
            self._build_phrase()
        elif self.index_type == "positional":
            self._build_positional()
        else:
            print("index type has to be one of the following: 'single', 'stem', 'phrase', and 'positional'")
        finish_temp = time.time()
        temp_time = finish_temp - start_temp
        print(f"done: {temp_time*1000} milliseconds")
        
    
        """
        m-way merge
        """
        print("\nmerging posting lists")
        start_merge = time.time()
        if len(os.listdir(self.out_dir + r"\\" + "pls")) == 1:
            if self.index_type == "phrase":
                shutil.copy(self.out_dir + r"\\pls\\pl_0.txt",
                            self.out_dir + r"\\" + f"{self.n_gram}_gram_posting_list.txt")
            else:
                shutil.copy(self.out_dir + r"\\pls\\pl_0.txt",
                            self.out_dir + r"\\" + f"{self.index_type}_posting_list.txt")
        else:
            if self.index_type == "positional":
                merged = self._m_way_merge_positional()
            else:
                merged = self._m_way_merge()
            if self.index_type == "phrase":
                with open(self.out_dir + r"\\" + f"{self.n_gram}_gram_posting_list.txt", "w") as out:
                    for pl in merged:
                        out.write("\t".join([str(item) for item in pl]) + "\n")
            else:
                with open(self.out_dir + r"\\" + f"{self.index_type}_posting_list.txt", "w") as out:
                    for pl in merged:
                        out.write("\t".join([str(item) for item in pl]) + "\n")     
        finish_merge = time.time()
        merge_time = finish_merge - start_merge
        print(f"done: {merge_time*1000} milliseconds")
    
        """
        filtering collection frequency for phrase index
        """
        if self.index_type == "phrase":
            print("\nfiltering phrases")
            self._cf_filter()
            print("done")
        
        finish_time = time.time()
        total_time = finish_time - start_parse

        """
        printing staistics
        """
        print("\nputting stats together")
        if self.nostats == False:
            self._print_stats()

        print(f"the total time: {total_time*1000} milliseconds")

    def _generate_doc_files(self):
        files = os.listdir(self.trec_path)
        for i, file in enumerate(files):
            if self.verbose:
                print(f"processing {i+1}/{len(files)}")
            file_path = self.trec_path + r"\\" + file
            file2doc.doc_writer(file_path, self.out_dir)
    
    def _build_single(self):
        n = 0  # to keep track of the number of intermediate posting list files
        doc_dir = self.out_dir + r"\\" + "doc_files"
        doc_files = os.listdir(doc_dir)
        inv_ind = inverted_index.InvertedIndex(mmry_cstrt=self.mmry_cstrt)
        for i, doc_file in enumerate(doc_files):
            if self.verbose:
                print(f"processing {i+1}/{len(doc_files)}")
            inv_ind.read_doc(doc_file)
            tokens = self._generate_line(doc_dir + r"\\" + doc_file)
            doc_length = 0
            for token in tokens:
                doc_length += 1
                if re.search(r"[a-zA-Z0-9]", token) is not None:                
                    inv_ind.update_index(token.strip('\n'))
            inv_ind.doc_map[inv_ind.did].append(doc_length)
            if inv_ind.is_above_constraint():
                self._output(inv_ind.posting_list, n)
                n += 1
                inv_ind.posting_list = set()
                inv_ind.store_pl()
            else:
                inv_ind.store_pl()
                if i == len(doc_files) - 1:
                    self._output(inv_ind.posting_list, n)
                    inv_ind.posting_list = set()
        self._write_lexicon(self.out_dir + r"\\" + "single_lexicon.txt", inv_ind.lexicon)
        return inv_ind

    def _build_stem(self):
        n = 0  # to keep track of the number of intermediate posting list files
        doc_dir = self.out_dir + r"\\" + "doc_files"
        doc_files = os.listdir(doc_dir)
        inv_ind = inverted_index.InvertedIndex(mmry_cstrt=self.mmry_cstrt, stem=True)
        for i, doc_file in enumerate(doc_files):
            if self.verbose:
                print(f"processing {i+1}/{len(doc_files)}")
            inv_ind.read_doc(doc_file)
            tokens = self._generate_line(doc_dir + r"\\" + doc_file)
            for token in tokens:
                if re.search(r"[a-zA-Z0-9]", token) is not None:                
                    inv_ind.update_index(token.strip('\n'))
            if inv_ind.is_above_constraint():
                self._output(inv_ind.posting_list, n)
                n += 1
                inv_ind.posting_list = set()
                inv_ind.store_pl()
            else:
                inv_ind.store_pl()
                if i == len(doc_files) - 1:
                    self._output(inv_ind.posting_list, n)
                    inv_ind.posting_list = set()
        self._write_lexicon(self.out_dir + r"\\" + "stem_lexicon.txt", inv_ind.lexicon)
        return inv_ind
    
    
    def _build_positional(self):
        n = 0  # to keep track of the number of intermediate posting list files
        doc_dir = self.out_dir + r"\\" + "doc_files"
        doc_files = os.listdir(doc_dir)
        inv_ind = inverted_index.InvertedIndex(mmry_cstrt=self.mmry_cstrt, exclude_stop_words=False)
        for i, doc_file in enumerate(doc_files):
            if self.verbose:
                print(f"processing {i+1}/{len(doc_files)}")
            inv_ind.read_doc(doc_file)
            tokens = self._generate_line(doc_dir + r"\\" + doc_file)
            for j, token in enumerate(tokens):
                if re.search(r"[a-zA-Z0-9]", token) is not None:
                    inv_ind.update_positional_index(j, token.strip('\n'))
            if inv_ind.is_above_constraint():
                self._output(inv_ind.posting_list, n)
                n += 1
                inv_ind.posting_list = set()
                inv_ind.store_positional_pl()
            else:
                inv_ind.store_positional_pl()
                if i == len(doc_files) - 1:
                    self._output(inv_ind.posting_list, n)
                    inv_ind.posting_list = set()
        self._write_lexicon(self.out_dir + r"\\" + "positional_lexicon.txt", inv_ind.lexicon)
        return inv_ind
    
    def _build_phrase(self):
        n = 0  # to keep track of the number of intermediate posting list files
        doc_dir = self.out_dir + r"\\" + "doc_files"
        doc_files = os.listdir(doc_dir)
        inv_ind = inverted_index.InvertedIndex(mmry_cstrt=self.mmry_cstrt)
        for i, doc_file in enumerate(doc_files):
            if self.verbose:
                print(f"processing {i+1}/{len(doc_files)}")
            inv_ind.read_doc(doc_file)
            window = []
            with open(doc_dir + r"\\" + doc_file) as tokens:
                token = "dummy"  # just so that while loop can start
                while token:
                    while len(window) < self.n_gram:
                        token = tokens.readline()
                        window.append(token.strip("\n"))
                    is_phrase = True
                    for j, item in enumerate(window):
                        if re.search(r"[a-zA-Z0-9]", item) is None or item in stop_words:
                            is_phrase = False
                            window = window[j+1:]
                            break
                    if is_phrase and len(window) == self.n_gram:
                        phrase = " ".join(window)
                        inv_ind.update_index(phrase)
                        window = window[1:]
            if inv_ind.is_above_constraint():
                self._output(inv_ind.posting_list, n)
                n += 1
                inv_ind.posting_list = set()
                inv_ind.store_pl()
            else:
                inv_ind.store_pl()
                if i == len(doc_files) - 1:
                    self._output(inv_ind.posting_list, n)
                    inv_ind.posting_list = set()
        self._write_lexicon(self.out_dir + r"\\" + f"{self.n_gram}_gram_lexicon.txt", inv_ind.lexicon)
        return inv_ind
    
    def _generate_line(self, file):
        with open(file) as f:
            for line in f:
                yield line
    
    def _output(self, pl_list, n):
        pl_list = list(pl_list)
        pl_list.sort(key=lambda x: x[1])
        pl_list.sort()
        pl_dir = self.out_dir + r"\\" + "pls"
        out_path = pl_dir + r"\\" + f"pl_{n}.txt"
        if os.path.isdir(pl_dir) is False:
            os.mkdir(pl_dir)
        with open(out_path, "w") as out:
            for pl in pl_list:
                out.write("\t".join([str(item) for item in pl]) + "\n")
    
    def _write_lexicon(self, out_path, lexicon):
        tids = list(lexicon.values())
        terms = list(lexicon.keys())
        mapped = zip(tids, terms)
        with open(out_path, "w") as out:
            for pair in mapped:
                out.write("\t".join([str(item) for item in pair]) + "\n")
            
    def _pl_to_list(self, pl):
        pl_list = []
        for tid in pl:
            for did in pl[tid]:
                tf = pl[tid][did]
                pl_list.append((tid, did, tf))
        return pl_list
    
    def _m_way_merge(self):
        pl_dir = self.out_dir + r"\\" + "pls"
        itrmdt_pls = os.listdir(pl_dir)
        name_orders = [(int(itrmdt_pl.strip(".txt").split("_")[1]), itrmdt_pl) for itrmdt_pl in itrmdt_pls]
        name_orders.sort()
        itrmdt_pls = [name_order[1] for name_order in name_orders]
        temps = []
        for itrmdt_pl in itrmdt_pls:
            temps.append(open(pl_dir + r"\\" + itrmdt_pl))
        total = len(temps)
        m_lines = []
        final = []
        min_index = None
        min_tuple = None
        while temps:
            if min_index is None:  # the first iteration       
                for i, temp in enumerate(temps):
                    line = temp.readline().strip("\n").split("\t")
                    if len(line) == 3:
                        line = [int(item) for item in line]
                        m_lines.append(line)
            else:  # the second iteration and onward
                for i, temp in enumerate(temps):
                    if i == min_index:
                        line = temp.readline().strip("\n").split("\t")
                        if len(line) == 3:
                            line = [int(item) for item in line]
                            m_lines.insert(i+1, line)
                            m_lines.pop(i)
                        else:
                            m_lines.pop(i)
                            temps[i].close()
                            temps.pop(i)
                            os.remove(pl_dir + r"\\" + itrmdt_pls[i])
                            itrmdt_pls.pop(i)
                            if self.verbose:
                                print(f"{len(temps)}/{total} left!")
                        break
            if len(temps) > 0:
                m_tids = [int(line[0]) for line in m_lines]
                min_index = m_tids.index(min(m_tids))
                min_tuple = m_lines[min_index]
                final.append(min_tuple)
        return final
    
    def _m_way_merge_positional(self):
        pl_dir = self.out_dir + r"\\" + "pls"
        itrmdt_pls = os.listdir(pl_dir)
        name_orders = [(int(itrmdt_pl.strip(".txt").split("_")[1]), itrmdt_pl) for itrmdt_pl in itrmdt_pls]
        name_orders.sort()
        itrmdt_pls = [name_order[1] for name_order in name_orders]
        temps = []
        for itrmdt_pl in itrmdt_pls:
            temps.append(open(pl_dir + r"\\" + itrmdt_pl))
        total = len(temps)
        m_lines = []
        final = []
        min_index = None
        min_tuple = None
        while temps:
            if min_index is None:  # the first iteration       
                for i, temp in enumerate(temps):
                    raw_line = temp.readline().strip("\n").split("\t")
                    if len(raw_line)== 3:
                        line = [int(item) for item in raw_line[:-1]]
                        line.append(tuple(int(item) for item in raw_line[-1].strip("(),").split(",")))
                        m_lines.append(line)
            else:  # the second iteration and onward
                for i, temp in enumerate(temps):
                    if i == min_index:
                        raw_line = temp.readline().strip("\n").split("\t")
                        if len(raw_line) == 3:
                            line = [int(item) for item in raw_line[:-1]]
                            line.append(tuple(int(item) for item in raw_line[-1].strip("(),").split(",")))
                            m_lines.insert(i+1, line)
                            m_lines.pop(i)
                        else:
                            m_lines.pop(i)
                            temps[i].close()
                            temps.pop(i)
                            os.remove(pl_dir + r"\\" + itrmdt_pls[i])
                            itrmdt_pls.pop(i)
                            if self.verbose:
                                print(f"{len(temps)}/{total} left!")
                        break
            if len(temps) > 0:
                m_tids = [line[0] for line in m_lines]
                min_index = m_tids.index(min(m_tids))
                min_tuple = m_lines[min_index]
                final.append(min_tuple)
        return final
    
    def _cf_filter(self, threshold=1):
        lexicon = dict()
        posting_list = dict()
        lexs_path = (self.out_dir + r"\\" + f"{self.n_gram}_gram_lexicon.txt")
        pls_path = (self.out_dir + r"\\" + f"{self.n_gram}_gram_posting_list.txt")
        prev_tid = 0
        pls = self._generate_line(pls_path)
        for pl in pls:
            line = [int(item) for item in pl.strip("\n").split("\t")]
            tid = line[0]
            if tid != prev_tid:
                posting_list[tid] = []
            posting_list[tid].append(tuple(line[1:]))
            prev_tid = tid
        lexs = self._generate_line(lexs_path)    
        for lex in lexs:
            line = lex.strip("\n").split("\t")
            lexicon[int(line[0])] = line[1]
        cf_list = [sum(pl[1] for pl in posting_list[tid]) for tid in posting_list]
        filtered = set([i+1 for i in range(len(cf_list)) if cf_list[i] >= 2])
        filtered_posting_list = [tuple(posting_list[tid]) for tid in posting_list if tid in filtered]
        filtered_lexicon = [lexicon[tid] for tid in lexicon if tid in filtered]
        with open(lexs_path, "w") as out:
            for i, lex in enumerate(filtered_lexicon):
                line = [str(i+1)] + [lex]
                out.write("\t".join(line) + "\n")        
        with open(pls_path, "w") as out:
            for i, pl in enumerate(filtered_posting_list):
                for doc in pl:                
                    line = [str(i+1)] + [str(item) for item in doc]
                    out.write("\t".join(line) + "\n")

    def _write_doc_map(self, out_path, doc_map):
        dids = list(doc_map.keys())
        doc_names = [item[0] for item in doc_map.values()]
        doc_lengths = [item[1] for item in doc_map.values()]
        mapped = zip(dids, doc_names, doc_lengths)
        with open(out_path, "w") as out:
            for triplet in mapped:
                out.write("\t".join([str(item) for item in triplet]) + "\n")

    def _print_stats(self):
        lexicon = dict()
        posting_list = dict()
        prev_tid = 0
        if self.index_type == "positional":
            pls = self._generate_line(self.out_dir + r"\\" + f"{self.index_type}_posting_list.txt")
            lexs = self._generate_line(self.out_dir + r"\\" + f"{self.index_type}_lexicon.txt")
            for pl in pls:
                line = pl.strip("\n").split("\t")
                tf = len([position for position in line[2].split(",") if len(position) > 0]) # omitting empty string
                tid = int(line[0])
                if tid != prev_tid:
                    posting_list[tid] = []
                posting_list[tid].append((int(line[1]), tf))
                prev_tid = tid
        else:
            if self.index_type == "phrase":
                pls = self._generate_line(self.out_dir + r"\\" + f"{self.n_gram}_gram_posting_list.txt")
                lexs = self._generate_line(self.out_dir + r"\\" + f"{self.n_gram}_gram_lexicon.txt")
            else:
                pls = self._generate_line(self.out_dir + r"\\" + f"{self.index_type}_posting_list.txt")
                lexs = self._generate_line(self.out_dir + r"\\" + f"{self.index_type}_lexicon.txt")        
            for pl in pls:
                line = [int(item) for item in pl.strip("\n").split("\t")]
                tf = line[2]
                tid = line[0]
                if tid != prev_tid:
                    posting_list[tid] = []
                posting_list[tid].append(tuple(line[1:]))
                prev_tid = tid

        for lex in lexs:
            line = lex.strip("\n").split("\t")
            lexicon[int(line[0])] = line[1]
        df_list = [len(posting_list[key]) for key in posting_list]
        min_df = min(df_list)
        max_df = max(df_list)
        max_df_term = lexicon[df_list.index(max_df) + 1]
        mean_df = sum(df_list)/len(df_list)
        med_df = df_list[len(df_list)//2]
        print(f"Lexicon:\t{len(lexicon)}")
        print(f"Max df:\t\t{max_df} ({max_df_term})")
        print(f"Min df:\t\t{min_df}")
        print(f"Mean df:\t{round(mean_df, 3)}")
        print(f"Median df:\t{med_df}")

if __name__ == '__main__':
	main()