# search-engine

## Index Building

- `index_builder.py` will parse raw text files, segment them into individual documents, and create lexicon and posting list files.
- usage: `python index_builder.py [trec-files-directory-path] [index-type] [output-dir] [--ngram] [--memory_constraint] [--noparse] [--nostats] [--verbose]`
  - `[trec-files-directory-path]`: path to the directory that contains raw text files
  - `[index-type]`: index type. `single`, `stem`, `positional`, or `phrase`
  - `[output-dir]`: path to the directory where the generated lexicon and posting list files will be written out.
  - `[--ngram] [-n]`: _n_ for n-gram in phrase index. default: 2.
  - `[--memory_constraint] [-m]`: memory constraint. default: 10,000.
  - `[--noparse]`: an option _not to_ parse raw text files (useful once the individual document files are created).
  - `[--nostats]`: an option _not to_ print the statistics of the generated lexicon and posting list files.
  - `[--verbose]`: an option to print out the status of the process.
  - For more details, run: `python index_builder.py --help`

- example: `python index_builder.py data\trec_raw phrase data\index  -n 3 -m 10000 --noparse --nostats --verbose`
  - This will run the program that builds 3-phrase index with the memory constraint of 10,000, assuming that parsing has already been done, no printing of statistics at the end, while printing all the details of the process.

## Query Processing
### Static Query Processing
  - usage: `python query_static.py [index_directory_path] [query_file_pathh] [retrieval_model] [index_type] [results_file] [--verbose] [--evaluate]`
    - `[index_directory_path]`: path of the directory that contains all lexicon and posting list files.
    - `[query_file_path]`: path of the query file.
    - `[retrieval_model]`: retrieval model: `cosine`, `bm25`, or `lm`.
    - `[index_type]`: index type: `single` or `stem`.
    - `[results_file]`: path of output file, where the generated relevance ranking will be written out.
    - `[--verbose]`: an option to print every step of the process.
    - `[--evaluate]`: an option to print the evaluation using treceval.exe.
    - For more details, run: `python query_static.py --help`

  - example: `python  query_static.py  data\index  data\queryfile.txt  bm25  single  data\result.txt  --verbose –evaluate`
    - This will run the program that sends queries to single term index; retrieve documents using bm25 probabilistic model, and write the relevance ranking onto a file called result.txt. Also, this command will print details of the process and print the evaluation result on the terminal.

### Dynamic Query Processing
  - usage: `python query_dynamic.py [index_directory_path] [query_file_pathh] [results_file] [--ngram] [--window_size] [--retrieval_threshold] [--verbose] [--evaluate]`
    - `[index_directory_path]`: path of the directory that contains all lexicon and posting list files.
    - `[query_file_path]`: path of the query file.
    - `[results_file]`: path of output file, where the generated relevance ranking will be written out.
    - `[--ngram]`: _n_ for n-gram in phrase index. default: 2.
    - `[--window_size]`: window size for proximity search. default: 10.
    - `[--retrieval_threshold]`: retrieval threshold for phrase and proximity search. default: 20
    - `[--verbose]`: an option to print every step of the process.
    - `[--evaluate]`: an option to print the evaluation using treceval.exe.
    - For more details, run: `python query_static.py --help`

  - example: `python  query_dynamic.py  data\index  data\queryfile.txt  data\result.txt  -n 2 -w 5 -t 10 --verbose --evaluate`
    - This will run the dynamic query processing using 2-term phrase index and the proximity search with window size of 5, with the retrieval threshold value of 10. This will  write out the relevance ranking to a text file called “result.txt" and will also print out details of the process and the evaluation result onto the terminal.

## Query Reformulation
  - usage: `python query_expansion_reduction.py [index_directory_path] [query_file_path] [retrieval_model] [index_type] [--n_doc] [--m_term] [--num_iter] [--q_threshold] [results_file] [--long] [--verbose] [--evaluate] [--show]`
    - `[index_directory_path]`: path of the directory that contains all lexicon and posting list files.
    - `[query_file_path]`: path of the query file.
    - `[retrieval_model]`: retrieval model: `cosine`, `bm25`, or `lm`.
    - `[index_type]`: index type: `single` or `stem`.
    - `[results_file]`: path of output file, where the generated relevance ranking will be written out.
    - `[--n_doc] [-n]`: top _n_ documents to retrieve expansion terms from. default is 10.
    - `[--m_term] [-m]`: top _m_ terms to expand the query terms by. default is 3.
    - `[--num_iter] [-i]`: number of query expansion iteration. default is 0 (no expansion).
    - `[--q_threshold] [-q]`: top 100*_p_ % of the query terms will be used. default is 1 (no reduction).
    - `[--long]`: an option to use long query. default is False (short query).
    - `[--verbose]`: an option to print every step of the process.
    - `[--evaluate]`: an option to print the evaluation using treceval.exe.
    - `[--show]`: this will show what query term was added and/or removed.
    - For more details, run: `python query_static.py --help`
    - For expansion-only model with short query (Report 1), make sure to specify `-q 1` (no reduction) and to exclude `--long` (short query). For reduction-only model with long query (Report 2), make sure to specify `-i 0` (expansion) and to specify `--long` (long query).
  - example: `python  query_expansion_reduction.py  data\index  data\queryfile.txt cosine single data\result.txt  -n 4 -m 17 -i 1 -q 0.4 --long --verbose --evaluate --show`
    - This will run the program that sends queries to single term index; retrieve documents using cosine measure, and write the relevance ranking onto a file called result.txt. The query terms will be first reduced to top 40% (-q 0.4), and one iteration (-i 1) of query expansion will be performed using top 4 retrieved documents (-n 4), adding top 17 terms (-m 17) to the original query. Also, this command will print details of the process and print the evaluation result on the terminal, while showing sets of original, removed, and added terms on the terminal.
