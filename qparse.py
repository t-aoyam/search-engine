import re

def query_parser(path):
    queries = []
    lines = _generate_line(path)
    for line in lines:
        if line.startswith("<num>"):
            match = re.search(r"Number: ([0-9]+)", line)
            query = [int(match[1])]
        elif line.startswith("<title>"):
            match = re.search(r"Topic: (.+)", line)
            query.append(re.sub(r"[\n\t\s]$", "", match[1]))
            query = tuple(query)
            queries.append(query)
    return queries

def query_parser_long(path):
    queries = []
    lines = _generate_line(path)
    narr = False
    for line in lines:
        if line.startswith("<num>"):
            match = re.search(r"Number: ([0-9]+)", line)
            query = [int(match[1])]
            text = ""
        elif line.startswith("</top>"):
            narr = False
            text = re.sub(r"[\n\t]", r" ", text)
            query.append(text)
            query = tuple(query)
            queries.append(query)
        if narr:            
            line = re.sub(r" +", " ", line)
            line = re.sub(r"[\n\t]$", "", line)
            line = re.sub(r"\n\t", r" ", line)
            text += line
        elif line.startswith("<narr>"):
            narr = True
    return queries

def _generate_line(file):
    with open(file) as f:
        for line in f:
            yield line

from pathlib import Path
fp = Path("data/queryfile.txt")
test = query_parser_long(fp)