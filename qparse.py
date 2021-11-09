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

def _generate_line(file):
    with open(file) as f:
        for line in f:
            yield line
