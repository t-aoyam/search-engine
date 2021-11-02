import re
import os

"""
pre-defined set of vocabulary to be used later
"""
month = {"january" : "01", "february" : "02", "march" : "03", "april" : "04",
          "may" : "05", "june" : "06", "july" : "07", "august" : "08",
          "september" : "09", "october" : "10", "november": "11", "december" : "12"}
mon = {"jan" : "01", "feb" : "02", "mar" : "03", "apr" : "04", "jun" : "06",
       "jul" : "07", "aug" : "08", "sep" : "09", "oct" : "10", "nov" : "11", "dec" : "12"}
prefixes = {"pre", "post", "anti", "over", "re", "co", "extra", "hyper",
            "auto", "inter", "intra", "non", "dis", "mono", "multi", "trans", "up"}

def doc_writer(input_path, out_dir):
    """
    put all internal functions below together here, so that this function
    can be called and used from outside.
    """
    doc_dir = out_dir + r"\\doc_files"
    if os.path.isdir(doc_dir) is False:
        os.mkdir(doc_dir)
    lines = _generate_line(input_path)
    docs = _doc_generator(lines)
    for doc in docs:
        doc_id = doc[0][0]
        doc_text = " ".join(doc[1])
        out_path = doc_dir + r"\\" + doc_id + r".txt"
        doc = Doc(doc_id, doc_text)
        tokens = doc._tokenization()
        with open(out_path, "w") as out:
            for token in tokens:
                tokenizer = Tokenizer(token)
                for lexeme in tokenizer.lexemes:
                    if len(lexeme) > 0:
                        out.write(lexeme + "\n")

def _generate_line(file):
    with open(file) as f:
        for line in f:
            yield line

def _doc_generator(lines):
    """
    generator function that reads the raw TREC files
    and yileds raw text file of each document
    """
    escape = {"&blank;" : " ", "&hyph;" : "-", "&sect;" : "§", "&times;" : "×", "&para;" : "¶"}
    text = False
    doc = ([],[])
    for line in lines:
        if line.startswith("<DOCNO>"):
            match = re.match(r"<DOCNO>(.+)</DOCNO>", line)
            doc_id = match[1].strip()
            doc[0].append(doc_id)
        if text:
            if re.search(r"(<!--.*-->)", line) is None:
                if re.search(r"<.+>(.+)</.+>", line) is not None:
                    match = re.search(r"<.+>(.+)</.+>", line)
                    line = match[1]

                for key in escape:
                    if key in line:
                        line = re.sub(key, escape[key], line)
                if re.search(r"[^ ].+", line) is not None:
                    line = re.sub("\n", " ", line)
                    doc[1].append(line)
        if line.startswith(r"<TEXT>"):
            text = True
        elif line.startswith(r"</TEXT>"):
            text = False
        elif line.startswith(r"</DOC>"):
            yield doc
            doc = ([],[])


class Tokenizer:
    """
    Each word will be instantiated as this Class Tokenizer,
    so that any special pattern can be identified at word-level.
    This class will essentially take a word and return normalized
    version of the word in list, because some words (e.g., compounds)
    have to be tokenized as multiple words.
    """
    def __init__(self, token):
        self.token = token
        self.lexemes = []

        begins_with_special = re.search(r"^([^a-zA-Z0-9])", self.token)
        ends_with_special = re.search(r"([^a-zA-Z0-9])$", self.token)
        if begins_with_special is not None:
            self.token = self.token[1:]
            self.lexemes.append(begins_with_special[0])
        if ends_with_special is not None:
            self.token = self.token[:-1]

        normalized = self._special()
        for item in normalized:
            self.lexemes.append(item)

        if ends_with_special is not None:
            self.lexemes.append(ends_with_special[0])

    def _clean_up(self, token):
        return re.sub(r"[^a-zA-Z0-9/]", "", token).lower().strip("/")

    def _special(self):
        normalized = []

        if re.search(r"[^a-zA-Z]", self.token) is None:  # Express Lane- this is a normal token
            normalized.append(self.token.lower())

        elif re.search(r".+@([a-zA-Z]+\.)+[a-zA-Z]+", self.token) is not None:  # E-mail address
            match = re.search(r".+@([a-zA-Z]+\.)+[a-zA-Z]+", self.token)
            normalized.append(match[0].lower())

        elif re.search(r"https?://www\..+", self.token) is not None:  # URL
            match = re.search(r"https?://www\..+", self.token)
            normalized.append(match[0].lower())

        elif re.search(r"([0-9]{1,3}\.){3}([0-9]{1,3})", self.token) is not None:  # IP address
            match = re.search(r"([0-9]{1,3}\.){3}([0-9]{1,3})", self.token)
            normalized.append(match[0].lower())

        elif re.search(r"([a-zA-Z]+)-([0-9]+)", self.token) is not None:  # Alphabet-digit combination
            match = re.search(r"([a-zA-Z]+)-([0-9]+)", self.token)
            num_alpha = len(match[1])
            normalized.append(self._clean_up(re.sub(r"-", "", self.token)))
            if num_alpha >= 3:
                normalized.append(self._clean_up(match[1]))

        elif re.search(r"([0-9]+)-([a-zA-Z]+)", self.token) is not None:  # Digit-alphabet combination
            match = re.search(r"([0-9]+)-([a-zA-Z]+)", self.token)
            num_alpha = len(match[2])
            normalized.append(self._clean_up(re.sub(r"-", "", self.token)))
            if num_alpha >= 3:
                normalized.append(self._clean_up(match[2]))

        elif re.search(r"([$£€¥][0-9,]+(\.[0-9]+)?)", self.token) is not None:  # money
            normalized.append(self._clean_up(re.sub(r"[$£€¥]", "", self.token)))

        elif re.search(r"\b([a-zA-Z]{1,2}\.([a-zA-Z]\.?)+)(?!\s[A-Z])",
                     self.token) is not None:  # acronyms
            normalized.append(self._clean_up(re.sub(r"\.", "", self.token)))

        elif re.search(r"([a-zA-Z]+-)+([a-zA-Z]+)", self.token) is not None:  # Hyphenated words
            match = re.search(r"([a-zA-Z]+-)+([a-zA-Z]+)", self.token)
            if match is not None:
                parts = self.token.split("-")
                whole = self._clean_up("".join(parts))
                normalized.append(whole)
                for part in parts:
                    if part not in prefixes:
                        normalized.append(self._clean_up(part))

        elif re.search(r"([a-zA-Z]+/)([a-zA-Z]+)", self.token) is not None:  # Slash-separated words
            match = re.search(r"([a-zA-Z]+/)([a-zA-Z]+)", self.token)
            parts = self.token.split("/")
            for part in parts:
                normalized.append(self._clean_up(part))

        elif re.search(r"(.+)?\.([a-zA-Z0-9]{2,4})", self.token) is not None:  # File type
            match = re.search(r"(.+)?\.([a-zA-Z0-9]{2,4})", self.token)
            if match[1] is not None:
                normalized.append(self._clean_up(match[1]))
            if match[2] is not None:
                normalized.append(self._clean_up(match[2]))
        else:
            normalized.append(self._clean_up(self.token))

        return normalized


class Doc:
    def __init__(self, doc_id, text):
        self.id = doc_id
        self.text = self._nrmlz_digit(self._nrmlz_date(text))

    def _tokenization(self):
        tokens = self.text.split()
        del self.text
        for token in tokens:
            yield token

    def _nrmlz_digit(self, doc):
        def no_comma(match):
            if match.group() is not None:
                return re.sub(r",", "", match[0])
        def no_decimal(match):
            if match.group() is not None:
                return re.sub(r"\.0+", "", match[0])
        doc = re.sub(r"\b([0-9]+,)+(\.[0-9]+)?\b", no_comma, doc)
        doc = re.sub(r"\b(?<!\.)([0-9]+,?)+(\.0{1,3})\b", no_decimal, doc)
        return doc

    def _nrmlz_date(self, doc):

        def date_formatting(mdy):
            month = mdy[0]
            if len(month) == 1:
                month = "0" + month
            day = mdy[1]
            if len(day) == 1:
                day = "0" + day
            year = mdy[2]
            if len(year) == 2:
                if int(year) >= 22:
                    year = "19" + year
                else:
                    year = "20" + year
            return "/".join([month, day, year])
            
        # MM/DD/YY & MM/DD/YYYY
        pattern_1 = re.compile(r"(([01]?[0-9])/([0-3]?[0-9])/([0-9]{2,4}))")
        matches_1 = re.findall(pattern_1, doc)
        if matches_1 is not None:
            for match_1 in matches_1:        
                mdy = [match_1[1], match_1[2], match_1[3]]
                normalized = date_formatting(mdy)
                doc = re.sub(match_1[0], normalized, doc)
    
        # MM-DD-YY & MM-DD-YYYY
        pattern_2 = re.compile(r"(([01]?[0-9])-([0-3]?[0-9])-([0-9]{2,4}))")
        matches_2 = re.findall(pattern_2, doc)
        if matches_2 is not None:        
            for match_2 in matches_2:
                mdy = [match_2[1], match_2[2], match_2[3]]
                normalized = date_formatting(mdy)
                doc = re.sub(match_2[0], normalized, doc)
    
        # MMM/DD/YY & MMM/DD/YYYY
        pattern_3 = re.compile(r"(([a-zA-Z]{3})-([0-3]?[0-9])-([0-9]{2,4}))")
        matches_3 = re.findall(pattern_3, doc)
        if matches_3 is not None:
            for match_3 in matches_3:
                if match_3[1].lower() in set(mon):            
                    mdy = [mon[match_3[1].lower()], match_3[2], match_3[3]]
                    normalized = date_formatting(mdy)
                    doc = re.sub(match_3[0], normalized, doc)
    
        # Month Name, DD, YYYY
        pattern_4 = re.compile(r"(([a-zA-Z]{3,9}),?\s?([0-3]?[0-9]),?\s?([0-9]{2,4}))")
        matches_4 = re.findall(pattern_4, doc)
        if matches_4 is not None:
            for match_4 in matches_4:
                if match_4[1].lower() in set(month):
                    mdy = [month[match_4[1].lower()], match_4[2], match_4[3]]
                    normalized = date_formatting(mdy)
                    doc = re.sub(match_4[0], normalized, doc)
        return doc