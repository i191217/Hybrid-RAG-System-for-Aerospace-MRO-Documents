#!/usr/bin/env python3
"""
aero_spellcheck.py

A script to spell-correct and clean aerospace-domain text using a
custom JSON dictionary of standard terms, with:
 - No spaCy dependency
 - Data-driven phrase protection
 - Hyphens and slashes preserved (e.g. in-house, HMU/FMU)
 - Heuristic splitting only for unknown long tokens
"""

import re, sys, json, unicodedata, argparse
from rapidfuzz import process, fuzz
from spellchecker import SpellChecker

def build_lookup(dict_path):
    with open(dict_path, encoding="utf-8") as f:
        data = json.load(f)
    lookup = {}
    for e in data:
        std = str(e.get("standard_spelling") or e.get("term") or "").strip()
        if not std:
            continue
        variants = {
            e.get("term",""), e.get("acronym",""),
            *e.get("aliases",[]), *e.get("common_misspellings",[])
        }
        for v in variants:
            if not v:
                continue
            s = str(v).strip()
            lookup[s.lower()] = std
    return lookup

def normalize_unicode(text):
    text = unicodedata.normalize("NFKC", text)
    return (text
        .replace("—","-").replace("–","-")
        .replace("•"," ").replace(""," ")
    )

def fix_hyphens(text):
    return re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

def protect_phrases(text, lookup):
    phrases = sorted([k for k in lookup if " " in k], key=lambda x: -len(x))
    for ph in phrases:
        rx = re.compile(rf"\b{re.escape(ph)}\b", flags=re.IGNORECASE)
        text = rx.sub(lambda m: m.group(0).replace(" ", "_"), text)
    return text

def tokenize(text):
    # include letters, numbers, underscores, hyphens, apostrophes, slashes
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-'/]*", text)

def try_split(tok, lookup, spell):
    low = tok.lower()
    # if token is already valid (domain or English), don't split
    if low in lookup or low in spell.known([low]):
        return [tok]
    n = len(tok)
    if n < 8:
        return [tok]
    # try all splits leaving at least 3 chars each side
    for i in range(3, n-3):
        a, b = low[:i], low[i:]
        if ((a in lookup or a in spell.known([a])) and
            (b in lookup or b in spell.known([b]))):
            return [tok[:i], tok[i:]]
    return [tok]

def correct_and_filter(tokens, lookup, spell, threshold=90):
    cleaned = []
    keys = list(lookup.keys())

    for tok in tokens:
        # 1) protected phrase
        if "_" in tok:
            orig = tok.replace("_", " ")
            corr = lookup.get(orig.lower(), orig)

        # 2) skip hyphen or slash tokens
        elif "-" in tok or "/" in tok:
            corr = tok

        # 3) otherwise correct
        else:
            norm = tok.lower()
            if norm in lookup:
                corr = lookup[norm]
            else:
                best = process.extractOne(norm, keys,
                                          scorer=fuzz.ratio,
                                          score_cutoff=threshold)
                if best:
                    corr = lookup[best[0]]
                else:
                    cand = spell.correction(norm) or tok
                    corr = lookup.get(cand.lower(), cand)

        # 4) attempt split only for unknown-long tokens
        pieces = try_split(corr, lookup, spell)

        # 5) filter out garbage
        for p in pieces:
            if re.fullmatch(r"[\W_]+", p):
                continue
            if len(p) <= 2 and p.lower() not in lookup:
                continue
            cleaned.append(p)

    return cleaned

def process_text(raw, lookup):
    raw = normalize_unicode(raw)
    raw = fix_hyphens(raw)
    raw = protect_phrases(raw, lookup)
    toks = tokenize(raw)
    spell = SpellChecker()
    clean = correct_and_filter(toks, lookup, spell)
    return " ".join(clean)

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("-d","--dict",   required=True,
                   help="Path to aero_dict_enriched.json")
    p.add_argument("-i","--input",  required=True,
                   help="Text or JSON chunk file")
    p.add_argument("-o","--output", help="Where to write cleaned output")
    args = p.parse_args()

    if args.input.lower().endswith(".json"):
        j = json.load(open(args.input, encoding="utf-8"))
        raw = j.get("chunk_info", {}).get("content") or sys.exit("Bad JSON format")
    else:
        raw = open(args.input, encoding="utf-8").read()

    lookup = build_lookup(args.dict)
    out = process_text(raw, lookup)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out)

if __name__ == "__main__":
    main()
