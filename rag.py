"""A small document question-answering system (RAG) in plain Python.

Usage:
    python3 rag.py chunks               show every passage the documents are split into
    python3 rag.py search "question"    show the passages retrieved for a question
    python3 rag.py ask "question"       answer a question, with its source
"""

import json
import math
import re
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs"

# ---------------------------------------------------------------------------
# Step 2: chunking
# ---------------------------------------------------------------------------

SENTENCES_PER_CHUNK = 3  # how many sentences go into one passage
SENTENCE_OVERLAP = 1     # how many sentences a passage shares with the next one


def load_documents():
    """Read docs/sources.json and the text file of every document listed in it."""
    sources = json.loads((DOCS_DIR / "sources.json").read_text(encoding="utf-8"))
    for source in sources:
        source["text"] = (DOCS_DIR / f"{source['doc_id']}.txt").read_text(encoding="utf-8")
    return sources


def split_sentences(paragraph):
    """Split a paragraph into sentences.

    A sentence ends at . ! or ? when it is followed by a space and then a
    capital letter or an opening quote. That rule keeps numbers such as
    "5.20 AU" in one piece, because the dot there is followed by a digit.
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', paragraph.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def make_chunks(documents):
    """Cut every document into small overlapping passages ("chunks").

    Each chunk is a few neighbouring sentences from one paragraph. Neighbouring
    chunks share a sentence, so a fact that needs two sentences to make sense
    is not cut in half. Chunks never cross a paragraph border, because a new
    paragraph usually starts a new subject.
    """
    step = SENTENCES_PER_CHUNK - SENTENCE_OVERLAP
    chunks = []

    for document in documents:
        paragraphs = [line for line in document["text"].split("\n") if line.strip()]
        for paragraph_number, paragraph in enumerate(paragraphs, start=1):
            sentences = split_sentences(paragraph)
            chunk_number = 1
            for start in range(0, len(sentences), step):
                chunks.append({
                    # e.g. "mars-p3-c2" = mars document, paragraph 3, chunk 2
                    "chunk_id": f"{document['doc_id']}-p{paragraph_number}-c{chunk_number}",
                    "doc_id": document["doc_id"],
                    "title": document["title"],
                    "url": document["url"],
                    "text": " ".join(sentences[start:start + SENTENCES_PER_CHUNK]),
                })
                chunk_number += 1
                if start + SENTENCES_PER_CHUNK >= len(sentences):
                    break  # this chunk already reached the end of the paragraph

    return chunks


# ---------------------------------------------------------------------------
# Step 3: retrieval (BM25 keyword search)
# ---------------------------------------------------------------------------

TOP_K = 3  # how many passages to retrieve for a question

# Standard BM25 settings.
BM25_K1 = 1.5   # how quickly repeating a word stops adding to the score
BM25_B = 0.75   # how strongly long passages are penalised (0 = not at all, 1 = fully)

# Very common words that say nothing about the subject of a question.
STOPWORDS = set("""
a an and are as at be been by did do does for from had has have how in is it its
many much of on or than that the their there this to was were what when where which
who whom whose why will with
""".split())


def tokenize(text):
    """Turn text into a list of comparable keywords.

    "How many moons does Mars have?" -> ["moon", "mar"]

    Lower-case everything, keep letters and digits, drop stopwords, and cut a
    final "s" so that "moons" matches "moon". The cut is crude ("mars" becomes
    "mar") but harmless, because questions and passages are treated the same.
    """
    words = re.findall(r"[a-z0-9]+", text.lower().replace("'s", ""))
    tokens = []
    for word in words:
        if word in STOPWORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    return tokens


def build_index(chunks):
    """Prepare the numbers BM25 needs, once, before any question is asked."""
    # The document name ("mars") is added to the searchable words of each
    # chunk. Many sentences say "It" or "the planet" instead of the planet's
    # name, and without this those chunks could not be found by name.
    chunk_tokens = [tokenize(chunk["doc_id"] + " " + chunk["text"]) for chunk in chunks]

    # In how many chunks does each word appear?
    chunks_containing = {}
    for tokens in chunk_tokens:
        for word in set(tokens):
            chunks_containing[word] = chunks_containing.get(word, 0) + 1

    return {
        "chunks": chunks,
        "chunk_tokens": chunk_tokens,
        # keyword form of each document name -> document, e.g. "mar" -> "mars"
        "document_names": {tokenize(chunk["doc_id"])[0]: chunk["doc_id"] for chunk in chunks},
        "chunks_containing": chunks_containing,
        "average_length": sum(len(tokens) for tokens in chunk_tokens) / len(chunk_tokens),
    }


def word_rarity(index, word):
    """IDF: a word found in few chunks is worth more than a word found in many."""
    total = len(index["chunks"])
    containing = index["chunks_containing"].get(word, 0)
    return math.log(1 + (total - containing + 0.5) / (containing + 0.5))


def bm25_score(index, question_tokens, tokens):
    """Score one chunk against the question. Higher means more relevant."""
    relative_length = len(tokens) / index["average_length"]
    score = 0.0
    for word in set(question_tokens):
        count = tokens.count(word)
        if count == 0:
            continue
        # Grows with every repeat of the word, but less each time, and is
        # reduced for chunks that are longer than average.
        frequency = count * (BM25_K1 + 1) / (count + BM25_K1 * (1 - BM25_B + BM25_B * relative_length))
        score += word_rarity(index, word) * frequency
    return score


def search(index, question, top_k=TOP_K):
    """Return the top_k best-scoring chunks for a question, best first.

    If the question names one or more of the documents ("... on Mercury?"),
    only those documents are searched. Otherwise a passage about Venus that
    happens to share more words with the question can outrank the right one.
    """
    question_tokens = tokenize(question)
    keywords = set(question_tokens)
    named_documents = {index["document_names"][word] for word in keywords
                       if word in index["document_names"]}
    results = []
    for chunk, tokens in zip(index["chunks"], index["chunk_tokens"]):
        if named_documents and chunk["doc_id"] not in named_documents:
            continue
        score = bm25_score(index, question_tokens, tokens)
        if score > 0:
            matched = sorted(keywords & set(tokens))
            results.append({
                **chunk,
                "score": round(score, 2),
                "matched_words": matched,
                "missing_words": sorted(keywords - set(tokens)),
                # share of the question's keywords that this chunk contains
                "coverage": round(len(matched) / len(keywords), 2),
            })
    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Step 4: answering, with a citation, or declining to answer
# ---------------------------------------------------------------------------

# The best passage must contain at least this share of the question's
# keywords, otherwise we say the documents do not support an answer.
MIN_COVERAGE = 0.6

ANSWER_SENTENCES = 2  # the answer quotes at most this many sentences

NOT_SUPPORTED = "The documents do not support an answer to this question."


def pick_answer_sentences(index, question_tokens, chunk):
    """Choose the sentences of the passage that share the most with the question.

    Each sentence scores the summed rarity of the question keywords it
    contains. The best ones are returned in their original order.
    """
    # The passage's own document name is ignored here: in a passage about Mars
    # nearly every sentence says "Mars", so it does not help to choose one.
    keywords = set(question_tokens) - set(tokenize(chunk["doc_id"]))
    scored = []
    for position, sentence in enumerate(split_sentences(chunk["text"])):
        matched = keywords & set(tokenize(sentence))
        score = sum(word_rarity(index, word) for word in matched)
        if score > 0:
            scored.append((score, position, sentence))

    best = sorted(scored, key=lambda item: -item[0])[:ANSWER_SENTENCES]
    return [sentence for _, _, sentence in sorted(best, key=lambda item: item[1])]


def answer_question(index, question):
    """Answer a question from the documents, or decline.

    The answer is extractive: it is quoted word for word from the best
    passage, so the system cannot make anything up. It declines when the best
    passage is missing too many of the question's keywords.
    """
    retrieved = search(index, question)
    result = {
        "question": question,
        "answered": False,
        "answer": NOT_SUPPORTED,
        "reason": "",
        "citation": None,
        "retrieved": retrieved,
    }

    if not retrieved:
        result["reason"] = "No passage shares a keyword with the question."
        return result

    best = retrieved[0]
    if best["coverage"] < MIN_COVERAGE:
        result["reason"] = (f"The best passage [{best['chunk_id']}] contains only "
                            f"{best['coverage']:.0%} of the question's keywords "
                            f"(missing: {', '.join(best['missing_words'])}).")
        return result

    sentences = pick_answer_sentences(index, tokenize(question), best)
    result["answered"] = True
    result["answer"] = " ".join(sentences)
    result["reason"] = (f"The best passage contains {best['coverage']:.0%} "
                        f"of the question's keywords.")
    result["citation"] = {
        "chunk_id": best["chunk_id"],
        "title": best["title"],
        "url": best["url"],
        "passage": best["text"],
    }
    return result


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def show_answer(question):
    index = build_index(make_chunks(load_documents()))
    result = answer_question(index, question)
    print(f"Question: {question}\n")
    print(f"Answer:   {result['answer']}")
    print(f"Why:      {result['reason']}\n")
    if result["answered"]:
        citation = result["citation"]
        print(f"Source:   {citation['title']} - {citation['url']}")
        print(f"Passage:  [{citation['chunk_id']}] {citation['passage']}\n")
    print("Retrieved passages:")
    for rank, passage in enumerate(result["retrieved"], start=1):
        print(f"  {rank}. [{passage['chunk_id']}] score {passage['score']}, "
              f"coverage {passage['coverage']:.0%}")


def show_search(question):
    index = build_index(make_chunks(load_documents()))
    print(f"Question: {question}")
    print(f"Keywords: {tokenize(question)}\n")
    results = search(index, question)
    if not results:
        print("No passage shares a keyword with the question.")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. [{result['chunk_id']}] score {result['score']}  "
              f"matched: {result['matched_words']}  missing: {result['missing_words']}")
        print(f"   {result['text']}\n")


def show_chunks():
    chunks = make_chunks(load_documents())
    for chunk in chunks:
        print(f"[{chunk['chunk_id']}] ({len(chunk['text'].split())} words)")
        print(chunk["text"])
        print()
    sizes = [len(chunk["text"].split()) for chunk in chunks]
    print(f"{len(chunks)} chunks, {min(sizes)}-{max(sizes)} words each, "
          f"average {sum(sizes) / len(sizes):.0f}")


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "chunks":
        show_chunks()
    elif command == "search" and len(sys.argv) > 2:
        show_search(" ".join(sys.argv[2:]))
    elif command == "ask" and len(sys.argv) > 2:
        show_answer(" ".join(sys.argv[2:]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
