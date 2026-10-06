# Small document Q&A system (RAG take-home)

A command-line tool that answers questions about four Solar System planets
using only four short Wikipedia texts. Every answer is quoted from a passage
and cites it. When the passages do not support an answer, it says so.

It uses keyword search (BM25) and extractive answers. There is no language
model, no API, and nothing to install beyond Python.

**Evaluation result: 6 of 8 test questions passed.** Both failures are
explained below.

## Setup and run

Requirements: Python 3 (tested on 3.13.7, macOS). No third-party packages.

```bash
python3 rag.py ask "What is Jupiter's atmosphere made of?"   # answer one question
python3 evaluate.py                                           # run the 8 test questions
python3 rag.py search "What is the Great Red Spot?"           # show retrieved passages only
python3 rag.py chunks                                         # show all 44 passages
python3 fetch_docs.py                                         # optional: re-download the documents
```

Example:

```
$ python3 rag.py ask "What is Jupiter's atmosphere made of?"
Answer:   Jupiter's atmosphere consists of 76% hydrogen and 24% helium by mass, with a denser interior.
Why:      The best passage contains 67% of the question's keywords.
Source:   Jupiter - https://en.wikipedia.org/wiki/Jupiter
Passage:  [jupiter-p2-c1] Jupiter was the first of the Sun's planets to form, ...

$ python3 rag.py ask "How many rings does Saturn have?"
Answer:   The documents do not support an answer to this question.
Why:      The best passage [jupiter-p5-c2] contains only 50% of the question's keywords (missing: saturn).
```

Results are deterministic: running `evaluate.py` twice gives identical output.

## Documents

The introduction section of four English Wikipedia articles, about 500 words
each. Text is licensed CC BY-SA 4.0 by Wikipedia contributors. Copies are kept
in `docs/` so results do not change when Wikipedia does.

| File | Article | Exact version used |
|---|---|---|
| `docs/mercury.txt` | https://en.wikipedia.org/wiki/Mercury_(planet) | https://en.wikipedia.org/w/index.php?oldid=1377442586 |
| `docs/venus.txt` | https://en.wikipedia.org/wiki/Venus | https://en.wikipedia.org/w/index.php?oldid=1375690330 |
| `docs/mars.txt` | https://en.wikipedia.org/wiki/Mars | https://en.wikipedia.org/w/index.php?oldid=1375821898 |
| `docs/jupiter.txt` | https://en.wikipedia.org/wiki/Jupiter | https://en.wikipedia.org/w/index.php?oldid=1378749852 |

The same list, with download timestamps, is in `docs/sources.json`.

## Approach

All of it is in `rag.py`, in the order below.

**1. Chunking.** Each paragraph is split into sentences, and sentences are
grouped three at a time with one sentence of overlap. Chunks never cross a
paragraph. This gives 44 chunks of 25 to 103 words (average 62). Each has an
ID such as `mars-p3-c2` (Mars, paragraph 3, chunk 2), which is what answers
cite.

**2. Retrieval.** BM25 keyword scoring, written by hand (k1 = 1.5, b = 0.75).
Questions and chunks are lower-cased, stopwords are removed, and a final "s"
is cut so "moons" matches "moon". Two additions:

- The document name is added to each chunk's keywords, because many sentences
  say "It" or "the planet" instead of the name.
- If the question names a document ("... on Mercury?"), only that document is
  searched.

The top 3 chunks are returned.

**3. Answering.** The answer is the one or two sentences of the top chunk that
contain the rarest question keywords, quoted word for word. The chunk ID, the
passage and the source URL are printed with it.

**4. Abstaining.** If the top chunk contains less than 60% of the question's
keywords, the system answers "The documents do not support an answer to this
question" and names the missing keywords.

**Model and prompt:** none. No language model or embedding model is used.

## Evaluation

`eval/questions.json` holds 8 questions (6 answerable, 2 not) with expected
answers. `evaluate.py` runs them and writes `eval/results.md` and
`eval/results.json`, which contain the expected answer, retrieved passages,
actual answer and judgment for each.

Judgment rule: an answerable question passes if the answer contains the
expected phrase (for example "11.86"). An unanswerable one passes if the
system abstains.

| ID | Question | Should answer? | Did answer? | Result |
|---|---|---|---|---|
| Q1 | How long does Jupiter take to orbit the Sun? | yes | yes | PASS |
| Q2 | What is the largest crater on Mercury? | yes | yes | PASS |
| Q3 | How many moons does Mars have? | yes | yes | FAIL |
| Q4 | What is the atmosphere of Venus mostly made of? | yes | yes | PASS |
| Q5 | Which spacecraft was the first to orbit Mars? | yes | yes | PASS |
| Q6 | Which is the smallest planet in the Solar System? | yes | yes | PASS |
| Q7 | How many moons does Saturn have? | no | no | PASS |
| Q8 | When did humans first land on Venus? | no | yes | FAIL |

### Failures

- **Q3, wrong answer.** The document says Mars has "two natural satellites"
  and never uses the word "moons". The right passage was not retrieved, and
  the system quoted a sentence about a spacecraft orbiting "the Moon".
- **Q8, should have abstained.** A passage contains "humans" (who observed
  Venus), "first" and "Venus". That is 75% of the keywords, so the system
  answered with the 1961 Venera 1 flyby. The words are present but the passage
  is not about a human landing.

Both have one cause: the system matches words, not meaning.

### Weaknesses inside the passes

- Q1 passed with exactly 60% coverage, the minimum. A slightly different
  wording would have made the system abstain on a question it can answer.
- Q4, Q5 and Q6 include a second sentence that is not needed.
- The pass rule does not penalise that extra text.

### One improvement to try next

Add meaning-based retrieval: embed chunks and questions with a small local
embedding model and combine that score with BM25. This targets Q3 directly,
since "moons" and "natural satellites" are close in meaning. It would probably
not fix Q8, which needs a check that the passage actually answers the question
(for example a small entailment model or a local LLM asked "does this passage
answer the question?").

## Limitations

- **Keyword matching only.** Synonyms and paraphrases are missed (Q3).
- **Abstention checks words, not meaning.** A passage with the right words on
  a different subject is accepted (Q8).
- **The 60% threshold was tuned on few questions.** It was chosen after
  looking at 16 informal trial questions, where it separated answerable from
  unanswerable ones. Q3 was one of those trial questions and Q7 is a variant
  of one; the other six test questions were new. The system was not changed
  after the test was run.
- **Answers are quoted sentences, not direct replies.** "Which is the smallest
  planet?" returns a sentence containing "Mercury", not the word "Mercury".
- **Only the top chunk is used.** A question needing facts from two passages
  cannot be answered.
- **Small scale.** Four documents, 44 chunks, 8 test questions. The 6 of 8
  score is not a reliable estimate of accuracy.
- **English only, crude word handling.** The plural rule just cuts a final
  "s", so "volcano" does not match "volcanoes".
- **Tested on one machine** (Python 3.13.7, macOS).

## What I learned

<!-- TODO: rewrite these in your own words, and remove any you do not agree with. -->

- A retrieval score cannot decide when to abstain. Unanswerable questions
  scored higher than some answerable ones because they matched common words.
  Looking at which keywords were *missing* worked better.
- Most errors came from retrieval, not from answering. When the right passage
  was found, the right sentence was almost always picked.
- A threshold tuned on the same questions it is tested on looks better than it
  is. A new question (Q8) broke it immediately.
- Extractive answers cannot invent facts, but they can still be wrong by
  quoting a real sentence that answers a different question.

## AI assistance

<!-- TODO: edit so that every statement here is true for you. -->

- **What I used:** Claude Code (Claude Opus) wrote the code, the test
  questions and the first draft of this README, one step at a time. I chose
  the topic and the approach (keyword search with extractive answers, no LLM).
- **What I checked myself:** TODO, for example: read the four documents; ran
  every command in this README; compared each expected answer in
  `eval/questions.json` with the documents; read `eval/results.md` and agreed
  with each pass/fail judgment; read `rag.py` and can explain each step.

## Time spent

TODO: approximately X hours.

## Files

```
rag.py              chunking, retrieval, answering (the whole system)
evaluate.py         runs the test questions and writes the results
fetch_docs.py       downloads the documents (optional)
docs/               the four documents and sources.json
eval/questions.json the 8 test questions with expected answers
eval/results.md     evaluation results, readable
eval/results.json   evaluation results, complete data
```
