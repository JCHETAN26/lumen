"""The official VizWiz-VQA accuracy metric, ported line for line.

Source: `PythonEvaluationTools/vqaEvaluation/vqaEval.py` inside the VizWiz API.zip (sha256 pinned
in configs/pins.yaml). tests/test_vizwiz_metric.py checks this port against outputs of that file.

accuracy(prediction) = mean over the 10 leave-one-annotator-out subsets of
                       min(1, #annotators in the subset who gave the prediction / 3)

Faithfulness notes. These are quirks of the official code, kept on purpose so our numbers are
comparable with published VizWiz numbers:

- Only the *prediction* is normalized. Ground-truth answers are compared as-is, so a prediction
  "two" (normalized to "2") scores 0 against ten ground truths of "two".
- `_PERIOD_STRIP` is written `(?!<=\\d)` in the original, a typo for the lookbehind `(?<!\\d)`.
  As written, it only requires that the period is not followed by a digit, so "3." -> "3" but
  "3.5" is kept. We copy the pattern exactly.
- Punctuation is either deleted or replaced by a space depending on whether it borders a space
  anywhere in the text, or the text contains a digit-comma-digit sequence; the test is made on
  the original text, not the progressively edited one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn’t’ve", "couldnt’ve": "couldn’t’ve",
    "didnt": "didn’t", "doesnt": "doesn’t", "dont": "don’t", "hadnt": "hadn’t",
    "hadnt’ve": "hadn’t’ve", "hadn'tve": "hadn’t’ve", "hasnt": "hasn’t", "havent": "haven’t",
    "hed": "he’d", "hed’ve": "he’d’ve", "he’dve": "he’d’ve", "hes": "he’s", "howd": "how’d",
    "howll": "how’ll", "hows": "how’s", "Id’ve": "I’d’ve", "I’dve": "I’d’ve", "Im": "I’m",
    "Ive": "I’ve", "isnt": "isn’t", "itd": "it’d", "itd’ve": "it’d’ve", "it’dve": "it’d’ve",
    "itll": "it’ll", "let’s": "let’s", "maam": "ma’am", "mightnt": "mightn’t",
    "mightnt’ve": "mightn’t’ve", "mightn’tve": "mightn’t’ve", "mightve": "might’ve",
    "mustnt": "mustn’t", "mustve": "must’ve", "neednt": "needn’t", "notve": "not’ve",
    "oclock": "o’clock", "oughtnt": "oughtn’t", "ow’s’at": "’ow’s’at", "’ows’at": "’ow’s’at",
    "’ow’sat": "’ow’s’at", "shant": "shan’t", "shed’ve": "she’d’ve", "she’dve": "she’d’ve",
    "she’s": "she’s", "shouldve": "should’ve", "shouldnt": "shouldn’t",
    "shouldnt’ve": "shouldn’t’ve", "shouldn’tve": "shouldn’t’ve", "somebody’d": "somebodyd",
    "somebodyd’ve": "somebody’d’ve", "somebody’dve": "somebody’d’ve",
    "somebodyll": "somebody’ll", "somebodys": "somebody’s", "someoned": "someone’d",
    "someoned’ve": "someone’d’ve", "someone’dve": "someone’d’ve", "someonell": "someone’ll",
    "someones": "someone’s", "somethingd": "something’d", "somethingd’ve": "something’d’ve",
    "something’dve": "something’d’ve", "somethingll": "something’ll", "thats": "that’s",
    "thered": "there’d", "thered’ve": "there’d’ve", "there’dve": "there’d’ve",
    "therere": "there’re", "theres": "there’s", "theyd": "they’d", "theyd’ve": "they’d’ve",
    "they’dve": "they’d’ve", "theyll": "they’ll", "theyre": "they’re", "theyve": "they’ve",
    "twas": "’twas", "wasnt": "wasn’t", "wed’ve": "we’d’ve", "we’dve": "we’d’ve",
    "weve": "we've", "werent": "weren’t", "whatll": "what’ll", "whatre": "what’re",
    "whats": "what’s", "whatve": "what’ve", "whens": "when’s", "whered": "where’d",
    "wheres": "where's", "whereve": "where’ve", "whod": "who’d", "whod’ve": "who’d’ve",
    "who’dve": "who’d’ve", "wholl": "who’ll", "whos": "who’s", "whove": "who've",
    "whyll": "why’ll", "whyre": "why’re", "whys": "why’s", "wont": "won’t",
    "wouldve": "would’ve", "wouldnt": "wouldn’t", "wouldnt’ve": "wouldn’t’ve",
    "wouldn’tve": "wouldn’t’ve", "yall": "y’all", "yall’ll": "y’all’ll", "y’allll": "y’all’ll",
    "yall’d’ve": "y’all’d’ve", "y’alld’ve": "y’all’d’ve", "y’all’dve": "y’all’d’ve",
    "youd": "you’d", "youd’ve": "you’d’ve", "you’dve": "you’d’ve", "youll": "you’ll",
    "youre": "you’re", "youve": "you’ve",
}  # fmt: skip
_NUMBER_WORDS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}  # fmt: skip
_ARTICLES = frozenset({"a", "an", "the"})
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")  # sic: see module docstring
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCT = (";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
          ">", "<", "@", "`", ",", "?", "!")  # fmt: skip

UNANSWERABLE = "unanswerable"


def _process_punctuation(text: str) -> str:
    out = text
    has_digit_comma = _COMMA_STRIP.search(text) is not None
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or has_digit_comma:
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return _PERIOD_STRIP.sub("", out)


def _process_digit_article(text: str) -> str:
    words = [_NUMBER_WORDS.get(w, w) for w in text.lower().split()]
    words = [w for w in words if w not in _ARTICLES]
    return " ".join(_CONTRACTIONS.get(w, w) for w in words)


def normalize_prediction(answer: str) -> str:
    """Normalize a model's answer exactly as the official evaluator does."""
    text = answer.replace("\n", " ").replace("\t", " ").strip()
    return _process_digit_article(_process_punctuation(text))


def vqa_accuracy(prediction: str, gt_answers: Sequence[str]) -> float:
    """Official VizWiz accuracy of one prediction against its (normally 10) human answers."""
    if not gt_answers:
        raise ValueError("gt_answers is empty")
    pred = normalize_prediction(prediction)
    scores = []
    for i in range(len(gt_answers)):
        matches = sum(1 for j, a in enumerate(gt_answers) if j != i and a == pred)
        scores.append(min(1.0, matches / 3))
    return sum(scores) / len(scores)


@dataclass
class AccuracyReport:
    """Accuracy in percent. `overall_official` is rounded to 2 places like the official code;
    `overall` keeps full precision for significance tests and deltas."""

    n: int
    overall: float
    overall_official: float
    per_answer_type: dict[str, float] = field(default_factory=dict)
    per_answer_type_n: dict[str, int] = field(default_factory=dict)
    per_question: dict[str, float] = field(default_factory=dict)


def evaluate_accuracy(
    predictions: Mapping[str, str],
    annotations: Iterable[Mapping],
    *,
    keep_per_question: bool = True,
) -> AccuracyReport:
    """Score predictions (question_id -> answer) against VizWiz annotation records.

    Every annotated question must have a prediction; a missing one is an error, not a silent 0,
    so a crashed or truncated run can't pass as a low score.
    """
    per_q: dict[str, float] = {}
    by_type: dict[str, list[float]] = {}
    missing = []
    for ann in annotations:
        qid = ann["question_id"]
        if qid not in predictions:
            missing.append(qid)
            continue
        acc = vqa_accuracy(predictions[qid], [a["answer"] for a in ann["answers"]])
        per_q[qid] = acc
        by_type.setdefault(ann["answer_type"], []).append(acc)
    if missing:
        raise KeyError(f"{len(missing)} annotated questions have no prediction, e.g. {missing[:3]}")
    if not per_q:
        raise ValueError("no annotations to evaluate")
    overall = 100 * sum(per_q.values()) / len(per_q)
    return AccuracyReport(
        n=len(per_q),
        overall=overall,
        overall_official=round(overall, 2),
        per_answer_type={t: round(100 * sum(v) / len(v), 2) for t, v in sorted(by_type.items())},
        per_answer_type_n={t: len(v) for t, v in sorted(by_type.items())},
        per_question=per_q if keep_per_question else {},
    )
