"""
BERTopic complaint-theme extraction pipeline.

Turns the complaint corpus (phase2_complaint_corpus.csv) into BERTopic
topics: embed each review's cons_ko text, cluster with HDBSCAN, label
clusters with Kiwi-tokenized c-TF-IDF, and soft-assign each document to
every topic above a probability threshold.

Built to run unchanged at two scales:
    - Dry run: a few hundred to low-thousand real complaint segments, run
      locally, to sanity-check the pipeline end to end and settle the
      probability threshold empirically.
    - Full run: every complaint segment, embeddings computed on Kaggle,
      clustering run locally afterward on the resulting matrix (HDBSCAN
      doesn't need a GPU).

Design choices:

  1. Cons-only, whole-segment input, no clause/sentence splitting.
     Splitting was tried, empirically tested (31/40 segments split
     cleanly via a POS-tag whitelist), and reverted after published
     Korean ABSA research (a 2022 national-competition corpus, a 2024
     cosmetics-specific paper) and Hwahae's own production system all
     showed the field feeding whole, unsplit text into these models
     rather than pre-splitting it. Soft multi-topic assignment
     (calculate_probabilities=True + threshold) replaces splitting as the
     fix for one segment covering several distinct complaints.
  2. Embeddings: jhgan/ko-sroberta-multitask, not a newer option like
     KURE-v1. KURE is retrieval-optimized (query-to-document matching);
     ko-sroberta-multitask is trained on KorSTS/KorNLI (sentence
     similarity), the better task-match for BERTopic's clustering step.
     MTEB scores retrieval and clustering as separate capabilities --
     recency isn't the same axis as task-fit here.
  3. Tokenizer: Kiwi (kiwipiepy), not Mecab-ko (the incumbent industry
     default) or BERTopic's default whitespace splitter (which
     fragments Korean's agglutinative morphology). Kiwi is a pure C++
     core with no Java/JDK install dependency, actively maintained, and
     reports ~86.7% average disambiguation accuracy against ~50-70% for
     other open-source Korean analyzers per its own published
     benchmark.
  4. Topic count: BERTopic's default HDBSCAN, not a fixed K -- lets the
     data set the number of themes instead of guessing one in advance.
  5. POS filter: KIWI_KEEP_POS below is a content-word POS whitelist
     (nouns, verb/adjective stems, foreign/hanja/number tokens) applied
     after Kiwi's morphological analysis, dropping particles and sentence
     endings before they reach c-TF-IDF -- a standard choice for Korean
     topic modeling.

Two options on top of the core pipeline:

  - Tier 0 pre-filter (is_no_complaint_segment / filter_no_complaint_
    segments, below): drops "no complaint" meta-negation segments (e.g.
    "아쉬운 점은 없었어요") before embedding -- otherwise they form their
    own incoherent clusters, not real complaint themes. Applied in main()
    right after load_corpus(), reported with a real count/percentage
    removed plus a sample CSV of what got dropped -- never silently. Kept
    out of load_corpus() itself so that function keeps returning the
    exact same sample other scripts reproduce. See the function's own
    docstring for the false-positive classes it guards against (e.g.
    "아쉬운 점은 없지만 X" complaint-softening pivots, a genuine Korean
    construction).
  - Top-k assignment (soft_assign_topics_topk, below): an alternative to
    soft_assign_topics()'s absolute-probability threshold -- takes the
    top-k topics per document regardless of absolute magnitude, above a
    small display floor. Targets documents with flat probability
    distributions, where nothing clears even 0.07 anywhere -- a pattern
    no absolute threshold can catch no matter how low it's set. Wired
    into --dry-run via --topk-values.

--seed (defaults to RANDOM_SEED): at n=1,000 the topic-count/-1-share
numbers are sensitive to small input changes -- a single document moving
from dropped to kept swung the -1 share by 3+ points, because UMAP fits
jointly on the whole sample. Compare independent same-size draws to
separate a real effect of a config change from sampling noise before
trusting any single dry run's numbers.

Usage:
    Dry run -- sample N rows, evaluate candidate thresholds plus the
    top-k comparison, and report per-threshold/per-k
    topic-count stats plus the Tier 0 pre-filter's real removal count:
        python build_bertopic_pipeline.py \\
            --input phase2_complaint_corpus.csv \\
            --sample-size 1000 \\
            --dry-run \\
            --thresholds 0.1 0.2 \\
            --topk-values 2 3

    Full run, once a threshold is picked (at full scale the embeddings
    are computed on Kaggle and loaded with --embeddings-file):
        python build_bertopic_pipeline.py \\
            --input phase2_complaint_corpus.csv \\
            --threshold 0.1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Force UTF-8 on stdout/stderr regardless of the terminal's own codepage or
# the system locale: on a console that falls back to a legacy non-Unicode
# codepage (e.g. cp1251), printing a topic representation containing Korean
# text raises UnicodeEncodeError -- which once crashed a ~24h full-scale run
# before any output was saved. errors="replace" is a second layer: printing a
# replacement glyph is strictly better than crashing after hours of compute
# over what is, in the end, just a display convenience.
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding is None or sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# See the module docstring's point 2 for why this model.
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"

# min_cluster_size: the default (20) suits dry-run samples only. At
# n=10,000/seed=42, 20 beat 10 (93 topics, 40.5% outlier share) and 15
# (55 topics) on every measured axis, but it does not scale: on the full
# 543,760-segment corpus it produced 1,035 topics. The production run (the
# 12-month window corpus, 179,263 segments) passes --min-cluster-size 200
# explicitly: 68 topics, 45.41% outliers, largest topic 4.43% of the corpus.
# 200 was picked over 150 (a similar profile, more topics) and 100 (the
# lowest outlier share, but one topic held 9.41% of the corpus). BERTopic's
# own tuning guidance suggests 100-500 for corpora approaching a million
# documents. Kept as an explicit default rather than BERTopic's own (10),
# so a command that omits the flag can't silently run a different config.
DEFAULT_MIN_CLUSTER_SIZE = 20
# None -> HDBSCAN itself defaults min_samples to min_cluster_size.
DEFAULT_MIN_SAMPLES = None

# Content-word POS tags kept after Kiwi tokenization, dropped before
# c-TF-IDF -- see the module docstring's point 5. NNG/NNP/NNB = common/
# proper/bound nouns, VV/VA = verb/adjective stems, XR = root, SL/SH/SN =
# foreign-alphabet/hanja/number tokens. Particles (J*) and sentence
# endings (E*) are deliberately dropped -- they're grammatical glue, not
# topic content, and Korean's agglutinative morphology means keeping
# them would fragment otherwise-identical roots into distinct tokens.
# VA-I: Kiwi tags Korean's whole ᄇ-irregular adjective class
# (아쉽다/무겁다/가볍다/차갑다/덥다/어렵다 and others) with this separate
# tag, not plain VA; without it these words would be invisible to every
# topic label -- and 무겁다/가볍다 ("feels heavy" vs "feels light") are
# stock K-beauty texture words.
KIWI_KEEP_POS = frozenset(
    {"NNG", "NNP", "NNB", "VV", "VA", "VA-I", "XR", "SL", "SH", "SN"}
)

# Tags exempt from the single-character length filter below -- see
# _KIWI_LENGTH_EXEMPT_TAGS' use in kiwi_tokenizer(). Each tag is here
# because a real one-syllable content word was caught testing against
# actual Kiwi output, not guessed:
#   1. NNG/NNP/NNB, for nouns like 향 ("scent") and 색
#      ("color/shade") -- "the scent is too strong" is a stock K-beauty
#      complaint pattern, and a blanket length>1 filter would silently
#      drop it.
#   2. VV/VA: 없 ("없다", "to not exist/have" -- VA) is a single
#      character too. 없다 is arguably more consequential than 향 --
#      "효과가 없어요" ("no effect / doesn't work") and "자극이 없어요"
#      ("no irritation") are both extremely common review patterns, one
#      a real complaint, one a compliment, and both would silently
#      collapse to just ["효과"] / ["자극"] with 없 erased. VV is
#      included alongside VA for the same underlying reason (Korean verb
#      stems are just as often naturally one syllable -- 하, 되, 오, 가,
#      보), even though no concrete failing example was found for VV
#      specifically. XR/SL/SH/SN remain subject to the length filter --
#      no evidence yet that single-character tokens under those tags
#      are anything but fragment noise, unlike the noun/verb/adjective
#      case.
#   3. VA-I, same reasoning as VA --
#      at least one confirmed ᄇ-irregular stem is single-character
#      (덥, "hot"/"덥다"), which without this exemption would survive
#      the KIWI_KEEP_POS addition above only to be dropped right back
#      out here by the length filter.
_KIWI_LENGTH_EXEMPT_TAGS = frozenset({"NNG", "NNP", "NNB", "VV", "VA", "VA-I"})

RANDOM_SEED = 42


# =============================================================================
# TOKENIZATION (Kiwi, for c-TF-IDF topic labeling only -- NOT for the
# embedding step, which reads raw text directly per sentence-transformers'
# own convention)
# =============================================================================

_KIWI_INSTANCE = None


def _get_kiwi():
    """Lazy singleton -- Kiwi's model load is not free, don't repeat it
    per document."""
    global _KIWI_INSTANCE
    if _KIWI_INSTANCE is None:
        from kiwipiepy import Kiwi

        _KIWI_INSTANCE = Kiwi()
    return _KIWI_INSTANCE


def kiwi_tokenizer(text: str) -> list[str]:
    """Tokenize Korean text with Kiwi, keeping only content-bearing POS
    tags (see KIWI_KEEP_POS above). Wired into BERTopic's
    vectorizer_model as a CountVectorizer tokenizer function -- not
    BERTopic's default whitespace splitter.

    KNOWN, VERIFIED LIMITATION -- negation is not handled, and two of
    the three common Korean negation patterns are silently invisible to
    the resulting bag-of-words, confirmed by actually running Kiwi (not
    guessed):
      - "-지 않다" auxiliary negation (e.g. "끈적하지 않고"): the negator
        is tagged VX, which KIWI_KEEP_POS excludes. Verified:
        "끈적하지 않고" and "끈적해요" both reduce to the single token
        "끈적" -- negated and affirmative are indistinguishable here.
      - "안 X" pre-verbal negation (e.g. "안 끈적여요"): 안 is tagged
        MAG (adverb), also excluded. Same collapse.
      - "없다" existential negation (e.g. "자극이 없어요") is the
        exception -- 없다 is its own adjective predicate (VA, which IS
        kept), not an auxiliary attached to another stem, so its token
        survives (e.g. "자극이 없어요" keeps both "자극" and "없").
        Scope still isn't bound (bag-of-words has no concept of which
        noun "없" negates), but the negator's own signal isn't erased
        the way it is in the other two patterns.
    Not fixed here -- see count_negation_patterns() below for the real
    prevalence check this needs before deciding whether to build a fix.

    Two more negation patterns are ALSO confirmed invisible here, same
    mechanism as the two above -- "아니다"/"아니라" copula negation
    (tagged VCN, not in KIWI_KEEP_POS) and "-지 못하다"/bare "못"
    incapacity negation (tagged VX form "못하" or MAG form "못",
    neither in KIWI_KEEP_POS). Full detail, including why "-지 못하다"
    needed two different detection checks (Kiwi tokenizes it two
    different ways depending on context), is in
    count_negation_patterns()'s docstring below, not repeated here.
    """
    kiwi = _get_kiwi()
    tokens = kiwi.tokenize(text)
    return [
        t.form
        for t in tokens
        if t.tag in KIWI_KEEP_POS
        and (len(t.form) > 1 or t.tag in _KIWI_LENGTH_EXEMPT_TAGS)
    ]


def count_negation_patterns(texts: list[str]) -> dict:
    """Diagnostic, not a fix: counts how often each Korean negation
    pattern actually occurs in the real corpus, using Kiwi's own POS
    tags rather than regex guessing. Purely informational -- doesn't
    change tokenization or clustering behavior. Run in --dry-run mode
    so there's a real prevalence number (not a guess) before deciding
    whether negation-aware tokenization is worth building. See
    kiwi_tokenizer()'s docstring for what each pattern looks like and
    why it currently isn't distinguished from its affirmative form.

    Five patterns are counted. Besides the three in kiwi_tokenizer()'s
    docstring, two more occur at comparable prevalence in a 12,000-row
    cons_ko sample (directional only; this function's own output, run
    against the real corpus, is the number that counts):
      - "아니다"/"아니라" copula negation -- tagged VCN, form "아니".
        Confirmed empirically against 17 real cons_ko examples: fires
        cleanly and only on this pattern, no false positives observed.
      - "-지 못하다"/bare "못" incapacity negation -- confirmed
        empirically to tokenize two different ways depending on
        context (not assumed to be one form): Kiwi sometimes fuses it
        into a single auxiliary token (VX, form "못하" -- e.g. "빼내지
        못하는거"), and sometimes splits it into a separate adverb
        token (MAG, form "못") plus the following verb analyzed on its
        own (e.g. "느끼지 못했어요" -> MAG "못" + VV "하"). Both forms
        are checked below; checking only one would silently undercount
        roughly half the real occurrences (9 of 17 examples in testing
        hit the split MAG form, not the fused VX form).

    The "-지 않다" check requires form == "않" specifically, not any VX
    tag: "-지 못하다"'s fused form is ALSO tagged VX (see above), so a
    plain VX check would count 못하다-negation as 않다-negation too.
    """
    kiwi = _get_kiwi()
    aux_neg_docs = 0   # "-지 않다" -- VX tag with form '않' specifically
    prefix_neg_docs = 0  # "안 X" -- MAG tag with form '안'
    existential_neg_docs = 0  # "없다" -- VA tag with form '없'
    copula_neg_docs = 0  # "아니다"/"아니라" -- VCN tag with form '아니'
    incapacity_neg_docs = 0  # "-지 못하다"/bare "못" -- VX form '못하' OR MAG form '못'
    for text in texts:
        tags_seen = {(t.tag, t.form) for t in kiwi.tokenize(text)}
        if ("VX", "않") in tags_seen:
            aux_neg_docs += 1
        if ("MAG", "안") in tags_seen:
            prefix_neg_docs += 1
        if ("VA", "없") in tags_seen:
            existential_neg_docs += 1
        if ("VCN", "아니") in tags_seen:
            copula_neg_docs += 1
        if ("VX", "못하") in tags_seen or ("MAG", "못") in tags_seen:
            incapacity_neg_docs += 1
    n = len(texts)
    return {
        "n_documents": n,
        "aux_negation_-지_않다_count": aux_neg_docs,
        "aux_negation_-지_않다_pct": aux_neg_docs / n if n else 0.0,
        "prefix_negation_안_count": prefix_neg_docs,
        "prefix_negation_안_pct": prefix_neg_docs / n if n else 0.0,
        "existential_negation_없다_count": existential_neg_docs,
        "existential_negation_없다_pct": existential_neg_docs / n if n else 0.0,
        "copula_negation_아니다_count": copula_neg_docs,
        "copula_negation_아니다_pct": copula_neg_docs / n if n else 0.0,
        "incapacity_negation_못_count": incapacity_neg_docs,
        "incapacity_negation_못_pct": incapacity_neg_docs / n if n else 0.0,
    }


# =============================================================================
# TIER 0 -- "NO COMPLAINT" META-NEGATION PRE-FILTER
#
# DIFFERENT from count_negation_patterns() above: that function measures
# real complaint clauses whose own negation is invisible to c-TF-IDF.
# This measures segments that are, in substance, saying there ISN'T a
# complaint -- e.g. "아쉬운 점은 없었어요" ("no particular complaints"),
# "딱히 단점은 없어요" ("no real shortcomings"). These formed their own
# incoherent BERTopic clusters (10.2% of a 1,000-doc sample), not real
# complaint themes -- sentiment scoring sometimes classifies this kind of
# text as cons_is_complaint anyway, and this filter removes it before it
# reaches embedding.
# =============================================================================

# Cue forms for the "no complaint" framing itself. 아쉽다 ("regrettable/
# disappointing") tags as VA-I (irregular adjective), NOT plain VA -- a
# ("VA", "아쉽") cue failed 10/12 real positive test cases. "아쉬움" (the nominalized form)
# also tokenizes two different ways depending on context, same lesson
# as count_negation_patterns()'s "-지 못하다" find: sometimes its own
# NNG noun token, sometimes VA-I "아쉽" + an ending -- both included so
# neither form is silently missed.
_NO_COMPLAINT_CUE_FORMS = frozenset(
    {("VA-I", "아쉽"), ("NNG", "아쉬움"), ("NNG", "단점")}
)

# Connective endings that, when they IMMEDIATELY follow a "없" (VA)
# token (skipping past tense/aspect markers like "었" -- EP tag -- to
# find the real ending), signal a contrastive PIVOT ("no complaints,
# BUT X") rather than a genuine closing "no complaint" statement.
# Found empirically, not guessed: "토너 자체에 대한 아쉬운 점은
# 없지만 굳이 아쉬운 점이라면 용량이 많아..." is a REAL complaint
# segment that a filter without this guard drops, because
# it also contains "아쉬운 점은 없지만" as a rhetorical softener before
# still naming a real complaint -- a genuine, common Korean construction,
# not an edge case invented for testing. Scoped to the three forms that
# caused real failures ("지만", "는데", and "으나", as in "아쉬운점은
# 없었으나 대용량으로...") -- not a general adversative-
# clause parser, and not claimed to be an exhaustive list of Korean
# contrastive connectives.
_NO_COMPLAINT_PIVOT_EC_FORMS = frozenset({"지만", "는데", "으나"})


def _tokenize_sentence(sentence_text: str) -> list[tuple[str, str]]:
    """(tag, form) tokens for one sentence, per Kiwi. Split out so
    is_no_complaint_segment() can tokenize sentence-by-sentence
    instead of the whole document at once."""
    kiwi = _get_kiwi()
    return [(t.tag, t.form) for t in kiwi.tokenize(sentence_text)]


def _sentence_no_complaint_verdict(tokens: list[tuple[str, str]]) -> bool:
    """True if THIS SENTENCE ALONE states a genuine 'no complaint'
    claim -- i.e. a cue (아쉽다/아쉬움/단점) and a genuine 없-closer are
    both present within it. Checking across the WHOLE document instead
    lets an unrelated cue and an unrelated, harmless 없 anywhere else in
    a multi-sentence review combine into a false verdict. Real cases: a
    two-paragraph review with two genuine complaints (wants a
    fragrance-free product, this one isn't; hydration doesn't last),
    dropped only because of one unrelated, purely descriptive "향도
    거의 없는 편입니다" ("there's barely any scent") several sentences
    later; and a review where the pivot-guard correctly
    protects "아쉬운건 없지만" in one sentence, but a second, unrelated
    없 about a friend's experience in a LATER sentence ("친구는 효과가
    없었다고") overrode that protection and dropped the doc anyway.
    Confirmed via kiwi.split_into_sents() that both false-positive
    cases have their cue and their unrelated 없 in different sentences.
    """
    tags_seen = set(tokens)
    has_cue = bool(_NO_COMPLAINT_CUE_FORMS & tags_seen)
    if not has_cue:
        return False

    for i, (tag, form) in enumerate(tokens):
        if tag != "VA" or form != "없":
            continue

        # Exclude "-(으)ㄹ 수 없다" ("cannot X") -- this 없 means
        # "cannot", not "there is no [complaint]", and is NOT evidence
        # of a genuine no-complaint claim. Shape: [...ETM] + NNB "수" +
        # VA "없". Confirmed empirically: "용기로인한
        # 깔끔하게 사용할수없는점이 아쉬워요" ("disappointed I can't
        # use it cleanly because of the container") is a 100%-complaint
        # sentence that is dropped without this guard, because "없" here tags
        # identically (VA, immediately followed by ETM) to the "no
        # complaint" sense of 없 -- the two are indistinguishable by
        # tag+form alone without checking what precedes them.
        if (
            i >= 2
            and tokens[i - 1] == ("NNB", "수")
            and tokens[i - 2][0] == "ETM"
        ):
            continue

        # Exclude "없지 (는/도) 않다" ("not [that there is] none" =
        # "there IS X") -- double negation ON 없 itself, meaning the
        # OPPOSITE of "no complaint." Found by a full-corpus
        # pattern-coverage scan, then verified directly against the real
        # corpus: 114 real segments contain both a cue
        # word and this construction, e.g. "수분감이 살짝 부족한감이
        # 없지않아 있네요", and confirmed by direct token trace that at
        # least one -- "...건조한 느낌이 없지 않아 크림까지 발라줘야
        # 해서 보습력이 조금 아쉬워요" (a genuine complaint about
        # dryness) -- would get a false "no complaint" verdict without
        # this guard. Same class of confound as the "수 없다" guard
        # just above, different construction. Allows an optional JX
        # particle (는/도) between 지 and 않, since both "없지 않다" and
        # "없지는/없지도 않다" occur in real text.
        if i + 1 < len(tokens) and tokens[i + 1] == ("EC", "지"):
            k = i + 2
            if k < len(tokens) and tokens[k][0] == "JX" and tokens[k][1] in ("는", "도"):
                k += 1
            if k < len(tokens) and tokens[k] == ("VX", "않"):
                continue  # "없지(는/도) 않다" -- double negation, not a real closer

        # Forward EC pivot guard (see _NO_COMPLAINT_PIVOT_EC_FORMS above).
        j = i + 1
        while j < len(tokens) and tokens[j][0] == "EP":
            j += 1
        nxt = tokens[j] if j < len(tokens) else None
        if nxt is not None and nxt[0] == "EC" and nxt[1] in _NO_COMPLAINT_PIVOT_EC_FORMS:
            continue  # pivot ("...없지만 X") -- not a real closer

        # Backward "말고(는)" pre-posed exception guard -- a
        # complaint named BEFORE 없 as something explicitly excluded
        # from the "no complaint" claim, e.g. "너무
        # 비싼거 말고는 아쉬운건 없는거 같아요" ("no complaints except
        # for it being too expensive"). The forward EC guard above
        # can't catch this since nothing pivots off 없 itself -- the
        # exception sits earlier in the sentence, marked by VV "말" +
        # EC "고".
        if any(
            tokens[k] == ("VV", "말") and tokens[k + 1] == ("EC", "고")
            for k in range(i - 1)
        ):
            continue

        return True
    return False


# Sentence-initial (tag, form) pairs that, when they open the sentence
# immediately AFTER a flagged "no complaint" sentence, signal a real
# complaint pivoting across the sentence boundary -- not caught by
# _NO_COMPLAINT_PIVOT_EC_FORMS since these aren't EC endings attached
# to 없, they open their own separate sentence. Three forms, each
# confirmed from real evidence, not guessed:
#   - ("NNG", "대신") ("instead"): "그다지
#     단점으로 꼽을게 없는거 같아요. 대신 비싸요."
#   - ("MAG", "다만") ("however"): "저한테 아쉬운 점은 없었어요!
#     다만 바이럴 되는 만큼 블랙헤드가...무조건 쏙쏙 빠지진
#     않았습니다..." -- structurally the same cross-sentence pivot as
#     대신, different word, different tag (MAG, not NNG).
#   - ("MAG", "굳이") ("if [I] had to [pick one]"), found by a
#     full-corpus pattern-coverage scan: 1,528 occurrences across the real 623,478-row
#     corpus -- MORE frequent than 다만's own 970 -- e.g. "딱히
#     아쉬웠던 점은 없는데 꼭 써야하네요ㅜㅜ 굳이 꼽자면 이 제품은
#     사용감이 촉촉한데...당기는 느낌이 있는 제품이면 그것까지
#     커버하지는 못해요~". Structurally identical to the 다만/대신
#     pattern.
_NO_COMPLAINT_CROSS_SENTENCE_PIVOT_FORMS = frozenset(
    {("NNG", "대신"), ("MAG", "다만"), ("MAG", "굳이")}
)


def _sentence_opens_with_cross_sentence_pivot(tokens: list[tuple[str, str]]) -> bool:
    """True if a sentence's first token is one of
    _NO_COMPLAINT_CROSS_SENTENCE_PIVOT_FORMS -- see that set's comment
    for what this catches and why it's separate from the in-sentence EC
    pivot guard. NOT an exhaustive list of Korean contrastive adverbs --
    same "scoped to real, observed failures" discipline as
    _NO_COMPLAINT_PIVOT_EC_FORMS, not a general adversative-clause
    parser."""
    return bool(tokens) and tokens[0] in _NO_COMPLAINT_CROSS_SENTENCE_PIVOT_FORMS


def is_no_complaint_segment(text: str) -> bool:
    """True if `text` contains the 'no complaint' meta-negation pattern
    (아쉽다/아쉬움/단점 co-occurring with 없다's existential-negation
    stem 없) -- see the module-level comment block above for what this
    targets and why it's separate from count_negation_patterns().

    Checks cue+closer per SENTENCE (kiwi.split_into_sents()) rather than
    across the whole document at once -- see _sentence_no_complaint_verdict()'s
    docstring for the false-positive class this avoids. A document is dropped if ANY
    of its sentences states a genuine no-complaint claim on its own,
    UNLESS the immediately following sentence pivots into a real
    complaint (_sentence_opens_with_cross_sentence_pivot(), forms in
    _NO_COMPLAINT_CROSS_SENTENCE_PIVOT_FORMS) -- the within-sentence
    지만/는데/으나 pivot guard still handles pivots that stay inside
    one sentence.

    KNOWN, NOT FULLY RESOLVED, flagged rather than assumed away:
      - Only catches 아쉽다/단점 cues, not every way Korean expresses
        "no complaint" -- e.g. "특별히 불편한 점은 없어서 모르겠네요"
        (a real example) uses 불편 ("inconvenience")
        instead, and isn't caught. 불편 was deliberately left out of
        _NO_COMPLAINT_CUE_FORMS: it's a much more generic complaint
        word than 아쉽다/단점 (a real inconvenience IS a real
        complaint), so adding it risks meaningfully more false
        positives across the full 623K-row corpus than it's worth for
        a "cheap, no model change" Tier 0 pass -- a recall trade-off,
        not an oversight.
      - Real corpus typos (e.g. "아쉬은점" instead of standard
        "아쉬운점") can tokenize differently and slip through --
        observed in testing, not fixed (out of scope for a cheap
        keyword-style filter).
      - No rule for plain "no complaint, and I wish X" wish-statements
        with no contrastive connector at all (e.g. "아쉬운점은 없고
        가격세일을 자주하면 좋겠어요"): this pattern isn't a clean
        grammatical rule the way the others are, and it starts to blur
        into a keyword/wish-detection layer, so it's left as a known,
        documented residual gap.
      - The pivot guards above are scoped to the specific forms that
        caused real, observed failures -- a segment using some other,
        rarer contrastive construction to pivot from "no complaint"
        into a real one could still be dropped incorrectly. Spot-check
        the dropped-sample CSV this filter writes (see main()) before
        trusting it at full scale.
    """
    kiwi = _get_kiwi()
    sentences = kiwi.split_into_sents(text)
    if not sentences:
        return False

    sentence_tokens = [_tokenize_sentence(s.text) for s in sentences]
    verdicts = [_sentence_no_complaint_verdict(toks) for toks in sentence_tokens]

    for idx, verdict in enumerate(verdicts):
        if not verdict:
            continue
        if idx + 1 < len(sentence_tokens) and _sentence_opens_with_cross_sentence_pivot(
            sentence_tokens[idx + 1]
        ):
            continue  # "없어요. 대신/다만 X" -- pivot across the sentence boundary
        return True
    return False


def filter_no_complaint_segments(
    df: pd.DataFrame, text_col: str = "cons_ko"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies is_no_complaint_segment() to `df[text_col]`, splitting it
    into (kept, dropped). Pure function, no I/O or printing -- main()
    is responsible for reporting the real count/share removed and
    writing a sample of what got dropped, per this pipeline's
    never-silently-drop convention (same as count_negation_patterns()
    being surfaced before any tokenizer change, not applied quietly)."""
    mask = df[text_col].apply(is_no_complaint_segment)
    kept = df[~mask].reset_index(drop=True)
    dropped = df[mask].reset_index(drop=True)
    return kept, dropped


# =============================================================================
# EMBEDDING
# =============================================================================


def embed_documents(
    texts: list[str], model_name: str = EMBEDDING_MODEL, batch_size: int = 64
) -> np.ndarray:
    """Embed cons_ko text with the locked sentence-transformers model.

    At full scale this step runs on Kaggle (a BERT-base-sized forward
    pass across every complaint segment) -- the same call works locally
    for the dry run and, unchanged, inside a Kaggle notebook.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
    )


def load_precomputed_embeddings(
    embeddings_path: Path, review_ids_path: Path | None = None
) -> tuple[np.ndarray, list[str]]:
    """Load an embedding matrix computed on Kaggle (embed on Kaggle,
    cluster locally) plus its
    companion review_id list (written by embed_on_kaggle.py in the same
    row order as the matrix).

    Returns both pieces; does NOT verify alignment itself -- main()
    compares review_ids against its own locally Tier-0-filtered
    df["review_id"] and fails loudly on a mismatch, since a silent
    misalignment here would corrupt every downstream cluster assignment
    with no visible symptom.
    """
    embeddings = np.load(embeddings_path)
    if review_ids_path is None:
        review_ids_path = embeddings_path.with_name(
            embeddings_path.stem + "_review_ids.csv"
        )
    review_ids = (
        pd.read_csv(review_ids_path, encoding="utf-8-sig")["review_id"]
        .astype(str)
        .tolist()
    )
    if len(review_ids) != embeddings.shape[0]:
        raise ValueError(
            f"{review_ids_path} has {len(review_ids):,} review_ids but "
            f"{embeddings_path} has {embeddings.shape[0]:,} rows -- these "
            "must come from the same embed_on_kaggle.py run; something is "
            "mismatched."
        )
    return embeddings, review_ids


# =============================================================================
# CLUSTERING
# =============================================================================


def build_topic_model(
    texts: list[str],
    embeddings: np.ndarray,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_samples: int | None = DEFAULT_MIN_SAMPLES,
    calculate_probabilities: bool = True,
):
    """Fit BERTopic on precomputed embeddings with default HDBSCAN
    (soft multi-topic assignment enabled by default) and Kiwi-tokenized
    c-TF-IDF.

    Returns the fitted topic_model plus BERTopic's own hard-assignment
    topics and per-document, per-topic probability matrix -- the
    threshold-based soft assignment happens as a separate step
    (soft_assign_topics below), since the threshold value is meant
    to be tried at more than one value against the same fit, not baked
    into the clustering call itself.

    calculate_probabilities=False (--skip-probabilities):
    skips the separate all_points_membership_vectors soft-probability
    computation entirely -- measured at ~7h53m on the full 543,760-row
    corpus, the single largest cost in the whole pipeline. Safe to skip
    for anything that only needs topic count / sizes / outlier share
    (a min_cluster_size sweep, e.g.): HDBSCAN's hard `topics` assignment
    is computed by the clustering fit itself and does not depend on
    this step -- cluster_summary() below reads only `topics`, never
    `probs`. probs comes back None from BERTopic in this case; callers
    must not pass it to soft_assign_topics() or compare_thresholds().
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    vectorizer_model = CountVectorizer(tokenizer=kiwi_tokenizer, token_pattern=None)

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # BERTopic's default UMAP step has NO random_state set unless a
    # umap_model is passed in explicitly, so RANDOM_SEED alone would only
    # control which rows load_corpus() samples, never UMAP's own randomness.
    # Confirmed empirically: refitting on the identical 1,000-row sample
    # (same RANDOM_SEED, same min_cluster_size) gave 17 topics in one run
    # and 16 in the next. These are BERTopic's own documented default UMAP parameters (n_neighbors=15,
    # n_components=5, min_dist=0.0, metric="cosine") -- only random_state is
    # new here, so this pins reproducibility without otherwise changing
    # clustering behavior. Caveat: umap-learn's own docs note that random_state alone doesn't always
    # guarantee bit-for-bit reproducibility under multi-threaded BLAS --
    # verify by literally re-running twice and diffing the topic count
    # before trusting this fully closes the gap.
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_SEED,
    )

    topic_model = BERTopic(
        embedding_model=None,  # embeddings are precomputed and passed in directly
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=calculate_probabilities,
        verbose=True,
        # BERTopic's `language` constructor arg defaults to "english", and
        # at that default its internal _preprocess_text() step strips every
        # character that isn't a Latin letter, digit, or space -- silently
        # destroying 100% of the Korean text before it ever reaches
        # kiwi_tokenizer(). Confirmed empirically: a 1,000-doc dry run's
        # fitted vocabulary was 43 terms, every one a bare number or
        # English/Latin fragment, zero Korean tokens; the same Korean text
        # survives intact under language="multilingual", BERTopic's own
        # documented setting for non-English text.
        language="multilingual",
    )

    # TIMING INSTRUMENTATION ONLY -- no behavior change. BERTopic's own
    # verbose logger prints exactly one "Cluster - Start
    # clustering the reduced embeddings" / "Cluster - Completed" pair that
    # covers BOTH the HDBSCAN clustering fit (hdbscan_model.fit(...)) AND,
    # because calculate_probabilities=True above, the separate, subsequent
    # all_points_membership_vectors() soft-probability computation -- with
    # nothing printed in between the two, so on a multi-hour run "still
    # working" and "hung" look the same.
    #
    # Fix: wrap the real hdbscan.all_points_membership_vectors function --
    # the same one bertopic.cluster._utils.hdbscan_delegator calls
    # internally (confirmed directly against the installed package's source,
    # not assumed) -- so a start line prints the moment the clustering fit
    # hands off into the probability step, and a second line prints when
    # that step finishes with its own elapsed time. This changes NOTHING
    # about what gets computed: calculate_probabilities stays True, the
    # same hdbscan.all_points_membership_vectors call still runs with the
    # same arguments and the same return value, and topic_model.fit_transform
    # below is called exactly as before. Only visibility into the boundary
    # between the two sub-steps is new.
    import hdbscan as _hdbscan_module

    _original_all_points_membership_vectors = _hdbscan_module.all_points_membership_vectors

    def _timed_all_points_membership_vectors(model):
        n_clusters = int(model.labels_.max()) + 1 if model.labels_.max() >= 0 else 0
        print(
            f"[timing] HDBSCAN clustering fit complete -- {model.labels_.shape[0]:,} "
            f"points, {n_clusters} clusters found (excluding outlier topic -1). "
            f"Starting soft-probability computation (all_points_membership_vectors)...",
            flush=True,
        )
        _t0 = time.time()
        result = _original_all_points_membership_vectors(model)
        print(
            f"[timing] Soft-probability computation complete in {time.time() - _t0:.1f}s",
            flush=True,
        )
        return result

    _hdbscan_module.all_points_membership_vectors = _timed_all_points_membership_vectors

    topics, probs = topic_model.fit_transform(texts, embeddings)
    # When calculate_probabilities=False, BERTopic returns
    # probs=None -- np.asarray(None) would silently turn that into a useless
    # 0-d object array instead of a clear, checkable None. Keep it None.
    return topic_model, np.asarray(topics), (np.asarray(probs) if probs is not None else None)


def soft_assign_topics(probs: np.ndarray, threshold: float) -> list[list[int]]:
    """For each document, return every topic index whose soft-assignment
    probability exceeds `threshold` -- this is the actual multi-topic
    assignment mechanism (the threshold plugs in here), separate
    from BERTopic's own single hard-assigned `topics` array.

    probs has shape (n_documents, n_topics_excluding_outlier) when
    calculate_probabilities=True; column index i corresponds to
    topic id i (topic -1 is not a column here -- see note below on
    how outlier is folded back in by the caller).
    """
    assignments = []
    for row in probs:
        above = np.where(row > threshold)[0].tolist()
        assignments.append(above)
    return assignments


def soft_assign_topics_topk(
    probs: np.ndarray, k: int, min_prob: float = 0.02
) -> list[list[int]]:
    """Alternative to soft_assign_topics()'s absolute
    threshold: for each document, return its top-k
    topics by probability, regardless of absolute magnitude, as long as
    probability clears `min_prob` (a display floor to exclude
    essentially-zero noise, not a real threshold).

    Targets a pattern an absolute threshold structurally cannot catch:
    documents whose probabilities are flat across every topic and nothing cleared even
    0.07 anywhere -- no absolute cutoff, however low, would ever assign
    those documents a topic, but their probability mass still peaks
    somewhere. Whether that peak is a MEANINGFUL topic (as opposed to
    noise dressed up as a ranking) needs a separate test against real
    documents -- this function alone doesn't establish that, it just
    makes the assignment mode available to test.
    """
    assignments = []
    for row in probs:
        order = np.argsort(row)[::-1][:k]
        assignments.append([int(i) for i in order if row[i] > min_prob])
    return assignments


# =============================================================================
# SUMMARY (the run's topic-level output, and what the spot-checks read from)
# =============================================================================


def cluster_summary(topic_model, texts: list[str], topics: np.ndarray) -> pd.DataFrame:
    """Topic-level summary: size, top c-TF-IDF terms, share of corpus.
    -1 (outlier) is included as its own row: it is kept labeled rather
    than dropped, and excluded downstream only where display calls for it."""
    info = topic_model.get_topic_info()  # includes -1 by default
    total_docs = len(texts)
    info["share_of_corpus"] = info["Count"] / total_docs
    return info


def check_topic_collapse(summary: pd.DataFrame, share_threshold: float = 0.5) -> str | None:
    """Collapse detector. A noise-floor test -- running --seed 43/44 at --sample-size 1000 and --seed 42/43 at
    --sample-size 3000 (all min_cluster_size=10) surfaced a real,
    previously-unknown failure mode distinct from ordinary run-to-run
    noise -- --sample-size 3000 --seed 43 produced only 2 topics, with
    ONE non-(-1) topic absorbing 2,610/2,640 (98.86%) of the corpus into
    an undifferentiated bucket (top terms: 같/것/있/쓰/좋/사용/거/바르/하/없
    -- pure generic filler, no thematic content), while -1 share read
    0.00% -- which on a glance looks like a GOOD outlier rate, not the
    tell that something broke. The script gave no other signal: it
    exited cleanly and printed "2 topics found" same as any healthy run.

    This check is the fix for that blind spot -- diagnostic only, it
    does not change clustering behavior, min_cluster_size, or anything
    else. It fires when a single non-(-1) topic's share of the corpus
    exceeds `share_threshold`. The default (0.5) is set with a wide
    margin: across the 5 real runs seen so far (2 sample sizes, 4
    seeds), every HEALTHY run's single largest non-(-1) topic stayed
    under ~13% of kept docs (n=1,000 seed=44's topic 0, 106/857 =
    12.4%, the highest of the four healthy runs) -- nowhere near 50%,
    let alone 98.86%. This runs on the SAME shared reporting path
    (cluster_summary() -> main()) that both --dry-run and the full run
    use -- deliberately, since the full 623K-row run is a
    one-shot: it won't get re-run 3x by hand the way a dry-run seed
    sweep can, so this is the one thing standing between it and a
    silently-collapsed, unusable topic set going unnoticed."""
    real_topics = summary[summary["Topic"] != -1]
    if real_topics.empty:
        return None
    worst = real_topics.loc[real_topics["share_of_corpus"].idxmax()]
    share = float(worst["share_of_corpus"])
    if share > share_threshold:
        return (
            f"topic {int(worst['Topic'])} contains {int(worst['Count']):,} docs "
            f"({share:.1%} of the corpus) -- this run likely COLLAPSED rather "
            f"than converged (every healthy run seen so far had its largest "
            f"single topic under ~13% of the corpus). Treat this output as "
            f"unreliable -- don't use it for a min_cluster_size or threshold "
            f"comparison; re-run with a different seed and/or investigate "
            f"before trusting anything downstream of it."
        )
    return None


def representative_docs_per_topic(
    topic_model, texts: list[str], n: int = 5
) -> dict[int, list[str]]:
    """BERTopic's own global representative-document excerpts per
    topic -- native output, no extra step needed. NOTE: this is global
    to a topic, not scoped to a product/month -- the per-(topic,
    product, month) `representative_phrase` needs its own selection
    rule; this function only supplies the topic-labeling input."""
    reps = {}
    for topic_id in topic_model.get_topics().keys():
        try:
            docs = topic_model.get_representative_docs(topic_id) or []
        except (KeyError, ValueError):
            # Some BERTopic versions don't index -1 the same way as real
            # topics for this call -- -1 should still be represented,
            # so fall back to a manual sample rather than dropping it.
            docs = []
        reps[topic_id] = docs[:n]
    return reps


# =============================================================================
# DRY-RUN THRESHOLD COMPARISON
# =============================================================================


def compare_thresholds(probs: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    """For each candidate threshold, report the distribution of
    per-document topic counts -- the empirical check for picking the
    threshold: a sane outcome lands most multi-point segments around
    3-4 topics, not 1 (too strict) and not 15 (too loose)."""
    rows = []
    for threshold in thresholds:
        counts = [len(np.where(row > threshold)[0]) for row in probs]
        counts = np.asarray(counts)
        rows.append(
            {
                "threshold": threshold,
                "mean_topics_per_doc": counts.mean(),
                "median_topics_per_doc": float(np.median(counts)),
                "max_topics_per_doc": int(counts.max()) if len(counts) else 0,
                "pct_docs_zero_topics": float((counts == 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def compare_topk(probs: np.ndarray, k_values: list[int], min_prob: float = 0.02) -> pd.DataFrame:
    """Top-k counterpart to compare_thresholds() -- for each
    candidate k, report per-document assigned-topic-count stats under
    soft_assign_topics_topk(). Unlike the threshold comparison,
    "pct_docs_zero_topics" here specifically isolates documents whose
    ENTIRE probability row is at or below min_prob (truly flat/near-
    zero everywhere, not just below some threshold) -- top-k has no
    other way to produce zero topics."""
    rows = []
    for k in k_values:
        assignments = soft_assign_topics_topk(probs, k, min_prob=min_prob)
        counts = np.asarray([len(a) for a in assignments])
        rows.append(
            {
                "k": k,
                "mean_topics_per_doc": counts.mean(),
                "median_topics_per_doc": float(np.median(counts)),
                "pct_docs_zero_topics": float((counts == 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def topk_rescue_report(
    probs: np.ndarray, threshold: float, k: int, min_prob: float = 0.02
) -> dict:
    """Direct test of what top-k assignment is for: of the documents
    that get ZERO topics under the absolute `threshold` (soft_assign_
    topics), how many gain at least one under top-k instead? Top-k targets
    exactly the flat-probability pattern an absolute threshold
    structurally can't reach, no matter how low it's set."""
    threshold_assignments = soft_assign_topics(probs, threshold)
    topk_assignments = soft_assign_topics_topk(probs, k, min_prob=min_prob)
    zero_under_threshold = [
        i for i, a in enumerate(threshold_assignments) if len(a) == 0
    ]
    rescued = [i for i in zero_under_threshold if len(topk_assignments[i]) > 0]
    return {
        "threshold": threshold,
        "k": k,
        "n_zero_under_threshold": len(zero_under_threshold),
        "n_rescued_by_topk": len(rescued),
        "pct_rescued": (len(rescued) / len(zero_under_threshold)) if zero_under_threshold else 0.0,
    }


# =============================================================================
# GATE 3 -- OUTLIER RESCUE with reduce_outliers (--gate3-rescue), run on the
# existing fit rather than re-clustering
# =============================================================================


def load_soft_rescue_reference(path: Path) -> set[str]:
    """Load the review_ids that are hard -1 in the production run but
    were soft-assigned to a real topic anyway by the separate 0.1
    probability threshold (13,593 rows). These rows already have a
    disposition from that mechanism -- Gate 3 leaves them alone rather
    than double-processing them.

    Expects `path` (typically phase2_window12mo_final_assignments.csv)
    to have review_id and assigned_topics columns -- assigned_topics is
    the JSON-list column this same script's soft_assign_topics() branch
    writes; a row is in the protected set if that list is non-empty.
    """
    ref = pd.read_csv(path, usecols=["review_id", "assigned_topics"], dtype={"review_id": str})

    def _has_soft_assignment(cell) -> bool:
        if pd.isna(cell):
            return False
        try:
            parsed = json.loads(cell)
        except (TypeError, ValueError):
            return False
        return isinstance(parsed, list) and len(parsed) > 0

    protected = ref.loc[ref["assigned_topics"].apply(_has_soft_assignment), "review_id"]
    return set(protected.astype(str).tolist())


def gate3_rescue_dry_run(
    topic_model, texts: list[str], topics, thresholds: list[float], strategy: str,
    embeddings: np.ndarray | None = None, exclude_topics: list[int] | None = None,
    protected_ids: set[str] | None = None, review_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Gate 3 threshold comparison -- mirrors compare_thresholds()'s
    role for the soft-assignment threshold, but for reduce_outliers'
    own threshold. For each candidate, runs reduce_outliers once
    (nothing here is written back to topic_model or to disk as a final
    assignment) and reports how many originally-outlier rows would
    move, how many of those moves would land in a Non-Complaint/
    Excluded topic (and so get reverted by the real pass's post-hoc
    filter -- see gate3_apply_rescue below), and excludes protected
    rows (the already soft-assigned overlap) from the "would rescue" count
    -- previewing exactly what a real run at that threshold would keep.
    Does NOT pick a threshold; that is read from this table by eye, not
    derived here.
    """
    exclude_topics_set = set(exclude_topics or [])
    protected_ids = protected_ids or set()
    topics_arr = np.asarray(topics)
    n_outlier = int((topics_arr == -1).sum())
    protected_mask_base = (
        np.array([rid in protected_ids for rid in review_ids])
        if review_ids is not None and protected_ids else np.zeros_like(topics_arr, dtype=bool)
    )

    rows = []
    for t in thresholds:
        new_topics = topic_model.reduce_outliers(
            texts, list(topics_arr), strategy=strategy, threshold=t, embeddings=embeddings,
        )
        new_topics_arr = np.asarray(new_topics)
        moved_mask = (topics_arr == -1) & (new_topics_arr != -1)
        excluded_mask = moved_mask & np.isin(new_topics_arr, list(exclude_topics_set))
        protected_moved_mask = moved_mask & protected_mask_base & ~excluded_mask
        kept_mask = moved_mask & ~excluded_mask & ~protected_mask_base
        dest_counts = pd.Series(new_topics_arr[kept_mask]).value_counts().head(5).to_dict()
        rows.append({
            "threshold": t,
            "n_outlier_before": n_outlier,
            "n_moved_raw": int(moved_mask.sum()),
            "n_reverted_excluded_topic": int(excluded_mask.sum()),
            "n_reverted_soft_rescue_protected": int(protected_moved_mask.sum()),
            "n_kept_rescued": int(kept_mask.sum()),
            "pct_of_outliers_rescued": (int(kept_mask.sum()) / n_outlier) if n_outlier else 0.0,
            "top5_destination_topics": dest_counts,
        })
    return pd.DataFrame(rows)


def gate3_apply_rescue(
    topic_model, texts: list[str], topics, strategy: str, threshold: float,
    exclude_topics: list[int] | None, embeddings: np.ndarray | None = None,
    protected_ids: set[str] | None = None, review_ids: list[str] | None = None,
):
    """Gate 3's real, single rescue pass. Runs
    reduce_outliers once at (strategy, threshold), then applies two
    post-hoc reverts back to -1, since reduce_outliers has no native
    way to prevent either (checked against the official docs):

      1. Rows reassigned into a Non-Complaint/Excluded topic (a
         contamination risk -- re-injecting exactly what topic labeling
         removed).
      2. Rows in the protected set (already soft-assigned by the
         separate threshold path) -- left as they were, not
         double-processed.

    Returns (final_topics_array, kept_rescued_mask, report_dict) --
    kept_rescued_mask marks only genuinely-new, non-reverted rescues,
    the population the manual validation sample should be drawn from.
    Nothing here silently overwrites; every count is in report_dict.
    """
    exclude_topics_set = set(exclude_topics or [])
    protected_ids = protected_ids or set()
    topics_arr = np.asarray(topics)
    n_outlier_before = int((topics_arr == -1).sum())

    raw_new_topics = topic_model.reduce_outliers(
        texts, list(topics_arr), strategy=strategy, threshold=threshold, embeddings=embeddings,
    )
    raw_new_topics_arr = np.asarray(raw_new_topics)
    final_topics_arr = raw_new_topics_arr.copy()

    moved_mask = (topics_arr == -1) & (raw_new_topics_arr != -1)
    excluded_topic_mask = moved_mask & np.isin(raw_new_topics_arr, list(exclude_topics_set))
    protected_mask_raw = (
        moved_mask & np.array([rid in protected_ids for rid in review_ids])
        if review_ids is not None and protected_ids else np.zeros_like(moved_mask)
    )
    # excluded_topic_mask and protected_mask_raw can overlap (a row can land
    # in an excluded topic AND already be in the soft-rescue-protected set),
    # so counting them as independent tallies would make the report's
    # sub-counts sum to MORE than the rows actually reverted (on the real
    # 0.2 threshold run: a 1,242-row overlap). As in gate3_rescue_dry_run,
    # protected_moved_mask excludes excluded_mask, so the three reported
    # categories are mutually exclusive and sum exactly to n_raw_moved.
    # (revert_mask below is a correct union either way.)
    protected_mask = protected_mask_raw & ~excluded_topic_mask
    both_mask = excluded_topic_mask & protected_mask_raw

    revert_mask = excluded_topic_mask | protected_mask_raw
    final_topics_arr[revert_mask] = -1
    kept_rescued_mask = moved_mask & ~revert_mask

    report = {
        "strategy": strategy,
        "threshold": threshold,
        "exclude_topics": sorted(exclude_topics_set),
        "n_protected_ids_supplied": len(protected_ids),
        "n_outlier_before": n_outlier_before,
        "n_raw_moved_by_reduce_outliers": int(moved_mask.sum()),
        "n_reverted_excluded_topic": int(excluded_topic_mask.sum()),
        "n_reverted_soft_rescue_protected": int(protected_mask.sum()),
        "n_reverted_both_excluded_and_protected": int(both_mask.sum()),
        "n_kept_rescued": int(kept_rescued_mask.sum()),
        "n_still_outlier_after_gate3": int((final_topics_arr == -1).sum()),
        "pct_of_original_outliers_rescued": (
            int(kept_rescued_mask.sum()) / n_outlier_before if n_outlier_before else 0.0
        ),
    }
    return final_topics_arr, kept_rescued_mask, report


def hard_topics_sidecar_path(model_path: Path) -> Path:
    """Gate 3 model caching: the sidecar CSV that travels
    with a --save-model / --load-model pickle.

    Deliberately NOT trusting topic_model.topics_ to survive the
    save/load round-trip intact -- BERTopic's own serialization docs
    confirm pickle mode "saves the complete model" but don't explicitly
    document whether topics_/probabilities_ specifically persist, and
    "probably fine" isn't good enough ahead of a decision that skips an
    ~1h11m refit. Reconstructing
    topics by review_id on load, not by array position, is the same
    defensive pattern as --embeddings-file's own alignment check below
    -- if anything drifted between the save and load runs, this catches
    it as a missing-key error instead of a silent misalignment.
    """
    return Path(str(model_path) + ".hard_topics.csv")


def save_hard_topics_sidecar(model_path: Path, review_ids: list[str], topics) -> None:
    sidecar = pd.DataFrame({"review_id": review_ids, "topic": np.asarray(topics)})
    sidecar.to_csv(hard_topics_sidecar_path(model_path), index=False)


def load_hard_topics_sidecar(model_path: Path, expected_review_ids: list[str]) -> list[int]:
    path = hard_topics_sidecar_path(model_path)
    sidecar = pd.read_csv(path, dtype={"review_id": str})
    lookup = dict(zip(sidecar["review_id"], sidecar["topic"]))
    missing = [rid for rid in expected_review_ids if rid not in lookup]
    if missing:
        raise ValueError(
            f"--load-model's hard-topics sidecar ({path}) is missing "
            f"{len(missing):,} review_id(s) present in this run's "
            "Tier-0-filtered corpus -- most likely --input/--sample-size/"
            "--seed/--skip-no-complaint-filter don't match the run that "
            "produced --save-model. Not filling gaps silently; re-check "
            "the command matches the original fit exactly."
        )
    return [int(lookup[rid]) for rid in expected_review_ids]


# =============================================================================
# MAIN
# =============================================================================


def load_corpus(input_path: Path, sample_size: int | None, seed: int = RANDOM_SEED) -> pd.DataFrame:
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df = df[df["cons_ko"].notna() & (df["cons_ko"].str.strip() != "")]
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="phase2_complaint_corpus.csv")
    parser.add_argument("--sample-size", type=int, default=None, help="Dry-run sample size")
    parser.add_argument("--dry-run", action="store_true", help="Print threshold comparison, don't write final assignments")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2], help="Candidate thresholds to compare in --dry-run mode")
    parser.add_argument("--threshold", type=float, default=None, help="Locked threshold for a non-dry-run assignment pass")
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument(
        "--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
        help="Decouples HDBSCAN's min_samples from min_cluster_size, which otherwise "
        "move in lockstep (a 20->200 min_cluster_size move is silently also a 10x "
        "jump in this parameter). "
        "Defaults to DEFAULT_MIN_SAMPLES (None -> HDBSCAN itself sets min_samples "
        "= min_cluster_size), so every prior run and result is unaffected unless "
        "this is passed explicitly. A min_samples sweep of [5, 10, 25, 50, 100, 200] at "
        "fixed --min-cluster-size 200 uses the 200 point as a control: it must "
        "reproduce 68 topics / 45.41%% outliers exactly, same as the production run.",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Which RANDOM_SEED-style draw load_corpus() samples with. "
        "Defaults to RANDOM_SEED (42), matching every prior run. Override this "
        "ONLY to draw a different independent 1,000-row sample for noise-floor "
        "testing (see the module docstring) -- e.g. --seed 43, --seed 44 "
        "at the same --sample-size, to see how much topic count/-1 share vary "
        "just from which rows got sampled, not from any real pipeline change.",
    )
    parser.add_argument("--output-prefix", type=str, default="bertopic_dryrun")
    parser.add_argument(
        "--topk-values", type=int, nargs="+", default=[2, 3],
        help="Candidate k values for the relative top-k assignment comparison in --dry-run mode",
    )
    parser.add_argument(
        "--topk-min-prob", type=float, default=0.02,
        help="Display floor for top-k assignment -- excludes essentially-zero noise from the top-k pick, not a real threshold",
    )
    parser.add_argument(
        "--skip-no-complaint-filter", action="store_true",
        help="Skip the Tier 0 'no complaint' meta-negation pre-filter -- off by default, use this to compare against the pre-Tier-0 behavior",
    )
    parser.add_argument(
        "--embeddings-file", type=Path, default=None,
        help="Load a precomputed embedding matrix (.npy, from "
        "embed_on_kaggle.py) instead of calling embed_documents() "
        "locally -- at real scale the embedding forward pass runs on "
        "Kaggle, not on a local machine. Verified against this run's own "
        "Tier-0-filtered review_id order before use; see "
        "--embeddings-review-ids.",
    )
    parser.add_argument(
        "--embeddings-review-ids", type=Path, default=None,
        help="review_id list (one column, same row order as "
        "--embeddings-file's matrix) written by embed_on_kaggle.py. "
        "Defaults to --embeddings-file with its suffix replaced by "
        "'_review_ids.csv' if omitted.",
    )
    parser.add_argument(
        "--skip-probabilities", action="store_true",
        help="Skip calculate_probabilities / the "
        "all_points_membership_vectors soft-probability step entirely -- "
        "measured at ~7h53m on the full 543,760-row corpus, the single "
        "largest cost in the pipeline. Writes ONLY {output-prefix}_cluster_"
        "summary.csv (topic count, sizes, outlier share -- none of which "
        "depend on probabilities, since HDBSCAN's hard cluster assignment "
        "happens independently of the soft-probability step). For a "
        "min_cluster_size sweep answering 'how many topics does this "
        "config produce' ONLY -- never for the real assignment pass, "
        "which needs probabilities to threshold. Off by default; every "
        "existing and future real run is unaffected unless this is passed "
        "explicitly. Incompatible with --dry-run and --threshold (both "
        "require probabilities) -- hard error if combined.",
    )
    parser.add_argument(
        "--gate3-rescue", action="store_true",
        help="Run reduce_outliers against the freshly-fit "
        "model instead of the normal threshold-based assignment pass. "
        "Verified against BERTopic's own docs (not assumed): neither the "
        "c-tf-idf nor the embeddings strategy needs the probability matrix "
        "-- only 'probabilities'/'distributions' strategies do -- so this "
        "is normally combined with --skip-probabilities. Incompatible with "
        "--dry-run (a different, unrelated diagnostic mode -- the "
        "soft-assignment threshold comparison, not this one) -- hard error if combined. "
        "See --rescue-dry-run/--rescue-strategy/--rescue-threshold/"
        "--rescue-exclude-topics/--gate3-soft-rescue-reference below.",
    )
    parser.add_argument(
        "--rescue-dry-run", action="store_true",
        help="With --gate3-rescue: compare --rescue-thresholds side by side "
        "(how many rows would be rescued, how many reverted by the "
        "exclusion filter, at each candidate) without writing a final "
        "assignment -- mirrors --dry-run's role for the soft-assignment "
        "threshold, for this gate's own threshold instead. Requires --gate3-rescue.",
    )
    parser.add_argument(
        "--rescue-thresholds", type=float, nargs="+",
        default=[0.05, 0.1, 0.2, 0.3, 0.5],
        help="Candidate reduce_outliers thresholds to compare under "
        "--rescue-dry-run. Arbitrary starting spread, not a proposal -- "
        "the threshold is picked from what this table shows, not derived or "
        "defaulted blind.",
    )
    parser.add_argument(
        "--rescue-strategy", type=str, choices=["c-tf-idf", "embeddings"],
        default="c-tf-idf",
        help="reduce_outliers strategy for the locked (non-dry-run) pass. "
        "Chain fast-to-slow, c-tf-idf first -- run 'embeddings' "
        "separately over whatever's still -1 after a c-tf-idf pass if "
        "that pass under-moves (per discussion #1866, a strategy that "
        "changes nothing means the threshold is too high, not that the "
        "strategy is done).",
    )
    parser.add_argument(
        "--rescue-threshold", type=float, default=None,
        help="Locked reduce_outliers threshold for the real (non-dry-run) "
        "--gate3-rescue pass. Required outside --rescue-dry-run, same "
        "pattern as --threshold/--dry-run above.",
    )
    parser.add_argument(
        "--rescue-exclude-topics", type=int, nargs="+",
        default=[19, 24, 27, 33, 35, 54],
        help="Post-hoc filter: any row reduce_outliers "
        "reassigns into one of these topics is reverted back to -1 rather "
        "than kept, since reduce_outliers has no native way to restrict "
        "eligible destination topics (checked against the official docs, "
        "not assumed). Defaults to this project's own Non-Complaint/"
        "Excluded topic set -- override only with a "
        "reason, this is the guard against re-injecting excluded content.",
    )
    parser.add_argument(
        "--gate3-soft-rescue-reference", type=Path, default=None,
        help="Path to phase2_window12mo_final_assignments.csv (or "
        "equivalent) -- used to identify the 13,593 rows "
        "that are hard -1 but already soft-assigned by the separate "
        "0.1 threshold path (non-empty assigned_topics column). "
        "Gate 3 leaves these rows alone rather than "
        "double-processing them. Required with --gate3-rescue unless "
        "--gate3-skip-soft-rescue-protection is passed explicitly -- no "
        "silent default, this is a deliberate safeguard.",
    )
    parser.add_argument(
        "--gate3-skip-soft-rescue-protection", action="store_true",
        help="Explicit opt-out of the --gate3-soft-rescue-reference "
        "protection above. Only use this with a real reason -- it means "
        "Gate 3 may reassign rows the soft-assignment threshold already "
        "gave a topic to.",
    )
    parser.add_argument(
        "--save-model", type=Path, default=None,
        help="Cache the fitted model to disk (pickle) after this run's "
        "refit, so a later --load-model run can skip repeating it: --gate3-rescue's "
        "own --rescue-dry-run and locked-threshold passes are two "
        "separate invocations, and without this, each one independently "
        "pays the ~1h11m refit measured for this exact config (window "
        "corpus, mcs=200, --skip-probabilities, from the original mcs "
        "sweep's own timing -- not guessed). Uses serialization='pickle' "
        "deliberately, not safetensors/pytorch: BERTopic's own docs "
        "confirm pickle 'saves the complete model' but don't document "
        "whether the lighter formats retain what reduce_outliers needs "
        "(they explicitly drop the HDBSCAN/UMAP objects) -- pickle's "
        "larger file size (100s of MB) and slower load are irrelevant "
        "costs for a same-machine, same-session intermediate file, so "
        "there's no reason to take that risk here. Also writes a small "
        "<path>.hard_topics.csv sidecar (review_id, hard topic) -- see "
        "--load-model.",
    )
    parser.add_argument(
        "--load-model", type=Path, default=None,
        help="Load a model written by --save-model instead of "
        "refitting: skips --embeddings-file loading and the whole "
        "UMAP/HDBSCAN/build_topic_model/fit_transform sequence "
        "entirely. Still runs load_corpus()/Tier 0 filtering (a few "
        "minutes, not hours) to reconstruct texts/review_ids the exact "
        "same deterministic way as the original fit, and reads hard "
        "topic labels from <path>.hard_topics.csv keyed by review_id "
        "(not by row position -- verified aligned, not assumed) rather "
        "than trust topic_model.topics_ survived the round-trip. Only "
        "wired up for --gate3-rescue right now (the --dry-run/"
        "--threshold branches need probs, which this path doesn't "
        "have) -- hard error if combined with anything else.",
    )
    args = parser.parse_args()

    if args.skip_probabilities and args.dry_run:
        raise SystemExit(
            "--skip-probabilities cannot be combined with --dry-run: the "
            "dry-run threshold/top-k comparisons both require probabilities. "
            "Use --skip-probabilities alone (no --dry-run, no --threshold) "
            "for a cheap topic-count/outlier-share-only read."
        )
    if args.skip_probabilities and args.threshold is not None:
        raise SystemExit(
            "--skip-probabilities cannot be combined with --threshold: no "
            "probabilities are computed, so there is nothing to threshold. "
            "Use --skip-probabilities alone -- it writes only "
            f"{args.output_prefix}_cluster_summary.csv, no assignments."
        )
    if args.gate3_rescue and args.dry_run:
        raise SystemExit(
            "--gate3-rescue cannot be combined with --dry-run: --dry-run is "
            "the soft-assignment threshold comparison, unrelated to Gate "
            "3's rescue. Use --rescue-dry-run instead to compare Gate 3's "
            "own threshold candidates."
        )
    if (args.rescue_dry_run or args.rescue_threshold is not None
            or args.gate3_soft_rescue_reference is not None
            or args.gate3_skip_soft_rescue_protection) and not args.gate3_rescue:
        raise SystemExit(
            "--rescue-dry-run/--rescue-threshold/--gate3-soft-rescue-reference/"
            "--gate3-skip-soft-rescue-protection all require --gate3-rescue."
        )
    if args.gate3_rescue and not args.rescue_dry_run and args.rescue_threshold is None:
        raise SystemExit(
            "--rescue-threshold is required with --gate3-rescue outside "
            "--rescue-dry-run mode (threshold set from evidence "
            "shown by --rescue-dry-run, not the library default)."
        )
    if (args.gate3_rescue and not args.gate3_skip_soft_rescue_protection
            and args.gate3_soft_rescue_reference is None):
        raise SystemExit(
            "--gate3-rescue needs --gate3-soft-rescue-reference (pointing "
            "at phase2_window12mo_final_assignments.csv) to protect Gate "
            "0a's 13,593-row soft-rescued overlap, or an explicit "
            "--gate3-skip-soft-rescue-protection to consciously skip that "
            "guard. Not defaulted silently either way."
        )
    if args.save_model and args.load_model:
        raise SystemExit(
            "--save-model and --load-model are contradictory in the same "
            "run: --save-model caches a fresh fit, --load-model skips "
            "fitting entirely. Use --save-model on the run that fits "
            "(e.g. the --rescue-dry-run pass), then --load-model alone "
            "on the later run that reuses it."
        )
    if args.load_model and not args.gate3_rescue:
        raise SystemExit(
            "--load-model is only wired up for --gate3-rescue -- the "
            "--dry-run/--threshold branches need the probability matrix, "
            "which a loaded model doesn't have (this refit always runs "
            "with --skip-probabilities). Combine with --gate3-rescue."
        )

    print("BERTopic pipeline")
    print("=" * 50)

    df = load_corpus(args.input, args.sample_size, seed=args.seed)
    n_before_prefilter = len(df)
    print(f"\nLoaded {n_before_prefilter:,} non-empty complaint segments"
          + (f" (sampled from a larger file, seed={args.seed})" if args.sample_size else ""))

    # Tier 0: drop "no complaint" meta-negation segments before
    # embedding -- see the module-level comment block above
    # is_no_complaint_segment() for what this targets. Applied here in
    # main(), not inside load_corpus(), so load_corpus() keeps returning
    # the identical sample other scripts reproduce.
    if args.skip_no_complaint_filter:
        dropped_no_complaint = df.iloc[0:0]
        print("\n--skip-no-complaint-filter set: Tier 0 pre-filter NOT applied.")
    else:
        df, dropped_no_complaint = filter_no_complaint_segments(df)
        n_dropped = len(dropped_no_complaint)
        pct_dropped = n_dropped / n_before_prefilter if n_before_prefilter else 0.0
        print(f"\nTier 0 pre-filter: dropped {n_dropped:,} / {n_before_prefilter:,} "
              f"'no complaint' meta-negation segments ({pct_dropped:.1%}).")
        if n_dropped:
            sample_cols = [c for c in ("review_id", "cons_ko") if c in dropped_no_complaint.columns]
            dropped_sample_path = f"{args.output_prefix}_prefilter_dropped_sample.csv"
            dropped_no_complaint[sample_cols].head(30).to_csv(
                dropped_sample_path, index=False, encoding="utf-8-sig"
            )
            print(f"  Sample of dropped rows (up to 30) written to {dropped_sample_path} -- "
                  "spot-check before trusting this at full scale.")

    texts = df["cons_ko"].tolist()
    print(f"\n{len(texts):,} segments remain after the Tier 0 pre-filter.")

    review_ids_for_run = df["review_id"].astype(str).tolist()

    if args.load_model:
        print(f"\n--load-model set: loading the cached fit from "
              f"{args.load_model} instead of refitting -- skipping "
              "--embeddings-file/embedding and the whole "
              "UMAP/HDBSCAN/build_topic_model step entirely.")
        from bertopic import BERTopic as _BERTopicForLoad
        topic_model = _BERTopicForLoad.load(args.load_model)
        topics = load_hard_topics_sidecar(args.load_model, review_ids_for_run)
        probs = None
        embeddings = None
        # Only needed if this run's --rescue-strategy is 'embeddings';
        # 'c-tf-idf' (the default) needs neither this nor the model's
        # own UMAP/HDBSCAN objects, which .load() does not restore.
        if args.embeddings_file and args.rescue_strategy == "embeddings":
            embeddings, review_ids_loaded = load_precomputed_embeddings(
                args.embeddings_file, args.embeddings_review_ids
            )
            if review_ids_loaded != review_ids_for_run:
                raise ValueError(
                    "--embeddings-file review_id mismatch under "
                    "--load-model -- same hard error as the normal fit "
                    "path, same reason: a mismatched embeddings file "
                    "would silently corrupt --rescue-strategy "
                    "embeddings' similarity computation."
                )
        print(f"  Loaded {len(topics):,} hard topic labels from "
              f"{hard_topics_sidecar_path(args.load_model)}, realigned "
              "by review_id (not trusted by position).")
    else:
        if args.embeddings_file:
            embeddings, review_ids_loaded = load_precomputed_embeddings(
                args.embeddings_file, args.embeddings_review_ids
            )
            if review_ids_loaded != review_ids_for_run:
                raise ValueError(
                    f"--embeddings-file review_id mismatch: the loaded matrix "
                    f"has {len(review_ids_loaded):,} rows, this run's "
                    f"Tier-0-filtered corpus has {len(review_ids_for_run):,}. "
                    "Loading embeddings computed against a different "
                    "--input/--sample-size/--seed/--skip-no-complaint-filter "
                    "combination than this run would silently corrupt every "
                    "cluster assignment, so this is a hard error, not a "
                    "warning. Re-check embed_on_kaggle.py ran against the "
                    "exact same corpus and Tier 0 settings as this command."
                )
            print(f"\nLoaded precomputed embeddings from {args.embeddings_file} "
                  f"({embeddings.shape[0]:,} x {embeddings.shape[1]}), "
                  "review_id alignment verified against this run's "
                  "Tier-0-filtered corpus.")
        else:
            print(f"\nEmbedding with {EMBEDDING_MODEL}...")
            embeddings = embed_documents(texts)

        skip_prob_note = " [--skip-probabilities: soft-probability step will be skipped]" if args.skip_probabilities else ""
        min_samples_note = args.min_samples if args.min_samples is not None else "None (-> min_cluster_size, HDBSCAN default)"
        print(f"\nClustering (min_cluster_size={args.min_cluster_size}, min_samples={min_samples_note}){skip_prob_note}...")
        topic_model, topics, probs = build_topic_model(
            texts, embeddings, min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            calculate_probabilities=not args.skip_probabilities,
        )

        if args.save_model:
            print(f"\n--save-model set: saving the fitted model to "
                  f"{args.save_model} (serialization='pickle') so a "
                  "later --load-model run can skip this refit...")
            topic_model.save(args.save_model, serialization="pickle")
            save_hard_topics_sidecar(args.save_model, review_ids_for_run, topics)
            print(f"  Saved. Sidecar: {hard_topics_sidecar_path(args.save_model)}")

    summary = cluster_summary(topic_model, texts, topics)

    # Save every artifact this run produces to disk IMMEDIATELY after it's
    # computed, before any of it gets printed to the console: a display
    # step that crashes (see the sys.stdout/stderr.reconfigure block near
    # the top of this file) must never take a multi-hour run's results with
    # it. Printing is a convenience; saving is the point of the run. So: for
    # the full-scale (non-dry-run) path specifically, compute and write
    # {output_prefix}_assignments.csv (the per-row topic assignments) here
    # too, before any of the diagnostic printing below.
    # (The --dry-run branch further down is left as-is -- there, each
    # print is intentionally paired with its own smaller-scale save, and
    # a --dry-run session is an interactive tuning run, not an unattended
    # multi-hour one, so the print-before-save cost of a hypothetical
    # future crash there is minutes of reruns, not a full day.)
    summary.to_csv(f"{args.output_prefix}_cluster_summary.csv", index=False)

    # --skip-probabilities writes ONLY cluster_summary.csv
    # -- no probs exist to threshold, so the assignments block below (which
    # the --threshold guard protects) is skipped entirely here, not deferred.
    # See --skip-probabilities' own --help text; validated at parse time above
    # so this branch and --dry-run's branch are mutually exclusive by then.
    if args.skip_probabilities or args.gate3_rescue:
        pass
    elif not args.dry_run:
        if args.threshold is None:
            raise SystemExit("--threshold is required outside --dry-run mode (pick it from a dry run first).")
        assignments = soft_assign_topics(probs, args.threshold)
        df["assigned_topics"] = [json.dumps(a) for a in assignments]
        df["hard_topic"] = topics
        assignments_output_path = Path(f"{args.output_prefix}_assignments.csv")
        df.to_csv(assignments_output_path, index=False, encoding="utf-8-sig")

    n_topics = int((summary["Topic"] != -1).sum())
    outlier_row = summary[summary["Topic"] == -1]
    outlier_share = float(outlier_row["share_of_corpus"].iloc[0]) if len(outlier_row) else 0.0
    print(f"\n{n_topics} topics found (excluding -1).")
    print(f"-1 (outlier) share of corpus: {outlier_share:.2%}")

    # Collapse detector -- see check_topic_collapse()'s own docstring. Runs
    # every time (dry run and full run alike), since a
    # collapsed run otherwise looks identical to a healthy one in the
    # console output above.
    collapse_warning = check_topic_collapse(summary)
    if collapse_warning:
        print(f"\n{'!' * 70}")
        print(f"WARNING: {collapse_warning}")
        print(f"{'!' * 70}")

    print("\nTop 10 topics by size:")
    print(summary.sort_values("Count", ascending=False).head(10).to_string(index=False))

    if args.dry_run:
        # --- Diagnostic: several topics in an initial dry
        # run showed BLANK Representation columns in get_topic_info() (all
        # 10 slots empty) despite kiwi_tokenizer() producing plenty of real
        # content words when tested directly against those topics'
        # representative docs -- ruling out the tokenizer as the cause.
        # This prints BERTopic's raw get_topic() output (actual (word,
        # score) pairs, no DataFrame truncation/padding) plus vocabulary
        # size, to tell a genuine all-zero c-TF-IDF score apart from a
        # display-only artifact in get_topic_info(). Not a guess -- direct
        # inspection of the object BERTopic actually computed.
        print("\nRaw get_topic() output for each found topic (diagnostic --")
        print("blank Representation columns above, verifying against the")
        print("model's own scores rather than assuming a cause):")
        vocab_size = len(topic_model.vectorizer_model.vocabulary_)
        print(f"  Vectorizer vocabulary size: {vocab_size:,} terms")
        for topic_id in sorted(summary["Topic"].tolist()):
            raw = topic_model.get_topic(topic_id)
            n_nonzero = sum(1 for _, score in raw if score > 0)
            print(f"  Topic {topic_id}: {n_nonzero}/{len(raw)} non-zero-score terms -- {raw[:5]}")

        print("\nChecking negation-pattern prevalence (see kiwi_tokenizer()'s")
        print("docstring -- four of five patterns are currently invisible")
        print("to the topic-labeling step)...")
        negation_stats = count_negation_patterns(texts)
        print(f"  -지 않다 (invisible):        {negation_stats['aux_negation_-지_않다_count']:,} / {negation_stats['n_documents']:,} "
              f"({negation_stats['aux_negation_-지_않다_pct']:.1%})")
        print(f"  안 X (invisible):            {negation_stats['prefix_negation_안_count']:,} / {negation_stats['n_documents']:,} "
              f"({negation_stats['prefix_negation_안_pct']:.1%})")
        print(f"  없다 (partially kept):       {negation_stats['existential_negation_없다_count']:,} / {negation_stats['n_documents']:,} "
              f"({negation_stats['existential_negation_없다_pct']:.1%})")
        print(f"  아니다/아니라 (invisible):   {negation_stats['copula_negation_아니다_count']:,} / {negation_stats['n_documents']:,} "
              f"({negation_stats['copula_negation_아니다_pct']:.1%})")
        print(f"  -지 못하다/못 (invisible):   {negation_stats['incapacity_negation_못_count']:,} / {negation_stats['n_documents']:,} "
              f"({negation_stats['incapacity_negation_못_pct']:.1%})")
        with open(f"{args.output_prefix}_negation_stats.json", "w", encoding="utf-8") as f:
            json.dump(negation_stats, f, ensure_ascii=False, indent=2)

        print(f"\nComparing thresholds {args.thresholds}...")
        comparison = compare_thresholds(probs, args.thresholds)
        print(comparison.to_string(index=False))
        comparison.to_csv(f"{args.output_prefix}_threshold_comparison.csv", index=False)

        # Relative top-k assignment as an alternative to the absolute
        # threshold above -- see soft_assign_topics_topk()'s docstring for
        # what this targets (flat-probability documents, which no absolute
        # threshold can ever reach).
        print(f"\nComparing top-k assignment {args.topk_values} ("
              f"min_prob={args.topk_min_prob})...")
        topk_comparison = compare_topk(probs, args.topk_values, min_prob=args.topk_min_prob)
        print(topk_comparison.to_string(index=False))
        topk_comparison.to_csv(f"{args.output_prefix}_topk_comparison.csv", index=False)

        lowest_threshold = min(args.thresholds)
        print(f"\nRescue check: of the docs with zero topics at threshold={lowest_threshold} "
              f"(the most lenient candidate), how many gain >=1 topic under top-k instead?")
        rescue_rows = []
        for k in args.topk_values:
            rescue = topk_rescue_report(probs, lowest_threshold, k, min_prob=args.topk_min_prob)
            rescue_rows.append(rescue)
            print(f"  k={k}: {rescue['n_rescued_by_topk']:,} / {rescue['n_zero_under_threshold']:,} "
                  f"rescued ({rescue['pct_rescued']:.1%})")
        with open(f"{args.output_prefix}_topk_rescue_report.json", "w", encoding="utf-8") as f:
            json.dump(rescue_rows, f, ensure_ascii=False, indent=2)

        reps = representative_docs_per_topic(topic_model, texts)
        with open(f"{args.output_prefix}_representative_docs.json", "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in reps.items()}, f, ensure_ascii=False, indent=2)

        print(f"\nDry-run outputs written:")
        print(f"  {args.output_prefix}_cluster_summary.csv")
        print(f"  {args.output_prefix}_threshold_comparison.csv")
        print(f"  {args.output_prefix}_topk_comparison.csv")
        print(f"  {args.output_prefix}_topk_rescue_report.json")
        print(f"  {args.output_prefix}_representative_docs.json")
        if not args.skip_no_complaint_filter and len(dropped_no_complaint):
            print(f"  {args.output_prefix}_prefilter_dropped_sample.csv")
        print("\nNext: inspect a handful of clusters by eye, pick a threshold")
        print("(or a top-k mode) from the comparisons above, then re-run")
        print("without --dry-run at full scale with --threshold set.")
    elif args.gate3_rescue:
        print(f"\n--gate3-rescue set: reduce_outliers pass "
              f"(strategy={args.rescue_strategy!r}, "
              + (f"DRY RUN over {args.rescue_thresholds}" if args.rescue_dry_run
                 else f"threshold={args.rescue_threshold}") + ").")

        protected_ids: set[str] = set()
        if args.gate3_skip_soft_rescue_protection:
            print("  !! --gate3-skip-soft-rescue-protection set: the 13,593-row "
                  "hard-outlier-but-soft-rescued overlap is NOT protected this run. "
                  "These rows are meant to be left alone -- confirm "
                  "this is intentional before trusting the output.")
        else:
            print(f"  Loading soft-rescue protection reference from "
                  f"{args.gate3_soft_rescue_reference}...")
            protected_ids = load_soft_rescue_reference(args.gate3_soft_rescue_reference)
            print(f"  {len(protected_ids):,} review_ids protected (already soft-rescued "
                  "by the soft-assignment threshold -- Gate 3 will not touch these).")

        review_ids_list = df["review_id"].astype(str).tolist()
        embeddings_for_rescue = embeddings if args.rescue_strategy == "embeddings" else None

        if args.rescue_dry_run:
            comparison = gate3_rescue_dry_run(
                topic_model, texts, topics, args.rescue_thresholds,
                strategy=args.rescue_strategy, embeddings=embeddings_for_rescue,
                exclude_topics=args.rescue_exclude_topics,
                protected_ids=protected_ids, review_ids=review_ids_list,
            )
            print(comparison.to_string(index=False))
            dry_run_path = f"{args.output_prefix}_gate3_rescue_dry_run.csv"
            comparison.to_csv(dry_run_path, index=False)
            print(f"\nGate 3 dry-run output written: {dry_run_path}")
            print("Not a decision -- pick a threshold from this table by eye, then "
                  "re-run with --gate3-rescue --rescue-threshold <value> (no "
                  "--rescue-dry-run).")
        else:
            final_topics, rescued_mask, report = gate3_apply_rescue(
                topic_model, texts, topics, strategy=args.rescue_strategy,
                threshold=args.rescue_threshold, exclude_topics=args.rescue_exclude_topics,
                embeddings=embeddings_for_rescue, protected_ids=protected_ids,
                review_ids=review_ids_list,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))

            df["gate3_old_topic"] = topics
            df["gate3_new_topic"] = final_topics
            df["gate3_rescued"] = rescued_mask
            gate3_assignments_path = f"{args.output_prefix}_gate3_assignments.csv"
            df.to_csv(gate3_assignments_path, index=False, encoding="utf-8-sig")

            report_path = f"{args.output_prefix}_gate3_rescue_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            print(f"\nGate 3 outputs written:")
            print(f"  {gate3_assignments_path} ({len(df):,} rows -- gate3_rescued=True "
                  "marks newly-rescued rows; draw the manual validation sample from "
                  "exactly this subset)")
            print(f"  {report_path}")
            print("\nNOT run automatically: update_topics() to refresh c-TF-IDF "
                  "representations after this rescue. That step belongs after the topic "
                  "merge/exclude decisions are applied -- confirm before running it.")
    elif args.skip_probabilities:
        print(f"\n--skip-probabilities set: only {args.output_prefix}_cluster_summary.csv "
              "was written (topic count / sizes / outlier share only -- no "
              "probabilities, no assignments). Diagnostic sweep output, not a "
              "real run's deliverable.")
    else:
        # Assignments were already computed and written to disk above,
        # before any printing -- see the comment there. This just reports it.
        print(f"\nAssignments written to {assignments_output_path} ({len(df):,} rows).")


if __name__ == "__main__":
    main()
