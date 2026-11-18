"""The knowledge pipeline behind `solution.KnowledgeSystem` (which only exposes the three contract
methods and hands each call to this class).

ingest():  page store (A) -> page classifier (B) -> fact extractor (C) -> fact history (D) -> vocabulary,
           passage index (+ embeddings with a key), conflict register (G).
answer():  hybrid. The rules path parses the question (H), selects the snapshot view (I) and answers from
           resolved facts (E/F/J/K) or from passage search; when the rules are not confident, gpt-4o-mini
           reads the question (llm/understand.py) and the answer is still built from the site's own text.
conflicts(): contradictions found within each snapshot (conflicts.py).
"""
import copy

from contract import Answer, Snapshot

from knowledge.answering.facts import RATE_ATTRS, FactAnswerer
from knowledge.ingest.classify import PageClassifier
from knowledge.conflicts import ConflictRegister
from knowledge.facts.history import FactHistory
from knowledge.llm.client import LLM
from knowledge.facts.resolve import Resolver
from knowledge.ingest.store import PageStore
from knowledge.answering.passages import PassageAnswerer, PassageIndex
from knowledge.query.question import parse
from knowledge.facts.timeline import Timeline
from knowledge.llm.understand import Understander
from knowledge.query.vocab import Vocabulary

STRONG_MATCH = 0.75     # rules product-match score above which the rules fact answer is used without the LLM
STRONG_COVERAGE = 0.8   # passage coverage above which the rules passage answer is used without the LLM


class Pipeline:
    """Everything behind the three contract methods: ingest builds the stores, answer runs the hybrid
    (rules first, gpt-4o-mini for uncertain questions), conflicts reads the conflict register."""
    def __init__(self):
        self.store = PageStore()
        self.classifier = PageClassifier()
        self.classify_errors = 0
        self.facts = FactHistory(self.store)
        self.resolver = Resolver(self.facts)
        self.timeline = Timeline(self.resolver)
        self.vocab = Vocabulary(self.facts)
        self.fact_answerer = FactAnswerer(self)
        self.passages = PassageIndex(self.store)
        self.passage_answerer = PassageAnswerer(self)
        # OpenAI is optional: checked once here; without a key everything runs on rules.
        self.llm = LLM()
        self.register = ConflictRegister(self)
        self.understander = Understander(self)
        self._qvecs: dict = {}
        self._last_understanding = None
        self._last_from_facts = False
        self._last_meaning_only = False
        self._cache: dict = {}
        self.answer_errors = 0
        self.passage_errors = 0

    def ingest(self, snapshot: Snapshot) -> None:
        rec = self.store.add_snapshot(snapshot)
        for key in sorted(rec.members):
            try:
                self.classifier.classify_version(self.store.versions[key][-1])
            except Exception:
                self.classify_errors += 1
        self.facts.add_snapshot(rec)
        try:
            self.passages.add_snapshot(rec)
        except Exception:
            self.passage_errors += 1
        try:
            self.vocab.ensure()          # built at ingest so questions don't pay for it
        except Exception:
            pass
        try:
            self.passages.embed_pending(self.llm)
        except Exception:
            self.passage_errors += 1
        self.register.add_view(rec.index)       # contradictions within this snapshot (G)
        try:
            self.understander.embed_products()
        except Exception:
            self.passage_errors += 1

    def _question_vector(self, text: str):
        """Embedding of a question (cached), or None without an LLM."""
        if not self.llm.enabled:
            return None
        key = " ".join((text or "").lower().split())
        if key not in self._qvecs:
            m = self.llm.embed([text], timeout=3.0)
            self._qvecs[key] = m[0] if m is not None else None
        return self._qvecs[key]

    # ------------------------------------------------------------------ I: view selector
    def _select_view(self, ref_date):
        """Latest snapshot taken on or before `ref_date` (None = latest). None if before the first."""
        snaps = self.store.snapshots
        if not snaps:
            return None
        if ref_date is None:
            return len(snaps) - 1
        idx = None
        for rec in snaps:
            if rec.taken_on <= ref_date:
                idx = rec.index
        return idx

    def answer(self, question, as_of=None):
        ck = (" ".join((question or "").lower().split()), as_of, len(self.store.snapshots))
        hit = self._cache.get(ck)
        if hit is not None:
            return hit
        ans = self._answer(question, as_of)
        self._cache[ck] = ans
        return ans

    def _answer(self, question, as_of=None):
        """Hybrid: the rules answer first; when the rules are confident it is returned as is (fast, stable).
        Otherwise gpt-4o-mini reads the question and its answer wins; if the LLM is unavailable or fails,
        the rules answer stands."""
        self.vocab.ensure()
        q = parse(question, self.vocab)
        rules_ans, strong = self._rules_answer(q, as_of)
        if strong or not self.llm.enabled:
            return rules_ans
        rules_products = set(self.fact_answerer.last_products) if self._last_from_facts else set()
        q2 = copy.deepcopy(q)
        llm_ans = self._llm_answer(q2, as_of, rules_terms=q.terms)
        if llm_ans is None:
            return rules_ans
        # The LLM may only override the rules when it adds something:
        if not llm_ans.confident and rules_ans.confident and not self._last_meaning_only:
            # A refusal doesn't replace an answer the rules found on real evidence (a rate fact, or a
            # passage with the question's own words); it does replace a passage that was only close in
            # meaning ("timeshare loan" -> an unrelated loan page).
            return rules_ans
        u = self._last_understanding
        if (u is not None and u["route"] == "rate" and rules_products and rules_ans.confident
                and ("list" in q.intents or "list" in q2.intents)
                and set(u["products"]) < rules_products):
            return rules_ans                     # a narrower product list than the rules already gave
        return llm_ans

    # ------------------------------------------------------------------ rules path
    def _view(self, q, as_of):
        # A date in a "how did it change since May" question is the start of the change, not the view.
        ref = as_of or (None if "change" in q.intents else q.ref_date)
        return ref, self._select_view(ref)

    def _too_early(self):
        return Answer(text="I don't have information from that early; Dupaco's website records here start "
                           "later than that date, so I can't confirm what applied then.", confident=False)

    def _rules_answer(self, q, as_of):
        """(answer, confident?) from the rules path."""
        self._last_from_facts = False
        self._last_meaning_only = False
        ref, index = self._view(q, as_of)
        if index is None:
            return self._too_early(), True
        rec = self.store.snapshots[index]
        ref_date = ref if ref and ref >= rec.taken_on else rec.taken_on
        past = bool(as_of) or (q.ref_date is not None and index < len(self.store.snapshots) - 1)
        try:
            ans = self.fact_answerer.answer(q, index, ref_date, rec.taken_on, past)
        except Exception:
            self.answer_errors += 1
            ans = None
        if ans is not None:
            self._last_from_facts = True
            site_wide = q.attribute and q.attribute not in RATE_ATTRS and not q.products and ans.confident \
                and self.fact_answerer.last_score == 0.0
            strong = ans.confident and (site_wide or self.fact_answerer.last_score >= STRONG_MATCH)
            # A refusal with a concrete reason (asked variant not offered, product discontinued) is final.
            strong = strong or self.fact_answerer.last_refusal in ("missing_variant", "discontinued")
            return ans, strong
        if q.attribute in RATE_ATTRS:
            # Rates are only ever answered from extracted facts; no matching fact = not published.
            return FactAnswerer._refusal(q), False
        try:
            ans = self.passage_answerer.answer(q, index, past, rec.taken_on)
        except Exception:
            self.answer_errors += 1
            return FactAnswerer._refusal(q), False
        self._last_meaning_only = self.passage_answerer.last_meaning_only
        return ans, ans.confident and self.passage_answerer.last_coverage >= STRONG_COVERAGE

    # ------------------------------------------------------------------ LLM path
    def _llm_answer(self, q, as_of, rules_terms=()):
        """gpt-4o-mini reads the question (route, product, qualifiers, which passages answer it); the
        answer is still built from the site's own facts and sentences. None if the LLM gave nothing."""
        ref, index = self._view(q, as_of)
        if index is None:
            return self._too_early()
        self._last_understanding = None
        try:
            u = self.understander.understand(q, index)
        except Exception:
            self.answer_errors += 1
            return None
        if u is None:
            return None
        self._last_understanding = u
        if u["route"] == "rate" and not self._products_explain(q, u["products"]):
            return None                          # the LLM picked a product the question doesn't name
        Understander.apply(q, u)
        if len(rules_terms) > len(q.terms):
            q.terms = list(rules_terms)          # the question names more terms than the LLM returned
            q.term = q.terms[0]
        Understander.fix_route(q, u)
        ref, index = self._view(q, as_of)
        if index is None:
            return self._too_early()
        rec = self.store.snapshots[index]
        ref_date = ref if ref and ref >= rec.taken_on else rec.taken_on
        past = bool(as_of) or (q.ref_date is not None and index < len(self.store.snapshots) - 1)
        return self._answer_understood(q, u, index, ref_date, rec, past)

    def _products_explain(self, q, products) -> bool:
        """The chosen products must account for the question's product words: a word the site uses for
        products or variants ("construction", "adjustable") that none of the chosen products carries means
        the LLM substituted a different product. Words unknown to the site ("car") are not held against it."""
        V = self.vocab
        words = set()
        for p in products:
            info = V.products.get(p)
            if info is not None:
                words |= info.key_words | info.section_words | info.page_words
        variant_words = V.variant_words()       # words the site uses for product variants ("adjustable")
        for k in self.facts.by_key:
            if k[0] in products:
                for v in dict(k[2]).get("variant", ()):
                    words |= set(v.split())
        for w in q.content:
            if w in words or (w not in V.idf and w not in variant_words):
                continue
            return False
        return True

    def _answer_understood(self, q, u, index, ref_date, rec, past):
        if u["route"] == "rate":
            try:
                ans = self.fact_answerer.answer(q, index, ref_date, rec.taken_on, past)
            except Exception:
                self.answer_errors += 1
                ans = None
            return ans if ans is not None else FactAnswerer._refusal(q)
        if u["route"] == "passage" and (q.attribute in RATE_ATTRS or u["attribute"] in RATE_ATTRS):
            # Rates are only ever answered from extracted facts: a rate question the facts can't answer
            # is not published, even if an article talks about such rates in general.
            return FactAnswerer._refusal(q)
        if u["route"] == "passage" and (u["passages"] or u["quotes"]):
            try:
                return self.passage_answerer.answer_from(q, u["passages"][:3], u["query"], past, rec.taken_on,
                                                         quotes=u["quotes"])
            except Exception:
                self.answer_errors += 1
        return self.passage_answerer._refusal()

    def conflicts(self):
        try:
            return self.register.conflicts()
        except Exception:
            return []
