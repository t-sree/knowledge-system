"""Question understanding with gpt-4o-mini (component H, used by the hybrid for uncertain questions).

One JSON call per question decides what the rules used to guess:
  * route      rate fact / website passage / not on the site
  * products   chosen ONLY from a short list of real product keys (closed vocabulary)
  * attribute, terms, model year, amount, variants, business, intent
  * sentences  which numbered sentences of the retrieved passages answer the question
The answer itself is still built by code from the site's own facts and sentences (no generated
numbers). Anything the model returns outside the given lists is dropped. The reply is kept short
(short keys, only non-empty fields, sentence ids instead of copied text) because output length is
what makes the call slow. If the call fails or there is no key, the caller keeps the rules answer.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from knowledge.query.question import AVAILABLE_WORDS, CHANGE_WORDS

RATE_ATTRS = {"apy", "apr", "rate", "dividend_rate"}
ATTRIBUTES = ["apy", "apr", "rate", "dividend_rate", "fee", "min_open", "min_earn"]
INTENTS = {"list": "list", "highest": "high", "lowest": "low", "change": "change", "available": "available",
           "link": "url", "yesno": "yesno"}
_LINK_WORDS = re.compile(r"\b(link|url|website|web ?page|page|site)\b", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])|\n+")

N_PRODUCTS = 12              # candidates by name match + by meaning, each
N_PASSAGES = 8
N_LEAD = 3                   # lead paragraphs of the top pages added to the passages
PASSAGE_CHARS = 480          # sentences shown per passage, up to about this many characters
MAX_SENTENCES = 6
MAX_OUTPUT_TOKENS = 90
TIMEOUT = 2.5                # hard deadline per call (llm/client.py enforces it); measured chat p90 1.7 s, max 2.6 s

SYSTEM = """You map a credit union member's question to a structured lookup over the credit union's website.
You get candidate PRODUCTS and PASSAGES (sentences numbered S<passage>.<sentence>). Use only these lists.
Reply with ONE compact JSON object. Include only keys that apply (omit empty lists, nulls and false):
 "r": route, one of "rate", "passage", "none"   (always present)
 "p": product ids, e.g. ["P1"]
 "a": one of "apy","apr","rate","dividend_rate","fee","min_open","min_earn"
 "t": terms in months, e.g. [12, 24]  (5 years = 60, 30-year = 360)
 "y": vehicle model year
 "m": balance or loan amount in dollars
 "v": variants from VARIANTS that the question states
 "b": true if about business accounts
 "i": intents from "list","highest","lowest","change","available","link","yesno"
 "s": sentence ids that answer the question, e.g. ["S2.1","S4.3"]  (at most 3, most useful first)
Rules:
- "r":"rate" when the question asks for a rate, APY, APR, dividend rate, fee or minimum of one of the PRODUCTS,
  including how it changed, whether it is still offered, the highest/lowest tier, a list of such rates, the
  web page/link with those rates, or "which <products> do you offer" when such PRODUCTS exist (intent "list").
- "r":"passage" when sentences in PASSAGES answer it (hours, addresses, programs, events, counts, how-to,
  eligibility, services). Give the answering sentences in "s". When sentences disagree (counts, rankings,
  hours), list the most recently dated one first.
- "r":"none" when neither a listed product nor any sentence answers it. If the question asks for the rate of a
  product that is not in PRODUCTS, use "none" - never substitute a different product.
- "p": only the product(s) the question names or whose kind it names ("savings", "checking", "certificate",
  "loan" mean products of that kind only). If it names one product, return just that one. Broad questions
  ("credit card rates") get every matching product. Prefer source "rate pages" over the same product from
  "promo" or "articles".
- "a": omit when the question asks for "the rate" in general.
- "i": "change" for changes over time; "available" ONLY for "is it still offered / available / discontinued";
  "list" for broad listings; "link" for URL / web page questions; "yesno" for yes/no checks of a stated value.
"""


def _loose(text: str) -> str:
    """Comparison form: lower case, unified dashes and quotes, single spaces."""
    t = text.lower().replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(t.split())


def _numbered(text: str) -> list[str]:
    """The first sentences / lines of a passage, up to PASSAGE_CHARS."""
    out, size = [], 0
    for s in _SENT_SPLIT.split(text or ""):
        s = " ".join(s.split())
        if not s:
            continue
        if out and (size + len(s) > PASSAGE_CHARS or len(out) >= MAX_SENTENCES):
            break
        out.append(s[:PASSAGE_CHARS])
        size += len(s)
    return out


class Understander:
    def __init__(self, ks) -> None:
        self.ks = ks
        self.product_vecs: dict[str, object] = {}
        self.errors = 0

    # ------------------------------------------------------------------ ingest
    def embed_products(self) -> None:
        """Embed product keys (with their section words) not embedded yet."""
        llm = self.ks.llm
        if llm is None or not llm.enabled:
            return
        V = self.ks.vocab
        new = [p for p in self._rate_products() if p not in self.product_vecs]
        if not new:
            return
        texts = [f"{p}. {' '.join(sorted(V.products[p].section_words)[:12])}" for p in new]
        m = llm.embed(texts, retries=2, budget=60.0)
        if m is None:
            return
        for p, row in zip(new, m):
            self.product_vecs[p] = row

    def _rate_products(self) -> list[str]:
        V = self.ks.vocab
        return [p for p, info in V.products.items()
                if p and any(a in RATE_ATTRS or a in ("fee", "min_open", "min_earn") for a in info.attributes)]

    # ------------------------------------------------------------------ question
    def _candidate_products(self, q, index: int) -> list[str]:
        V = self.ks.vocab
        products = {p: set(V.products[p].attributes) for p in self._rate_products()}
        lexical = [p for _, p, _ in self.ks.fact_answerer.score_products(q, products, index)[:N_PRODUCTS]]
        semantic = []
        qv = self.ks._question_vector(q.text)
        if qv is not None and self.product_vecs:
            sims = [(float(vec @ qv), k) for k, vec in self.product_vecs.items()]
            semantic = [k for _, k in sorted(sims, reverse=True)[:N_PRODUCTS]]
        out = []
        for p in lexical + semantic:
            if p not in out:
                out.append(p)
        return out

    def understand(self, q, index: int) -> Optional[dict]:
        llm = self.ks.llm
        if llm is None or not llm.enabled:
            return None
        V = self.ks.vocab
        in_view = {k[0] for k in self.ks.resolver.view_index(index)}
        products = self._candidate_products(q, index)
        priors = self.ks.fact_answerer.view_priors(index)
        passages, query = self.ks.passage_answerer.candidates(q, index, N_PASSAGES)
        passages = self._with_lead_paragraphs(passages)
        lines = []
        for i, p in enumerate(products, 1):
            info = V.products[p]
            attrs = sorted(a for a in info.attributes if a in ATTRIBUTES)
            tiers = sorted(k for k in info.qualifier_kinds if k not in ("variant", "account_type"))
            kinds = "personal+business" if "account_type" in info.qualifier_kinds else "personal"
            where = "current" if p in in_view else "older snapshots only"
            prior = priors.get(p, 0.6)
            source = "rate pages" if prior >= 0.85 else "promo/info" if prior >= 0.7 else "articles/calculators"
            lines.append(f"P{i}: {p} | {', '.join(attrs)} | tiers: {', '.join(tiers) or '-'} | {kinds} | "
                         f"source: {source} | {where}")
        sentences = []                                   # (passage, sentence text) by "S<i>.<j>"
        plines = []
        for i, r in enumerate(passages, 1):
            v = r[3].version
            title = v.title.split("|")[0].strip()[:70]
            date = (v.info.content_date if v.info else None) or "undated"
            nums = _numbered(r[2].text)
            parts = [f"S{i}.{j} {s}" for j, s in enumerate(nums, 1)]
            sentences.append(nums)
            plines.append(f"S{i} [{title}; {date}]: " + " ".join(parts))
        variants = sorted(V.rate_variants)
        user = (f"QUESTION: {q.text}\nVARIANTS: {', '.join(variants)}\nPRODUCTS:\n" + "\n".join(lines or ["-"])
                + "\nPASSAGES:\n" + "\n".join(plines or ["-"]))
        out = llm.chat(SYSTEM, user, max_tokens=MAX_OUTPUT_TOKENS, timeout=TIMEOUT, json_mode=True)
        if not out:
            return None
        try:
            data = json.loads(out)
        except (ValueError, TypeError):
            self.errors += 1
            return None
        if not isinstance(data, dict):
            return None
        u = self._validate(data, products, passages, sentences, query, variants)
        if u is not None and len(u["products"]) > 1:
            # Same product from a weaker source (a promo or calculator copy of a rate table): drop it.
            top = max(priors.get(p, 0.6) for p in u["products"])
            u["products"] = [p for p in u["products"] if priors.get(p, 0.6) >= top - 0.15]
        return u

    def _with_lead_paragraphs(self, passages):
        """Articles state their key fact first: add the lead paragraph of the top pages when missing."""
        P = self.ks.passages
        out = list(passages)
        seen_text = {r[2].text for r in out}
        pages = []
        for r in passages:
            if r[3].version.key not in [e.version.key for e in pages]:
                pages.append(r[3])
        for entry in pages[:N_LEAD]:
            v = entry.version
            if v.info is None or v.info.page_type not in ("newsroom", "article", "info", "product", "promo"):
                continue
            for b in v.blocks:
                cids = P._block_chunks.get(b.key) or []
                c = next((P.chunks[c] for c in cids if len(P.chunks[c].text) > 120), None)
                if c is not None:
                    if c.text not in seen_text:
                        out.append((0.0, 0.0, c, entry, b.index, 0.0))
                        seen_text.add(c.text)
                    break
        return out

    # ------------------------------------------------------------------ validation
    @staticmethod
    def _ids(values, prefix: str, n: int, names=None) -> list[int]:
        """Positions of the chosen items: by id ("P3") or, for products, by their exact listed name."""
        out = []
        if isinstance(values, (str, int)):
            values = [values]
        for v in values or []:
            v = str(v).strip()
            m = re.fullmatch(prefix + r"(\d+)", v, re.I)
            i = int(m.group(1)) - 1 if m else (names.index(v.lower()) if names and v.lower() in names else -1)
            if 0 <= i < n and i not in out:
                out.append(i)
        return out

    @staticmethod
    def _sentences(values, passages, sentences) -> list[tuple[object, str]]:
        """(passage, sentence) for each valid sentence id "S<i>.<j>"."""
        out = []
        if isinstance(values, str):
            values = [values]
        for v in (values or [])[:3]:
            m = re.fullmatch(r"S(\d+)\.(\d+)", str(v).strip(), re.I)
            if not m:
                continue
            i, j = int(m.group(1)) - 1, int(m.group(2)) - 1
            if 0 <= i < len(passages) and 0 <= j < len(sentences[i]):
                pair = (passages[i], sentences[i][j])
                if pair not in out:
                    out.append(pair)
        return out

    def _validate(self, data: dict, products, passages, sentences, query, variants) -> Optional[dict]:
        route = data.get("r")
        if route not in ("rate", "passage", "none"):
            return None
        attr = data.get("a")
        num = (int, float)
        intents = data.get("i") or []
        quotes = self._sentences(data.get("s"), passages, sentences)
        u = {
            "route": route,
            "products": [products[i] for i in self._ids(data.get("p"), "P", len(products), products)],
            "attribute": attr if attr in ATTRIBUTES else None,
            "terms": [int(t) for t in (data.get("t") if isinstance(data.get("t"), list) else [data.get("t")])
                      if isinstance(t, num) and 0 < t < 600],
            "model_year": int(data["y"]) if isinstance(data.get("y"), num) and 1950 < data["y"] < 2100 else None,
            "amount": float(data["m"]) if isinstance(data.get("m"), num) and data["m"] > 0 else None,
            "variants": {v for v in (data.get("v") or []) if v in variants},
            "business": bool(data.get("b")),
            "intents": {INTENTS[x] for x in ([intents] if isinstance(intents, str) else intents) if x in INTENTS},
            "passages": [r for r, _ in quotes],
            "quotes": quotes,
            "query": query,
        }
        if route == "rate" and not u["products"]:
            u["route"] = "none"
        return u

    @staticmethod
    def fix_route(q, u: dict) -> None:
        """A change / rate question about a chosen product belongs to the rate path, whatever the label."""
        if u["route"] != "rate" and u["products"] and ("change" in q.intents or q.attribute in RATE_ATTRS):
            u["route"] = "rate"

    @staticmethod
    def apply(q, u: dict) -> None:
        """Write the LLM reading into the rules Query (the LLM decides; numbers it gives are validated)."""
        q.products = u["products"] or None
        if u["route"] == "rate":
            q.attribute = u["attribute"] or ("rate" if q.attribute not in RATE_ATTRS else q.attribute)
        q.terms = u["terms"]
        q.term = u["terms"][0] if u["terms"] else None
        q.model_year = u["model_year"]
        q.amounts = [u["amount"]] if u["amount"] else []
        q.variants = set(q.variants) | set(u["variants"])     # rules' variant words are kept
        q.business = u["business"]
        # The rules' readings of change / still-available / link / highest-lowest are kept as well.
        keep = q.intents & {"past", "change", "available", "url", "high", "low"}
        q.intents = keep | u["intents"]
        # "change" only when the question uses change words ("as of May" alone is a point in time)
        if "change" in q.intents and not CHANGE_WORDS.search(q.text):
            q.intents.discard("change")
        # "available" only when the question asks whether something is still offered
        if "available" in q.intents and not AVAILABLE_WORDS.search(q.text):
            q.intents.discard("available")
        # "link" only when the question itself asks for a link / URL / web page
        if "url" in q.intents and not _LINK_WORDS.search(q.text):
            q.intents.discard("url")
        if "yesno" in q.intents and not q.percents:
            q.intents.discard("yesno")
