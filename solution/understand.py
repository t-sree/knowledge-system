"""Question understanding with gpt-4o-mini (component H, LLM-first).

One JSON call per question decides what the rules used to guess:
  * route     rate fact / website passage / not on the site
  * products  chosen ONLY from a short list of real product keys (closed vocabulary)
  * attribute, terms, model year, amount, variants, business, intent
  * passages  which of the retrieved passages actually answer the question
The answer itself is still built by code from the site's own facts and sentences (no generated
numbers). Anything the model returns outside the given lists is dropped. If the call fails or there
is no key, the caller keeps the rules-only path.
"""
from __future__ import annotations

import json
import re
from typing import Optional

RATE_ATTRS = {"apy", "apr", "rate", "dividend_rate"}
_LINK_WORDS = re.compile(r"\b(link|url|website|web ?page|page|site)\b", re.I)


from question import _AVAILABLE as _AVAILABLE_WORDS  # noqa: E402
from question import _CHANGE as _CHANGE_WORDS  # noqa: E402


def _loose(text: str) -> str:
    """Comparison form for quotes: lower case, unified dashes and quotes, single spaces."""
    t = text.lower().replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(t.split())
ATTRIBUTES = ["apy", "apr", "rate", "dividend_rate", "fee", "min_open", "min_earn"]
INTENTS = {"list": "list", "highest": "high", "lowest": "low", "change": "change", "available": "available",
           "link": "url", "yesno": "yesno"}
N_PRODUCTS = 12              # candidates by name match + by meaning, each
N_PASSAGES = 10
N_LEAD = 4                   # lead paragraphs of the top pages added to the passages
PASSAGE_CHARS = 420
TIMEOUT = 8.0                # generous for now; tightened after timing is measured

SYSTEM = """You map a credit union member's question to a structured lookup over the credit union's website.
You get candidate PRODUCTS (with the rate attributes the site publishes for them) and candidate PASSAGES
from the site. Use only these lists. Reply with one JSON object:
{
 "route": "rate" | "passage" | "none",
 "products": ["P1", ...],
 "attribute": "apy" | "apr" | "rate" | "dividend_rate" | "fee" | "min_open" | "min_earn" | null,
 "terms_months": [integers],
 "model_year": integer or null,
 "amount": number or null,
 "variants": [strings from VARIANTS],
 "business": true | false,
 "intent": [any of "list", "highest", "lowest", "change", "available", "link", "yesno"],
 "passages": ["S1", ...],
 "quotes": [{"passage": "S1", "text": "exact sentence(s) copied from that passage"}]
}
Rules:
- route "rate": the question asks for a rate, APY, APR, dividend rate, fee or minimum of one of the PRODUCTS,
  including how it changed over time, whether it is still offered, the highest/lowest tier, a list of such rates,
  or the web page/link where those rates are shown. Also "which <products> do you offer / have" when the
  matching PRODUCTS exist (use intent "list").
- route "passage": the question is answered by facts in PASSAGES (hours, addresses, programs, events, counts,
  how-to, eligibility, services). List in "passages" every passage that contains part of the answer (up to 3),
  most useful first. When passages disagree (counts, rankings, hours), put the most recently dated one first.
- route "none": neither a listed product nor any passage answers the question. If the question asks for the
  rate of a product that is not in PRODUCTS, use "none" - never substitute a different product.
- "products": only the product(s) the question names or whose kind it names. If it names one product
  (e.g. "Share Savings"), return just that one, even when other products pay more. A kind of account
  ("savings", "checking", "certificate", "loan") means products of that kind only, not other kinds. Broad questions ("credit card rates", "savings rates")
  get every matching product. Prefer products whose source is "rate pages" over the same product from
  "promo" or "articles". [] when route is not "rate".
- "attribute": null when the question asks for "the rate" in general; otherwise the specific one.
- "terms_months": terms asked for (12-month = 12, 5 years = 60, 30-year = 360). Several when the question compares terms.
- "model_year": vehicle model year if given. "amount": balance or loan amount in dollars if given.
- "variants": only words from VARIANTS that the question states (e.g. jumbo, special, purchase, refinance, intro).
- "business": true only if the question is about business accounts.
- "intent": "change" for questions about changes over time; "available" for "is it still offered/available";
  "list" for broad listings; "link" for URL / web page questions; "yesno" for yes/no checks of a stated value.
  Use "available" ONLY when the question asks whether something is still offered / available / discontinued.
- "quotes": for route "passage", copy word for word the sentence(s) or line(s) from the passages that answer the
  question (at most 3, each from the passage named). Copy exactly; do not rephrase or combine. [] otherwise.
- Answer with the ids ("P1", "S2") from the lists.
"""


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
        qv = self.ks.question_vector(q.text)
        if qv is not None and self.product_vecs:
            keys = list(self.product_vecs)
            sims = [(float(self.product_vecs[k] @ qv), k) for k in keys]
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
        vidx = self.ks.resolver.view_index(index)
        in_view = {k[0] for k in vidx}
        products = self._candidate_products(q, index)
        priors = self.ks.fact_answerer.view_priors(index)
        passages, query = self.ks.passage_answerer.candidates(q, index, N_PASSAGES)
        passages = self._with_lead_paragraphs(passages, index)
        lines = []
        for i, p in enumerate(products, 1):
            info = V.products[p]
            attrs = sorted(a for a in info.attributes if a in ATTRIBUTES)
            tiers = sorted(k for k in info.qualifier_kinds if k not in ("variant", "account_type"))
            kinds = "personal and business" if "account_type" in info.qualifier_kinds else "personal"
            where = "on the current site" if p in in_view else "only in older snapshots"
            prior = priors.get(p, 0.6)
            source = "rate pages" if prior >= 0.85 else "promo or info pages" if prior >= 0.7 else "articles or calculators"
            lines.append(f"P{i}: {p} | publishes: {', '.join(attrs)} | tiers: {', '.join(tiers) or 'none'} | "
                         f"accounts: {kinds} | source: {source} | {where}")
        plines = []
        for i, r in enumerate(passages, 1):
            v = r[3].version
            title = v.title.split("|")[0].strip()
            ptype = v.info.page_type if v.info else "page"
            date = (v.info.content_date if v.info else None) or "undated"
            text = " ".join(r[2].text.split())[:PASSAGE_CHARS]
            plines.append(f"S{i} [{title}; {ptype}; {date}]: {text}")
        variants = sorted(V.rate_variants)
        user = (f"QUESTION: {q.text}\n\nVARIANTS: {', '.join(variants)}\n\nPRODUCTS:\n" + "\n".join(lines or ["(none)"])
                + "\n\nPASSAGES:\n" + "\n".join(plines or ["(none)"]))
        out = llm.chat(SYSTEM, user, max_tokens=220, timeout=TIMEOUT, json_mode=True)
        if not out:
            return None
        try:
            data = json.loads(out)
        except (ValueError, TypeError):
            self.errors += 1
            return None
        u = self._validate(data, products, passages, query, variants)
        if u is not None and len(u["products"]) > 1:
            # Same product from a weaker source (a promo or calculator copy of a rate table): drop it.
            top = max(priors.get(p, 0.6) for p in u["products"])
            u["products"] = [p for p in u["products"] if priors.get(p, 0.6) >= top - 0.15]
        return u

    def _with_lead_paragraphs(self, passages, index):
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

    @staticmethod
    def _ids(values, prefix: str, n: int, names=None) -> list[int]:
        """Positions of the chosen items: by id ("P3") or, for products, by their exact listed name."""
        out = []
        if isinstance(values, str):
            values = [values]
        for v in values or []:
            v = str(v).strip()
            m = re.fullmatch(prefix + r"(\d+)", v, re.I)
            i = int(m.group(1)) - 1 if m else (names.index(v.lower()) if names and v.lower() in names else -1)
            if 0 <= i < n and i not in out:
                out.append(i)
        return out

    def _validate(self, data: dict, products, passages, query, variants) -> Optional[dict]:
        route = data.get("route")
        if route not in ("rate", "passage", "none"):
            return None
        attr = data.get("attribute")
        u = {
            "route": route,
            "products": [products[i] for i in self._ids(data.get("products"), "P", len(products), products)],
            "attribute": attr if attr in ATTRIBUTES else None,
            "terms": [int(t) for t in data.get("terms_months") or [] if isinstance(t, (int, float)) and 0 < t < 600],
            "model_year": int(data["model_year"]) if isinstance(data.get("model_year"), (int, float))
            and 1950 < data["model_year"] < 2100 else None,
            "amount": float(data["amount"]) if isinstance(data.get("amount"), (int, float))
            and data["amount"] > 0 else None,
            "variants": {v for v in data.get("variants") or [] if v in variants},
            "business": bool(data.get("business")),
            "intents": {INTENTS[i] for i in ([data["intent"]] if isinstance(data.get("intent"), str)
                                              else data.get("intent") or []) if i in INTENTS},
            "passages": [passages[i] for i in self._ids(data.get("passages"), "S", len(passages))],
            "quotes": self._quotes(data.get("quotes"), passages),
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
    def _quotes(values, passages) -> list[tuple[object, str]]:
        """(passage, quote) pairs whose quote really appears, word for word, in that passage."""
        out = []
        if not isinstance(values, list):
            return out
        for item in values[:3]:
            if not isinstance(item, dict):
                continue
            m = re.fullmatch(r"S(\d+)", str(item.get("passage", "")).strip(), re.I)
            text = " ".join(str(item.get("text", "")).split())
            if not m or not text or not (1 <= int(m.group(1)) <= len(passages)):
                continue
            r = passages[int(m.group(1)) - 1]
            source = " ".join(r[2].text.split())
            if len(text) >= 12 and _loose(text) in _loose(source):
                out.append((r, text))
        return out

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
        q.variants = set(u["variants"])
        q.business = u["business"]
        # The rules' readings of change / still-available / link / highest-lowest are kept as well.
        keep = q.intents & {"past", "change", "available", "url", "high", "low"}
        q.intents = keep | u["intents"]
        # "change" only when the question uses change words ("as of May" alone is a point in time)
        if "change" in q.intents and not _CHANGE_WORDS.search(q.text):
            q.intents.discard("change")
        # "available" only when the question asks whether something is still offered
        if "available" in q.intents and not _AVAILABLE_WORDS.search(q.text):
            q.intents.discard("available")
        # "link" only when the question itself asks for a link / URL / web page
        if "url" in q.intents and not _LINK_WORDS.search(q.text):
            q.intents.discard("url")
        if "yesno" in q.intents and not q.percents:
            q.intents.discard("yesno")
