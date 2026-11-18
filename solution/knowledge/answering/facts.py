"""Matcher (component J, fact path) and Answer Composer (component K, templates).

J  finds the products a question is about (closed vocabulary from query/vocab.py), the fact keys of
   those products in the selected snapshot view whose qualifiers fit the question, and resolves
   each with the resolver (E).
K  turns the winning facts into a member-facing answer with sources. Values rejected by the
   resolver go into `conflict_note` only, never into the answer text.

Default tier rules (agreed):
  * menu products (rows keyed by term / name / model year) -> list every row
  * tiered products (rows keyed by balance or loan amount) -> base tier first, then the others
  * a named tier (amount, term, model year, variant in the question) -> that tier only
  * "highest" / "lowest" -> that tier
Returns None when the fact path has no confident answer, so the caller can fall back.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from contract import Answer, Source

from knowledge.ingest.extract import is_product_phrase, parse_label, product_key
from knowledge.facts.resolve import same_value

RATE_ATTRS = {"apy", "apr", "rate", "dividend_rate"}
DEPOSIT_PAIR = ("dividend_rate", "apy")
LOAN_PAIR = ("rate", "apr")
SUPPORT_ATTRS = ("min_open", "min_earn")
ATTR_LABEL = {"apy": "APY", "apr": "APR", "dividend_rate": "dividend rate", "rate": "interest rate",
              "points": "points", "min_open": "minimum to open", "min_earn": "minimum balance to earn",
              "min_balance": "minimum balance", "max_balance": "maximum balance", "fee": "fee",
              "routing_number": "routing number", "phone": "phone number", "address": "address"}
MIN_SCORE = 0.5
MIN_SCORE_NARROWED = 0.3
MAX_ROWS = 30


def _fmt_date(iso: Optional[str]) -> str:
    if not iso:
        return ""
    try:
        d = date.fromisoformat(iso)
    except ValueError:
        return iso
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def _tier(quals: dict):
    return quals.get("balance_tier") or quals.get("loan_amount")


class FactAnswerer:
    def __init__(self, ks) -> None:
        self.ks = ks
        self._priors: dict[tuple[int, int], dict[str, float]] = {}
        self.last_score = 0.0                   # best product-match score of the last answer()

    def view_priors(self, index: int) -> dict[str, float]:
        """How product-like each key is in this view: the best evidence it has here.
        Tables on rate/product pages 1.0, labelled lines there 0.9, tables elsewhere 0.75, else 0.6."""
        ck = (index, len(self.ks.store.snapshots))
        hit = self._priors.get(ck)
        if hit is not None:
            return hit
        out: dict[str, float] = {}
        pages: dict[str, set] = {}
        for vs in self.ks.facts.sightings_in_view(index):
            s = vs.sighting
            if vs.present and s.fact.attribute in RATE_ATTRS:
                pages.setdefault(s.fact.product_key, set()).add(s.page_key)
            good = s.page_type in ("rate_sheet", "product")
            ext = s.fact.extractor
            v = 1.0 if good and ext == "table" else 0.9 if good and ext == "kv" else                 0.75 if ext == "table" and s.page_type in ("promo", "info") else 0.6
            if v > out.get(s.fact.product_key, 0):
                out[s.fact.product_key] = v
        # Products published on many pages are the site's main products (small boost).
        for p in out:
            out[p] *= 0.7 + 0.3 * min(len(pages.get(p, ())), 5) / 5
        self._priors = {ck: out}
        return out

    # ================================================================== J: products
    def score_products(self, q, view_products: dict[str, set], index: int) -> list[tuple[float, str, set]]:
        """(score, product key, leftover question words) for products visible in the view."""
        V = self.ks.vocab
        priors = self.view_priors(index)
        Q = set(q.content)
        if not Q:
            return []
        max_idf = max(V.idf.values()) if V.idf else 1.0
        # Words the site never uses for any product can't point elsewhere: they count less.
        qsum = sum(V.idf.get(t, 0.3 * max_idf) for t in Q)
        out = []
        for p in view_products:
            info = V.products.get(p)
            if info is None or not p:
                continue
            kw, sw, pw = info.key_words, info.section_words, info.page_words
            matched = 0.0
            hit = set()
            for t in Q:
                w = V.idf.get(t, max_idf)
                if t in kw:
                    matched += w
                elif t in sw or t in pw:
                    matched += 0.6 * w              # section heading above its table / title or URL of its page
                else:
                    continue
                hit.add(t)
            if not hit:
                continue
            qcov = matched / qsum
            key_hit = kw & Q
            if key_hit:
                pcov = sum(V.idf.get(t, max_idf) for t in key_hit) / sum(V.idf.get(t, max_idf) for t in kw)
                qcov *= 0.7 + 0.3 * pcov                  # product names with many unasked words
            out.append((qcov * priors.get(p, 0.6), p, Q - hit))
        out.sort(key=lambda x: (-x[0], x[1]))
        return out

    # ================================================================== J: keys
    @staticmethod
    def _compatible(q, quals: dict, kinds: set, allow_business: bool, asked_variants=None) -> bool:
        acct = quals.get("account_type")
        if q.business and acct != "business":
            return False
        if not q.business and acct == "business" and not allow_business:
            return False
        variants = set(quals.get("variant", ()))
        if variants & {"floor", "cap"} and not (q.variants & {"floor", "cap"}):
            return False
        asked = q.variants if asked_variants is None else asked_variants
        if not asked <= variants:
            return False
        terms = set(q.terms) | ({q.term} if q.term is not None else set())
        if terms:
            if "term" in quals:
                if quals["term"] not in terms:
                    return False
            elif "term" in kinds:
                return False
        if q.model_year is not None:
            if "model_year" in quals:
                lo, hi = quals["model_year"]
                if not lo <= q.model_year <= hi:
                    return False
            elif "model_year" in kinds:
                return False            # the product's rates depend on model year; this fact states none
        tier = _tier(quals)
        if q.amounts and tier:
            lo, hi = tier
            x = q.amounts[0]
            if not (lo - 0.01 <= x <= hi + 0.01):
                return False
            if len(q.amounts) > 1 and q.amounts[1] > hi + 1.0:     # "$10,000 and $24,999" vs 24,999.99
                return False
        return True

    def _wanted_attrs(self, q, keys_attrs: set) -> tuple[str, ...]:
        if q.attribute and q.attribute not in RATE_ATTRS:
            return (q.attribute,)
        if keys_attrs & {"apy", "dividend_rate"}:
            return DEPOSIT_PAIR
        if keys_attrs & {"apr", "rate"}:
            return LOAN_PAIR
        return ()

    # ================================================================== entry point
    def answer(self, q, index: int, ref_date: str, snapshot_date: str, past: bool) -> Optional[Answer]:
        R = self.ks.resolver
        V = self.ks.vocab
        vidx = R.view_index(index)
        self.last_score = 0.0
        self.last_refusal = ""
        self.last_products = []
        rate_question = q.attribute in RATE_ATTRS or q.attribute is None and (
            q.intents & {"yesno", "high", "low", "url", "change", "available"})
        if not rate_question and not q.attribute:
            return None

        # products visible in this view that carry the asked attribute class
        view_products: dict[str, set] = {}
        for k in vidx:
            if rate_question and k[1] in RATE_ATTRS or not rate_question and k[1] == q.attribute:
                view_products.setdefault(k[0], set()).add(k[1])
        if "available" in q.intents and rate_question or q.products:
            # products seen in any snapshot, so a discontinued one can still be recognised
            for k in self.ks.facts.by_key:
                if k[1] in RATE_ATTRS:
                    view_products.setdefault(k[0], set()).add(k[1])

        # site-wide attributes (routing number): no product needed
        if not rate_question and "" in view_products:
            keys = [k for k in vidx if k[0] == "" and k[1] == q.attribute
                    and self._compatible(q, dict(k[2]), set(), True)]
            if keys:
                return self._compose_simple(keys, index, ref_date, past, snapshot_date)

        min_score = MIN_SCORE
        if q.products:
            # Products chosen by the LLM step from the closed vocabulary: no name matching needed.
            scored = [(1.0, p, set()) for p in q.products if p in view_products]
        else:
            scored = self.score_products(q, view_products, index)
            if (not scored or scored[0][0] < MIN_SCORE) and q.model_year is not None:
                # Fallback only: a model year fits products tiered by model year ("2025 car loan").
                narrowed = {p: a for p, a in view_products.items()
                            if p in V.products and "model_year" in V.products[p].qualifier_kinds}
                if narrowed:
                    scored = self.score_products(q, narrowed, index)
                    min_score = MIN_SCORE_NARROWED
        if not scored or scored[0][0] < min_score:
            return None
        best = scored[0][0]
        self.last_score = best
        if "list" in q.intents:
            chosen = [(s, p, left) for s, p, left in scored if s >= 0.8 * best]
        else:
            # equally good matches ("savings" vs several "X Savings" products): keep them all
            chosen = [(s, p, left) for s, p, left in scored if s >= best - 1e-6]
        leftover = set.intersection(*(left for _, _, left in chosen)) if chosen else set()
        # A variant word seen elsewhere on the site that these products never have
        # ("adjustable" for a product with only fixed rates): the asked variant isn't published.
        variant_words = V.variant_words()       # words the site uses for product variants ("adjustable")
        missing_variant = {w for w in leftover if w in variant_words}
        if missing_variant and rate_question:
            self.last_refusal = "missing_variant"      # the question names a variant this product lacks
            return self._refusal(q)

        if "url" in q.intents:
            return self._compose_url(q, [p for _, p, _ in chosen], index)

        # Candidate groups: the best match(es), then each further product above the threshold.
        # Pass 1 only accepts groups that explain every question word known to the vocabulary;
        # pass 2 accepts the best group that has fitting rows ("share certificate").
        groups = [chosen] + [[c] for c in scored if c not in chosen and c[0] >= min_score][:5]
        for strict in (True, False):
            for group in groups:
                left = set.intersection(*(lf for _, _, lf in group))
                if strict and any(w in V.idf for w in left):
                    continue
                rows = self._group_rows(q, [p for _, p, _ in group], vidx, index, ref_date)
                if rows:
                    self.last_products = sorted({p for p, _, _ in rows})
                    if "available" in q.intents:
                        return self._compose_available(q, rows, past, snapshot_date)
                    return self._compose_rows(q, rows, past, snapshot_date)
        if rate_question:
            # Listed in an earlier snapshot but not in this one: the product was discontinued.
            gone = self._last_listed(q, [p for _, p, _ in chosen], index)
            if gone is not None:
                self.last_refusal = "discontinued"
                return self._discontinued(q, gone)
            return self._refusal(q)
        return None

    # ================================================================== discontinued products
    def _last_listed(self, q, products, index):
        """Snapshot record in which a matching fact was last stated, if it was stated before `index`."""
        store = self.ks.store
        last = None
        for k, sightings in self.ks.facts.by_key.items():
            if k[0] not in products or k[1] not in RATE_ATTRS:
                continue
            kinds = self.ks.vocab.products[k[0]].qualifier_kinds if k[0] in self.ks.vocab.products else set()
            if not self._compatible(q, dict(k[2]), kinds, True):
                continue
            for s in sightings:
                if s.snapshot_index < index and s.page_type in ("rate_sheet", "product", "promo", "info"):
                    if last is None or s.snapshot_index > last:
                        last = s.snapshot_index
        if last is None:
            return None
        return store.snapshots[last]

    @staticmethod
    def _discontinued(q, rec) -> Answer:
        month = date.fromisoformat(rec.taken_on).strftime("%B")
        return Answer(text=(f"That isn't listed on Dupaco's website anymore: it no longer appears on the rate pages "
                            f"(it was last listed in {month}). Please contact Dupaco for current options."),
                      sources=[], confident=False)

    def _compose_available(self, q, rows, past, snapshot_date) -> Answer:
        ans = self._compose_rows(q, rows, past, snapshot_date)
        ans.text = "Yes, it is still listed. " + ans.text
        return ans

    def _group_rows(self, q, products, vidx, index, ref_date):
        allow_business = "list" in q.intents
        rows = self._rows(q, products, vidx, index, ref_date, allow_business)
        if not rows and not q.business and not allow_business:
            rows = self._rows(q, products, vidx, index, ref_date, True)
        return rows

    # ================================================================== rows
    def _rows(self, q, products, vidx, index, ref_date, allow_business):
        """Rows = (product, qualifiers) with resolved attribute values."""
        R = self.ks.resolver
        V = self.ks.vocab
        rows = {}
        for p in products:
            kinds = V.products[p].qualifier_kinds
            keys = [k for k in vidx if k[0] == p]
            attrs_here = {k[1] for k in keys}
            wanted = self._wanted_attrs(q, attrs_here)
            if not wanted:
                continue
            # Asked variant words this product never has are ignored ("standard purchase APR" on a card).
            have = {v for k in keys for v in dict(k[2]).get("variant", ())}
            asked = q.variants & have
            for k in keys:
                if k[1] not in wanted and k[1] not in SUPPORT_ATTRS:
                    continue
                quals = dict(k[2])
                if not self._compatible(q, quals, kinds, allow_business, asked):
                    continue
                # Column-header variants (introductory / after promotional) belong to the same row.
                row_quals = tuple((a, b) for a, b in k[2])
                res = R.resolve(k, index, ref_date)
                if res.winner is None:
                    continue
                rows.setdefault((p, row_quals), {})[k[1]] = res
        # a row needs at least one of the wanted attributes
        out = []
        for (p, rq), attrs in rows.items():
            wanted = self._wanted_attrs(q, {a for a in attrs})
            if any(a in attrs for a in wanted) or (q.attribute and q.attribute in attrs):
                out.append((p, rq, attrs))
        # a product stated in a table: prose restatements of it are dropped
        tabled = {p for p, rq, attrs in out if any(r.fact.extractor == "table" for r in attrs.values())}
        out = [(p, rq, attrs) for p, rq, attrs in out
               if p not in tabled or any(r.fact.extractor == "table" for r in attrs.values())]
        # a named amount: keep the tightest tier containing it
        if q.amounts:
            best = {}
            for p, rq, attrs in out:
                d = dict(rq)
                t = _tier(d)
                width = (t[1] - t[0]) if t else float("inf")
                base = (p, tuple((a, b) for a, b in rq if a not in ("balance_tier", "loan_amount")))
                if base not in best or width < best[base][0]:
                    best[base] = (width, (p, rq, attrs))
            out = [v[1] for v in best.values()]
        return out

    # ================================================================== K: text
    def _product_display(self, fact) -> str:
        for c in reversed(fact.context):
            if is_product_phrase(c) and product_key(parse_label(c).product) == fact.product_key:
                return c
        return fact.product_key.title() if fact.product_key else ""

    def _row_name(self, p, quals: dict, fact) -> str:
        label = (fact.label or "").strip()
        if fact.extractor == "table" and label:
            first = label.split(" · ")[0]
            name = label if is_product_phrase(first) else f"{self._product_display(fact)} {label}".strip()
        else:
            name = self._product_display(fact)
            bits = []
            if quals.get("term"):
                t = quals["term"]
                bits.append(f"{t // 12}-year" if t % 12 == 0 and t >= 60 else f"{t}-month")
            if quals.get("variant"):
                bits.extend(quals["variant"])
            if bits:
                name = f"{name} ({', '.join(bits)})"
        if quals.get("account_type") == "business" and "business" not in name.lower():
            name = "Business " + name
        return name

    @staticmethod
    def _fmt_value(attr: str, value: str) -> str:
        v = value.strip()
        if attr == "apy" and "apy" not in v.lower():
            return f"{v} APY"
        if attr == "apr" and "apr" not in v.lower():
            return f"{v} APR"
        if attr == "dividend_rate":
            return f"{v} dividend rate"
        if attr == "rate":
            return f"{v} interest rate"
        if attr in ("min_open", "min_earn"):
            return f"{ATTR_LABEL[attr]} {v}"
        return v

    def _row_text(self, p, rq, attrs, q) -> str:
        quals = dict(rq)
        any_res = next(iter(attrs.values()))
        name = self._row_name(p, quals, any_res.winner.sighting.fact)
        order = [a for a in (q.attribute,) if a and a not in RATE_ATTRS] + list(DEPOSIT_PAIR) + list(LOAN_PAIR) \
            + ["points"] + list(SUPPORT_ATTRS)
        parts = []
        shown = []
        for a in order:
            if a in attrs and a not in [x for x, _ in parts]:
                f = attrs[a].fact
                # a "rate" cell that is really the APR text again ("12.50% – 19.50% APR")
                if a == "rate" and "apr" in f.value.lower() and any(same_value(f, g) for g in shown):
                    continue
                if a in SUPPORT_ATTRS and not f.numbers:
                    continue                                # "None" / "—" minimums add nothing
                shown.append(f)
                parts.append((a, self._fmt_value(a, attrs[a].value)))
        variants = set(quals.get("variant", ()))
        note = " (introductory)" if "introductory" in variants and "introductory" not in name.lower() else ""
        note = " (after the promotional period)" if "after promotional" in variants else note
        return f"{name}{note}: " + ", ".join(t for _, t in parts)

    def _sort_key_for(self, q):
        def key(row):
            d = dict(row[1])
            extra = len(set(d.get("variant", ())) - q.variants)     # exact variant matches first
            return (extra,) + self._sort_key(row)
        return key

    @staticmethod
    def _sort_key(row):
        p, rq, _ = row
        d = dict(rq)
        tier = _tier(d) or (0, 0)
        my = d.get("model_year", (0, 0))
        return (p, d.get("account_type", ""), tuple(d.get("variant", ())), -my[1], d.get("term", 0), tier[1], tier[0])

    def _compose_changes(self, q, rows) -> Answer:
        """How a value moved across the snapshots (from the fact timeline)."""
        T = self.ks.timeline
        lines, used = [], []
        # The attribute the question names; a plain "rate" means the interest rate where one is published.
        order = {"apy": ("apy", "dividend_rate"), "apr": ("apr", "rate"), "dividend_rate": ("dividend_rate", "apy"),
                 "rate": ("rate", "apy", "apr", "dividend_rate")}.get(q.attribute or "", ("apy", "apr", "rate",
                                                                                         "dividend_rate"))
        for p, rq, attrs in rows[:8]:
            attr = next((a for a in order if a in attrs), None)
            if attr is None:
                continue
            res = attrs[attr]
            periods = T.periods(res.key)
            if not periods:
                continue
            name = self._row_name(p, dict(rq), res.winner.sighting.fact)
            label = ATTR_LABEL.get(attr, attr)
            if len(periods) == 1:
                pr = periods[0]
                lines.append(f"{name}: no change, {self._fmt_value(attr, pr.value)} in every snapshot from "
                             f"{_fmt_date(pr.first_date)} to {_fmt_date(pr.last_date)}")
            else:
                steps = []
                for i, pr in enumerate(periods):
                    when = (f"from {_fmt_date(pr.first_date)}" if i == len(periods) - 1 else
                            f"{_fmt_date(pr.first_date)}" + (f" to {_fmt_date(pr.last_date)}"
                                                             if pr.last_date != pr.first_date else ""))
                    steps.append(f"{pr.value} ({when})")
                lines.append(f"{name} {label}: " + " → ".join(steps))
            used.append((p, rq, attrs))
        if not lines:
            return self._compose_rows_plain(q, rows, False, "")
        text = lines[0] + "." if len(lines) == 1 else "How the rates changed:\n" + "\n".join(f"- {ln}" for ln in lines)
        return self._finish(text, used)

    def _compose_rows(self, q, rows, past, snapshot_date) -> Answer:
        if "change" in q.intents:
            return self._compose_changes(q, sorted(rows, key=self._sort_key_for(q) if q.variants else self._sort_key))
        return self._compose_rows_plain(q, rows, past, snapshot_date)

    def _compose_rows_plain(self, q, rows, past, snapshot_date) -> Answer:
        rows = sorted(rows, key=self._sort_key_for(q) if q.variants else self._sort_key)
        if "yesno" in q.intents and q.percents:
            return self._compose_verify(q, rows, past, snapshot_date)
        if q.intents & {"high", "low"} and len(rows) > 1:
            rows = [self._extreme(rows, "high" in q.intents)]
        rows = rows[:MAX_ROWS]
        lines = [self._row_text(p, rq, attrs, q) for p, rq, attrs in rows]
        prefix = f"As of {_fmt_date(snapshot_date)}, " if past else ""
        if len(lines) == 1:
            text = prefix + lines[0] + "."
        else:
            text = (prefix + "here are the published rates:" if prefix else "Here are the published rates:") \
                + "\n" + "\n".join(f"- {ln}" for ln in lines)
        text += self._effective_note(rows)
        return self._finish(text, rows)

    def _primary(self, attrs):
        for a in ("apy", "apr", "dividend_rate", "rate"):
            if a in attrs and attrs[a].fact.numbers:
                return attrs[a].fact.numbers[0][0]
        return None

    def _extreme(self, rows, high: bool):
        scored = [(self._primary(attrs), r) for r in rows for attrs in [r[2]] if self._primary(attrs) is not None]
        if not scored:
            return rows[0]
        scored.sort(key=lambda x: x[0], reverse=high)
        return scored[0][1]

    def _compose_verify(self, q, rows, past, snapshot_date) -> Answer:
        target = q.percents[0]
        hits = [r for r in rows if any(abs(n - target) < 1e-9 and u in ("%", "")
                                       for res in r[2].values() for n, u in res.fact.numbers)]
        if hits:
            lines = [self._row_text(p, rq, attrs, q) for p, rq, attrs in hits[:MAX_ROWS]]
            text = "Yes. " + "; ".join(lines) + "."
            use = hits
        else:
            top = self._extreme(rows, True)
            lines = [self._row_text(p, rq, attrs, q) for p, rq, attrs in rows[:MAX_ROWS]]
            text = (f"No. None of the published rates is {q.percents[0]:g}%. The highest is "
                    f"{self._row_text(top[0], top[1], top[2], q)}.\nAll published rates:\n"
                    + "\n".join(f"- {ln}" for ln in lines))
            use = rows
        text += self._effective_note(use)
        return self._finish(text, use)

    def _effective_note(self, rows) -> str:
        dates = {res.winner.sighting.content_date for _, _, attrs in rows for res in attrs.values()
                 if res.winner and res.winner.sighting.content_date}
        if len(dates) == 1:
            return f" (Rates effective {_fmt_date(dates.pop())}.)"
        return ""

    def _finish(self, text: str, rows) -> Answer:
        # Sources: the winning pages, then pages that state the same value (product pages first).
        sources, seen = [], set()
        notes = []
        backups = []
        for _, _, attrs in rows:
            for res in attrs.values():
                s = res.winner.sighting
                if s.url not in seen and len(sources) < 3:
                    seen.add(s.url)
                    sources.append(Source(url=s.url, snapshot_id=s.snapshot_id, quote=s.fact.quote[:200]))
                for a in res.agreeing:
                    if a.present:
                        backups.append(a.sighting)
                for d in res.disagreeing[:1]:
                    ds = d.sighting
                    crit = res.reason.split(" on ", 1)[1].split(" (")[0] if " on " in res.reason else "rank"
                    note = f"{ds.fact.value} on {ds.url} not used ({crit})"
                    if note not in notes and len(notes) < 3:
                        notes.append(note)
        backups.sort(key=lambda s: ({"product": 0, "rate_sheet": 1, "info": 2}.get(s.page_type, 3), s.url))
        for s in backups:
            if len(sources) >= 5:
                break
            if s.url not in seen:
                seen.add(s.url)
                sources.append(Source(url=s.url, snapshot_id=s.snapshot_id, quote=s.fact.quote[:200]))
        return Answer(text=text, sources=sources, confident=True,
                      conflict_note="; ".join(notes) if notes else None)

    def _compose_simple(self, keys, index, ref_date, past, snapshot_date) -> Optional[Answer]:
        R = self.ks.resolver
        results = [R.resolve(k, index, ref_date) for k in keys]
        results = [r for r in results if r.winner]
        if not results:
            return None
        # most agreed value first
        results.sort(key=lambda r: -len(r.agreeing))
        r = results[0]
        label = ATTR_LABEL.get(r.fact.attribute, r.fact.attribute.replace("_", " "))
        prefix = f"As of {_fmt_date(snapshot_date)}, " if past else ""
        num = r.fact.numbers[0][0] if r.fact.numbers else None
        value = f"{int(num)}" if num is not None and r.fact.attribute == "routing_number" else r.value
        text = f"{prefix}Dupaco's {label} is {value}." if prefix == "" else f"{prefix}Dupaco's {label} was {value}."
        rows = [("", (), {r.fact.attribute: r})]
        # the same value stated under other keys (e.g. on business pages) are sources too
        rows += [("", (), {o.fact.attribute: o}) for o in results[1:] if same_value(o.fact, r.fact)]
        return self._finish(text[0].upper() + text[1:], rows)

    def _compose_url(self, q, products, index) -> Optional[Answer]:
        counts = {}
        for vs in self.ks.facts.sightings_in_view(index):
            s = vs.sighting
            if s.fact.product_key not in products or s.fact.attribute not in RATE_ATTRS:
                continue
            if (dict(s.fact.qualifiers).get("account_type") == "business") != q.business:
                continue
            rank = {"product": 0, "rate_sheet": 1}.get(s.page_type, 2)
            c = counts.setdefault(s.url, [rank, 0, s])
            c[1] += 1
        if not counts:
            return None
        url, (rank, n, s) = min(counts.items(), key=lambda kv: (kv[1][0], -kv[1][1], kv[0]))
        name = self._product_display(s.fact) or "these"
        text = f"You can see current {name} rates at {url}"
        return Answer(text=text, sources=[Source(url=url, snapshot_id=s.snapshot_id, quote=s.fact.quote[:200])])

    @staticmethod
    def _refusal(q) -> Answer:
        return Answer(
            text=("I couldn't find a published rate for that on Dupaco's website, so I can't confirm one. "
                  "Please contact Dupaco directly for a current quote."),
            sources=[], confident=False)
