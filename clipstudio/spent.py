"""What Muse cost: one line for every answer a mind gave, with its tokens and what they cost by the provider's list
price, in a file of the studio's own (work/spent.jsonl), so the page can say what each message, this chat, today and
this month came to.

Muse Glimmer on this computer costs nothing. An answer whose price is not known is counted as not priced, never
guessed. A question cut short by Stop or by new words is not in it: a hosted API says what an answer cost only at
its end, so a cut one cannot be counted (the page says so).
"""
import datetime
import json
import os
import threading
import time
import urllib.request

RATES = "https://open.er-api.com/v6/latest/USD"
RATES_FROM = "open.er-api.com"
RATE_KEPT = 12 * 3600
RATE_RETRY = 600
_lock = threading.Lock()


def _fetch():
    with urllib.request.urlopen(RATES, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def rate(currency, path):
    """{currency, per_usd[, day, source]}: how to show dollars in the person's own money. The day's rates are fetched
    once and kept in path for 12 hours; the request carries nothing of the person's. US dollars, a currency the rates
    do not have, or rates that cannot be fetched give dollars (per_usd 1), the last two with why."""
    currency = (currency or "USD").upper()
    if currency == "USD" or not (len(currency) == 3 and currency.isalpha()):
        return {"currency": "USD", "per_usd": 1.0}
    with _lock:
        try:
            with open(path, encoding="utf-8") as f:
                kept = json.load(f)
        except (OSError, ValueError):
            kept = {}
        age = time.time() - kept.get("fetched", 0)
        if not kept.get("rates") and age > RATE_RETRY or kept.get("rates") and age > RATE_KEPT:
            try:
                got = _fetch()
                kept = {"fetched": time.time(), "day": got["time_last_update_utc"][:16], "rates": got["rates"]}
            except (OSError, ValueError, KeyError) as e:
                kept = dict(kept, fetched=time.time()) if kept.get("rates") else {"fetched": time.time(), "why": str(
                    getattr(e, "reason", None) or e)}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(kept, f)
    per = (kept.get("rates") or {}).get(currency)
    if not per:
        why = ("The exchange rates could not be fetched from %s (%s), so amounts are in US dollars."
               % (RATES_FROM, kept["why"]) if kept.get("why") else
               "%s has no rate for %s, so amounts are in US dollars." % (RATES_FROM, currency))
        return {"currency": "USD", "per_usd": 1.0, "why": why}
    return {"currency": currency, "per_usd": per, "day": kept["day"], "source": RATES_FROM}


def add(path, chat, job, brain, numbers):
    """Writes one answer's line."""
    row = {"t": time.time(), "chat": chat, "job": job, "mind": getattr(brain, "key", brain.label),
           "label": brain.label, "model": getattr(brain, "model_id", None), "read_new": numbers.get("read_new"),
           "reused": numbers.get("reused"), "written": numbers.get("written"), "usd": numbers.get("usd")}
    with _lock:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def totals(path, chat=None, now=None):
    """{chat, jobs, today, month, all, minds, unpriced}: US dollars, by this computer's calendar. jobs are the
    messages of the chat, each {job, usd, answers, minds}; minds what each mind came to in all; unpriced how many
    answers had no price."""
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    day = datetime.datetime.fromtimestamp(now or time.time())
    out = {"chat": 0.0, "jobs": [], "today": 0.0, "month": 0.0, "all": 0.0, "minds": {}, "unpriced": 0}
    jobs = {}
    for r in rows:
        usd = r.get("usd")
        if usd is None:
            out["unpriced"] += 1
            continue
        at = datetime.datetime.fromtimestamp(r["t"])
        out["all"] += usd
        out["month"] += usd if (at.year, at.month) == (day.year, day.month) else 0
        out["today"] += usd if at.date() == day.date() else 0
        out["minds"][r["label"]] = out["minds"].get(r["label"], 0.0) + usd
        if chat and r.get("chat") == chat:
            out["chat"] += usd
            job = jobs.setdefault(r["job"], {"job": r["job"], "usd": 0.0, "answers": 0, "minds": []})
            job["usd"] += usd
            job["answers"] += 1
            if r["label"] not in job["minds"]:
                job["minds"].append(r["label"])
    out["jobs"] = [jobs[k] for k in sorted(jobs)]
    return out
