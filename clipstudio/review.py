"""A product review told over several takes, cut the way C cut Tarm's yeti review by hand (10-28): Muse only says
what each line does in a review, and the program builds the cut from whole lines, in the review order, inside the
length asked.

Why the work is split this way: asked to arrange the eight yeti takes itself, Glimmer was refused eleven times in a
row (it sent the same 68.9 s five times) and Spark opened on a count-in. Saying what one line does is a small
question a model of any size answers well; the order, the seconds and the length are the program's. Tarm on C's cut:
"ใช้ได้สำหรับคนทั่วไป สำหรับการเริ่มต้นคือดีเลย" and the person can still ask Muse for another cut.

The order is C's: show it working first (the viewer decides in two seconds, and a test they can hear is the reason
to stay), then what it is, the one thing it does well, what it works with, what there is to see, the person's verdict
and price, and the ask to comment.
"""
import json

ROLES = (
    ("proof", "the product shown or heard working right now: a test the viewer can see or hear, such as talking from "
              "far away to a microphone, or the person saying what the viewer is seeing it do"),
    ("what", "says what the product is: its name, its brand or what kind of thing it is"),
    ("point", "the one thing that makes it worth having: what it does well, what problem it solves"),
    ("use", "what it works with or how it is used, such as what it plugs into"),
    ("look", "something about it to see: its light, its colours, its shape"),
    ("verdict", "the person's own opinion of it after using it, or its price"),
    ("ask", "asks the viewers to do something: comment, follow, click a link"),
    ("other", "anything else: hello and thanks, setting up, a setting explained step by step, a line cut short, a "
              "line said again in another take, talk to someone off camera"),
)
NAMES = tuple(r for r, _ in ROLES)
ORDER = ("proof", "what", "point", "use", "look", "verdict", "ask")
# Kept first when the cut is too long: the hook, what it is and the verdict make a review (its CORE); a second look
# goes before any job does.
KEEP = ("proof", "what", "verdict", "point", "ask", "use", "look")
CORE = 3
MOST = {"look": 2}
JOIN_GAP = 1.5
PIECE_MOST = 9.0
HOOK = 4.0
SHORT = ("proof", "ask")
OVER = 1.15
# Why each piece of the program's cut goes in, for the person, when the cut is used as it is.
SAYS = {"proof": "shows it working", "what": "says what it is", "point": "its best point", "use": "what it works with",
        "look": "something to see on it", "verdict": "the verdict and the price", "ask": "asks the viewers to comment"}

ASK = """These are the lines a person says in the takes of a product review, as speech recognition heard them, so
some words are misheard. Say what each line does in a review. The jobs:
%s

Give every line one job. A line that starts a thought goes with the job of the thought. When the same thing is said in
two takes, give the job to the clearer one and "other" to the rest.

Then name the hook: the one line the clip should open on, the moment that makes someone scrolling past stay. Best is
a line said while the product is seen or heard working.

The lines, as number, take, second, words:
%s"""


def question(lines):
    """(messages, schema) asking what each of lines ([{"n", "clip", "from", "to", "text"}]) does in a review."""
    jobs = "\n".join("- %s: %s" % r for r in ROLES)
    listed = "\n".join("%d. take %d, %g s: %s" % (ln["n"], ln["clip"], round(ln["from"], 1), ln["text"]) for ln in lines)
    schema = {"type": "object", "properties": {"lines": {"type": "array", "items": {
        "type": "object", "properties": {"n": {"type": "integer"}, "job": {"type": "string", "enum": list(NAMES)}},
        "required": ["n", "job"]}}, "hook": {"type": "integer"}}, "required": ["lines", "hook"]}
    return [{"role": "user", "content": ASK % (jobs, listed)}], schema


def read(answer, lines):
    """{"jobs": {line number: job}, "hook": line number or None} from Muse's answer; a line it skipped or gave an
    unknown job is "other", and the hook is None unless it is one of lines. Raises ValueError when the answer is not
    the JSON asked for."""
    got = json.loads(answer)
    known = {ln["n"] for ln in lines}
    jobs = {n: "other" for n in known}
    for item in got["lines"]:
        if item.get("n") in known and item.get("job") in NAMES:
            jobs[item["n"]] = item["job"]
    return {"jobs": jobs, "hook": got.get("hook") if got.get("hook") in known else None}


def is_review(jobs):
    """True when the lines Muse labelled ({line number: job}) make a review: one says what the product is, and one its
    point or the verdict. The takes of a trip someone talks through come back mostly other."""
    found = set(jobs.values())
    return "what" in found and bool(found & {"point", "verdict"})


def stretches(lines, jobs, alone=()):
    """[{"clip", "from", "to", "job", "lines": [n]}]: lines next to each other in one take with the same job (other
    than "other"), JOIN_GAP s apart at most, made one stretch of at most PIECE_MOST s; a longer run is split into
    several, a single longer line stays whole, and a line in alone is a stretch by itself."""
    out = []
    for ln in sorted(lines, key=lambda x: (x["clip"], x["from"])):
        job = jobs.get(ln["n"], "other")
        if job == "other":
            continue
        last = out[-1] if out else None
        if (last and last["clip"] == ln["clip"] and last["job"] == job and ln["from"] - last["to"] <= JOIN_GAP
                and ln["to"] - last["from"] <= PIECE_MOST and ln["n"] not in alone
                and not set(last["lines"]) & set(alone)):
            last["to"], last["lines"] = max(last["to"], ln["to"]), last["lines"] + [ln["n"]]
            continue
        out.append({"clip": ln["clip"], "from": ln["from"], "to": ln["to"], "job": job, "lines": [ln["n"]]})
    return out


def _best(job, found, taken=()):
    """The stretch of job to use: for the hook and the ask (SHORT) the one nearest HOOK seconds long, for the rest the
    longest; a second one of a job from another take than taken's when there is one, so the picture changes."""
    other = [s for s in found if s["clip"] not in {t["clip"] for t in taken}] or found
    if job in SHORT:
        return min(other, key=lambda s: (abs(s["to"] - s["from"] - HOOK), s["clip"], s["from"]))
    return max(other, key=lambda s: (s["to"] - s["from"], -s["clip"], -s["from"]))


def draft(clips, lines, jobs, length, hook=None):
    """The cut, {"clips": [{"clip", "start", "end", "job", "lines"}], "left_out": [{"clip", "why"}], "length"}: the
    hook line alone first when Muse named one, then for each job in ORDER its best stretch (two for MOST's); while the
    cut is longer than length by OVER, the second ones go first, then the jobs from the end of KEEP but its CORE; and
    every clip of clips ([{"n", "talking"}]) not used is left out with the reason."""
    jobs = jobs if hook is None else {**jobs, hook: "proof"}
    found = stretches(lines, jobs, alone=() if hook is None else (hook,))
    picked = []
    for job in ORDER:
        left = [s for s in found if s["job"] == job and (job != "proof" or hook is None or hook in s["lines"])]
        for _ in range(MOST.get(job, 1)):
            if not left:
                break
            best = _best(job, left, [s for s in picked if s["job"] == job])
            picked.append(best)
            left.remove(best)

    def seconds(chosen):
        return sum(s["to"] - s["from"] for s in chosen)

    first = {}
    for s in picked:
        first.setdefault(s["job"], s)
    drops = [s for s in picked if first[s["job"]] is not s][::-1] + [
        first[job] for job in reversed(KEEP[CORE:]) if job in first]
    for s in drops:
        if seconds(picked) <= length * OVER:
            break
        picked.remove(s)
    picked.sort(key=lambda s: (ORDER.index(s["job"]), s["clip"], s["from"]))
    used = {s["clip"] for s in picked}
    left_out = []
    for c in clips:
        if c["n"] in used:
            continue
        if not c.get("talking"):
            why = "no one talks through it, and a review is told by what is said"
        else:
            mine = sorted({jobs[ln["n"]] for ln in lines if ln["clip"] == c["n"]} - {"other"})
            why = ("what is said in it is said better in another take: %s" % ", ".join(mine) if mine
                   else "nothing said in it is part of the review: setting up, settings step by step or talk off "
                        "the script")
        left_out.append({"clip": c["n"], "why": why})
    return {"clips": [{"clip": s["clip"], "start": round(s["from"], 2), "end": round(s["to"], 2), "job": s["job"],
                       "lines": s["lines"]} for s in picked],
            "left_out": left_out, "length": round(seconds(picked), 1)}
