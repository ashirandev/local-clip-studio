"""Words Muse writes, tidied by the program before they are shown or read back.

Glimmer sometimes writes the same Thai mark twice ("ค้้น"). Read back in the next question, the slip is copied and
grows step by step (measured 10-27: 2, 3, 4, then 8 marks in a row, then 15,000 tokens of thinking without an
answer). Thai never stacks the same mark twice, so a run of one mark is cut to one. Glimmer also writes half a
Thai letter now and then, which reaches the studio as U+FFFD, the sign for a broken letter (seen 10-27 after the
tone mark of พื้นที่ and of ชั้น); that sign never means anything, so it is taken out.

Muse is asked for plain text, and still writes some markdown (seen 10-27: * bullets and ** bold). The page shows
text as it is, so bold marks and heading marks are dropped and * bullets become - bullets, before the answer is
shown or read back.

Glimmer misspells Thai late in a long run, and not early (measured 10-27, 16 answers each, through the studio's own
Mind: with only the question, no bit any dictionary lacks was misspelled; after a run's 12,500 tokens, 5 to 8 bits
in 100 Thai words, in half the answers; temperature 0.1 instead of 0.5 only halved it). So the words a person keeps
are fixed in a short conversation of their own (spelled): on the 16 long-run answers the pointer flagged, 59 bits
went to 1, every number was kept, and 97% of the text stayed as it was. One question for many texts is cheaper
than one each (8 texts in 40 s, against about 15 s each), because the thinking is shared.
"""
import difflib
import json
import re

from . import thai

BROKEN = "\ufffd"
THAI_MARK_RUN = re.compile("([ัิ-ฺ็-๎])\\1+")
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
HEADING = re.compile(r"^#{1,6}[ \t]+", re.M)
BULLET = re.compile(r"^([ \t]*)[*+][ \t]+", re.M)
MARKUP = re.compile(r"<\|[a-z_]+\|>|</?atem:")


def tidy(text):
    return mend(THAI_MARK_RUN.sub(r"\1", text.replace(BROKEN, ""))) if isinstance(text, str) else text


def tool_markup(text):
    """True when the answer holds a chat template's tags or a tool call written out as words, which no person can read
    (Spark Open, 11-01: <|start|>assistant to=mute_sound<|message|><atem:function_calls>)."""
    return isinstance(text, str) and bool(MARKUP.search(text))


def unmarked(text):
    """The words of an answer without those tags and without the tool call written out after them."""
    if not tool_markup(text):
        return text
    kept = re.sub(r"<\|start\|>[^<]*<\|message\|>", "", text.split("<atem:")[0])
    return MARKUP.sub(" ", kept).strip()


def plain(text):
    """An answer without the markdown Muse was asked not to write."""
    if not isinstance(text, str):
        return text
    return BULLET.sub(r"\1- ", HEADING.sub("", BOLD.sub(r"\1", text)))


THAI = re.compile("[\u0e01-\u0e5b]")
CREATOR_WORDS = {"มินิมอล", "ไตเติล", "แคปชัน", "แคปชั่น", "แฮชแท็ก", "คอมโบ", "รีล", "รีลส์", "ช็อต", "ทรานซิชัน", "แอป", "เทส",
                 "บอท", "บอต", "เกต", "คอปปี้", "คอมมิชชัน", "บูลีน", "สวีต", "มิลลิวินาที", "เรนเดอร์",
                 "สติกเกอร์", "สติ๊กเกอร์", "เอฟเฟกต์", "เอฟเฟ็กต์", "สตรีมเมอร์", "จอย", "โกล", "คัลเจอร์"}
_WORDS = []
SPELL_ASK = ("Fix only the spelling in each of these Thai texts: tone marks, vowels and letters that are wrong. Keep "
             "every word, every number, every hashtag and the meaning; do not reword or add anything. Give back the "
             "same number of texts, in order, each with its line breaks.\n\n%s")
SPELL_SCHEMA = {"type": "object", "properties": {"texts": {"type": "array", "items": {"type": "string"}}},
                "required": ["texts"], "additionalProperties": False}
MOST_CHANGED = 0.2
GROWS = 2


SLIPS = {"\u0e37": "\u0e4c", "\u0e49": "\u0e48\u0e31", "\u0e25": "\u0e23"}
SLIP_ROUNDS = 4
MET = 100
UNMARKED = re.compile("(\u0e40[\u0e01-\u0e2e]{1,2}\u0e2d\u0e23)(?![\u0e01-\u0e2e\u0e30-\u0e3a\u0e47-\u0e4e\ufffd])")


def marked(text):
    """text with the \u0e4c put back on a loanword's last syllable written \u0e40-\u0e2d\u0e23 without it (\u0e04\u0e2d\u0e21\u0e1e\u0e34\u0e27\u0e40\u0e15\u0e2d\u0e23, \u0e42\u0e1f\u0e25\u0e40\u0e14\u0e2d\u0e23,
    \u0e1b\u0e4a\u0e2d\u0e1b\u0e04\u0e31\u0e25\u0e40\u0e08\u0e2d\u0e23): where the syllable ends the word, it takes the \u0e4c when the Thai National Corpus has met it so MET
    times and without it fewer. Thai words hardly end so: the corpus has two such words, met 7 times in all (the name
    \u0e40\u0e2d\u0e21\u0e2d\u0e23 6 of them), against 37,799 times for words ending \u0e2d\u0e23\u0e4c. Measured 10-29 on everything Muse had written: \u0e2d\u0e23\u0e4c 664 times, and 24 without the \u0e4c,
    every one of them dropped (\u0e2a\u0e15\u0e23\u0e35\u0e21\u0e40\u0e21\u0e2d\u0e23, \u0e40\u0e23\u0e19\u0e40\u0e14\u0e2d\u0e23, \u0e04\u0e2d\u0e21\u0e1e\u0e34\u0e27\u0e40\u0e15\u0e2d\u0e23). The dictionary flags none of them, as the bare
    syllables cut into words it has (\u0e40\u0e08\u0e2d and \u0e23), and it flags the right ones, which it lacks; so neither odd_thai nor
    Muse's spelling check could put the \u0e4c back. A \u0e23 a word goes on from (\u0e40\u0e08\u0e2d\u0e23\u0e16, \u0e40\u0e18\u0e2d\u0e23\u0e39\u0e49) is left as it is."""
    if not isinstance(text, str) or not UNMARKED.search(text):
        return text
    counts = _counts()[0]
    return UNMARKED.sub(lambda m: m.group(1) + "\u0e4c" if counts.get(m.group(1) + "\u0e4c", 0) >= MET > counts.get(
        m.group(1), 0) else m.group(1), text)


def unslipped(text):
    """text with Glimmer's slips of the Thai marks put back where one swap makes words of bits no dictionary word
    makes up: an ื written for ์ (เรนเดอรือีก), an ้ for ่ or ั (ที้, ห้ว). The slips grow as a job goes on and are
    the model's own first choice, not the sampling: beat 1's step 14 replayed 20 times wrote one in every answer at
    temperature 0.5, at 0.2, and with every earlier picture taken out (10-29). On beat 1's 25 why lines and 40
    replayed answers the swaps took 117 such bits to 25, every change right; any look-alike swap made two wrong
    words, and adding a missing ์ put it in the wrong place, so neither is done. The word a swap makes must be one the
    Thai National Corpus has met MET times, or one of CREATOR_WORDS: อีกคร้ง (ครั้ง with its ั dropped) became คร่ง,
    which the corpus never met, and then ครัง, met 6 times; the right fixes make ที่, ช่วง, ใส่, ตัว (12,059 to
    818,364). A bit no single swap mends stays as written. A ์ dropped from a last syllable is put back first (marked).
    A ล written for ร goes back the same way: hearing wrote อย่าลุมดิ and Glimmer's titles took โดนลุม from it, in 4 of
    its Sifu titles, and its own check kept it. On 1,168 texts Muse wrote, ล to ร changed 12, all right (10-31); ร to ล
    made คลิปโต of คริปโต and ปลากฎ of ปรากฎ, so a ร is left alone."""
    text = marked(text)
    for _ in range(SLIP_ROUNDS):
        bits = odd_thai(text)
        best, fewest = None, len(bits)
        for bit in bits:
            at = text.find(bit)
            for i in range(max(0, at), at + len(bit) if at >= 0 else 0):
                for mark in SLIPS.get(text[i], ""):
                    tried = text[:i] + mark + text[i + 1:]
                    left = len(odd_thai(tried))
                    if left < fewest and _met(tried, at, at + len(bit)):
                        best, fewest = tried, left
        if best is None:
            return text
        text = best
    return text


def _met(text, a, b):
    """Whether every Thai word of text that overlaps letters a to b is one the Thai National Corpus has met MET times,
    is made of such words (อีกครั้ง, which the corpus counts as อีก and ครั้ง), or is one of CREATOR_WORDS: a swap
    can also make words of a wrong cut (อีกครัง read as อี, กค, รัง)."""
    from pythainlp.tokenize import word_tokenize
    counts, trie = _counts()
    at, ok = 0, True
    for w in word_tokenize(text, engine="newmm", keep_whitespace=True, custom_dict=_WORDS[1]):
        if at < b and a < at + len(w) and THAI.search(w):
            word = w.strip()
            parts = [p for p in word_tokenize(word, engine="newmm", custom_dict=trie) if p.strip()]
            ok = ok and (counts.get(word, 0) >= MET or word in CREATOR_WORDS
                         or (len(parts) > 1 and all(counts.get(p, 0) >= MET for p in parts)))
        at += len(w)
    return ok


def odd_thai(words):
    """Bits of Thai in words that no dictionary word makes up, each with the words on either side: the place a tone
    mark or a vowel may be wrong (Glimmer writes ที้ for ที่, เพื่ม for เพิ่ม). PyThaiNLP's own corrections were
    wrong on these (10-27), so the program only points, and Muse fixes the spelling.

    The repeat mark ๆ is not a word (สวยๆ is สวย said twice), and CREATOR_WORDS are loanwords for making clips and
    apps that the dictionary does not have (ไตเติล, แฮชแท็ก, แอป, เทส). Measured 10-27: in 32 answers Glimmer wrote with a short context, every
    bit pointed at was one of these."""
    if not THAI.search(words or ""):
        return []
    from pythainlp.tokenize import word_tokenize
    _dictionary()
    toks = word_tokenize(words, engine="newmm", keep_whitespace=True, custom_dict=_WORDS[1])
    out = []
    for i, w in enumerate(toks):
        base = w.strip().rstrip("ๆ").strip()
        if THAI.search(base) and base not in _WORDS[0]:
            near = [x for x in (toks[i - 1] if i else "", w, toks[i + 1] if i + 1 < len(toks) else "") if x.strip()]
            out.append("".join(near))
    return out


def known(text):
    """[(start, end)] of each word of text the dictionary odd_thai reads has, Thai cut into words by newmm: in
    เราสามารถ, (3, 9) for สามารถ (can), which has สาม (three) in it."""
    from pythainlp.tokenize import word_tokenize
    _dictionary()
    out, at = [], 0
    for w in word_tokenize(text, engine="newmm", keep_whitespace=True, custom_dict=_WORDS[1]):
        if w in _WORDS[0]:
            out.append((at, at + len(w)))
        at += len(w)
    return out


def _dictionary():
    if not _WORDS:
        from pythainlp.corpus import thai_words
        from pythainlp.util import dict_trie
        _WORDS.append(thai_words() | CREATOR_WORDS)
        _WORDS.append(dict_trie(_WORDS[0]))


TONES = "่้๊๋"
PATH = re.compile(r"\\|(?<!\d)/|/(?!\d)")


def mend(text):
    """text with Glimmer's slips taken out where the dictionary says so (seen 10-27 in Muse's reasons on the talk
    clip, where each step copied the one before and added a letter: ชั้่นส่วน, ชั้่นสส่วน, ชั้่นสสส่วน, ชั้่นสสสส่วน).
    Only deletions, and a mark written one letter late moved back (_swaps), so no word is made up; text with a path
    in it is left as it is (a / between two numbers, 4/4, is not a path). A vowel written after its tone mark is put
    before it first (ต้ัง is ตั้ง), as Thai always writes them, whatever the dictionary says, unless the same vowel
    stands before the mark too (เลื่ือก, below). Only where Thai cannot be
    written so: a tone mark after another tone mark, after ็ or after ์ (ชั้่น, เป็่น), a vowel written again around
    a tone mark (เพื่ือ), and a letter written twice (มมุม), each kept only when fewer bits of the text are no word;
    and three or more of one letter (สสส่วน), cut to one when that is no worse and leaves the text in fewer words. As
    in every slip seen, the last of the letters carries a mark, and a letter written twice is one the dictionary
    leaves standing alone: one that ends a word and starts the next (แบบบน, จอออก, เทสสี, บอทที่) is left as it is.
    Measured 10-27 on 1,500 lines of the team's own Thai notes: looser rules (any tone mark; any letter written twice
    when it left fewer words) changed 21 lines that were right, such as เล็กกว่า to เล็กว่า."""
    if not isinstance(text, str) or not THAI.search(text) or PATH.search(text):
        return text
    text = TONE_FIRST.sub(lambda m: m.group(0) if m.group(1) == m.group(3) else m.group(1) + m.group(3) + m.group(2),
                          text)
    odd, count = _odd(text)
    while True:
        for better, run in _mended(text):
            n, c = _odd(better)
            if (n < odd and (run is not None or c <= count)) or (run and n <= odd and c < count):
                text, odd, count = better, n, c
                break
        else:
            break
    return text


def _mended(text):
    """Each text a slip taken out would leave, with whether it cut a run of one letter: the deletions first, then
    each mark written one letter late put back a letter earlier."""
    for gone, run in _tries(text):
        yield "".join(ch for k, ch in enumerate(text) if k not in gone), run
    for i in _swaps(text):
        yield text[:i - 1] + text[i] + text[i - 1] + text[i + 1:], None


def _swaps(text):
    """[i]: a vowel or tone mark inside a bit that is no word, right after a letter it may belong before. Glimmer
    writes a mark one letter late, and in a long chat it copies the slips already there: 10-28, Tarm's question after
    an answer with ขอ้ got ไดเ้, ใหเ้, ตอ้ง, ตัง้ and ปรบั, about 5 in 100 Thai letters, against 0.6 when that one word
    was spelled right (lane 0, n=10 each). The answer is mended before Muse reads it back. A swap is kept only when
    it leaves fewer bits that are no word and no more words. Measured 10-28 with the order fix in mend: they mended 98
    of the 1,858 Thai texts Muse wrote in the studio (bits that are no word 183 to 91) and changed 5 of 34,665 lines
    of the team's own notes, 3 of them wrongly, loanwords the dictionary lacks (นิสต์, คอนทราสต์). Letting a swap
    split the text into more words had turned สกิล into สิกล."""
    from pythainlp.tokenize import word_tokenize
    out, at = [], 0
    for w in word_tokenize(text, engine="newmm", keep_whitespace=True, custom_dict=_WORDS[1]):
        base = w.strip().rstrip("ๆ").strip()
        if THAI.search(base) and base not in _WORDS[0]:
            out += [i for i in range(max(at, 2), at + len(w)) if text[i] in MARKS and THAI.match(text[i - 1])]
        at += len(w)
    return out


MEND_REACH = 4
MARKS = "ัิีึืฺุู็" + TONES + "์ํ๎"
NO_TONE_AFTER = TONES + "\u0e47\u0e4c"
TONE_FIRST = re.compile("(.)([%s])([\u0e31\u0e34\u0e35\u0e36\u0e37\u0e38\u0e39])" % TONES)


def _tries(text):
    """[(the places to delete, whether it is a run of three or more of one letter)]: each such run cut to one, each
    slip on its own, and each two slips within MEND_REACH letters."""
    from pythainlp.tokenize import word_tokenize

    def carried(i):
        return i + 1 < len(text) and text[i + 1] in MARKS

    lone, at = set(), 0
    for w in word_tokenize(text, engine="newmm", keep_whitespace=True, custom_dict=_WORDS[1]):
        if len(w) == 1:
            lone.add(at)
        at += len(w)
    slips = [i for i, ch in enumerate(text) if i and THAI.match(ch) and (
        (ch in TONES and text[i - 1] in NO_TONE_AFTER) or (ch == text[i - 1] and carried(i) and i - 1 in lone) or
        (i > 1 and ch in MARKS and text[i - 1] in TONES and ch == text[i - 2]) or
        (ch in TONES and text[i - 1] in MARKS and i + 1 < len(text) and text[i + 1] == text[i - 1]))]
    out, i = [], 0
    while i < len(text):
        j = i
        while j + 1 < len(text) and text[j + 1] == text[i]:
            j += 1
        if j >= i + 2 and THAI.match(text[i]) and carried(j):
            out.append((tuple(range(i + 1, j + 1)), True))
        i = j + 1
    out += [((i,), False) for i in slips]
    out += [((i, j), False) for i in slips for j in slips if i < j <= i + MEND_REACH]
    return out


def _odd(text):
    """(how many bits of text are not words, how many words it is cut into), compared in that order: the bits
    odd_thai points at and the Thai letters standing alone, then the words. A letter written twice can still cut into
    words (สสส่วน into สส and ส่วน, การมมี into กา, รม and มี), so of two spellings with as many bits that are not
    words, the one in fewer words is the one meant."""
    from pythainlp.tokenize import word_tokenize
    odd = len(odd_thai(text))
    toks = [w for w in word_tokenize(text, engine="newmm", keep_whitespace=True, custom_dict=_WORDS[1]) if w.strip()]
    lone = sum(1 for w in toks if len(w) == 1 and "\u0e01" <= w <= "\u0e2e")
    return odd + lone, len(toks)


RARE = 1000


def split_up(before, after):
    """odd_thai(before) when the fix after cut before into more words and one of them is rare, else []: Glimmer's ขึ้อน for
    ขึ้น became ขี้อน, the words ขี้ and อน (the corpus has 315 of อน), so odd_thai sees nothing after it (11-01, 015633-a2c5)."""
    from pythainlp.tokenize import word_tokenize
    odd = odd_thai(before)
    if not odd:
        return []
    counts, trie = _counts()

    def cut(text):
        return [w for w in word_tokenize(text, engine="newmm", keep_whitespace=True, custom_dict=_WORDS[1]) if w.strip()]

    def rare(word):
        return any(counts.get(p, 0) < RARE for p in word_tokenize(word, engine="newmm", custom_dict=trie) if p.strip())

    was, now = cut(before), cut(after)
    return odd if len(now) > len(was) and any(rare(w) for w in now if w not in was and THAI.search(w)) else []


def spelled(ask, texts, every=False, kept=None):
    """(texts with their Thai spelling fixed, [(before, after)] for each one fixed). The texts odd_thai flags, or with
    every all Thai texts, are asked about in one question to Muse in a conversation of their own: ask(messages, schema)
    returns the answer's words. A fix is kept when it keeps every number and either leaves fewer flagged bits and
    changes at most a fifth of the text, or changes only tone marks and vowels and flags no more bits; otherwise the
    text stays as it was written. kept, a dict, keeps each fix and each text Muse gave back as it was, so it is not
    asked about again; a change these rules refused is not kept, and the text is asked about again next time, when
    the texts around it may tell Muse more: alone, Muse wrote ต้องสู้ for ต้อสู้ five times of five; with the post text
    beside it, ต่อสู้ three of three.

    every is for what the viewer reads. Glimmer's slips there are often real words, so the dictionary flags nothing:
    ต้อสู้ for ต่อสู้, ตังค่า for ตั้งค่า, ต้อวัน for ต่อวัน (10-27 and 10-28, in titles and post text). Measured 10-28 on the
    111 Thai titles, captions and hashtags Muse had written, all asked: Muse changed 11, these rules kept the 9 right
    fixes and refused the 2 wrong ones (a reworded line, a word dropped), and none of the 100 right texts changed.
    Round 8 the same day, Muse made ตั้งแต่ into ตั้งแต้, a tone mark only: such a change is now kept only when the words
    it touches are commoner after it (_commoner); on those 112 texts that keeps the 9 right fixes and refuses it.
    And Muse dropped the ง of ห้องบอลรูม in an XM clip (ห้อ is a word: to gallop); asked, it wrote ห้อง five times of
    five, which no rule kept. A change that adds or swaps at most GROWS letters, drops none, and makes the words it
    touches commoner is kept too (_grows_little); on the same texts it keeps that one and no wrong one."""
    texts, kept = list(texts), {} if kept is None else kept
    todo = [i for i, t in enumerate(texts) if isinstance(t, str) and (THAI.search(t) if every else odd_thai(t))]
    todo = [i for i in todo if texts[i] not in kept]
    if todo:
        answer = ask([{"role": "user", "content": SPELL_ASK % json.dumps([texts[i] for i in todo],
                                                                          ensure_ascii=False)}], SPELL_SCHEMA)
        try:
            back = json.loads(answer)["texts"]
        except (ValueError, KeyError, TypeError):
            back = []
        for i, after in zip(todo, back if isinstance(back, list) else []):
            before = texts[i]
            if isinstance(after, str) and after != before and thai.values(after) == thai.values(before) and (
                    (len(odd_thai(after)) < len(odd_thai(before))
                     and difflib.SequenceMatcher(None, before, after).ratio() >= 1 - MOST_CHANGED)
                    or (_skeleton(after) == _skeleton(before) and len(odd_thai(after)) <= len(odd_thai(before))
                        and _commoner(before, after))
                    or (_grows_little(before, after) and _commoner(before, after))):
                kept[before] = after
            elif after == before:
                kept[before] = before
    changed = []
    for i, before in enumerate(texts):
        after = kept.get(before, before) if isinstance(before, str) else before
        if after != before:
            texts[i] = after
            changed.append((before, after))
    return texts, changed


_COUNTS = []


def _counts():
    """The Thai National Corpus's word counts, and a dictionary of its words, loaded once."""
    if not _COUNTS:
        from pythainlp.corpus import tnc
        from pythainlp.util import dict_trie
        _COUNTS.append(dict(tnc.word_freqs()))
        _COUNTS.append(dict_trie(set(_COUNTS[0])))
    return _COUNTS


def _commoner(before, after):
    """Whether the words a change touches are commoner after it than before, counted in the Thai National Corpus:
    ต้อสู้ to ต่อสู้ (361 to 4,604) and ตังค่า to ตั้งค่า (339 to 21,261) are fixes, ตั้งแต่ to ตั้งแต้ (21,505 to 116) is not."""
    ops = [o for o in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes() if o[0] != "equal"]
    return _least(after, [(j1 - 1, max(j2, j1 + 1)) for _, _, _, j1, j2 in ops]) > \
        _least(before, [(i1 - 1, max(i2, i1 + 1)) for _, i1, i2, _, _ in ops])


def _grows_little(before, after):
    """Whether after is before with at most GROWS letters added or swapped and none taken away."""
    ops = [o for o in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes() if o[0] != "equal"]
    return bool(ops) and all(j2 - j1 >= i2 - i1 for _, i1, i2, j1, j2 in ops) and \
        sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in ops) <= GROWS


def _least(text, spans):
    """The least corpus count of the words of text that overlap any of spans ((from, to) in characters); 0 for a word
    the corpus does not have."""
    from pythainlp.tokenize import word_tokenize
    counts, trie = _counts()
    least, at = None, 0
    for w in word_tokenize(text, engine="newmm", keep_whitespace=True, custom_dict=trie):
        a, at = at, at + len(w)
        if w.strip() and any(a < y and x < at for x, y in spans):
            n = counts.get(w.strip(), 0)
            least = n if least is None else min(least, n)
    return least or 0


VOWELS_AND_MARKS = set("ัิีึืุู็่้๊๋์ํ")


def _skeleton(text):
    """text without its tone marks and the vowels written above and below the letters."""
    return "".join(ch for ch in text if ch not in VOWELS_AND_MARKS)


def tidy_all(value):
    """Every string inside a JSON value, tidied."""
    if isinstance(value, str):
        return tidy(value)
    if isinstance(value, list):
        return [tidy_all(v) for v in value]
    if isinstance(value, dict):
        return {k: tidy_all(v) for k, v in value.items()}
    return value


def tidy_arguments(raw):
    """Tool call arguments (a JSON string) with every string tidied, and the brackets and the text closed when Glimmer
    stopped before them; JSON that closing does not make readable is left as it came."""
    if not isinstance(raw, str):
        return tidy_all(raw)
    try:
        value = json.loads(raw)
    except ValueError:
        whole = closed(raw)
        if whole is None:
            return raw
        value = json.loads(whole)
        return json.dumps(tidy_all(value), ensure_ascii=False)
    fixed = tidy_all(value)
    return raw if fixed == value else json.dumps(fixed, ensure_ascii=False)


def closed(raw):
    """raw with the text and the brackets it left open closed, or None when that does not make it JSON. Seen 10-28:
    Glimmer ended a hand-over's arguments after the hashtags, without the last "}", and the job ended there, as
    llama-server could not read that call in the conversation on the next question."""
    shut, inside, escaped = [], False, False
    for ch in raw:
        if inside:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                inside = False
        elif ch == '"':
            inside = True
        elif ch in "{[":
            shut.append("}" if ch == "{" else "]")
        elif ch in "}]" and (not shut or shut.pop() != ch):
            return None
    tail = ('"' if inside else "") + "".join(reversed(shut))
    if not tail:
        return None
    try:
        json.loads(raw + tail)
    except ValueError:
        return None
    return raw + tail
