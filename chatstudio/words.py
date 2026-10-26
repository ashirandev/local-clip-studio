"""Words Muse writes, tidied by the program before they are shown or read back.

Glimmer sometimes writes the same Thai mark twice ("ค้้น"). Read back in the next question, the slip is copied and
grows step by step (measured 10-27: 2, 3, 4, then 8 marks in a row, then 15,000 tokens of thinking without an
answer). Thai never stacks the same mark twice, so a run of one mark is cut to one.
"""
import json
import re

THAI_MARK_RUN = re.compile("([ัิ-ฺ็-๎])\\1+")


def tidy(text):
    return THAI_MARK_RUN.sub(r"\1", text) if isinstance(text, str) else text


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
    """Tool call arguments (a JSON string) with every string tidied; unreadable JSON is left as it came."""
    if not isinstance(raw, str):
        return tidy_all(raw)
    try:
        value = json.loads(raw)
    except ValueError:
        return raw
    fixed = tidy_all(value)
    return raw if fixed == value else json.dumps(fixed, ensure_ascii=False)
