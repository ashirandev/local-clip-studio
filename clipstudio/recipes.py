"""Recipes: one file per kind of video in recipes/, each a short guide Muse reads before it plans that kind of clip.

A recipe is Markdown with front matter, like a SKILL.md file: name, title, chip (the button that switches to it),
description (when to pick it), length (seconds), style (the house style it starts with) and map (what the map ranks
by, such as "sound 1, screen 1"), and, when the program must read the screen for private words, read: screen. Then the
body. The body
says what a good clip of that kind is and which rules apply, in words, never in tool names, so it stays true as
tools are added. Muse picks the recipe; the loop makes sure the rules that must hold are checked.

recipes/house.md is not a recipe: it is the house standard, the way the studio's makers cut their own short clips
(the opening line, whole sentences, words a stranger knows, no other app named...), and it ends every recipe's body.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = os.path.join(ROOT, "recipes")
FIELDS = ("name", "title", "chip", "description", "length", "style", "map")
SIGNALS = ("sound", "screen")
HOUSE = "house.md"
READS = ("screen",)


def read(path):
    """{name, title, chip, description, length, style, map, read, body} from one recipe file."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    head, sep, body = text.partition("\n---\n")
    if not text.startswith("---\n") or not sep:
        raise ValueError("%s has no front matter between --- lines." % path)
    recipe = {}
    for line in head[4:].splitlines():
        key, _, value = line.partition(":")
        recipe[key.strip()] = value.strip()
    missing = [k for k in FIELDS if not recipe.get(k)]
    if missing:
        raise ValueError("%s is missing %s." % (path, ", ".join(missing)))
    recipe["length"] = int(recipe["length"])
    recipe["map"] = weights(recipe["map"], path)
    recipe["read"] = recipe.get("read") or None
    if recipe["read"] not in (None,) + READS:
        raise ValueError("%s: read is %s, not %s." % (path, " or ".join(READS), recipe["read"]))
    recipe["body"] = body.strip()
    return recipe


def weights(text, path):
    """{"sound": 1.0, "screen": 1.0} from "sound 1, screen 1". A signal left out counts 0."""
    out = dict.fromkeys(SIGNALS, 0.0)
    for part in text.split(","):
        name, _, value = part.strip().partition(" ")
        if name not in SIGNALS:
            raise ValueError("%s: the map ranks by %s, not by %s." % (path, " and ".join(SIGNALS), name))
        out[name] = float(value)
    return out


def load(folder=FOLDER):
    """The recipes in folder by name, each body ending with the house standard (HOUSE), the rules every clip
    follows whatever its kind."""
    with open(os.path.join(folder, HOUSE), encoding="utf-8") as f:
        house = f.read().strip()
    return {r["name"]: dict(r, body=r["body"] + "\n\n" + house)
            for r in (read(os.path.join(folder, n)) for n in sorted(os.listdir(folder))
                      if n.endswith(".md") and n != HOUSE)}


RECIPES = load()
