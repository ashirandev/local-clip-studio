"""Recipes: one file per kind of video in recipes/, each a short guide Muse reads before it plans that kind of clip.

A recipe is Markdown with front matter, like a SKILL.md file: name, title, chip (the button that switches to it),
description (when to pick it), length (seconds) and style (the house style it starts with), then the body. The body
says what a good clip of that kind is and which rules apply, in words, never in tool names, so it stays true as
tools are added. Muse picks the recipe; the loop makes sure the rules that must hold are checked.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = os.path.join(ROOT, "recipes")
FIELDS = ("name", "title", "chip", "description", "length", "style")


def read(path):
    """{name, title, chip, description, length, style, body} from one recipe file."""
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
    recipe["body"] = body.strip()
    return recipe


def load(folder=FOLDER):
    return {r["name"]: r for r in (read(os.path.join(folder, n)) for n in sorted(os.listdir(folder))
                                   if n.endswith(".md"))}


RECIPES = load()
