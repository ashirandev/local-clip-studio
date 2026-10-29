"""The mind: one way to ask Muse a question, whichever Muse it is.

Four minds, one request format (the OpenAI chat API):
- local: Muse Glimmer, Meta's open-weight model, on this computer through llama.cpp's own server (llama-server).
  The address is one setting, LCS_MIND_URL, so any server that runs an eligible Muse model works. Each question names
  the model, LCS_MIND_MODEL (muse-glimmer): a router llama-server sends it to that model, and one that serves a
  single model ignores the name.
- spark: Muse Spark on Meta's hosted API, with the person's own key in META_API_KEY. The key is read from the
  environment when a question is asked and is never written anywhere.
- wonder: Muse Wonder, Meta's model for long work with many tools, on the same API and key as Spark, at three to
  five times Spark's price (LCS_WONDER_MODEL names it).
- modal: Muse Spark Open, the open-weight Spark, on a Modal shared endpoint of the person's own workspace, with their
  proxy token in MODAL_KEY and MODAL_SECRET (LCS_MODAL_MODEL names the endpoint's model).

Every answer is streamed, so Muse's thinking can be shown while it is written, and a question in flight can be
cancelled at once: Asking.cancel() shuts the connection, and llama-server drops the work ("stop: cancel task" in its
log). Every answer comes back with its engine numbers: how much of the question was read new, how much was reused
from the question before, and how fast it read and wrote (llama-server's own timings; the hosted API gives token
counts).
"""
import hashlib
import http.client
import io
import json
import os
import queue
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

TIMEOUT = 900
LOAD_SECONDS = 120
CHAT_TEMPERATURE = 0.7
PRICES = {
    "muse-spark-1.3": (1.25, 0.15, 4.25),
    "muse-wonder-1.0": (4.00, 0.20, 20.00),
    "meta-models/Muse-Spark-Open-816B-A42B": (1.25, 0.15, 4.25),
}
HOSTED_ROOM = 16384
LOOPBACK = ("127.0.0.1", "localhost", "::1")
KEY_FILE = os.path.join(os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config"),
                        "LocalClipStudio", "keys.env")
PRICED = ("Meta's price list (dev.meta.ai/docs/pricing-rate-limits) and Modal's for Muse Spark Open, both read on "
          "28 Oct 2026")


def load_keys(named=None):
    """Puts the KEY=VALUE lines of the files LCS_KEYS names (separated by ; on Windows, : elsewhere), then KEY_FILE,
    into this process's environment, so a key can stay in a file of the person's own, outside the studio's folder; a
    key already in the environment wins. Lines may start with "export " and values may be quoted; blank lines and #
    lines are left out. Returns the names it set, never a value; a file that cannot be read is left out."""
    done = []
    named = os.environ.get("LCS_KEYS", "") + os.pathsep + KEY_FILE if named is None else named
    for path in [p.strip() for p in named.split(os.pathsep) if p.strip()]:
        try:
            with open(path, encoding="utf-8-sig") as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        for line in lines:
            name, eq, value = line.strip().removeprefix("export ").partition("=")
            name, value = name.strip(), value.strip()
            if not eq or not name or name.startswith("#") or not name.replace("_", "").isalnum():
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value and not os.environ.get(name):
                os.environ[name] = value
                done.append(name)
    return done


def api_names(url):
    """The settings an endpoint and its key go into: Modal's for a *.modal.direct address (its proxy token is a key and
    a secret), else Meta's API, the way Muse Spark is reached."""
    if (urllib.parse.urlsplit(url).hostname or "").endswith(".modal.direct"):
        return "LCS_MODAL_URL", ("MODAL_KEY", "MODAL_SECRET")
    return "LCS_SPARK_URL", ("META_API_KEY",)


def saved_api():
    """The endpoint the person saved in KEY_FILE ("" when none), and whether a key is saved with it. Never the key."""
    try:
        with open(KEY_FILE, encoding="utf-8-sig") as f:
            got = dict(line.partition("=")[::2] for line in f.read().splitlines() if "=" in line)
    except OSError:
        return "", False
    url = got.get("LCS_MODAL_URL") or got.get("LCS_SPARK_URL") or ""
    return url, bool(url) and all(got.get(name) for name in api_names(url)[1])


def save_api(url, key):
    """Keeps the endpoint and the API key the person typed in KEY_FILE, in their own user folder and never in the
    studio's, and puts them in this process's environment at once, over what was there. Returns None, or a sentence
    saying what to give instead. The key goes only to that endpoint, in the Authorization header of each question."""
    url, key = url.strip().rstrip("/"), key.strip()
    u = urllib.parse.urlsplit(url)
    if not u.hostname or not (u.scheme == "https" or (u.scheme == "http" and u.hostname in LOOPBACK)):
        return "Give the endpoint's address, starting with https://."
    if not key:
        return "Give the API key."
    url_name, key_names = api_names(url)
    if len(key_names) == 2:
        parts = [p for p in re.split(r"[.:\s]+", key, maxsplit=1) if p]
        if len(parts) != 2:
            return "A Modal proxy token has two parts: give the key and the secret joined by a dot."
        values = dict(zip(key_names, parts))
    else:
        values = {key_names[0]: key}
    values[url_name] = url
    gone = {"LCS_SPARK_URL", "LCS_MODAL_URL", "META_API_KEY", "MODAL_KEY", "MODAL_SECRET"}
    try:
        with open(KEY_FILE, encoding="utf-8-sig") as f:
            kept = [line for line in f.read().splitlines() if line.partition("=")[0].strip() not in gone]
    except OSError:
        kept = []
    os.makedirs(os.path.dirname(KEY_FILE), exist_ok=True)
    tmp = KEY_FILE + ".tmp"
    text = "\n".join(kept + ["%s=%s" % pair for pair in values.items()]) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    with open(tmp, encoding="utf-8") as f:
        if f.read() != text:
            return "The key could not be written to %s." % KEY_FILE
    os.replace(tmp, KEY_FILE)
    os.environ.update(values)
    return None


class MindError(Exception):
    """A sentence for the page: what went wrong and what to do about it."""


class TooLong(MindError):
    """The conversation and the answer do not fit in the model's memory (its context): asked is how many tokens they
    came to, room how many fit."""

    def __init__(self, sentence, asked, room):
        MindError.__init__(self, sentence)
        self.asked, self.room = asked, room


class Cancelled(Exception):
    """The question was cancelled while it was in flight."""


class Asking:
    """One question in flight. Another thread can cancel it at any moment, and the asker is free at once.

    The answer is read in a thread of its own and handed over piece by piece. Cancel shuts the connection, which
    makes llama-server drop the work, and wakes the asker straight away. It does not wait for the reading thread:
    measured 10-27 on Windows, a read stays blocked after the shutdown until the server closes its side, and a server
    that never notices would hold Stop for as long as it keeps writing.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.sock = None
        self.cancelled = False
        self.pieces = queue.Queue()

    def cancel(self):
        with self.lock:
            self.cancelled = True
            sock = self.sock
        self.pieces.put(("cancel", None))
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def stream(self, url, body, headers):
        """Posts the question and yields each streamed chunk as a dict. Raises Cancelled when cancelled."""
        threading.Thread(target=self._read, args=(url, body, headers), daemon=True).start()
        while True:
            kind, value = self.pieces.get()
            if self.cancelled:
                raise Cancelled()
            if kind == "error":
                raise value
            if kind == "done":
                return
            yield value

    def _read(self, url, body, headers):
        u = urllib.parse.urlsplit(url)
        kind = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        conn, resp = kind(u.hostname, u.port, timeout=TIMEOUT), None
        try:
            conn.connect()
            with self.lock:
                if self.cancelled:
                    return
                self.sock = conn.sock
            conn.request("POST", u.path, json.dumps(body).encode("utf-8"), headers)
            resp = conn.getresponse()
            if resp.status != 200:
                detail = io.BytesIO(resp.read(2000))
                raise urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, detail)
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    self.pieces.put(("chunk", json.loads(data)))
            self.pieces.put(("done", None))
        except (OSError, http.client.HTTPException, ValueError) as e:
            self.pieces.put(("error", e))
        finally:
            if resp is not None:
                resp.close()
            conn.close()


class Mind:
    def __init__(self, key, label, url, model=None, key_env=None, reasoning_line=None, where=""):
        self.key, self.label, self.url, self.model_id = key, label, url.rstrip("/"), model
        self.key_env, self.reasoning_line, self.where = key_env, reasoning_line, where

    def secret(self):
        return os.environ.get(self.key_env, "").strip() if self.key_env else ""

    def missing(self):
        return ("Set %s to your own Meta API key, or name a file that holds it in LCS_KEYS, then start the studio "
                "again." % self.key_env)

    def price(self):
        """US dollars per million tokens (read new, read from the provider's cache, written) by the provider's price
        list; nothing for a mind on this computer, and None for a hosted model whose price the studio does not know."""
        return PRICES.get(self.model_id) if self.key_env else (0.0, 0.0, 0.0)

    def cost(self, numbers):
        """What one answer cost in US dollars, or None when its price or its token counts are not known: never a
        guess."""
        price = self.price()
        if not price or numbers["read_new"] is None or numbers["written"] is None:
            return None
        new, cached, written = price
        return (numbers["read_new"] * new + (numbers["reused"] or 0) * cached + numbers["written"] * written) / 1e6

    def headers(self, messages=()):
        headers = {"Content-Type": "application/json"}
        if self.key_env:
            headers["Authorization"] = "Bearer " + self.secret()
        return headers

    def _call(self, path, body=None, timeout=10):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.url + path, data, self.headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def status(self):
        """{ok, key, label, where, local, name} or {ok: False, ..., why} with a sentence. local is True for a mind
        on this computer, False for one on someone's servers. Never sends a question."""
        out = {"key": self.key, "label": self.label, "where": self.where, "url": self.url, "local": not self.key_env}
        if self.key_env and not self.secret():
            return dict(out, ok=False, why=self.missing())
        if self.key_env:
            return dict(out, ok=True, name=self.model_id)
        if not listens(self.url):
            return dict(out, ok=False, why=self.unreachable(ConnectionRefusedError("nothing is listening there")))
        try:
            got = self._call("/models")
        except (OSError, ValueError) as e:
            return dict(out, ok=False, why=self.unreachable(e))
        ids = [m.get("id", "") for m in got.get("data") or []]
        if not ids:
            return dict(out, ok=False, why="The server at %s has no model loaded." % self.url)
        return dict(out, ok=True, name=self.model_id if self.model_id in ids else os.path.basename(ids[0]))

    def on_card(self):
        """Loads the model and waits for it when a router llama-server holds it unloaded. Started with
        --no-models-autoload, as the README shows, the router loads a model only when asked to, so nothing but the
        studio brings Glimmer back onto the graphics card while SAM 3.1 has it step aside (10-28: a question from
        another program loaded it 5 s into a SAM run, beside SAM and partly off the card, and it wrote 5.2 tokens a
        second instead of 47.5). Nothing to do for the hosted API, or for a server of one model, whose list of models
        carries no state."""
        if self.key_env or not self.model_id:
            return
        root = self.url[:-3] if self.url.endswith("/v1") else self.url
        until, asked = time.time() + LOAD_SECONDS, False
        while True:
            try:
                got = self._call("/models")
            except (OSError, ValueError):
                return
            row = next((m for m in got.get("data") or [] if m.get("id") == self.model_id), None)
            state = ((row or {}).get("status") or {}).get("value")
            if state in (None, "loaded"):
                return
            if state == "unloaded" and not asked:
                req = urllib.request.Request(root + "/models/load", json.dumps({"model": self.model_id}).encode(),
                                             {"Content-Type": "application/json"})
                try:
                    urllib.request.urlopen(req, timeout=LOAD_SECONDS).close()
                except urllib.error.HTTPError as e:
                    if e.code != 400:
                        raise MindError("%s could not be loaded onto the graphics card (HTTP %d)."
                                        % (self.label, e.code))
                except OSError as e:
                    raise MindError(self.unreachable(e))
                asked = True
            if time.time() > until:
                raise MindError("%s was not loaded onto the graphics card within %d s." % (self.label, LOAD_SECONDS))
            time.sleep(0.25)

    def unreachable(self, e):
        if self.key_env:
            return "Meta's API did not answer at %s (%s)." % (self.url, _reason(e))
        return ("Muse Glimmer is not running at %s (%s). Open Options and press Set up, or start llama-server as "
                "the README shows." % (self.url, _reason(e)))

    def ask(self, messages, tools=None, temperature=CHAT_TEMPERATURE, max_tokens=None, on_delta=None,
            asking=None, schema=None):
        """Asks once. Returns {text, thinking, tool_calls, message, engine}.

        on_delta(kind, piece) gets the thinking and the words as they are written ("thinking" or "text"). Pass an
        Asking to be able to cancel; a cancelled question raises Cancelled. A JSON schema makes the answer's words
        JSON of that shape. Anything else that goes wrong raises MindError with a sentence.

        message is the answer as it goes back into the conversation: the words and the tool calls, without the
        thinking. llama-server reuses everything before it; from this message on it reads again (a few hundred
        tokens), because the thinking it wrote is not sent back.
        """
        if self.key_env and not self.secret():
            raise MindError(self.missing())
        self.on_card()
        msgs = list(messages)
        if self.reasoning_line:
            if msgs and msgs[0].get("role") == "system":
                msgs[0] = dict(msgs[0], content=self.reasoning_line + "\n\n" + msgs[0]["content"])
            else:
                msgs.insert(0, {"role": "system", "content": self.reasoning_line})
        body = {"messages": msgs, "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True}}
        if self.model_id:
            body["model"] = self.model_id
        if tools:
            body["tools"] = tools
        if max_tokens:
            body["max_tokens"] = max(max_tokens, HOSTED_ROOM) if self.key_env else max_tokens
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "answer", "strict": True, "schema": _closed(schema) if self.key_env else schema}}
        asking = asking or Asking()
        text, thinking, calls, finish, tail = [], [], {}, None, {}
        t0 = time.time()
        try:
            for chunk in asking.stream(self.url + "/chat/completions", body, self.headers(msgs)):
                for key in ("timings", "usage"):
                    if chunk.get(key):
                        tail[key] = chunk[key]
                choice = (chunk.get("choices") or [{}])[0]
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                for kind, piece, into in (("thinking", delta.get("reasoning_content"), thinking),
                                          ("text", delta.get("content"), text)):
                    if piece:
                        into.append(piece)
                        if on_delta:
                            on_delta(kind, piece)
                for part in delta.get("tool_calls") or []:
                    call = calls.setdefault(part.get("index", len(calls)),
                                            {"type": "function", "function": {"name": "", "arguments": ""}})
                    if part.get("id"):
                        call["id"] = part["id"]
                    fn = part.get("function") or {}
                    call["function"]["name"] += fn.get("name") or ""
                    call["function"]["arguments"] += fn.get("arguments") or ""
        except urllib.error.HTTPError as e:
            said = e.read().decode("utf-8", "replace")
            sentence = "%s refused the question (HTTP %d): %s" % (self.label, e.code, said[:300])
            full = _full(said)
            if full:
                raise TooLong(sentence, *full)
            raise MindError(sentence)
        except (OSError, http.client.HTTPException, ValueError) as e:
            raise MindError(self.unreachable(e))
        numbers = engine(tail, time.time() - t0)
        numbers["usd"] = self.cost(numbers)
        if finish == "length" and numbers["written"] is not None and (not max_tokens or
                                                                      numbers["written"] < max_tokens):
            asked = (numbers["read_new"] or 0) + (numbers["reused"] or 0) + numbers["written"]
            raise TooLong("%s's memory filled while it wrote its answer (%d tokens), so the answer was cut off."
                          % (self.label, asked), asked, asked)
        words, calls = "".join(text), [calls[i] for i in sorted(calls)]
        if not words.strip() and not calls:
            if finish == "length":
                raise MindError("%s thought for %s tokens without reaching an answer, so the studio cut it off. "
                                "Ask again, or say it more simply." % (self.label, max_tokens))
            raise MindError("%s gave an empty answer (it stopped because: %s)." % (self.label, finish))
        back = {"role": "assistant", "content": words}
        if calls:
            back["tool_calls"] = calls
        return {"text": words.strip(), "thinking": "".join(thinking).strip(), "tool_calls": calls, "message": back,
                "engine": numbers}


class Shared(Mind):
    """A model on a Modal shared endpoint. Modal's proxy token is a key and a secret joined by a dot; the model must be
    one the workspace has made an endpoint for, or every question is refused as an unknown model; and each
    conversation is kept on one machine (Modal-Routing-Affinity-Key), so the part it has read before is reused, which
    Modal bills at an eighth of the price."""

    def secret(self):
        parts = [os.environ.get(name, "").strip() for name in self.key_env]
        return ".".join(parts) if all(parts) else ""

    def missing(self):
        return ("Set %s to the key and the secret of your Modal proxy token, or name a file that holds them in "
                "LCS_KEYS, then start the studio again." % " and ".join(self.key_env))

    def headers(self, messages=()):
        headers = Mind.headers(self)
        if messages:
            first = json.dumps(list(messages)[:2], sort_keys=True, ensure_ascii=False).encode("utf-8")
            headers["Modal-Routing-Affinity-Key"] = hashlib.sha256(first).hexdigest()[:32]
        return headers

    def status(self):
        out = {"key": self.key, "label": self.label, "where": self.where, "url": self.url, "local": False}
        if not self.secret():
            return dict(out, ok=False, why=self.missing())
        try:
            got = self._call("/models")
        except urllib.error.HTTPError as e:
            return dict(out, ok=False, why="Modal refused the proxy token (HTTP %d). Make a new one in Modal's "
                                           "settings and put it in your key file." % e.code)
        except (OSError, ValueError) as e:
            return dict(out, ok=False, why=self.unreachable(e))
        if self.model_id not in [m.get("id", "") for m in got.get("data") or []]:
            return dict(out, ok=False, why="Your Modal workspace has no shared endpoint for %s yet. Make one in "
                                           "Modal's dashboard, under Endpoints." % self.model_id)
        return dict(out, ok=True, name=self.model_id)

    def unreachable(self, e):
        return "Modal did not answer at %s (%s)." % (self.url, _reason(e))


def _closed(schema):
    """The schema as a hosted API's strict mode takes it: every object closed to properties it does not name, and
    every property it names required. Meta's API refuses any other (10-28: "'additionalProperties' is required to be
    supplied and to be false"), and every question with a schema had failed there."""
    if isinstance(schema, list):
        return [_closed(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: _closed(v) for k, v in schema.items()}
    if out.get("type") == "object" and isinstance(schema.get("properties"), dict):
        out["additionalProperties"] = False
        out["required"] = list(schema["properties"])
    return out


def _full(said):
    """(tokens asked, tokens that fit) from llama-server's refusal of a question longer than its memory, else None."""
    try:
        got = json.loads(said)
    except ValueError:
        return None
    error = got.get("error") if isinstance(got, dict) else None
    if not isinstance(error, dict) or error.get("type") != "exceed_context_size_error":
        return None
    try:
        return int(error["n_prompt_tokens"]), int(error["n_ctx"])
    except (KeyError, TypeError, ValueError):
        return None


def engine(got, seconds):
    """The engine numbers of one answer: llama-server's timings, or the usage block of a hosted API, whose prompt
    tokens include those it served from its cache (billed at the cached price) and whose written tokens include its
    thinking (billed as written)."""
    t, usage = got.get("timings") or {}, got.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    prompt = usage.get("prompt_tokens")
    return {"read_new": t.get("prompt_n", None if prompt is None else prompt - (cached or 0)),
            "reused": t.get("cache_n", cached),
            "read_per_s": round(t["prompt_per_second"]) if t.get("prompt_per_second") else None,
            "written": t.get("predicted_n", usage.get("completion_tokens")),
            "write_per_s": round(t["predicted_per_second"], 1) if t.get("predicted_per_second") else None,
            "seconds": round(seconds, 1)}


def _reason(e):
    return str(getattr(e, "reason", None) or e)


def listens(url):
    """False at once when nothing listens at a server on this computer (Windows takes 2 s to say that a connection
    to this computer was refused); True for a server elsewhere, which only asking can tell."""
    u = urllib.parse.urlsplit(url)
    if u.hostname not in LOOPBACK:
        return True
    try:
        socket.create_connection((u.hostname, u.port or 80), timeout=0.3).close()
        return True
    except OSError:
        return False


def minds():
    """The minds as the environment sets them now: made again once the key files are read, so a setting kept in a key
    file counts as much as one set before the studio starts."""
    return {
        "local": Mind("local", "Muse Glimmer", os.environ.get("LCS_MIND_URL", "http://127.0.0.1:8863/v1"),
                      model=os.environ.get("LCS_MIND_MODEL", "muse-glimmer"), reasoning_line="Reasoning strength: low",
                      where="on this computer"),
        "spark": Mind("spark", "Muse Spark", os.environ.get("LCS_SPARK_URL", "https://api.meta.ai/v1"),
                      model=os.environ.get("LCS_SPARK_MODEL", "muse-spark-1.3"), key_env="META_API_KEY",
                      where="on Meta's servers, with your own key"),
        "wonder": Mind("wonder", "Muse Wonder", os.environ.get("LCS_SPARK_URL", "https://api.meta.ai/v1"),
                       model=os.environ.get("LCS_WONDER_MODEL", "muse-wonder-1.0"), key_env="META_API_KEY",
                       where="on Meta's servers, with your own key"),
        "modal": Shared("modal", "Muse Spark Open",
                        os.environ.get("LCS_MODAL_URL", "https://inference.us-west.modal.direct/v1"),
                        model=os.environ.get("LCS_MODAL_MODEL", "meta-models/Muse-Spark-Open-816B-A42B"),
                        key_env=("MODAL_KEY", "MODAL_SECRET"), where="on Modal's servers, with your own token"),
    }


MINDS = minds()
DEFAULT = "local"


def get(key):
    return MINDS.get(key) or MINDS[DEFAULT]
