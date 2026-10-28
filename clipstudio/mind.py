"""The mind: one way to ask Muse a question, whichever Muse it is.

Two minds, one request format (the OpenAI chat API):
- local: Muse Glimmer, Meta's open-weight model, on this computer through llama.cpp's own server (llama-server).
  The address is one setting, LCS_MIND_URL, so any server that runs an eligible Muse model works. Each question names
  the model, LCS_MIND_MODEL (muse-glimmer): a router llama-server sends it to that model, and one that serves a
  single model ignores the name.
- spark: Muse Spark on Meta's hosted API, with the person's own key in META_API_KEY. The key is read from the
  environment when a question is asked and is never written anywhere.

Every answer is streamed, so Muse's thinking can be shown while it is written, and a question in flight can be
cancelled at once: Asking.cancel() shuts the connection, and llama-server drops the work ("stop: cancel task" in its
log). Every answer comes back with its engine numbers: how much of the question was read new, how much was reused
from the question before, and how fast it read and wrote (llama-server's own timings; the hosted API gives token
counts).
"""
import http.client
import io
import json
import os
import queue
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

TIMEOUT = 900
LOAD_SECONDS = 120
CHAT_TEMPERATURE = 0.7


def load_keys(named=None):
    """Puts the KEY=VALUE lines of the files LCS_KEYS names (separated by ; on Windows, : elsewhere) into this
    process's environment, so a key can stay in a file of the person's own, outside the studio's folder; a key already
    in the environment wins. Lines may start with "export " and values may be quoted; blank lines and # lines are left
    out. Returns the names it set, never a value; a file that cannot be read is left out."""
    done = []
    for path in [p.strip() for p in (os.environ.get("LCS_KEYS", "") if named is None else named).split(os.pathsep)]:
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

    def headers(self):
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
        """{ok, key, label, where, name} or {ok: False, ..., why} with a sentence. Never sends a question."""
        out = {"key": self.key, "label": self.label, "where": self.where, "url": self.url}
        if self.key_env and not self.secret():
            return dict(out, ok=False, why="Set %s to your own Meta API key, or name a file that holds it in "
                                           "LCS_KEYS, then start the studio again." % self.key_env)
        if self.key_env:
            return dict(out, ok=True, name=self.model_id)
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
        return ("Muse is not answering at %s (%s). Start llama-server with Muse Glimmer, as the README shows."
                % (self.url, _reason(e)))

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
            raise MindError("Set %s to your own Meta API key to use %s." % (self.key_env, self.label))
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
            body["max_tokens"] = max_tokens
        if schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "answer", "strict": True, "schema": schema}}
        asking = asking or Asking()
        text, thinking, calls, finish, tail = [], [], {}, None, {}
        t0 = time.time()
        try:
            for chunk in asking.stream(self.url + "/chat/completions", body, self.headers()):
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
    """The engine numbers of one answer: llama-server's timings, or the usage block of a hosted API."""
    t, usage = got.get("timings") or {}, got.get("usage") or {}
    return {"read_new": t.get("prompt_n", usage.get("prompt_tokens")),
            "reused": t.get("cache_n"),
            "read_per_s": round(t["prompt_per_second"]) if t.get("prompt_per_second") else None,
            "written": t.get("predicted_n", usage.get("completion_tokens")),
            "write_per_s": round(t["predicted_per_second"], 1) if t.get("predicted_per_second") else None,
            "seconds": round(seconds, 1)}


def _reason(e):
    return str(getattr(e, "reason", None) or e)


MINDS = {
    "local": Mind("local", "Muse Glimmer", os.environ.get("LCS_MIND_URL", "http://127.0.0.1:8863/v1"),
                  model=os.environ.get("LCS_MIND_MODEL", "muse-glimmer"), reasoning_line="Reasoning strength: low",
                  where="on this computer"),
    "spark": Mind("spark", "Muse Spark", os.environ.get("LCS_SPARK_URL", "https://api.meta.ai/v1"),
                  model=os.environ.get("LCS_SPARK_MODEL", "muse-spark-1.3"), key_env="META_API_KEY",
                  where="on Meta's servers, with your own key"),
}
DEFAULT = "local"


def get(key):
    return MINDS.get(key) or MINDS[DEFAULT]
