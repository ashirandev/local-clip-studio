"""The mind: one way to ask Muse a question, whichever Muse it is.

Two minds, one request format (the OpenAI chat API):
- local: Muse Glimmer, Meta's open-weight model, on this computer through llama.cpp's own server (llama-server).
  The address is one setting, LCS_MIND_URL, so any server that runs an eligible Muse model works.
- spark: Muse Spark on Meta's hosted API, with the person's own key in META_API_KEY. The key is read from the
  environment when a question is asked and is never written anywhere.

Every answer comes back with its engine numbers: how much of the question was read new, how much was reused from
the question before, and how fast it read and wrote (llama-server's own timings; the hosted API gives token counts).
"""
import json
import os
import time
import urllib.error
import urllib.request

TIMEOUT = 900
CHAT_TEMPERATURE = 0.7


class MindError(Exception):
    """A sentence for the page: what went wrong and what to do about it."""


class Mind:
    def __init__(self, key, label, url, model=None, key_env=None, reasoning_line=None, where=""):
        self.key, self.label, self.url, self.model_id = key, label, url.rstrip("/"), model
        self.key_env, self.reasoning_line, self.where = key_env, reasoning_line, where

    def secret(self):
        return os.environ.get(self.key_env, "").strip() if self.key_env else ""

    def _call(self, path, body=None, timeout=10):
        headers = {"Content-Type": "application/json"}
        if self.key_env:
            headers["Authorization"] = "Bearer " + self.secret()
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.url + path, data, headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def status(self):
        """{ok, key, label, where, name} or {ok: False, ..., why} with a sentence. Never sends a question."""
        out = {"key": self.key, "label": self.label, "where": self.where}
        if self.key_env and not self.secret():
            return dict(out, ok=False, why="Set %s to your own Meta API key, then start the studio again."
                                           % self.key_env)
        if self.model_id:
            return dict(out, ok=True, name=self.model_id)
        try:
            got = self._call("/models")
        except (OSError, ValueError) as e:
            return dict(out, ok=False, why=self.unreachable(e))
        ids = [m.get("id", "") for m in got.get("data") or []]
        if not ids:
            return dict(out, ok=False, why="The server at %s has no model loaded." % self.url)
        return dict(out, ok=True, name=os.path.basename(ids[0]))

    def unreachable(self, e):
        if self.key_env:
            return "Meta's API did not answer at %s (%s)." % (self.url, _reason(e))
        return ("Muse is not answering at %s (%s). Start llama-server with Muse Glimmer, as the README shows."
                % (self.url, _reason(e)))

    def ask(self, messages, tools=None, temperature=CHAT_TEMPERATURE, max_tokens=None):
        """Asks once. Returns {text, thinking, tool_calls, message, engine}; raises MindError with a sentence.

        message is the answer as it goes back into the conversation: the words and the tool calls, without the
        thinking. llama-server reuses everything before it; from this message on it reads again (a few hundred
        tokens), because the thinking it wrote is not sent back.
        """
        if self.key_env and not self.secret():
            raise MindError("Set %s to your own Meta API key to use %s." % (self.key_env, self.label))
        msgs = list(messages)
        if self.reasoning_line:
            if msgs and msgs[0].get("role") == "system":
                msgs[0] = dict(msgs[0], content=self.reasoning_line + "\n\n" + msgs[0]["content"])
            else:
                msgs.insert(0, {"role": "system", "content": self.reasoning_line})
        body = {"messages": msgs, "temperature": temperature}
        if self.model_id:
            body["model"] = self.model_id
        if tools:
            body["tools"] = tools
        if max_tokens:
            body["max_tokens"] = max_tokens
        t0 = time.time()
        try:
            got = self._call("/chat/completions", body, timeout=TIMEOUT)
        except urllib.error.HTTPError as e:
            raise MindError("%s refused the question (HTTP %d): %s"
                            % (self.label, e.code, e.read().decode("utf-8", "replace")[:300]))
        except (OSError, ValueError) as e:
            raise MindError(self.unreachable(e))
        seconds = time.time() - t0
        choice = (got.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text, calls = (msg.get("content") or "").strip(), msg.get("tool_calls") or []
        if not text and not calls:
            if choice.get("finish_reason") == "length":
                raise MindError("%s thought for %s tokens without reaching an answer, so the studio cut it off. "
                                "Ask again, or say it more simply." % (self.label, max_tokens))
            raise MindError("%s gave an empty answer (it stopped because: %s)."
                            % (self.label, choice.get("finish_reason")))
        back = {"role": "assistant", "content": msg.get("content") or ""}
        if calls:
            back["tool_calls"] = calls
        return {"text": text, "thinking": (msg.get("reasoning_content") or "").strip(), "tool_calls": calls,
                "message": back, "engine": engine(got, seconds)}


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
    "local": Mind("local", "Muse Glimmer (local)", os.environ.get("LCS_MIND_URL", "http://127.0.0.1:8863/v1"),
                  reasoning_line="Reasoning strength: low", where="on this computer's graphics card"),
    "spark": Mind("spark", "Muse Spark (Meta API)", os.environ.get("LCS_SPARK_URL", "https://api.meta.ai/v1"),
                  model=os.environ.get("LCS_SPARK_MODEL", "muse-spark-1.3"), key_env="META_API_KEY",
                  where="on Meta's servers, with your own key"),
}
DEFAULT = "local"


def get(key):
    return MINDS.get(key) or MINDS[DEFAULT]
