# Local Chat Studio

People who cannot edit video can still edit video. You drop in a long recording, and Meta's Muse model edits it as
an **agent**: it makes its own plan, runs it step by step, gives a reason for every choice, checks its own clip,
fixes what fails, and hands over a vertical clip ready to post. To change anything, you say what you want.

- **Live sellers first.** A two-hour selling live becomes one clip per product, and every price or number on screen
  is one the seller actually said.
- **Streamers and creators next.** It is the same engine, with a different recipe.
- **Local first.** The agent runs on Muse Glimmer, Meta's open-weight model, on your own graphics card, through
  llama.cpp. The recordings never leave the computer.

## Quick start

Python 3.11 or newer. The studio itself uses the standard library only.

```
python app.py
```

Then open http://127.0.0.1:8899/ . Set `LCS_PORT` to use another port.

## The mind: two ways to run Muse

The page has a switch for the two. Every answer shows which one answered, with its engine numbers: tokens read
new, tokens reused from the question before, reading and writing speed, and seconds.

**1. Muse Glimmer on your own graphics card (the default).** This is Meta's open-weight model, run by llama.cpp's
own server. This is the exact setup the demo runs on:

| | |
|---|---|
| Model file | `Muse-Glimmer-30B-UD-Q3_K_XL.gguf` from [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) |
| Size | 13,360,983,072 bytes |
| SHA-256 | `820d18e00c59376331eba62dd716a6bcdcaa305c2316f5da3a2fab36c7c37e15` |
| Server | llama.cpp `llama-server`, build b11716 |
| Graphics card | NVIDIA RTX 5070 Ti, 16 GB |

```
llama-server -m Muse-Glimmer-30B-UD-Q3_K_XL.gguf -c 32768 -np 1 --jinja -fit on --host 127.0.0.1 --port 8863
```

The studio looks for it at `http://127.0.0.1:8863/v1`. Set `LCS_MIND_URL` to point it at any other server that runs
an eligible Muse model.

**2. Muse Spark on Meta's API.** This route needs no graphics card. Set `META_API_KEY` to your own key before
starting the studio. `LCS_SPARK_MODEL` picks the model (default `muse-spark-1.3`). The key is read from the
environment and is never written anywhere.

## How the agent works

Each message starts a job. Muse makes a plan, then calls one tool at a time, and the page shows every step as it
happens: its number, what it did, why (one line in your language), the input, the result, and the engine numbers.

- **The loop keeps the rules, not Muse.** Wrong input is refused with a sentence that says what to send instead. A
  call that already got the same result twice is refused until something changes, so a fixed retry always runs.
  Plan steps left open get one reminder. A job stops after 30 steps.
- **The conversation only grows at its end.** The system message and the tool list never change, so llama-server
  reuses everything it has already read and reads only the newest messages.
- **You can steer it while it works.** Muse's thinking shows on the page as it is written. Type at any time: the
  question in flight is cancelled and your words go in next, so Muse answers them before anything else. Stop ends
  the job at once. Both close the connection, and llama-server drops the work.
- **Chats are kept.** New chat starts an empty one; Earlier chats opens an old one again, exactly as it was.
- **Every step is written down.** `work/chats/<chat>.events.jsonl` is the job log the page is drawn from, and
  `<chat>.messages.jsonl` is exactly what Muse read.
- **Recordings are only looked up, never copied.** `LCS_VIDEOS` lists the folders they are kept in (separated by `;`
  on Windows; the default is `~/Videos`).

The loop's rules are tested against a scripted mind, and the streamed answers against a small fake server:

```
python -m unittest discover -s tests
```

## Built during the hackathon (Oct 26 - Nov 4, 2026)

Every line in this repository was written during the Meta AI Developer Hackathon. The first commit is dated
Oct 26, 2026, after kickoff, and the commit history shows how it grew, one working step at a time.

Built with help from AI coding assistants.

## Prototyped privately before the event

The idea was tried on a private prototype before the event. Its lessons are listed under Measured below, with their
dates. No code from it is in this repository.

## Open source and models used

Each piece is listed with its licence. The Status column says whether this build uses it yet.

| Piece | What it does here | Licence | Status |
|---|---|---|---|
| [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) (Meta) | the agent's mind, on your own graphics card | Apache-2.0 | used |
| Muse Spark (Meta API) | the same mind without a graphics card, with your own key | Meta's API terms | used |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (llama-server) | runs Glimmer locally | MIT | used |
| [FFmpeg](https://ffmpeg.org/) | cuts, frames and renders the clip | LGPL-2.1+ (GPL for some builds) | planned |
| [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) (Meta) | writes down what is said, in many languages | Apache-2.0 | planned |
| [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (OpenCV Zoo) | finds faces, so text never covers one | MIT | planned |
| [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (OpenCV Zoo) | tells faces apart, so other people's faces can be blurred | Apache-2.0 | planned |
| [MMS-TTS Thai](https://huggingface.co/facebook/mms-tts-tha) (Meta) | an optional voice-over, off by default | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), **non-commercial** | planned |
| Kanit, Prompt (Google Fonts) | Thai and Latin text on the clip | SIL Open Font License 1.1 | planned |

Model weights are downloaded when they are first needed. None of them are committed here.

## Measured

| Date | What | Result |
|---|---|---|
| 2026-10-26, before the event, private prototype | Glimmer UD-Q3_K_XL against Q4_K_M on one RTX 5070 Ti (16 GB), same settings | writing 54.5 vs 20.0 tokens/s, reading 1,677 vs 1,031 tokens/s, Thai captions no worse |
| 2026-10-26, before the event, private prototype | Omnilingual ASR (CTC 1B) on Thai FLEURS, 60 sentences, with a fix for two lost Thai vowels | 7.9% character errors |

| 2026-10-27, this build | The agent loop on Glimmer Q3: how much of each question is read again | step 2 of a job read 232 tokens new and reused 1,084; the first step of the next message read 119 new and reused 2,782 |
| 2026-10-27, this build | A doubled Thai mark ("ค้้น") read back in the next question | it grew to 2, 3, 4, then 8 marks, then 15,000 tokens of thinking with no answer; now the program tidies it before it is read back, and each step stops at 4,096 tokens |
| 2026-10-27, this build | Steering a running job on Glimmer Q3 | a typed message cut the question in flight 0.05 s after it was sent, and the next step used it; Stop ended the job 0.08 s after it was pressed |

New measurements are added here as the build is tested, each with its date.

## Footage

The recordings in the demo are the author's own: a product review shot on an iPhone, a game stream, and clips
from Ray-Ban Meta glasses. All were recorded before the event. Every edit the demo shows is made by this build.

## Licence

MIT. See [LICENSE](LICENSE). Models, fonts and tools keep their own licences, listed above.
