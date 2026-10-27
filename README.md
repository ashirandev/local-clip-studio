# Local Clip Studio

People who cannot edit video can still edit video. You drop in a long recording, and Meta's Muse model edits it as
an **agent**: it makes its own plan, runs it step by step, gives a reason for every choice, checks its own clip,
fixes what fails, and hands over a vertical clip ready to post. To change anything, you say what you want.

- **Live sellers first.** A two-hour selling live becomes one clip per product, and every price or number on screen
  is one the seller actually said.
- **Streamers and creators next.** It is the same engine, with a different recipe.
- **Local first.** The agent runs on Muse Glimmer, Meta's open-weight model, on your own graphics card, through
  llama.cpp. The recordings never leave the computer.

## Quick start

Python 3.11 or newer, and [FFmpeg](https://ffmpeg.org/) (`ffmpeg` and `ffprobe` on the PATH). The studio itself
uses the Python standard library only.

```
python app.py
```

Then open http://127.0.0.1:8899/ . Set `LCS_PORT` to use another port, and `LCS_VIDEOS` to the folders your
recordings are in (separated by `;` on Windows; the default is `~/Videos`).

Drop a recording on the page, or tap one of your newest. It is opened where it is: a browser does not give the
path of a dropped file, so the studio finds it in those folders by its name, size and time. Nothing is copied or
uploaded, and the recording is only ever read.

The screen stays simple: what Muse plans, what it does and why. **Settings** has the switch between the two Muses,
the details of this computer's setup, and **Show how Muse works**, which adds each step's input and result, the files
it opens, its thinking as it is written, and its engine numbers.

**On a phone.** Start the studio with `LCS_HOST=0.0.0.0` and open `http://<this computer's address>:8899/` on a phone
on the same Wi-Fi. The phone picks from the recordings on the computer, and the clip shows as a small preview at the
top right that opens full screen with a tap. Do this only on a network you trust: anyone on it can open the
recordings in your folders. Two guards hold either way: the studio answers only to an address or `localhost`, never
to a web name (so a web page cannot reach it through a name it owns), and it takes only JSON posts (so another web
page cannot send it anything without asking first).

## The mind: two ways to run Muse

Settings has a switch for the two. With Show how Muse works on, every answer shows which one answered, with its
engine numbers: tokens read new, tokens reused from the question before, reading and writing speed, and seconds.

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
happens: its number, what it did and why (one line in your language), and with Show how Muse works on, the input,
the result and the engine numbers.

- **The loop keeps the rules, not Muse.** Wrong input is refused with a sentence that says what to send instead. A
  call that already got the same result twice is refused until something changes, so a fixed retry always runs.
  Plan steps left open get one reminder. A job stops after 30 steps.
- **The conversation only grows at its end.** The system message and the tool list never change, so llama-server
  reuses everything it has already read and reads only the newest messages.
- **You can steer it while it works.** Muse's thinking shows on the page as it is written. Type at any time: the
  question in flight is cancelled and your words go in next, so Muse answers them before anything else. Stop ends
  the job at once. Both close the connection, and llama-server drops the work.
- **Chats are kept.** New chat starts an empty one; Earlier chats opens an old one again, exactly as it was.
- **One recipe per kind of video, and no questions.** Muse picks the kind (`recipes/seller.md` for a product review
  or a selling live, `recipes/game.md` for a game stream), reads that recipe, and shows its guesses: the kind, the
  length, the style and the language. Each guess has a button to change it with one tap, and Muse plans again.
  What you type wins over the recipe and every guess. A kind with no recipe yet is refused, and Muse tells you so
  instead of using a recipe made for something else. A recipe is a Markdown file with front matter, like a
  SKILL.md file, so a new kind of video is a new file.
- **Every step is written down.** `work/chats/<chat>.events.jsonl` is the job log the page is drawn from, and
  `<chat>.messages.jsonl` is exactly what Muse read.
- **Muse looks before it guesses.** When a recording opens, Muse gets its length, picture size and sound tracks
  (from ffprobe) and one picture from it. File names mislead: the demo's product review is called `trip_...mov`,
  and from the name alone Muse called it a travel clip; with the picture it picked the product-review recipe.
- **A map before any look.** On a long recording Muse maps it first. One ffmpeg pass reads how loud every second is
  against the minutes around it, and how much the picture changes at every keyframe; only the keyframes are
  decoded, which is what makes it fast. The recipe says what ranks: sound and picture for a game stream, the
  picture alone for a product review (a person at a table is just as loud the whole time). Each stretch starts well
  before its peak, so what led up to the peak is in it. The map is kept per file, so mapping a recording again is
  instant, and the page shows how far it has got while it reads. Stop ends it at once.
- **Muse looks before it chooses.** Muse picks a stretch to look at, usually one the map lists, and gets 2 to 6
  pictures of it, each as its own picture right after its time; the page shows them in the step. To see what led
  up to a peak it can look earlier, or wider. The same look twice is refused, and its pictures are not sent again.
- **A moment Muse proposes is one it has seen.** Muse proposes up to three moments, best first. Each must sit inside
  one of its looks, and moments may not overlap. The page shows each as a card with Muse's reason and a picture
  from that look, and a rough 480p preview of the stretch is cut in the background, so the card plays a second or
  two later. Muse goes on with the best one by itself; a tap on another card switches. Until a tool has made the
  clip, the state Muse reads says "no clip made yet", so it cannot claim one.
- **Every page shows the chat the server has open.** When another chat opens, every page follows it, so neither an
  address nor a second tab can leave a page working on another chat.

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
| [FFmpeg](https://ffmpeg.org/) | reads each recording (ffprobe) and takes pictures from it; later cuts and renders the clip | LGPL-2.1+ (GPL for some builds) | used |
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
| 2026-10-27, this build | Opening the review clip by dropping it | found where it is in 0.16 s by name, size and time; the folder, size and time of the file unchanged; from the file name alone Muse called it travel, with one picture it picked the product-review recipe |
| 2026-10-27, this build | Mapping a 4-hour game stream (77 GB, 1080p, 60 fps, a keyframe every 4 s) | 1:54 for one pass that reads the sound and only the keyframes; the prototype took 4:48 in two passes, every frame decoded. Mapping it again: 0.13 s |
| 2026-10-27, this build | What that map ranks first | the top 6 stretches are all louder and busier than 97-99% of the stream; in 4 of them the peak is the game's death screen (the streamer shouts at each death), with the fight that led to it in the 21 s before the peak |
| 2026-10-27, this build | What a look costs Glimmer Q3 (vision on the processor) | 4 frames, each its own 540-pixel picture: 971 tokens read in 10.8 s; the same 4 frames in one picture: 293 tokens in 2.9 s |
| 2026-10-27, this build | From opening to proposed moments, Glimmer Q3, one run each | the review clip (3:06): 7 steps, 0 refused, 3 cards at 1:19; the 4-hour stream: 10 steps, 0 refused, cards at 2:04 after looking at the map's top 6, each moment ending before the death screen. A rough 30 s preview: 1.3-2.1 s to cut |
| 2026-10-27, this build | Muse's answer after the cards, before any clip existed | it said the clip was made, with captions and a blurred chat. Now the state Muse reads says "no clip made yet": next run, it said only that the moments were ready to preview (one run, not yet a measurement) |

New measurements are added here as the build is tested, each with its date.

## Footage

The recordings in the demo are the author's own: a product review shot on an iPhone, a game stream, and clips
from Ray-Ban Meta glasses. All were recorded before the event. Every edit the demo shows is made by this build.

## Licence

MIT. See [LICENSE](LICENSE). Models, fonts and tools keep their own licences, listed above.
