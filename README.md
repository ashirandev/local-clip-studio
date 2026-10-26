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

Python 3.11 or newer. This step uses the standard library only.

```
python app.py
```

Then open http://127.0.0.1:8899/ . Set `LCS_PORT` to use another port.

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
| [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) (Meta) | the agent's mind, on your own graphics card | Apache-2.0 | planned |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (llama-server) | runs Glimmer locally | MIT | planned |
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

New measurements are added here as the build is tested, each with its date.

## Footage

The recordings in the demo are the author's own: a product review shot on an iPhone, a game stream, and clips
from Ray-Ban Meta glasses. All were recorded before the event. Every edit the demo shows is made by this build.

## Licence

MIT. See [LICENSE](LICENSE). Models, fonts and tools keep their own licences, listed above.
