# Local Clip Studio

People who cannot edit video can still edit video. You drop in a long recording, and Meta's Muse model edits it as
an **agent**: it makes its own plan, runs it step by step, gives a reason for every choice, checks its own clip,
fixes what fails, and hands over a vertical clip ready to post. To change anything, you say what you want.

- **Live sellers first.** A two-hour selling live becomes one clip per product, and every price or number on screen
  is one the seller actually said.
- **Streamers and creators next.** It is the same engine, with a different recipe.
- **Local first.** The agent runs on Muse Glimmer, Meta's open-weight model, on your own graphics card, through
  llama.cpp. The recordings never leave the computer.
- **The program checks facts; the person keeps the taste.** Every check is one a program can prove on the finished
  file: a caption is a word that was said, a number on screen was said, no viewer's name or private line can be
  read. Whether a clip is funny or good is the person's call, so the studio gives no score for it.

## Quick start

**On Windows, double-click `setup.bat`.** It asks where to install (a folder window; it makes a `Local Clip Studio`
folder there, about 18 GB with Muse Glimmer), then the endpoint and the key of a Muse API, and Enter skips those.
Everything else it does by itself:
- copies the studio into the folder chosen, and keeps everything there from then on;
- installs its own Python 3.11.9 in `python\` (nothing system-wide) and the studio's packages;
- downloads the parts below that this computer is missing;
- puts **Local Clip Studio** on the desktop and opens the studio.

If anything stops, run `setup.bat` again: it carries on from where it stopped. Every file it downloads is checked
against a size and SHA-256 pinned in the repository first. Measured on 10-29, on a clean copy of the studio with no
FFmpeg on the PATH: Python, pip, the packages and FFmpeg, then the shortcut, and the studio answered on its own Python.
Installed again from the repository folder into another drive (H:), it copied 13 MB of the studio there, set up 369 MB
in all, and the shortcut and the studio ran from H:. To remove it, delete that folder, the shortcut, and
`%APPDATA%\LocalClipStudio` (the saved endpoint and key).

**Elsewhere, or by hand:** with Python 3.11 or newer,

```
pip install -r requirements.txt
python app.py
```

then open http://127.0.0.1:8899/ and press **Set up Muse** (it is also in Options). The studio shows what this
computer already has and what it still needs, with each size, and downloads the rest with one button
(`python -m clipstudio.parts` does the same in a console):

| Part | Size | From |
|---|---|---|
| FFmpeg 8.1.1, essentials build (Windows) | 109 MB | [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), on [GitHub](https://github.com/GyanD/codexffmpeg/releases/tag/8.1.1) |
| llama.cpp `llama-server`, build b11716 (Windows) | 579 MB (CUDA 13.4) or 659 MB (CUDA 12.4) | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/releases/tag/b11716) |
| Muse Glimmer 30B, UD-Q3_K_XL, with its vision part | 14.8 GB | [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF), the vision part from Meta's [meta-models/Muse-Glimmer-30B-GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF) |
| Hearing: Omnilingual ASR CTC 1B v2 as int8 ONNX | 1.0 GB | this repository's release [hearing-1](https://github.com/ashirandev/local-clip-studio/releases/tag/hearing-1), made from Meta's checkpoint by `hearing/export_onnx.py` (Apache-2.0, with its NOTICE); `LCS_HEARING_FROM` names another address |

Each file is used only after its size and SHA-256 match the ones pinned in `clipstudio/parts.py`; a file that does
not match is deleted, and a download that is cut off carries on from where it stopped. A part that cannot be set up
is left out with the reason, and the parts after it are still set up. Everything goes in `work/`
(`work/tools` and `work/models`), and a part already on the computer (on the PATH, or a server already running) is
used as it is. The graphics driver picks the llama.cpp build: CUDA 13.4 when the driver offers CUDA 13 (an RTX 50 card
needs it), else CUDA 12.4. Once Glimmer and llama-server are there, the studio starts llama-server itself, as
[the mind](#the-mind-three-ways-to-run-muse) shows, and stops it when the studio stops. Measured here on 10-29:
FFmpeg in 5 s and llama.cpp in 23 s from GitHub, `llama-server --version` ran from the unpacked folder, and the
hearing (from a server on this computer) started on its own once its download was checked.

Glimmer needs an NVIDIA card with 16 GB. Without one, setup leaves Glimmer and llama.cpp out and says why; Muse
Spark (below) needs no graphics card. On Linux and macOS, install FFmpeg (with libass) and llama.cpp's
`llama-server` from your packages; setup downloads Glimmer and the hearing model. The studio's own Python packages
are OpenCV to find faces, pythainlp to break Thai words onto lines, NumPy, onnxruntime with RapidOCR for the
hearing and for reading the screen, and segno for the QR code a phone opens the studio with.

The face finder's model (YuNet, 230 KB) is downloaded from OpenCV Zoo the first time it is needed; set
`LCS_FACE_MODEL` to the file if you have it. The house font (Google Sans), Kanit, Pattaya and Taviraj are in `fonts/`.

Then open http://127.0.0.1:8899/ . Set `LCS_PORT` to use another port. Muse finds your recordings by itself: with
nothing set, it looks on every drive (only names and sizes, leaving out system and program folders; 1.7 s for five
drives here) and keeps that list 30 s. Set `LCS_VIDEOS` to the folders your recordings are in (separated by `;` on
Windows) to keep it to those.

**Hearing what is said** uses Meta's Omnilingual ASR (CTC 1B v2), run by onnxruntime inside the studio, on the
processor, so the graphics card stays with Muse. It needs no Docker. Set up puts its model in `work/models/hearing`
(or the folder `LCS_HEARING_MODELS` names): `model.int8.onnx` (1.0 GB) and `letters.json`; `model.onnx` with
`model.onnx.data` (3.9 GB) works too and is used when the int8 file is not there. The studio starts the hearing by
itself, before Muse is asked anything; it takes about 1 GB of memory. Without the model the studio still works;
Muse just cannot hear, and says so.

The model file is made once from Meta's checkpoint by `hearing/export_onnx.py`, which runs in Docker because
Omnilingual's own loader (fairseq2) runs on Linux only; only whoever makes the file needs that (see the file).

Drop a recording on the page, or tap one of your newest. It is opened where it is: a browser does not give the
path of a dropped file, so the studio finds it in those folders by its name, size and time. Nothing is copied or
uploaded, and the recording is only ever read.

**From your phone**, over the same Wi-Fi or the same [Tailscale](https://tailscale.com/): the studio listens on the
computer's Wi-Fi address and its Tailscale address too (only those), and Options shows each address with a QR code
for the phone's camera. Windows asks once whether Python may use the network. On the phone, tap **+** beside the
message box, or **Send from this phone**, and pick photos and videos: they go into the `From phone` folder next to the
studio (a folder for each time you send), one video opens at once, and several open together for Muse to make one
clip from. Muse makes clips from videos; photos are kept there. Several people can use the studio at once, so a
group can try it from their own phones; there is nothing to pair, and anyone on that Wi-Fi or Tailscale can use it,
your API key included (Tarm, 10-29: it is open source, and who shares their Wi-Fi is their call). Phone: Off in
Options keeps it to this computer, and only this computer can set the key or turn Phone off. Measured on 10-29
through the Wi-Fi address: a 4 s video arrived with the same bytes and was first in the newest recordings, and two
photos sent from the page arrived whole. Not yet measured from a real phone.

The screen stays simple, with pictures and icons more than words. The first screen is your clips: each chat shows
as a picture of the clip it made, with which Muse made it (a computer for this one, a cloud for an API) and what it
cost, and the month's total sits at the top. In a chat, each message's work is one card: the plan as a bar, each step
as an icon with its reason, the moments Muse proposes as short videos, and what Muse is thinking while it writes;
each step opens to show how Muse decided, and a finished card can play its steps again. The editor shows the clip,
a timeline of its words, and icon buttons (Trim, Captions, Style, Colour, Sound, Words, Channel), each opening
pictures to pick from. **Options** has the switch between the Muses, **Simple or Detailed** steps (Detailed adds each step's
input and result, the files it opens and its engine numbers), light or dark, the money costs are shown in, and this
computer's setup.

**On a phone.** Start the studio with `LCS_HOST=0.0.0.0` and open `http://<this computer's address>:8899/` on a phone
on the same Wi-Fi. The phone picks from the recordings on the computer; the clips, the chat and the editor are
three screens there (on a wide screen the chat and the editor sit side by side), and the editor plays the clip full
screen at a tap. Do this only on a network you trust: anyone on it can open the
recordings in your folders. Two guards hold either way: the studio answers only to an address or `localhost`, never
to a web name (so a web page cannot reach it through a name it owns), and it takes only JSON posts (so another web
page cannot send it anything without asking first).

## The mind: three ways to run Muse

The chip under the title shows which Muse will answer and where it runs: local (on this computer) or API (on
someone's servers, paid per token). Tap it to switch. With Detailed steps, every answer shows which one
answered, with its engine numbers: tokens read new, tokens reused from the question before, reading and writing speed, and seconds.

**1. Muse Glimmer on your own graphics card (the default).** This is Meta's open-weight model, run by llama.cpp's
own server. This is the exact setup the demo runs on:

| | |
|---|---|
| Model file | `Muse-Glimmer-30B-UD-Q3_K_XL.gguf` from [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) |
| Size | 13,360,983,072 bytes |
| SHA-256 | `820d18e00c59376331eba62dd716a6bcdcaa305c2316f5da3a2fab36c7c37e15` |
| Vision part | `mmproj-Muse-Glimmer-30B-Q4_K_M.gguf` from Meta's own [meta-models/Muse-Glimmer-30B-GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF) (unsloth ships the same bytes as `mmproj-kquant.gguf`), 1,400,328,928 bytes, SHA-256 `f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6`: the studio shows Muse pictures, so Glimmer needs it |
| Server | llama.cpp `llama-server`, build b11716 |
| Graphics card | NVIDIA RTX 5070 Ti, 16 GB |
| Context | 49,152 tokens: 13.6 GB of the card in use, the same speed as at 32,768 (measured 10-28) |

Set up does all of this for you: it downloads both files, writes this preset to `work/tools/glimmer.ini` and starts
the server. To run it yourself, start it in router mode, from a preset file, so the studio can ask it to step off the
graphics card while SAM cuts a person out, and load it again after. `glimmer.ini` (use full paths to your files):

```ini
version = 1

[muse-glimmer]
model = Muse-Glimmer-30B-UD-Q3_K_XL.gguf
mmproj = mmproj-Muse-Glimmer-30B-Q4_K_M.gguf
mmproj-offload = true
c = 49152
parallel = 1
jinja = true
fit = on
load-on-startup = true
```

```
llama-server --models-preset glimmer.ini --models-max 1 --no-models-autoload --host 127.0.0.1 --port 8863
```

`--no-models-autoload` means only the studio loads Glimmer: a question from another program cannot bring it back
while SAM has the card. (Measured 10-28: brought back 5 s into a SAM run, it sat beside SAM, partly off the card, and
wrote 5.2 tokens a second instead of 47.5.) When a question finds Glimmer unloaded, the studio loads it first.

The studio looks for it at `http://127.0.0.1:8863/v1` and names the model `muse-glimmer` in every question
(`LCS_MIND_MODEL`). Set `LCS_MIND_URL` to point it at any other server that runs an eligible Muse model; a server
that runs one model without a preset works too, but then nothing that needs SAM can run beside it.

**2. Muse Spark on Meta's API.** This route needs no graphics card. Set `META_API_KEY` to your own key before
starting the studio, or type the endpoint (`https://api.meta.ai/v1`) and the key in Options > API.
`LCS_SPARK_MODEL` picks the model (default `muse-spark-1.3`). **Muse Wonder** (`muse-wonder-1.0`, Meta's model for
long work with many tools) is a fourth choice on the same address and key, at $4 a million tokens read and $20
written; `LCS_WONDER_MODEL` changes it. Measured 10-29: one short question in 7 s, $0.006.

**3. Muse Spark Open on a Modal shared endpoint.** The open-weight Spark (`meta-models/Muse-Spark-Open-816B-A42B`) on
Modal's machines, paid per token, with no graphics card of your own. Make a shared endpoint for it in your Modal
workspace (Endpoints), make a proxy token, and set `MODAL_KEY` and `MODAL_SECRET` to its two parts (or type the
endpoint and the token, its two parts joined by a dot, in Options > API). `LCS_MODAL_MODEL`
and `LCS_MODAL_URL` change the model and the address (default `https://inference.us-west.modal.direct/v1`). Each
conversation stays on one Modal machine, so what it read before is billed at the cached price. Options says so when
the workspace has no endpoint for the model yet.

**The endpoint and key typed in Options > API** (or asked by `setup.bat`) are kept in your own user folder, never in
the studio's: `%APPDATA%\LocalClipStudio\keys.env` on Windows, `~/.config/LocalClipStudio/keys.env` elsewhere. The
address decides which Muse it is for: a `*.modal.direct` address is Muse Spark Open on Modal, any other is Muse Spark.
The Muse they reach is ready at once, without a restart. Only the page on the computer the studio runs on can set them
or see the saved address, and the key is never shown again: from a phone on the same network, someone could otherwise
point Muse at a server of theirs and be sent the recording's pictures. An address on another computer must use https,
so the key is never sent in the clear.

Keys can also stay in a file of your own: name it in `LCS_KEYS` (`KEY=VALUE` lines; several files separated by `;` on
Windows, `:` elsewhere). The studio prints the names it read, never a value.

**What it costs.** Each chat on the clips screen, each work card and each finished clip shows what Muse cost to make
it ($0 on your own card, and a dash for a chat made before costs were written down, never a $0 that is not known);
tap a cost for the chat, today and this month, at the providers' list prices in US dollars per million tokens (Muse
Spark 1.3: $1.25 read new, $0.15 read from the provider's cache, $4.25 written, thinking included; Muse Spark Open on
Modal the same). Glimmer on your own card costs nothing. Every answer's line in Detailed steps carries its cost too.
Amounts are shown in the money of the browser's language (US dollars for English, baht for Thai; Options can switch
it to US dollars), at the day's rate
from open.er-api.com, fetched at most twice a day and carrying nothing of yours. A question cut short by Stop is not
counted: a provider says what an answer cost only at its end.

**SAM 3.1, for the cut-out looks (optional).** SAM finds the person frame by frame, so words can sit behind them. It
runs in its own Python as a worker, and Glimmer steps off the graphics card while it works, then comes back. A
6-second stretch took 54 s on the RTX 5070 Ti, and Glimmer came back at 47.8 tokens a second (measured 10-28).

| | |
|---|---|
| Weights | `sam3.1_multiplex_fp16.safetensors`: Meta's SAM 3.1 multiplex checkpoint in fp16, as repackaged by Comfy-Org ([Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors), not gated). Meta's original is `facebook/sam3.1` (`sam3.1_multiplex.pt`, gated) |
| Size | 1,745,546,848 bytes |
| SHA-256 | `9ba99c92703c2e8b4f47de2d34a539bb8e18923049e238b780d70dbe6368eb03`, the same as Hugging Face's own record (checked 10-28) |
| Code | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) at `f1647e8d`, in its own Python 3.11 with PyTorch 2.14.1 and CUDA 13.0; the install lines are at the top of `cutting/cut.py` |
| Licence | SAM License (Meta, last updated 19 Nov 2025) |

Set `LCS_SAM_PYTHON` to that Python's `python.exe` and `LCS_SAM_WEIGHTS` to the file. Without them, everything
else works the same.

## How the agent works

Each message that asks for work starts a job. Muse makes a plan, then calls one tool at a time, and the page shows
every step as it happens: an icon, what it did and why (one line in your language) and how Muse decided, and with
Detailed steps the input, the result and the engine numbers.

- **The loop keeps the rules, not Muse.** Wrong input is refused with a sentence that says what to send instead. A
  call that already got the same result twice is refused until something changes, so a fixed retry always runs.
  Plan steps left open get one reminder. A job stops after 30 steps.
- **The conversation only grows at its end.** The system message and the tool list never change, so llama-server
  reuses everything it has already read and reads only the newest messages.
- **You can steer it while it works.** Muse's thinking shows on the page as it is written. Type at any time: the
  question in flight is cancelled and your words go in next, so Muse answers them before anything else. Stop ends
  the job at once. Both close the connection, and llama-server drops the work.
- **An editor you can talk to.** A greeting, a thanks or a word about your day gets an answer, not a job. If Muse's
  first answer to something you typed calls a tool anyway, one short question apart asks whether you asked for
  work; if you did not, Muse is asked again with no tools at all, so nothing runs.
- **Chats are kept.** + starts an empty one; the clips screen opens an old one again, exactly as it was. After
  the first job that does work, Muse names the chat in a few words of your language ("รีวิวไมค์ Yeti Orb"), and
  the clips screen shows each chat as the clips it made, which Muse answered in it, local or API, and its cost.
- **An editor can finish it in Premiere Pro.** Premiere, under Save on a finished clip, saves a zip: the clip as a
  sequence in Final Cut Pro 7 XML (File > Import in Premiere), its captions as .srt and a note. Every piece on the
  timeline points at your own file, named by it, with its in and out points there, so an editor drags an edge to take
  more or less of it; a montage points back at the clips it was joined from, not at the joined file. The frame comes
  over as scale and position (a crop where a part does not fill the frame), the sound left and right, the music,
  and the titles and each piece's reason as markers. Looks, stickers and blurs do not; the note lists what the clip
  had and did not come over, a blur first, since a hidden face or chat shows again there.
- **One recipe per kind of video, and no questions.** Muse picks the kind (`recipes/seller.md` for a product review
  or a selling live, `recipes/game.md` for a game stream, `recipes/vlog.md` for a vlog or an edited episode, kept
  whole as a card so its own subtitles and logo are never cut), reads that recipe, and shows its guesses: the kind, the
  length, the style and the language. Each guess has a button to change it with one tap, and Muse plans again.
  What you type wins over the recipe and every guess. A kind with no recipe yet is refused, and Muse tells you so
  instead of using a recipe made for something else. A recipe is a Markdown file with front matter, like a
  SKILL.md file, so a new kind of video is a new file.
- **Every step is written down.** `work/chats/<chat>.events.jsonl` is the job log the page is drawn from, and
  `<chat>.messages.jsonl` is exactly what Muse read.
- **Muse looks before it guesses.** When a recording opens, Muse gets its length, picture size and sound tracks
  (from ffprobe) and one picture from it; a recording of 3 minutes or more comes as four moments across it, two by two,
  each marked with its time (about 570 tokens for an hour). File names mislead: the demo's product review is called
  `trip_...mov`, and from the name alone Muse called it a travel clip; with the picture it picked the product-review
  recipe. One picture misleads too: an hour-long travel episode opened on two people in a room, and Muse took it for a
  podcast (10-29).
- **A map before any look.** On a long recording Muse maps it first. One ffmpeg pass reads how loud every second is
  against the minutes around it, and how much the picture changes at every keyframe; only the keyframes are
  decoded, which is what makes it fast. The recipe says what ranks: sound and picture for a game stream, the
  picture alone for a product review (a person at a table is just as loud the whole time). Each stretch starts well
  before its peak, so what led up to the peak is in it. The map is kept per file, so mapping a recording again is
  instant, and the page shows how far it has got while it reads. Stop ends it at once.
- **Muse looks before it chooses.** Muse picks a stretch to look at, usually one the map lists, and gets 2 to 6
  pictures of it, each as its own picture right after its time; the page shows them in the step. To see what led
  up to a peak it can look earlier, or wider. The same look twice is refused, and its pictures are not sent again.
- **Muse hears what is said in each look.** Omnilingual ASR writes the stretch down, every phrase with its own time
  from the model's own frames (it decides one letter every 20 ms). The Thai marks it cannot write are put back, and
  so are number words it began with a digit ("5้าสิบ" becomes ห้าสิบ). Then Muse fixes misheard words from the
  context, in a question apart from the job, with every number taken out first as <N1>, <N2>...: a line whose answer
  loses or adds a number, or rewrites more than 40% of it, keeps what was heard. A stretch is heard once and kept.
- **A moment Muse proposes is one it has seen.** Muse proposes up to eight moments, best first. Each must sit inside
  one of its looks, and moments may not overlap. An edge that falls inside a line said is moved just outside it,
  so a clip never opens or closes on half a word; each look is heard 6 s past its edges, so a line the look cuts is
  heard whole and the clip ends where the sentence does. The page shows each as a card with Muse's reason and a picture
  from that look, and a rough 480p preview of the stretch is cut in the background, so the card plays a second or
  two later. Muse goes on with the best one by itself; a tap on another card switches. Until a tool has made the
  clip, the state Muse reads says "no clip made yet", so it cannot claim one.
- **Muse edits the clip with tools, and the program draws it.** From a moment, Muse makes the clip in one of four
  layouts: the game as a card with the streamer's face in a circle above it, the face above the game, the face alone,
  or one part of the picture filling the frame. Muse chooses the layout and the part of the picture to show; the
  program finds the face itself (YuNet, a picture every half second of the moment, the face that stays in one place)
  and works out every pixel. Letting the model place everything was measured first: it cut heads and showed the face
  camera twice.
- **The clip keeps the last take and drops the dead air.** A speaker who stumbles stops and says the line again;
  the program finds a line said again after a stop, letter for letter (a name heard two ways still matches), and cuts
  the first take with the stop. A silence over a second is cut down to 0.3 s a side, and the clip opens and closes
  0.3 s from a word, but only while nothing moves on the screen: the answer streaming in while the speaker waits is
  the payoff, not dead air. A game keeps every second. Muse is told what was cut and why.
- **In a conversation, the clip cuts between the faces.** For two or more people talking, whoever speaks fills the
  frame, with a short cut to the listener's reaction and to everyone when they laugh together. Two more layouts are
  there when you ask for everyone on screen at once: the others in circles in the top corner away from the speaker's
  head, or, for two people, the screen split into a top half and a bottom half, each person's captions under their
  own face (Tarm, 10-29, on a clip with circles: cut between faces instead). A Q&A's comment card is read like a tutorial's screen, and a viewer's handle on
  it is hidden.
- **Who speaks is heard, not guessed from lips.** On a real 720p Q&A, pixel change around two mouths could not tell
  the two speakers apart (every score about 1.1). Their voices could: every line one said sat at 97 to 137 Hz, every
  line the other said at 192 to 314 Hz. So the program measures the pitch of each line heard and parts the lines into
  two voices when they sit far enough apart; then Muse looks at a few frames from inside each voice's longest lines,
  each person's number drawn over their head, and says who is speaking. A voice goes to the person Muse names in
  most of its frames, and two voices must go to two people, or the voices are not used. Two voices too alike, such
  as two men of the same pitch, are left to the mouths.
- **Many short clips become one, in an order Muse can explain.** Point Muse at a folder of phone clips from a trip
  or an event. The program numbers them in the order the camera's clock says they were shot (an app that saves
  clips writes the minute it saved them into the file's own time, so that one is not trusted), hears each one, and
  shows Muse three frames of each in one picture. Muse says which go in, in what order, which seconds of each, a
  line of words over each part of the story, and why, and why each clip it leaves out is left out. The program joins
  the pieces into one upright recording (a clip shot lying down is shown whole over a blurred copy of itself), and
  the clip is made from it like any other. A piece holds at most 8 s and no second where the picture is frozen, the
  whole stays near the length asked for, and the first piece says where we are.
- **Words never cover a face or the product.** Every title is measured as libass will draw it, then refused at a
  time or a place where it would cover a face the program found, or the product Muse named, and the refusal lists
  the times and places that are clear. A blur over the face the clip shows is refused too: the face is the point.
- **Nothing private shows on a tutorial's screen.** In a screen explainer the program reads the screen at full size
  (RapidOCR, on the processor) and hides every line with an email address, a phone number, a card or ID number, a
  key, an account number, a password or someone's @handle, for as long as it shows. Patterns decide, not a model, and the words are
  never passed to Muse, so they cannot reach a title or a post. The last check reads the screen again and fails
  while a private line can be seen.
- **A tutorial's screen is shown where the action is, at a size a phone can read.** A whole 1920-pixel screen on a
  960-pixel card halves every word. In a screen explainer the program measures how tall the screen's lines are (from
  the same reading that hides private lines) and zooms until they are 16 px on a phone, then shows where the screen
  changes, second by second, panning when the action moves. It moves only when the new place keeps changing and stays
  changed, so a tooltip or a notification does not pull it away, and it never shows the face camera. When no one
  part changes (nothing does, or a blinking button, or a scrolling page), the card holds where the most lines are,
  from the first line and the first word. Muse is told how big the words come out on a phone.
- **Captions never cover the picture.** In the card layout the captions sit between the face and the card, where
  they read as what the face says and cover neither; every word stays above y 1250, clear of an ad's button.
- **A review told over several takes is cut in the review order, and Muse only says what each line does.** Asked to
  arrange eight takes of a mic review itself, the local Muse was refused eleven times in a row and the hosted one
  opened on a count-in. So the program asks one small question: what does each line do in a review (shows it working,
  says what it is, its best point, what it works with, what there is to see, the verdict and price, the ask), and
  which line should open the clip. It then builds the cut from whole lines in that order, inside the length asked:
  open on the proof, close on the verdict and the ask. On that mic review both Muses named the line a hand edit
  opened on, and the cut held 7 of its 8 pieces, in its order. Lines the hearing was less sure of are shown in a take
  someone talks through, marked, because the verdict and the price were among them. Over a talked review the hook
  stays on its own piece and the captions go one big line at a time.
- **Several clips in one go, each reported on its own.** When the person asks for every good moment, Muse proposes
  up to eight and gives each its title and post text; the program makes, checks and hands over each clip. Names are
  checked against the screen for every clip before any is made, a clip that fails a check is not handed over, and the
  answer says which clips are ready and what failed in the others, never one "done" for all.
- **Every change is one entry, and undo takes it back.** The clip is a list of entries (the moment and layout, the
  style, titles, captions, blurs, hits). The same list draws the preview and the finished file. After each change
  Muse sees the frame it made, and the page shows the clip with a timeline to any second. The person can undo on the
  page too, and Muse is told before it answers next.
- **The look and the words change at a tap, without waiting for Muse.** Under the clip, icon buttons open its
  captions (Off, Plain, Lit, Pop), the six text styles and the colour looks as pictures; on the timeline, every
  title and caption phrase sits on a line by its time. A tap runs the very tool Muse would call, with its checks, so a choice the tool would refuse is
  shown off with the tool's reason, and the page and Muse never disagree. Undo takes a tap back, and Muse is told
  before it answers next. Caption words the person types are kept as typed, also when the captions go on again, and
  the check on captions passes them as the person's own words. When the clip has changed since its video was made,
  Update video renders it as Muse's render does. Under Captions, As said puts them back in the words as
  said at a tap; English is a tap that asks Muse, which writes the translation.
- **Trim, or take in more, at a tap.** Muse picks the moment; under Trim the person taps a spot on the timeline and
  starts or ends the clip there, or takes in 2 s more before or after it at each tap, then Apply. The clip is made
  again in the same layout, crop and face, and its edges stay where the person put them. The captions, blurs, hits,
  stickers, sounds and slow pushes stay on what is said and shown, so a cut-off start takes its captions with it;
  the titles stay at their time in the clip, so the opening title still opens it. Seconds the moment never heard are
  heard first and captioned. Muse does the same when asked ("start 3 seconds earlier") with `change_ends`, and Undo
  puts the old ends back.
- **The preview changes at once.** The page plays the clip without its words (a base video, made once for each
  picture) and lays the words over it, drawn as see-through pictures by the same libass as the export. A change of
  words shows in about a second (0.55 s measured on the test clip, 10-28); only a change to the picture or the sound
  makes the base again, in the background, and the clip's frame shows until it is ready.
- **Ready to post says how little the person did:** how long Muse worked on the job, the messages the person typed
  and their taps, and Muse's steps.
- **The checks read red to green.** Each check of the finished file says Passed, Failed or Not needed, with a mark
  of its colour; one that failed on the clip's last check and passes now says Fixed. A failed check opens by itself,
  and the clip's card carries how many passed.
- **A clip is ready only when its file is.** Render returns when the file exists. Hand-over takes only a file made
  from the clip as it is now: a change after the render makes the file old, and the state Muse reads says so in
  words, so it cannot tell the person a clip is ready when it is not.
- **Every page shows the chat the server has open.** When another chat opens, every page follows it, so neither an
  address nor a second tab can leave a page working on another chat.

The loop's rules are tested against a scripted mind, and the streamed answers against a small fake server:

```
python -m unittest discover -s tests
```

## Built during the hackathon (Oct 26 - Nov 4, 2026)

Every line of code in this repository was written during the Meta AI Developer Hackathon. The first commit is
dated Oct 26, 2026, after kickoff, and the commit history shows how it grew, one working step at a time.

Built with help from AI coding assistants.

## Prototyped privately before the event

The idea was tried on a private prototype before the event. Its lessons are listed under Measured below, with their
dates. No code from it is in this repository.

## Open source and models used

Each piece is listed with its licence. The Status column says whether this build uses it yet. What it receives says what leaves this computer for it: on the home route (Glimmer, Omnilingual and SAM on your own card) nothing does; a Muse on an API, and Muse Wonder watching, are sent what that column says, with your own key.

| Piece | What it does here | What it receives | Licence | Status |
|---|---|---|---|---|
| [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) (Meta) | the agent's mind, on your own graphics card | nothing: it runs on this computer | Apache-2.0 | used |
| Muse Spark (Meta API) | the same mind without a graphics card, with your own key | each step's messages, the pictures Muse looks at, and what was said in them | Meta's API terms | used |
| Muse Wonder (Meta API) | the same mind, Meta's model for long work with many tools, with the same key; and, whichever Muse runs the chat, it watches a stretch of up to 60 s with its sound and names what the words miss (a scream, a laugh, a fail, a silent stare), with the effect for each (`watch_piece`) | as Spark when it runs the chat; for `watch_piece`, the stretch it watches, as a small video with its sound | Meta's API terms | used |
| [Muse Spark Open](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B) (Meta) on a Modal shared endpoint | the same mind on Modal's machines, with your own token | as Spark, on Modal's machines | Meta's model licence; Modal's terms | answers (one question measured 10-29); no whole clip made on it yet |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (llama-server) | runs Glimmer locally | nothing: it runs on this computer | MIT | used |
| [FFmpeg](https://ffmpeg.org/) | reads each recording (ffprobe) and takes pictures from it; later cuts and renders the clip | nothing: it runs on this computer | LGPL-2.1+ (GPL for some builds; the Windows build setup downloads is GPL-3.0) | used |
| [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) (Meta), CTC 1B v2 | writes down what is said, in many languages, with the time of every phrase | nothing: it runs on this computer | Apache-2.0 | used |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google), Thai *train* transcripts | counted once into `clipstudio/thai_marks.json`, which puts back the Thai vowel marks Omnilingual cannot write | nothing: files in this repo | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | used |
| [SAM 3.1](https://github.com/facebookresearch/sam3) (Meta) | finds the person frame by frame, for the cut-out looks and the titles behind the person (`cut_out`) | nothing: it runs on this computer | SAM License | used |
| [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (OpenCV Zoo) | finds faces, so the layouts frame them and words never cover one | nothing: it runs on this computer | MIT | used |
| [OpenCV](https://opencv.org/) (`opencv-python`) | runs YuNet | nothing: it runs on this computer | Apache-2.0 | used |
| [segno](https://github.com/heuer/segno) | draws the QR code a phone opens the studio with | nothing: it runs on this computer | BSD-3-Clause | used |
| [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) | breaks Thai titles and captions at word boundaries | nothing: it runs on this computer | Apache-2.0 | used |
| [NumPy](https://numpy.org/) | holds the pictures the face finder reads | nothing: it runs on this computer | BSD-3-Clause | used |
| [RapidOCR](https://github.com/RapidAI/RapidOCR) (`rapidocr-onnxruntime`), with the PP-OCRv4 models it ships | reads the words on the screen at full size, so private lines can be hidden | nothing: it runs on this computer | Apache-2.0 | used |
| [ONNX Runtime](https://onnxruntime.ai/) | runs RapidOCR's models on the processor | nothing: it runs on this computer | MIT | used |
| [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (OpenCV Zoo) | tells faces apart, so other people's faces can be blurred | nothing: it runs on this computer | Apache-2.0 | planned |
| [MMS-TTS Thai](https://huggingface.co/facebook/mms-tts-tha) (Meta) | an optional voice-over, off by default | nothing: it runs on this computer | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), **non-commercial** | planned |
| [Google Sans](https://github.com/googlefonts/googlesans), in `fonts/` with its licence | the house font for words on the clip, Thai and Latin | nothing: files in this repo | SIL Open Font License 1.1 | used |
| [Kanit](https://fonts.google.com/specimen/Kanit) (Google Fonts), in `fonts/` with its licence | Thai and Latin text on the clip, when chosen | nothing: files in this repo | SIL Open Font License 1.1 | used |
| [Pattaya](https://fonts.google.com/specimen/Pattaya) (Google Fonts), in `fonts/` with its licence | the glow look's script letters | nothing: files in this repo | SIL Open Font License 1.1 | used |
| [Taviraj](https://fonts.google.com/specimen/Taviraj) (Google Fonts), in `fonts/` with its licence | the classic look's italic letters | nothing: files in this repo | SIL Open Font License 1.1 | used |
| [Fluent Emoji 3D](https://github.com/microsoft/fluentui-emoji) (Microsoft), 60 of them in `stickers/` with `stickers/LICENSE-Fluent-Emoji.txt` | the studio's stickers | nothing: files in this repo | MIT | used |

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
| 2026-10-27, this build | Omnilingual ASR CTC 1B in Docker on the processor (float32, 8 threads), Thai FLEURS, the 60 test sentences, spaces and punctuation left out | 8.4% character errors after the Thai repair (12.3% as the model wrote it), 5.4-5.6x real time, 4.0 GB of memory. bfloat16 took 2.2 GB but had not finished after about 20 minutes (float32: 132 s), so float32 stays |
| 2026-10-28, this build | Omnilingual ASR CTC 1B v2 as ONNX, onnxruntime 1.27 on the processor (8 threads), against fairseq2 in Docker; the same 60 Thai FLEURS sentences | float32: the same token in all 2,253 frames of a 17 s and a 28 s check (exported at 10 s), 8.4% character errors after the Thai repair (12.4% as written), 7.3x real time, 3.6 GB of memory. int8 (its matrix products in 8 bits): 8.5% (653 of 7,657 letters wrong against 647), 9.4x real time, 1.0 GB of memory, a 1.0 GB file, so int8 is the one the studio uses |
| 2026-10-27, this build | Numbers in the review clip as the model wrote them | "5้าสิบ", "เจ็ดสิบ5้า", "1ึ่ง.ิด5": a digit where a number word begins. Muse's card said 70 m² where the seller said 75. Now written as said (ห้าสิบ, เจ็ดสิบห้า, หนึ่งจุดห้า); FLEURS unchanged at 8.4% |
| 2026-10-27, this build | Where the time of a 30 s look goes, Glimmer Q3 | hearing 5.3-6.2 s; Muse fixing the misheard words 20-42 s, mostly thinking (about 1,000 tokens written at "Reasoning strength: low"; with thinking switched off it wrote 2,600-2,900). So the fix runs only for the moments Muse proposes, and only the first before the cards show: Muse answers one question at a time, and fixing three moments kept the first clip waiting 195 s (10-27). Another moment is fixed when a clip is made from it. A question asked apart from the job kept the job's cache: 2,136 tokens reused, 5 read new |
| 2026-10-27, this build | Opening to cards with hearing, one run each | the review clip: 7 steps, 0 refused, 2:48 (1:19 without hearing); the 4-hour stream: 7 steps, 0 refused, 3:12 |
| 2026-10-27, this build | Do phrases start where speech starts? | 46 of 50 start at least 10 dB over their stretch's quiet floor; the other 4 at 8.6-9.9 dB, two of them cut by the first frame of the stretch |
| 2026-10-27, this build | Muse's answer after the cards, before any clip existed | it said the clip was made, with captions and a blurred chat. Now the state Muse reads says "no clip made yet": next run, it said only that the moments were ready to preview (one run, not yet a measurement) |
| 2026-10-27, this build | The editor, one live run each on Glimmer Q3, from opening the recording | the review clip: 14 steps, 3:00, no question to the person, a title clear of the face; it found four faults, all fixed: "the clip is ready" with no file made, a free window of 0.4 s offered, a key word inside "50–75", a title over the product. The 4-hour stream: 11 steps, 2:54, 0 refused, card layout with the face found in 37 of 40 pictures; it found two: the chat blur put over the streamer's face, and a Thai key word that coloured the whole title |
| 2026-10-27, this build | Drawing one preview frame of the clip | 0.1-0.45 s for a layout from a 1080p stream; one 1080x1920 frame with a title, captions and a hit in under 1 s |
| 2026-10-28, this build | Small talk in a new chat ("ฝนตกทั้งวันเลย", "ok", "เหนื่อยจัง ไลฟ์มาทั้งวัน"), 12 messages each | before: Glimmer called a tool for 4, Spark for 3. With the new instruction and the check: 0 of 40 small-talk messages ran a tool on either (12 in a new chat and 8 after a finished clip, twice each), and 8 of 8 requests for work still started work on each |
| 2026-10-28, this build | The question apart that sorts a message into work or talk, 36 talk and 34 work messages | 70 of 70 right with each, Spark in a median 3.1 s, Glimmer in 5.1 s (Glimmer missed "มีไฟล์อะไรบ้าง" until asking which recordings you have was named as work). So it is asked only when Muse's first answer calls a tool |
| 2026-10-29, this build | The Premiere export of the yeti review (7 pieces from 3 phone clips, 45.6 s), read back with OpenTimelineIO's Final Cut Pro 7 XML reader | 7 pieces on the video track and on the sound, each pointing at its own phone clip (83-109 s long) at the seconds the studio used, the timeline 45.60 s like the clip, the 6 reasons and the title as markers. Frames drawn from the originals with the exported scale and position matched the finished clip at each piece (mean difference 1.6-8.2 of 255, captions left out). Not yet opened in Premiere itself |
| 2026-10-29, this build | Seven of our own sets, each asked for once in Thai the way a person types it, on Spark and on Glimmer (the brother's podcast on Glimmer only): a mic review in 8 phone clips, horse riding (7), Akihabara (12), an event (19), a talk, a 4-hour SIFU stream, a 1-hour podcast | Spark: 6 of 6 made a clip that passed every check, $0.21-0.46 a clip ($1.83 in all), 2.8-8.1 min. Glimmer: 7 of 7 after three fixes the runs found (a title place -12 px tall in the speaker layout, a podcast's hosts listed to blur, a blur one pixel short of the viewers' names), 2.4-7.9 min a clip, 3.9 on average. The podcast's clip came out 6.7 s: the hearing reads that stretch badly, and the talk recipe keeps only clear speech |
| 2026-10-29, this build | Meta's own Glimmer file instead of unsloth's: `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` from [meta-models/Muse-Glimmer-30B-GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF), 16,756,683,904 bytes, SHA-256 `4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e` as Meta lists it, as a second preset in the same router, chosen with `LCS_MIND_MODEL`; the mic review set asked once in Thai, one run | a 29.0 s clip that passed every check, 7 steps, 0 refused, $0, 15.7 min (the same set on UD-Q3_K_XL: 7.9 min). The file is larger than the 16 GB card, so part of it runs on the processor; UD-Q3_K_XL stays the default |
| 2026-10-29, this build | Muse Wonder watching 30 s of the author's own game stream (2:51:08-2:51:38), cut to 360 pixels at 10 frames a second with mono sound (1.46 MB), through `watch_piece` | 36 s, $0.059 (2,735 tokens read, 2,375 written, most of them thinking); 6 moments, none of them said: the yelp at the first hit, a throw, the death from behind, the laugh at his own death, the age jumping from 38 to 44 (the big word it gave was Thai, "แก่!", as the stream is in Thai), and the run around the hedges |
| 2026-10-29, this build | One short Thai question to Muse Spark Open on the author's Modal endpoint, and to Muse Wonder on Meta's API | Spark Open: right answer in 1.7 s, $0.002; Wonder: 7 s, $0.006 |

New measurements are added here as the build is tested, each with its date.

## Footage

The recordings in the demo are the author's own: a product review shot on an iPhone, a game stream, and clips
from Ray-Ban Meta glasses. All were recorded before the event. Every edit the demo shows is made by this build.

## Licence

MIT. See [LICENSE](LICENSE). Models, fonts and tools keep their own licences, listed above.
