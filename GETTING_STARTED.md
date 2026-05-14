# Getting started with ctx-probe

A step-by-step guide for non-developers. If you've never used a terminal before, that's fine — this guide assumes nothing and walks through every step. Expected total time: **20 minutes**, most of which is one-time setup.

By the end, you'll have a chart on your screen that shows how well your AI model handles long documents.

---

## What this tool tells you (in plain English)

Modern AI models like Claude or GPT can read very long documents — sometimes a million words at a time. But the marketing claim and the reality are different. As documents get longer, accuracy drops. The model starts missing facts, especially ones buried in the middle.

**ctx-probe measures where your model starts to fail, on your own documents.** It produces one chart: a line that shows how often the model finds a hidden fact, depending on where in the document the fact is hidden.

If the line stays at 100% across the whole chart, your model handles your document length well. If it drops sharply at, say, 60%, your model starts missing facts past that point. Either way, you now know.

That's all. No fixes, no opinions, no recommendations. Just the measurement.

---

## Before you start

You need three things:

1. **A computer running macOS or Linux.** Windows works too, but the commands are slightly different — ask anyone on your team to translate if needed.
2. **An Anthropic account** (the company that makes Claude). You'll create an API key — think of it as a password that lets the tool talk to Claude on your behalf. Sign up at https://console.anthropic.com if you don't have one.
3. **A folder of documents** you want to test against. Plain text or Markdown files (`.txt` or `.md`). If your documents are PDFs or Word files, you'll need to convert them first — ask your team or use any free online converter.

You'll also need to install one piece of software, but the guide walks you through it.

---

## Step 1 — Open the Terminal

The Terminal is a built-in app on your Mac that lets you type commands instead of clicking. Don't be intimidated — you'll only paste in lines, nothing fancy.

**On macOS:** press `Cmd + Space`, type "Terminal", press Enter. A black or white window appears with text inside it. That's the Terminal.

**Throughout this guide:** when you see a code block like this:

```bash
some command here
```

you copy that command, paste it into the Terminal window, and press Enter. That's the entire interaction model.

---

## Step 2 — Check if Python is installed

ctx-probe runs on Python, a programming language that comes pre-installed on most Macs.

In your Terminal window, paste this and press Enter:

```bash
python3 --version
```

You'll see one of two things:

- **A version number like `Python 3.11.5` or higher** → great, skip to Step 3.
- **An error, or a version like `Python 3.9.x`** → you need to install or update Python. The easiest way is to download the installer from https://www.python.org/downloads/, run it, and re-open the Terminal. Then run the command above again to confirm.

You only do this once, ever.

---

## Step 3 — Install ctx-probe

This installs the tool from the Python package index. Paste:

```bash
pip install ctx-probe
```

The command takes about a minute. You'll see a wall of text scrolling — that's normal. When it's done you'll see "Successfully installed..." at the bottom.

Verify it worked:

```bash
ctx-probe --version
```

You should see `ctx-probe, version 0.1.1` or higher. If you do, you're set up. **You won't need to do Step 3 again** — ctx-probe is now installed permanently.

---

## Step 4 — Get your Anthropic API key

Go to https://console.anthropic.com/settings/keys in your browser. Sign in if you haven't. Click "Create Key", give it a name like "ctx-probe", and copy the key it shows you. **The key starts with `sk-ant-...` and is shown only once — copy it now.**

Treat this key like a password. Don't share it, don't email it, don't paste it into a chat.

Back in your Terminal, paste this command, replacing `PASTE_YOUR_KEY_HERE` with the key you just copied:

```bash
export ANTHROPIC_API_KEY=PASTE_YOUR_KEY_HERE
```

This stores the key in your Terminal session. You'll need to re-run this command every time you open a new Terminal window.

A note on cost: each test run costs a few cents to a few dollars, depending on how big your documents are. Anthropic bills the credit card on your account — you can set a hard monthly limit at https://console.anthropic.com/settings/limits to avoid surprises.

---

## Step 5 — Run a quick test (under 1 cent)

Before pointing at your own documents, let's verify everything works. Paste this single command:

```bash
ctx-probe demo
```

It should take 5–10 seconds. When it finishes you'll see a message like:

```
✓ 1/1 correct · report → demo-run/report.html
```

If you see that, **everything is working.** Open the report by pasting:

```bash
open ./demo-run/report.html
```

A chart should appear in your browser. The chart will be simple (one data point) — that's expected. The point is that the tool ran and produced an output.

If something went wrong, scroll to the "Common errors" section at the end of this guide.

---

## Step 6 — Run it on your own documents

Now the real measurement. Put the documents you want to test in any folder on your computer. They should be `.txt` or `.md` files. Note down the path to that folder — e.g. `/Users/yourname/Documents/my-clinical-standards`.

You'll also need to design a "needle" — a fake fact that you'll hide in the documents and ask the AI to find. The fake fact has to be something the AI couldn't possibly know from its training. Use a made-up identifier with a specific number.

**A good needle for a healthcare company might look like:**

- The fact: *"Per internal protocol XYZ-742, the threshold for nephrology referral is 2.3 mg/dL."*
- The question: *"What is the threshold specified in protocol XYZ-742?"*
- The expected answer: *"2.3"*

**A good needle for a fintech company might look like:**

- The fact: *"Per internal compliance rule FC-2026-118, the transaction review threshold is $4,750."*
- The question: *"What is the review threshold under rule FC-2026-118?"*
- The expected answer: *"4,750"*

**Bad needle (don't do this):**

- *"What's the maximum heart rate during exercise?"* — the AI knows this from training; you can't tell if it found the fact in the document or just made it up.

Once you have your needle, paste this command into the Terminal — but **replace the four values in CAPITAL LETTERS** with your own:

```bash
ctx-probe run \
  --model claude-sonnet-4-6 \
  --corpus YOUR_DOCUMENT_FOLDER_PATH \
  --context-length 100000 \
  --depths 10,25,50,75,90 \
  --samples 3 \
  --needles "" \
  --needle-text "YOUR_FAKE_FACT_HERE" \
  --needle-question "YOUR_QUESTION_HERE" \
  --needle-expected "YOUR_EXPECTED_ANSWER" \
  --out ./my-report
```

This will take 3–8 minutes and cost about $0.50–$1.50. While it runs, you'll see progress messages in the Terminal.

When it's done, open the report:

```bash
open ./my-report/report.html
```

---

## Step 7 — How to read your chart

The chart shows two things:

- **The horizontal axis (left to right):** where in the document the hidden fact was placed. 10% means near the start; 90% means near the end.
- **The vertical axis (top to bottom):** how often the AI found the fact at each position. 100% means it always found it; 0% means it never did.

**What different shapes mean:**

- **Flat line at 100% across the whole chart.** Your model handles your document length fine. If your AI system is still making mistakes, the problem is somewhere else (likely how documents get into the AI in the first place). The fact that the line is flat doesn't mean nothing's wrong — it just means *the model* isn't the problem.

- **Line drops sharply at some point, e.g. 60%.** That's your effective limit. Facts placed past that point in the document are unreliably retrieved. Whatever system feeds documents to the AI should be restructured so important facts land in the first 60% of context.

- **Jagged, inconsistent line.** Either your needle is ambiguous, or the document set is too small to make a fair measurement. Re-run with more samples (`--samples 5`).

The summary table below the chart shows additional numbers: cache hit rate (how efficiently the tool used Anthropic's caching), average latency, and total tokens used.

---

## What to do with this information

You have a chart. What now?

- **If the chart is flat at 100%:** the raw AI model isn't your bottleneck. Investigate the surrounding system — how documents are extracted, retrieved, or formatted before they reach the AI.
- **If the chart drops:** you've found the boundary. Anything more is a judgment call: restructure how content reaches the AI, switch to a different model that has a better curve, or accept the boundary and design around it.

If you want help interpreting the chart or planning what to fix, that's exactly what Opportune's [Effective Context Probe service](https://opportunedev.com/#services) does — book 20 minutes and we'll go through it with you.

---

## Common errors

**`ctx-probe: command not found`**
The install didn't complete or didn't add ctx-probe to your path. Try `pip install --user ctx-probe` and re-open the Terminal.

**`ANTHROPIC_API_KEY is not set` or `authentication_error`**
You haven't set the API key in this Terminal session, or the key is wrong. Re-run the `export ANTHROPIC_API_KEY=...` line from Step 4.

**`Corpus too small`**
The folder you pointed at has too little text to build a haystack of the requested size. Either add more documents or reduce `--context-length`.

**`No .txt or .md files found`**
ctx-probe only reads `.txt` and `.md` files. Convert your PDFs/Word docs first.

**`429 Rate limit exceeded`**
You're sending too many requests. Wait a few minutes and try again with `--samples 1` to reduce the volume.

**The command runs but you can't find the report**
After it finishes, the report is always at `<your-out-folder>/report.html`. Use `open` followed by that path to open it.

**Anything else weird**
Open an issue at https://github.com/OpportuneDev/ctx-probe/issues with the exact error message, or email hello@opportunedev.com.

---

## Quick reference

| What you want to do | What to paste in Terminal |
| --- | --- |
| Install (one time) | `pip install ctx-probe` |
| Set the API key | `export ANTHROPIC_API_KEY=sk-ant-...` |
| Run a quick smoke test | `ctx-probe demo` |
| Run on your documents | See Step 6 |
| See all options | `ctx-probe run --help` |
| Open a report | `open path/to/report.html` |

---

That's the whole guide. If you got to a chart, you're a ctx-probe user. If you got stuck anywhere, the failure is in this guide — tell us where and we'll fix it.
