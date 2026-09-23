# LinkedIn post: CardioCore AI

## Headline options (pick one for the first line)

1. **My AI scored 100% on its test. Then I gave it a real heart.**  (recommended)
2. Time-series data lies to you politely. Here's how I caught it.
3. Your model isn't 100% accurate. Your test data is just too friendly.

## Post (about 2,000 characters; LinkedIn's limit is 3,000)

My AI scored 100% on its test. Then I gave it a real heart. 🫀

I built CardioCore AI, an open-source teaching app: it takes an EKG signal, cleans it, finds the heartbeats, measures them, classifies the rhythm, and has Claude explain the result like a tutor.

On simulated data, the classifier was perfect. On a real patient recording full of premature beats, it confidently said "atrial fibrillation."

That gap is the whole story of time-series work. What actually mattered:

📉 Sample fast enough. At 12 samples per second, a heartbeat's sharp spike falls between the dots and vanishes. Nyquist isn't trivia.

🧹 Clean before you measure. Breathing drifts the baseline, muscle adds fuzz. A zero-phase filter keeps timing intact, but a live monitor can't use it, because it may only look at the past.

🔎 Find the events first. A 1985 algorithm (Pan-Tompkins) still detects beats better than most clever ideas I tried.

🧪 Synthetic data flatters you. My model learned a shortcut: "no P wave + irregular = AFib." Real ectopic beats broke it. The fix wasn't a bigger model, it was more realistic training data.

🏷️ Labels carry state. Cardiologists mark "AFib starts here" once. Slice a window from the middle and the marker is gone. My own loader labelled real AFib as "normal" until I caught it.

⚖️ Judge like production. Split by patient, and measure sensitivity and precision, not accuracy.

And in real devices, everything gets harder: endless streams, lead-off detection, motion artifacts, alarm fatigue, milliseconds that matter. One design rule I'd defend anywhere: the LLM explains results after the fact. It never decides whether an alarm sounds.

In my app, Claude never sees the waveform. Plain code measures; the LLM teaches from a validated fact sheet, with a structured, schema-checked answer.

Final score: 54 tests, 11 of 12 real recordings right, and a much better sense of what "works" means.

What's the sneakiest time-series bug you've hit? 👇

Code and a beginner-friendly illustrated guide: https://github.com/timroman234/cardiocore-ai

Educational demo, not a medical device.

#TimeSeries #MachineLearning #DataScience #SignalProcessing #HealthTech #Python #LLM

## Notes before posting

- **Repo visibility:** the link only works for readers if the GitHub repo is public. Check this first, or remove the last link line.
- **Image:** LinkedIn does not accept SVG. Use a screenshot of the running app (chart + the AFib badge), or export `docs/img/fig_sampling.svg` / `fig_nsr_vs_afib.svg` to PNG. The sampling figure makes the best scroll-stopper.
- **First line:** LinkedIn shows roughly the first 200 characters before "see more". The hook above fits inside that.
- **Claims:** "11 of 12" refers to the 12 slices tested (records 100, 119, 201, 208 at 0, 60 and 120 s), not a clinical accuracy figure. Keep that wording if you edit.
