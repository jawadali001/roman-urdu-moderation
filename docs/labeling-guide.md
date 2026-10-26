# Labeling guide

Rules for hand-labeling the evaluation set. The same rules apply to every batch and every labeler, so the labels stay consistent.

**What the labels are for:** scoring the model honestly. Hand-labeled messages are used for validation and test only and are **never used for training**.

## 1. What you are labeling

Every message gets exactly one label, based on **what the message says**, not on who wrote it or what you guess they meant.

| Label | Use when the message... | Mild example |
|---|---|---|
| `clean` | is a normal message: opinions, questions, jokes, criticism of ideas, plans or policies without insults, or reports someone else's abuse without repeating it as an attack | "yaar ye match to kamal ka tha" |
| `abusive` | insults, abuses or humiliates a specific person or a group | "tum bilkul bewakoof ho" |
| `profane` | swears or uses vulgar language **without a target** (anger at a situation, emphasis) | "saala ye traffic hi kharab hai" |
| `religious_hate` | attacks, mocks or shows hatred toward a religion, a sect, its followers, or its sacred figures and symbols | pattern: calls followers of a faith inferior, evil or subhuman |
| `sexual_hate` | is sexually abusive, harasses someone sexually, or demeans people because of gender or sexuality | pattern: demeans women or men as a group, or uses sexual insults at a person |
| `unsure` | cannot be decided after one careful read, usually because context is missing. Keep this under 5% of messages | |
| `unusable` | has no readable Latin-script text: only emoji, only a link or number, or text in another script | |

The examples are deliberately mild. Real messages will be harsher.

## 2. Decision order

Go through these in order and stop at the first one that applies:

1. Cannot read it (no Latin letters, only emoji, link or numbers)? → `unusable`
2. Attacks a religion, sect, believers or religious figures? → `religious_hate`
3. Sexually abusive, or demeans people for their gender or sexuality? → `sexual_hate`
4. Insults, abuses or humiliates a person or group? → `abusive`
5. Swearing or vulgar words with no target? → `profane`
6. None of the above → `clean`. Tick **looks risky but clean** (key H) if it contains strong words, religious or gender topics, or would trip a keyword filter.
7. Still cannot decide after one careful read → `unsure`

This is the "most specific wins" rule. If a message is both religious and sexual hate, pick the one that is more central and write "both" in the notes.

## 3. Tricky cases

| Situation | Label |
|---|---|
| Friends teasing with mild words ("oye pagal, kahan reh gaya tu?") | `clean`, tick risky, note "playful" |
| Teasing that uses a strong insult | `abusive`, note "playful" (the words decide) |
| Swearing at a situation | `profane` |
| Swearing at a person | `abusive` |
| Asking about or politely criticizing a religion's ideas or practices | `clean` |
| Calling believers inferior or evil, mocking sacred figures | `religious_hate` |
| Name-calling a politician, party, team or celebrity | `abusive` |
| Criticizing their policies or performance without insults | `clean` |
| Neutral talk about gender or sexuality | `clean` |
| Demeaning a gender as a group, sexual insults, sexual harassment | `sexual_hate` |
| Quoting or reporting abuse ("he called me names") | `clean` |
| Repeating abuse as the message's own attack | `abusive` |
| Sarcasm with obvious abuse | `abusive` |
| Sarcasm where you would have to guess | `unsure` |
| Threat of harm ("main tumhein dekh loon ga") | `abusive` for now, write "threat" in the notes (the threat class is planned) |
| Ads, links, "follow for follow" | `clean` for now, write "spam" in the notes |
| Phone number or address shared | `clean` for now, write "personal_info" in the notes |
| Roman Urdu mixed with English | label by meaning; the language mix does not matter |
| Hashtags and @mentions | ignore them, label the words |
| Part of the message is toxic | the toxic part wins (use the decision order) |

## 4. How to work

1. Open `scripts/label_tool.html` in a browser, then **Open CSV** and pick `data/custom/real_labeled/batch_01_to_label.csv`.
2. Keys: `1` clean, `2` abusive, `3` profane, `4` religious_hate, `5` sexual_hate, `U` unsure, `X` unusable, `H` looks risky but clean, `N` notes, arrow keys to go back or forward. The tool moves to the next unlabeled message by itself.
3. **Do not open the `*_sampling_meta.csv` file** until the batch is finished. It holds the model's predictions and would bias your labels. The tool never shows them.
4. Work in sessions of 30 to 45 minutes with breaks. About 12 seconds per message means roughly 2 hours for 640 messages. The messages can be unpleasant to read, so stop whenever you need to.
5. Progress saves in the browser automatically. The tool reminds you to **Export CSV** every 50 changes. Do it, since clearing browser data would delete the saved progress.
6. When the batch is done, export, and move `batch_01_labeled.csv` into `data/custom/real_labeled/`. Keep the original `to_label` file.
7. Check the counts shown above the message. The goal is **at least 50 messages per class**. If a class is short, create another batch with `sample_for_labeling.py --batch 2`.

## 5. Quality checks

- **Self-agreement:** after a week, re-label 50 random messages without looking at your earlier labels, and compare.
- **Second labeler:** ask a friend who reads Roman Urdu to label about 100 messages using this guide. We will compute Cohen's kappa. Aim for 0.6 or higher. A lower value means the rules need tightening, not that the work is bad.
- **Rule changes:** if you change a rule, add a line to the log below with the date, so later labels stay consistent with earlier ones.

## 6. How the labels are used

- Validation and test only, never training.
- Each batch mixes **random** rows (an unbiased picture of how common each class is) with **candidate** rows (picked by the model to find enough examples of the rare classes). Candidates tend to flatter the model, so results are reported for the random rows alone as well as for all rows.
- `unsure` and `unusable` rows are excluded from scoring, and their counts are reported.
- The messages come from the unlabeled RUHSOLD test tweets, which no model has trained on. That makes a leakage-free test, but it is the same kind of text as the RUHSOLD training data, not a new domain. The labels reflect one labeler's judgment (plus a second labeler for the agreement check).

## Rule changes log

| Date | Change |
|---|---|
| 2026-10-26 | First version |