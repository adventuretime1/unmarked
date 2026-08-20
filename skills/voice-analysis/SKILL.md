---
name: voice-analysis
description: Analyze a person's writing samples and produce a Unmarked voice profile describing how they write. Use when the user wants Unmarked to rewrite text in their own voice, asks to "analyze my writing", "capture my tone", "create a voice profile", or runs `unmark voice` and has no profile yet.
---

# Voice analysis

Read someone's writing, describe how they write, and save that description as a
Unmarked voice profile. Unmarked injects the description into rewrite prompts so
rewritten text sounds like the author instead of like a language model.

The output is prose, not a score. A person should be able to read it and say
"yes, that's me" or correct it directly.

## 1. Collect samples

Ask the user for writing samples if they have not supplied any. Useful guidance:

- Three to five pieces is plenty; one is enough to start.
- Prefer prose they actually wrote and did not heavily edit with an AI tool.
- Prefer the kind of writing they want Unmarked to produce. Someone who wants help
  with work email should not hand over their fiction.
- Anything readable works: files, pasted text, a link to something they wrote.

If the samples are mixed (a terse message and a long essay), ask which
register they want captured, or offer to make two profiles.

## 2. Read for these questions

Do not answer these as a checklist in the output. They are what to look for; the
output is a description written in your own words.

**Rhythm and sentence shape**
- Are sentences long and winding, short and clipped, or deliberately varied?
- Do they use fragments? Start sentences with "And" or "But"?
- How do paragraphs open and close? Do they land on a short line?

**Stance and certainty**
- Do they state claims flat, or hedge with "I think", "probably", "seems"?
- Do they qualify and caveat, or commit and move on?
- Do they argue, explain, narrate, or report?

**Register and distance**
- Formal, conversational, technical, wry, warm, blunt?
- Does the register shift by context, and how?
- First, second, or third person? Present or past tense?

**Texture**
- Punctuation habits: em dashes, semicolons, parentheticals, colons, lists.
- Do they use contractions? Rhetorical questions? Direct address?
- Recurring words, phrasings, or constructions they lean on.
- Metaphor and concrete examples, or abstract and plain?

**Negative space** — often the most useful part.
- What do they conspicuously avoid? Corporate filler, exclamation marks,
  hype words, hedging, jargon, emoji?
- What would immediately read as *not them*?

## 3. Write the description

Write 150-400 words of plain prose, addressed to whoever will do the rewriting.
Guidelines:

- Be specific and observable. "Short declarative sentences, rarely over 20
  words, often ending on a blunt three-word clause" beats "concise style".
- Quote or paraphrase a few characteristic constructions from the samples.
- Say what to avoid as well as what to do.
- Do not invent traits you did not observe. If the samples are too thin to judge
  something, leave it out and say the sample was limited.
- Do not include the samples themselves, personal details, names, employers, or
  anything else identifying. The description travels into model prompts.
- Describe *how they write*, never *what they wrote about*. Topic is not voice.

## 4. Save it

Ask Unmarked where profiles live rather than hard-coding a path:

```bash
unmark voice path              # the voices directory
unmark voice path work         # where a profile named "work" would go
```

Save the description:

```bash
unmark voice save work --from description.md --generated-by agent
```

or pipe it directly:

```bash
cat description.md | unmark voice save work --generated-by agent
```

Use `--force` only to deliberately replace an existing profile; check
`unmark voice list` first. Name profiles for the register they capture — `work`,
`casual`, `academic` — not after the person.

## 5. Confirm

Show the user the description and tell them how to use and correct it:

```bash
unmark voice show work
unmark edit draft.md --rewrite --voice work
```

Invite correction. Unmarked stores the description in a JSON voice-profile file;
use `unmark voice show` and `unmark voice save --force` to review or replace it.
