# Reflection

## What I Built

Basically, the service does 4 things:

1. **Parse the resume** - One LLM call turns the messy resume text into structured stuff (years of experience, skills, companies, visa status, etc)
2. **Find similar jobs** - FAISS vector search grabs the top 50 jobs that look relevant
3. **Filter out bad fits** - Drop jobs where the candidate clearly doesn't qualify (needs visa but job doesn't sponsor, way too junior for a senior role, and other stuff)
4. **Score what's left** - Match the resume against each job's green/red flags and calculate a final score

Main thing: I only call the LLM once per request. Everything else uses pre-computed embeddings so it stays fast and cheap.

## How I Got Here

### Started With The Obvious Thing

First idea was super simple: embed the resume, embed each job, return whatever has highest cosine similarity. Easy to build but it doesn't really work because:

- You get "plausible but wrong" matches (two jobs both say "software engineer" but they're completely different)
- Ignores hard stuff like visa sponsorship or seniority
- Can't explain *why* something matched

So embedding similarity became my first pass to narrow things down, but I needed more stuff on top.

### Why I Added The LLM Call

Resumes are messy. Keyword matching alone misses things like:
- How many years has this person actually worked?
- Do they need visa sponsorship?
- Do they want remote, or did they just happen to work remote before?

As people have different kid of resume formats, I added one LLM call upfront to pull out structured info: YOE, seniority, skills in categories, past companies, visa situation, work preferences.

Now I have actual signals to filter on without calling the LLM for every single job.

**Downside**: Adds some latency and LLMs hallucinate sometimes.
**Fix**: temperature=0, strict JSON format, and validation logic (more on this later).

### Two-Stage Setup

Once I had structured data, I set up:

1. **Recall** - FAISS grabs top 50 by embedding similarity
2. **Filter + Score** - Only run the heavy stuff on those 50

Could I just score all 300? Sure. But this scales better if jobs grow to like 30k.

### Hard Filters

Assignment says return empty list if nothing matches. So I need real filters, not just "lower scores."

Added filters for example, stuff like:
- **Sponsorship** - needs visa but job doesn't sponsor? gone
- **Experience** - junior person seeing Staff Engineer roles? nope. 2 YOE seeing "8+ required"? nope
- **Remote** - only filter this if the resume actually says they want remote (not just because they worked remote somewhere)

**Downside**: might filter out edge cases.
**Fix**: 1-year buffer on YOE, only apply location filter when there's real evidence.

### Flag Scoring

Jobs have `greenFlags` and `redFlags` already, stuff like "startup experience" or "needs to know Kubernetes." I wanted to use these without calling the LLM 300 times.

So I:
- Pre-embed all flags when the server starts
- Turn the resume into chunks (highlights, skills, company context)
- Embed those once per request  
- Match flags to chunks with cosine similarity

Now I get explanations like "Matches 3 criteria: startup experience, Python backend..." without extra API calls.

**Downside**: embeddings over-match on vague flags like "communicates well"
**Fix**: tiered scoring so strong matches count more. Would downweight generic flags given more time.

## Problems I Hit

### LLM Making Up Work Preferences

This one was annoying. The LLM would see "Company X (Remote)" in someone's history and decide they have a "remote only" preference. That's not how it works, just because you worked remote doesn't mean you require it.

Was causing bad filtering so I added validation:
- Only set work preference if resume actually says "seeking remote" or "remote only"
- Require evidence and run regex to double check
- If evidence isn't explicit, reset to null

Sometimes you just can't trust the LLM and need hard rules.

### YOE Is Never Accurate

Calculating years of experience was misleading too. Dates are formatted differently, internships vs full-time is unclear, roles overlap.

What I did:
- Use LLM estimate but add 1-year buffer
- No explicit YOE in job? Use title heuristics (Staff Engineer probably needs 8+)
- Penalized senior people applying to entry-level stuff

### Thresholds Were Guesswork

I just picked numbers that looked reasonable when testing.

Current setup:
- 50 jobs in recall
- Minimum score threshold
- Cap at 20 results (you can change these btw)

Would measure actual precision/recall with more time.

## Why Not Just LLM Score Everything?

Could ask the LLM "rate this resume vs this job 1-10" for each job. Didn't do it because:

- 300 API calls per request = expensive
- Latency all over the place
- Hard to debug
- You get a number but no real reason

LLM is good at one thing here: turning messy text into structured data. I let it do that, then use deterministic logic for the rest.

## What Works

- Precomputes everything at startup so requests are fast (1-2 sec)
- Filters catch obvious mismatches early
- Validation catches LLM hallucinations before they mess up filtering
- Flag matching gives decent explanations without extra calls

## What Doesn't Work Great

**Embeddings miss keywords** - if a job needs something very specifically, embeddings might not catch that. BM25 hybrid search would help.

**Generic flags match everything** - "moves fast" and "communicates well" match basically any resume. Should downweight these.

**YOE is fuzzy** - LLM guesses wrong sometimes. Buffer helps but edge cases slip through.

**Hand-tuned thresholds** - would be better to actually measure precision/recall on sample resumes.

## What I'd Do With More Time

- **Hybrid search** - BM25 + embeddings to catch specific skill keywords
- **Downweight generic flags** - or require keyword overlap for short ones
- **Vectorize the scoring loop** - matrix ops instead of Python for loop
- **Evaluation script** - run all sample resumes, measure what matches, tune from there
- **Cache resumes** - hash the text so repeated resumes don't redo extraction
- **Second LLM pass** - just for top 5-10 matches, write better explanations(exp again but not as exp as 300 calls)
