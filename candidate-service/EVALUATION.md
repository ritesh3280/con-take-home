# Reflection

## What I Built

The service has 4 steps:

1. **Parse the resume** - One LLM call makes resume text into structured info (years of experience, skills, companies, visa status, etc.)
2. **Find similar jobs** - FAISS vector search pulls the top 50 candidates
3. **Filter out bad fits** - Drop jobs where the candidate clearly doesn't qualify (needs visa but job doesn't sponsor, way too junior for a senior role, and other stuff)
4. **Score what's left** - Match the resume against each job's green/red flags to get a final score

The key idea: I only call the LLM once per request. Everything else uses pre-computed embeddings, so it's fast and cheap.

## How I Got Here

### I Started Simple

My first idea was the most basic thing: embed the resume, embed each job, return whatever has the highest cosine similarity. Quick to build, but I realized it doesn't actually work for this assignment because:

- It returns "plausible but wrong" matches (two jobs can both say "software engineer" but be completely different domains)
- It ignores hard constraints like visa sponsorship or seniority requirements
- Hard to explain *why* something matched

So I kept the embedding similarity as a first pass (recall), but knew I needed more on top.

### Added Structured Resume Extraction

Resumes are messy. Just matching keywords misses important stuff like how many years someone has worked, whether they need visa sponsorship, or if they actually want remote vs just happened to work remote before.

I added one LLM call at the start to pull out structured info: YOE, seniority level, skills broken into categories, companies they worked at, visa situation, and any explicit work preferences.

This gave me reliable signals for filtering without needing to call the LLM for every job.

**Tradeoff**: Adds latency and the LLM can hallucinate stuff.
**What I did**: Set temperature=0, use strict JSON response format, and added validation logic to catch obvious mistakes (more on this below).

### Two-Stage Pipeline

Once I had structured resume data, I set up a proper retrieval pipeline:

1. **Recall** - FAISS pulls top 50 jobs by embedding similarity
2. **Filter + Score** - Run the expensive logic only on those 50

With 300 jobs I could technically score everything, but this is the right architecture. If jobs grow to 30k later, it still works.

### Hard Constraint Filtering

The assignment says to return an empty list when there are no good matches. That means I need actual filters, not just "lower scores."

I added filters for:
- **Sponsorship** - If candidate needs visa sponsorship and job doesn't offer it, reject
- **Experience level** - Junior candidates don't see Staff Engineer roles, people with 2 YOE don't see "8+ years required"
- **Remote preference** - Only enforce this when the resume explicitly says they want remote (not just because they worked remote before)

**Tradeoff**: Strict filters might remove borderline matches.
**What I did**: Added a 1-year buffer on YOE, and only apply location constraints when there's actual evidence.

### Flag-Based Scoring

I wanted the ranking to be explainable. The jobs already have `greenFlags` (what they want) and `redFlags` (what they want to avoid), so I built scoring around these.

Instead of asking the LLM "does this resume match this flag?" 300 times, I:
- Pre-embed all flags at startup
- Turn the resume into "evidence chunks" (highlights, skill summaries, company context)
- Embed those chunks once per request
- Match flags to chunks with cosine similarity

This gives explanations like "Matches 3 key criteria: startup experience, Python backend..." without any per-job LLM calls.

**Tradeoff**: Embeddings can over-match on generic flags like "communicates well."
**What I did**: Tiered scoring (stronger matches get more points). With more time I'd downweight generic flags.

## Honest Issues I Ran Into

### LLM Hallucinating Work Preferences

Early on, the LLM would sometimes see "Company X (Remote)" in someone's work history and decide they have a "remote only" preference. That's wrong, working remote before doesn't mean they require it.

This was causing incorrect filtering, so I added strcit validation:
- Work preference is only set if the resume actually says something like "seeking remote" or "remote only"
- I require evidence and run regex patterns to verify
- If the evidence isn't explicit, I reset the preference to null

This is one place I chose deterministic checks over "trust the LLM"

### YOE Is Always Fuzzy

Years of experience is hard to calculate reliably. Date formats vary, internships vs full-time is unclear, sometimes roles overlap.

What I did:
- Use the LLM's estimate but add a 1-year buffer when filtering
- If the job doesn't have an explicit YOE range, fall back to title heuristics (if it says "Staff Engineer" it probably needs 8+ years)
- Penalize overqualification for entry-level jobs (senior folks applying to junior roles)

### Scoring Thresholds Are Hand-Tuned

I picked numbers that seemed reasonable from testing with the sample resumes. The main risk is returning too many weak matches.

Current approach:
- Pull 50 jobs in recall for good coverage
- Apply a minimum score threshold
- Cap results at 20 max(you can change any thresholds btw)

With more time I'd actually measure precision and recall using all the sample resumes.

## Why No Per-Job LLM Scoring

There was an idea to just ask the LLM "rate this resume against this job 1-10" for each job, but thats really dumb imo.

- Cost scales with job count (300 calls per request is expensive)
- Latency becomes unpredictable
- Harder to debug when something goes wrong
- You get a number but not really a reason

Instead I use the LLM only where it's strongest: turning messy text into structured data.

LLMS guess but dont think!!

## Why I Think This Works

- At start, it precomputes everything, so actual requests are quick (1-2 seconds)
- The constraint filter catches obvious mismatches early (a junior dev won't see staff engineer roles)
- The work preference validator catches LLM hallucinations before they cause bad filtering
- Flag matching gives decent explanations without extra API calls

## What Doesn't Work Great

**Embeddings miss specific keywords**: If a job absolutely needs some sort of experience, embeddings might not weight that heavily enough. I think a hybrid approach of BM25 (keyword search) along with embeddings would help us here.

**Some flags are too generic**: Flags like "moves fast" or "communicates well" match almost any resume. I could decrease the weight for these or require actual keyword overlap.

**YOE estimation is fuzzy**: The LLM calculates years of experience from the resume, and it's not always right. I added a 1 year buffer to be safe, but edge cases can always slip.

**Scoring thresholds are hand-tuned**: I picked values that seemed reasonable from testing, but with more time I'd run the sample resumes through and actually measure precision and recall.

## What I'd Add With More Time

- **Hybrid search** - Combine BM25 keyword matching with embeddings so we don't miss specific skill requirements
- **Better flag handling** - Decrease weight for generic flags, maybe require keyword overlap for short flags
- **Vectorized scoring** - Right now I loop through flags in Python, we could speed this up with matrix operations
- **Evaluation script** - Run all sample resumes, track which jobs they match, then we can tune the thresholds based on what looks right
- **Resume caching** - Hash the resume text so repeat uploads don't redo extraction
- **Second-pass LLM** - For just the top 5-10 matches, we can ask LLM to write better explanations (slightly more expensive but just for more explanations)
