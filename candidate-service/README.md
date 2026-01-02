## Overview
This service matches an uploaded resume to relevant jobs using a 4-stage retrieval + reranking pipeline.  
It makes **one LLM call per request** (resume → structured JSON). All job and flag embeddings are precomputed at startup.

See EVALUATION.md for design decisions, tradeoffs, and my personal reflection.

## How It Works

4 steps per resume:

1. **Extract** - one LLM call turns the resume into structured data (skills, YOE, companies, etc)
2. **Recall** - vector search grabs top 50 similar jobs
3. **Filter** - drop jobs that don't fit (visa issues, seniority mismatch, location, etc)
4. **Score** - score each remaining job based on green/red flags and embeddings

Everything is pre-computed at startup so the only API call per request is the one LLM extraction.

## Scoring

Each job gets a score out of 100:

- **Base similarity** (0-60) - how well resume matches job description
- **Green flags** (up to +25) - bonus for matching what the job wants
- **Red flags** (up to -35) - penalty for matching what they want to avoid
- **Company overlap** (+5) - worked at companies the job likes
- **YOE gap** (-3 per year) - under the minimum experience
- **Overqualification** (-5 to -15) - senior people applying to junior roles
- **Timezone** (+3) - candidate timezone matches job location

Flags are matched with embeddings, embed the flag text, embed resume chunks, check cosine similarity (threshold 0.45).

## Setup

```bash
cd candidate-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

Run:
```bash
python main.py
```

Server running at `http://localhost:8000`

## API

**POST /match**

```json
{
  "resume": {
    "content": "John Doe\nSenior Software Engineer at...",
    "filename": "resume.txt",
    "format": "txt"
  }
}
```

Returns ranked jobs with scores, explanations, and matched skills.

**GET /health** - status and job count

## Config

In `main.py`, you can tune them::

| Setting | Default | What it does |
|---------|---------|--------------|
| `TOP_N_RECALL` | 50 | jobs from vector search |
| `MAX_RESULTS` | 20 | max jobs returned |
| `GREEN_FLAG_THRESHOLD` | 0.45 | min similarity to count a flag |
| `MATCH_SCORE_THRESHOLD` | 25 | min score to show |

## Tradeoffs

**1 LLM call vs per-job** - could ask the LLM to score each job individually but that's 300 API calls per resume. parsing once is enough, embeddings handle the rest.

**FAISS vs hosted DB** - only 300 jobs, don't need a real vector db. FAISS runs locally and keeps things simple.

**Top-50 recall** - 50 candidates from vector search is good coverage. scoring all 300 would be slower and most won't be relevant anyway.

**Semantic flag matching** - jobs have green/red flags describing ideal candidates. embedding match is more flexible than keywords, though it over-matches on vague stuff like "communicates well".

## What I'd Add

- cache resume embeddings for repeat uploads
- BM25 hybrid search for better keyword matching
- batch endpoint for multiple resumes
- feedback loop to tune thresholds

## Also read EVALUATION.md
