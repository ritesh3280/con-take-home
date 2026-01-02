## Read README.md, then move to EVALUATION.md
## How It Works

The service runs a 4-phase pipeline for each resume:

1. **Extract** - Single LLM call parses the resume into structured data (skills, YOE, companies, etc.)
2. **Recall** - Vector search pulls the top 50 most similar jobs
3. **Filter** - Hard constraints knock out jobs that don't fit (visa, seniority, location)
4. **Score** - Each remaining job gets scored based on green/red flag matching

All job embeddings and flag embeddings are pre-computed at startup, so the only API call per request is the one LLM extraction.

## Scoring

Each job gets a score out of 100:

- **Base similarity** (0-60 pts) - how well the resume matches the job description
- **Green flags** (up to +25 pts) - bonus for matching what the job is looking for
- **Red flags** (up to -35 pts) - penalty for matching what the job wants to avoid
- **Company overlap** (+5 pts) - if candidate worked at companies the job considers "ideal"
- **YOE gap** (-3 pts per year) - if candidate is under the minimum experience
- **Overqualification** (-5 to -15 pts) - senior people applying to junior roles
- **Timezone bonus** (+3 pts) - if candidate's timezone matches job location

Green/red flags are matched semantically, we embed both the flag text and resume chunks, then check cosine similarity.          (threshold: 0.45).

## Setup

```bash
cd candidate-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file inside the candidate-servicd folder:
```
OPENAI_API_KEY=sk-your-key-here
```

Run:
```bash
python main.py
```

Server starts at `http://localhost:8000`

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

Returns ranked job matches with scores, explanations, and matched skills.

**GET /health** - returns status and job count

## Config

These are in `main.py` and you can tune them:

| Setting | Default | What it does |
|---------|---------|--------------|
| `TOP_N_RECALL` | 50 | Jobs to pull from vector search |
| `MAX_RESULTS` | 20 | Max jobs returned to user |
| `GREEN_FLAG_THRESHOLD` | 0.45 | Min similarity to count a green flag |
| `MATCH_SCORE_THRESHOLD` | 25 | Min score to include in results |

## Tradeoffs

- **1 LLM call vs per-job** - Can ask LLM to score each job individually, but that's 300 API calls per resume. One call to parse the resume is enough, then embeddings and constraints handle the rest.
- **FAISS vs hosted vector DB** - With only 300 jobs, no need for vector db. FAISS runs locally and it keeps things simple.
- **Top-50 recall** - Pulling 50 candidates from vector search gives good coverage. Scoring all 300 would be slower and most of the jobs won't be relevant anyway.
- **Semantic flag matching** - Jobs have green/red flags describing ideal candidates. Matching these with embeddings is more flexible than keyword rules, though it can over-match on generic flags like "communicates well".

## What I'd Add With More Time

- Cache resume embeddings for repeat uploads
- BM25 hybrid search for better keyword matching
- Batch endpoint for multiple resumes
- Feedback loop to tune thresholds based on user clicks

## Also read EVALUATION.md