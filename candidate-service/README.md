## Overview
This service matches an uploaded resume to relevant jobs using a 4-stage retrieval + reranking pipeline.  
It makes **one LLM call per request** (resume → structured JSON). All job and flag embeddings are precomputed at startup.

See EVALUATION.md for design decisions, tradeoffs, and my personal reflection.

## Loom Video:
https://www.loom.com/share/f79b9f61eccb43e09c915f621aaa6262

## How It Works

4 steps per resume:

1. **Extract** - one LLM call turns the resume into structured data (skills, YOE, companies, etc)
2. **Recall** - vector search grabs top 50 similar jobs
3. **Filter** - drop jobs that don't fit (visa issues, seniority mismatch, location, etc)
4. **Score** - score each remaining job based on green/red flags and embeddings

Everything is pre-computed at startup so the only API call per request is the one LLM extraction.

### Example structured resume JSON

This is a real example of the structured_resume output produced in phase 1:

```json
{
  "total_yoe": 7,
  "seniority": "senior",
  "candidate_location": {
    "city": "Portland",
    "region": "OR",
    "country": "USA"
  },
  "work_authorization": {
    "authorized_in_us": true,
    "needs_sponsorship": false,
    "visa_mentions": []
  },
  "open_to_us_roles": {
    "value": true,
    "evidence": null
  },
  "willing_to_relocate": {
    "value": false,
    "evidence": null
  },
  "engagement_pref": {
    "value": "full_time",
    "evidence": null
  },
  "timezone_overlap": [],
  "work_mode_exp": {
    "has_remote": true,
    "has_onsite": true,
    "has_hybrid": false
  },
  "work_pref": {
    "preference": null,
    "evidence": null
  },
  "companies": [
    {
      "name": "HashiCorp",
      "months": 25,
      "is_faang": false,
      "is_startup": false,
      "location": "Remote",
      "top_bullets": [
        "Lead platform engineering efforts for internal developer platform serving 500+ engineers",
        "Reduced cloud costs by 35% through resource right-sizing and spot instance usage"
      ]
    },
    {
      "name": "New Relic",
      "months": 30,
      "is_faang": false,
      "is_startup": false,
      "location": "Portland, OR",
      "top_bullets": [
        "Built CI/CD pipelines reducing deployment time from 2 hours to 15 minutes",
        "Migrated services from EC2 to Kubernetes improving resource utilization by 40%"
      ]
    },
    {
      "name": "Puppet",
      "months": 19,
      "is_faang": false,
      "is_startup": false,
      "location": "Portland, OR",
      "top_bullets": [
        "Implemented chaos engineering practices using Gremlin to test system resilience",
        "Optimized database performance reducing query latency by 60%"
      ]
    }
  ],
  "skills": {
    "languages": [
      "Python",
      "Bash",
      "Go",
      "YAML"
    ],
    "frameworks": [
      "Kubernetes",
      "Docker",
      "Helm"
    ],
    "infra": [
      "AWS",
      "GCP",
      "Terraform",
      "Ansible"
    ],
    "specializations": [
      "CI/CD",
      "Monitoring & Logging",
      "Cost Optimization"
    ]
  },
  "highlights": [
    "Built real-time notification service with Kubernetes + AWS, serving 500+ engineers",
    "Created disaster recovery procedures reducing downtime risk",
    "Led migration of 200+ services to Kubernetes with zero downtime",
    "Developed CLI tool for Kubernetes cost optimization, saving users $15K/month",
    "Implemented observability stack with Prometheus and Grafana, improving incident response"
  ],
  "education": [
    {
      "school": "Oregon State University",
      "degree": "Bachelor of Science in Computer Science",
      "is_top_tier": true
    }
  ],
  "signals": {
    "founder": false,
    "open_source": true,
    "publications": true,
    "leadership": true
  }
}
```

## Scoring

Each job gets a score out of 100:

- **Base similarity** (0-60) - how well resume matches job description
- **Green flags** (up to +25) - bonus for matching what the job wants
- **Red flags** (up to -35) - penalty for matching what they want to avoid
- **Company overlap** (+5) - worked at companies the job likes
- **YOE gap** (-3 per year) - under the minimum experience
- **Overqualification** (-5 to -15) - senior people applying to junior roles

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
