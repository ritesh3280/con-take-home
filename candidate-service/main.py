"""
Resume-to-Job Matching Service

4-phase pipeline:
1. LLM extracts structured resume (single call)
2. Vector recall top-N jobs via FAISS
3. Hard constraint filtering (sponsorship, seniority, location)
4. Semantic flag-evidence scoring
"""

import os
import json
import time
import re
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import faiss
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

# Config - tweak these to tune matching behavior
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_N_RECALL = 50          # how many jobs to pull from vector search
MAX_RESULTS = 20           # max jobs to return to user
GREEN_FLAG_THRESHOLD = 0.45  # min similarity to count as a green flag match
RED_FLAG_THRESHOLD = 0.45
MATCH_SCORE_THRESHOLD = 25   # min score to include in results, you can increase to tighten search

BASE_DIR = Path(__file__).parent.parent
JOBS_PATH = BASE_DIR / "data" / "jobs.json"

# Global state - loaded once at startup, reused for all requests
client: Optional[OpenAI] = None
jobs_data: list = []
job_embeddings: Optional[np.ndarray] = None
job_index: Optional[faiss.IndexFlatIP] = None  # FAISS index for fast similarity search
flag_embeddings: dict = {}  # pre-computed embeddings for job green/red flags


class ResumeInput(BaseModel):
    content: str
    filename: str = "resume.txt"
    format: str = "txt"


class MatchRequest(BaseModel):
    resume: ResumeInput


class JobMatch(BaseModel):
    job_id: str
    title: str
    company: str
    match_score: float
    explanation: str
    matching_skills: list[str]
    experience_alignment: str
    location: Optional[str] = None
    salary_range: Optional[str] = None
    job_category: Optional[str] = None
    green_flags_matched: list[str] = []
    red_flags_triggered: list[str] = []


class MatchMetadata(BaseModel):
    retrieval_method: str
    reranking_method: str
    processing_time_ms: int
    jobs_recalled: int
    jobs_after_filter: int
    resume_yoe: Optional[int] = None


class MatchResponse(BaseModel):
    matches: list[JobMatch]
    metadata: MatchMetadata


RESUME_EXTRACTION_PROMPT = """Extract key info from this resume as JSON. Be concise.

CRITICAL RULES:
1. work_pref.preference:
   - Set to "remote_only"/"hybrid_only"/"onsite_only" ONLY if resume has EXPLICIT constraint like "seeking remote", "remote only", "prefer hybrid", "must be onsite"
   - Set to "flexible" ONLY if resume EXPLICITLY indicates flexibility like "open to remote or onsite", "flexible on location", "hybrid or remote ok"
   - Set to null if NO explicit preference statement exists
2. Job location labels like "Company, Remote" or "NYC" indicate experience, NOT preference. These go in work_mode_exp only.
3. For visa/sponsorship: look for "H1B", "visa", "sponsorship", "authorized to work", "US citizen", "green card", etc.

{
  "total_yoe": number,
  "seniority": "junior|mid|senior|staff",
  "candidate_location": {"city": "str|null", "region": "str|null", "country": "str|null"},
  "work_authorization": {
    "authorized_in_us": true/false/null,
    "needs_sponsorship": true/false/null,
    "visa_mentions": ["exact quotes mentioning visa/authorization"]
  },
  "open_to_us_roles": {"value": true/false/null, "evidence": "quote or null"},
  "willing_to_relocate": {"value": true/false/null, "evidence": "quote or null"},
  "engagement_pref": {"value": "full_time|contract|either|null", "evidence": "quote or null"},
  "timezone_overlap": ["EST", "PST", etc] or [],
  "work_mode_exp": {"has_remote": bool, "has_onsite": bool, "has_hybrid": bool},
  "work_pref": {"preference": "remote_only|hybrid_only|onsite_only|flexible|null", "evidence": "EXPLICIT preference/flexibility quote or null"},
  "companies": [{"name": "str", "months": num, "is_faang": bool, "is_startup": bool, "location": "str", "top_bullets": ["1-2 key achievements"]}],
  "skills": {"languages": [], "frameworks": [], "infra": [], "specializations": []},
  "highlights": ["5-8 TECHNICAL achievements: must include WHAT you built + TECH STACK used + IMPACT/SCALE. Avoid soft claims like 'team player' unless backed by concrete example. Example: 'Built real-time notification service with Kafka + Redis serving 10M events/day'"],
  "education": [{"school": "str", "degree": "str", "is_top_tier": bool}],
  "signals": {"founder": bool, "open_source": bool, "publications": bool, "leadership": bool}
}

Resume:
"""

EXPLICIT_PREFERENCE_PATTERNS = [
    r"\b(seeking|prefer|only|must|require|looking for|need|want)\b.*\b(remote|onsite|hybrid|in-person|office)\b",
    r"\b(remote|onsite|hybrid|in-person|office)\b.*\b(only|required|preferred|must|seeking)\b",
    r"\bnot open to\b",
    r"\bwill not\b.*\b(relocate|commute|office)\b",
    r"\b(fully|100%)\s*remote\b",
]

EXPLICIT_FLEXIBILITY_PATTERNS = [
    r"\b(open to|flexible|either)\b.*\b(remote|onsite|hybrid|location)\b",
    r"\b(remote|onsite|hybrid)\b.*\b(or|and)\b.*\b(remote|onsite|hybrid|ok|okay|fine)\b",
    r"\bflexible on location\b",
    r"\bwilling to relocate\b",
]


def evidence_is_explicit_preference(evidence: str | None) -> bool:
    if not evidence:
        return False
    evidence_lower = evidence.lower()
    return any(re.search(p, evidence_lower) for p in EXPLICIT_PREFERENCE_PATTERNS)


def evidence_is_explicit_flexibility(evidence: str | None) -> bool:
    if not evidence:
        return False
    evidence_lower = evidence.lower()
    return any(re.search(p, evidence_lower) for p in EXPLICIT_FLEXIBILITY_PATTERNS)


def validate_and_fix_work_pref(structured_resume: dict) -> dict:
    """LLM sometimes hallucinates work preferences from job locations.
    This double-checks that any preference has real evidence backing it."""
    work_pref = structured_resume.get("work_pref", {})
    if not isinstance(work_pref, dict):
        structured_resume["work_pref"] = {"preference": None, "evidence": None}
        return structured_resume
    
    preference = work_pref.get("preference")
    evidence = work_pref.get("evidence")
    
    # if LLM set a preference but evidence doesn't back it up, reset to null
    if preference in ["remote_only", "hybrid_only", "onsite_only"]:
        if not evidence_is_explicit_preference(evidence):
            structured_resume["work_pref"] = {"preference": None, "evidence": None}
    elif preference == "flexible":
        if not evidence_is_explicit_flexibility(evidence):
            structured_resume["work_pref"] = {"preference": None, "evidence": None}
    
    return structured_resume


def validate_structured_resume(structured_resume: dict) -> dict:
    return validate_and_fix_work_pref(structured_resume)


def clean_html(html_text: str) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text(separator=" ")
    return re.sub(r'\s+', ' ', text).strip()


def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(response.data[0].embedding, dtype=np.float32)


def get_embeddings_batch(texts: list[str]) -> np.ndarray:
    """Batch embed texts - way faster than one at a time.
    OpenAI allows up to 2048 texts per request."""
    if not texts:
        return np.array([])
    
    all_embeddings = []
    batch_size = 2000
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings = [np.array(d.embedding, dtype=np.float32) for d in response.data]
        all_embeddings.extend(embeddings)
    
    return np.array(all_embeddings, dtype=np.float32)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize so we can use dot product as cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid div by zero
    return vectors / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def build_job_text(job: dict) -> str:
    parts = [
        job.get("title", ""),
        job.get("job_category", ""),
        job.get("requirements", ""),
        clean_html(job.get("responsibilities", "")),
    ]
    return " ".join(filter(None, parts))


def load_and_index_jobs():
    """One-time setup: load all jobs, embed them, build FAISS index.
    Also pre-embeds all green/red flags so we don't have to do it per-request."""
    global jobs_data, job_embeddings, job_index, flag_embeddings
    
    print("Loading jobs...")
    with open(JOBS_PATH, "r") as f:
        jobs_data = json.load(f)
    print(f"  {len(jobs_data)} jobs loaded")
    
    print("Building embeddings...")
    job_texts = [build_job_text(job) for job in jobs_data]
    job_embeddings = get_embeddings_batch(job_texts)
    job_embeddings = normalize_vectors(job_embeddings)
    
    dimension = job_embeddings.shape[1]
    job_index = faiss.IndexFlatIP(dimension)
    job_index.add(job_embeddings)
    print(f"  FAISS index built ({job_index.ntotal} vectors)")
    
    print("Pre-embedding flags...")
    all_flag_texts = []
    flag_mapping = []
    
    for idx, job in enumerate(jobs_data):
        for flag in job.get("greenFlags", []):
            all_flag_texts.append(flag)
            flag_mapping.append((idx, "green", flag))
        for flag in job.get("redFlags", []):
            all_flag_texts.append(flag)
            flag_mapping.append((idx, "red", flag))
    
    if all_flag_texts:
        all_flag_embeddings = get_embeddings_batch(all_flag_texts)
        all_flag_embeddings = normalize_vectors(all_flag_embeddings)
        
        for i, (job_idx, flag_type, flag_text) in enumerate(flag_mapping):
            job_id = jobs_data[job_idx]["job_id"]
            if job_id not in flag_embeddings:
                flag_embeddings[job_id] = {"green": [], "red": []}
            flag_embeddings[job_id][flag_type].append((flag_text, all_flag_embeddings[i]))
    
    print(f"  {len(all_flag_texts)} flags embedded")


def extract_resume_structure(resume_text: str) -> dict:
    """Use LLM to pull out structured info from resume text.
    This is the only LLM call per request - everything else uses embeddings."""
    truncated = resume_text[:6000] if len(resume_text) > 6000 else resume_text
    
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Extract resume info as JSON. Be concise. No markdown."},
            {"role": "user", "content": RESUME_EXTRACTION_PROMPT + truncated}
        ],
        temperature=0,
        max_tokens=1500,
        response_format={"type": "json_object"}
    )
    
    structured = json.loads(response.choices[0].message.content)
    return validate_structured_resume(structured)


def recall_top_jobs(resume_text: str, structured_resume: dict, top_n: int = TOP_N_RECALL) -> list[tuple[int, float]]:
    """Fast vector search to get candidate jobs. We combine resume text + skills
    into one query embedding and find nearest neighbors in FAISS."""
    skills = structured_resume.get("skills", {})
    query_parts = [
        resume_text[:2000],
        " ".join(skills.get("languages", [])),
        " ".join(skills.get("frameworks", [])),
        " ".join(skills.get("infra", [])),
        " ".join(skills.get("specializations", [])),
    ]
    query_text = " ".join(filter(None, query_parts))
    
    query_embedding = get_embedding(query_text)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = query_embedding.reshape(1, -1)
    
    scores, indices = job_index.search(query_embedding, top_n)
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]


def normalize_yoe_str(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", s)


def parse_yoe_band(yoe_str: str, yoe_min: int) -> tuple[int, Optional[int]]:
    if not yoe_str:
        return (yoe_min or 0, None)
    
    range_match = re.search(r"(\d+)\s*-\s*(\d+)", yoe_str)
    if range_match:
        return (int(range_match.group(1)), int(range_match.group(2)))
    
    plus_match = re.search(r"(\d+)\s*\+", yoe_str)
    if plus_match:
        return (int(plus_match.group(1)), None)
    
    return (yoe_min or 0, None)


def passes_constraints(job: dict, structured_resume: dict) -> tuple[bool, str]:
    """Hard filters - if any of these fail, job is excluded entirely.
    Checks: visa sponsorship, experience level, remote preferences."""
    work_auth = structured_resume.get("work_authorization", {})
    needs_sponsorship = work_auth.get("needs_sponsorship") if isinstance(work_auth, dict) else None
    if needs_sponsorship is None:
        needs_sponsorship = structured_resume.get("requires_visa")
    
    job_sponsors = job.get("h1b_sponsorship", False)
    if needs_sponsorship is True and not job_sponsors:
        return False, "Requires visa sponsorship but job doesn't offer it"
    
    resume_seniority = (structured_resume.get("seniority") or "").lower()
    resume_yoe = structured_resume.get("total_yoe", 0) or 0
    job_title = (job.get("title") or "").lower()
    job_yoe_min = job.get("yoe_min", 0) or 0
    job_yoe_norm = normalize_yoe_str(job.get("YOE", ""))
    band_min, band_max = parse_yoe_band(job_yoe_norm, job_yoe_min)
    has_explicit_band = bool(job_yoe_norm) or job_yoe_min > 0

    if resume_yoe + 1 < band_min:
        return False, f"Insufficient experience ({resume_yoe} years vs {band_min} required)"

    if not has_explicit_band:
        if resume_seniority == "junior" and resume_yoe <= 2 and any(k in job_title for k in ["senior", "staff", "principal", "lead"]):
            return False, f"Junior candidate not qualified for senior-level role: {job.get('title')}"
        if any(k in job_title for k in ["staff", "principal"]) and resume_yoe < 8:
            return False, f"Insufficient experience for Staff/Principal role ({resume_yoe} years, need 8+)"
        if "senior" in job_title and "staff" not in job_title and resume_yoe < 4 and resume_seniority in ["junior", ""]:
            return False, f"Insufficient experience for Senior role ({resume_yoe} years, need 4+)"
    
    work_pref_obj = structured_resume.get("work_pref", {})
    if isinstance(work_pref_obj, dict):
        work_pref = work_pref_obj.get("preference")
        work_pref_evidence = work_pref_obj.get("evidence")
    else:
        work_pref = work_pref_obj
        work_pref_evidence = None
    
    job_location_type = job.get("work_location_type", "")
    if work_pref == "remote_only" and evidence_is_explicit_preference(work_pref_evidence):
        if job_location_type == "In-person":
            return False, "Candidate explicitly requires remote but job is in-person only"
    
    return True, ""


def build_resume_evidence_chunks(structured_resume: dict) -> list[str]:
    """Turn structured resume into text chunks we can match against job flags.
    Each chunk is something like 'Backend engineering with Python, Django'."""
    chunks = []
    chunks.extend(structured_resume.get("highlights", []))
    
    skills = structured_resume.get("skills", {})
    languages = skills.get("languages", [])
    frameworks = skills.get("frameworks", [])
    infra = skills.get("infra", [])
    specializations = skills.get("specializations", [])
    
    if languages:
        chunks.append(f"Experienced with {', '.join(languages)}")
    if frameworks:
        chunks.append(f"Built systems using {', '.join(frameworks)}")
    if infra:
        chunks.append(f"Infrastructure experience with {', '.join(infra)}")
    if specializations:
        chunks.append(f"Specialized in {', '.join(specializations)}")
    
    backend_skills = [s for s in (languages + frameworks + infra) if s.lower() in 
                      ["java", "python", "go", "rust", "c++", "spring", "django", "flask", "fastapi", 
                       "node", "express", "postgresql", "mysql", "mongodb", "redis", "kafka", "rabbitmq"]]
    if backend_skills:
        chunks.append(f"Backend engineering with {', '.join(backend_skills[:5])}")
    
    frontend_skills = [s for s in (languages + frameworks) if s.lower() in 
                       ["javascript", "typescript", "react", "vue", "angular", "next.js", "nextjs", "svelte"]]
    if frontend_skills:
        chunks.append(f"Frontend development with {', '.join(frontend_skills[:5])}")
    
    cloud_skills = [s for s in infra if s.lower() in 
                    ["aws", "gcp", "azure", "kubernetes", "k8s", "docker", "terraform", "cloudformation"]]
    if cloud_skills:
        chunks.append(f"Cloud and infrastructure with {', '.join(cloud_skills)}")
    
    for company in structured_resume.get("companies", []):
        name = company.get("name", "")
        if company.get("is_faang"):
            chunks.append(f"Worked at top-tier company {name}")
        if company.get("is_startup"):
            chunks.append(f"Startup experience at {name}")
        top_bullets = company.get("top_bullets", [])
        chunks.extend(top_bullets[:2])
    
    for edu in structured_resume.get("education", []):
        edu_text = f"{edu.get('degree', '')} from {edu.get('school', '')}"
        if edu.get("is_top_tier"):
            edu_text += " (top-tier program)"
        chunks.append(edu_text)
    
    signals = structured_resume.get("signals", {})
    if signals.get("founder"):
        chunks.append("Founder or co-founder experience")
    if signals.get("open_source"):
        chunks.append("Open source contributions")
    if signals.get("publications"):
        chunks.append("Published research papers")
    if signals.get("leadership"):
        chunks.append("Engineering leadership experience")
    
    work_mode = structured_resume.get("work_mode_exp", {})
    if work_mode.get("has_remote"):
        chunks.append("Experience working remotely")
    
    chunks = [c for c in chunks if c and len(c) > 10]
    return chunks[:60]


def green_flag_points(sim: float) -> int:
    """Higher similarity = more points. Tiered so strong matches count more."""
    if sim >= 0.75:
        return 12
    elif sim >= 0.65:
        return 9
    elif sim >= 0.55:
        return 6
    return 3


def red_flag_penalty(sim: float) -> int:
    """Red flags hurt more than green flags help - being a bad fit matters."""
    if sim >= 0.75:
        return 18
    elif sim >= 0.65:
        return 13
    elif sim >= 0.55:
        return 9
    return 5


def score_job_with_flags(
    job: dict, 
    base_similarity: float,
    resume_chunks: list[str],
    chunk_embeddings: np.ndarray,
    structured_resume: dict
) -> tuple[float, list[str], list[str], str]:
    """Main scoring logic. Combines:
    - base similarity from vector search (0-60 pts)
    - green flag matches (up to +25 pts)
    - red flag penalties (up to -35 pts)
    - bonuses/penalties for YOE, company background, timezone
    """
    job_id = job["job_id"]
    job_flags = flag_embeddings.get(job_id, {"green": [], "red": []})
    
    green_matches = []
    red_matches = []
    
    for flag_text, flag_emb in job_flags["green"]:
        if len(chunk_embeddings) > 0:
            similarities = [cosine_similarity(flag_emb, chunk_emb) for chunk_emb in chunk_embeddings]
            max_sim = max(similarities) if similarities else 0
            if max_sim >= GREEN_FLAG_THRESHOLD:
                best_chunk_idx = similarities.index(max_sim)
                green_matches.append((flag_text, resume_chunks[best_chunk_idx], max_sim))
    
    for flag_text, flag_emb in job_flags["red"]:
        if len(chunk_embeddings) > 0:
            similarities = [cosine_similarity(flag_emb, chunk_emb) for chunk_emb in chunk_embeddings]
            max_sim = max(similarities) if similarities else 0
            if max_sim >= RED_FLAG_THRESHOLD:
                best_chunk_idx = similarities.index(max_sim)
                red_matches.append((flag_text, resume_chunks[best_chunk_idx], max_sim))
    
    ideal_companies = [c.lower() for c in job.get("idealCompanies", [])]
    resume_companies = [company.get("name", "").lower() for company in structured_resume.get("companies", [])]
    company_overlap = any(
        any(ideal in resume_co or resume_co in ideal for ideal in ideal_companies)
        for resume_co in resume_companies if resume_co
    )
    
    resume_yoe = structured_resume.get("total_yoe", 0) or 0
    job_yoe_min = job.get("yoe_min", 0)
    yoe_gap = max(0, job_yoe_min - resume_yoe)
    
    job_yoe_norm = normalize_yoe_str(job.get("YOE", ""))
    seniority = structured_resume.get("seniority", "")
    job_min, job_max = parse_yoe_band(job_yoe_norm, job_yoe_min)
    overqualification_penalty = 0
    
    is_entry_level = (job_max is not None and job_max <= 2) or "0-2" in job_yoe_norm
    
    if is_entry_level and seniority in ["senior", "staff"]:
        overqualification_penalty = 15
    elif job_max is not None:
        if resume_yoe >= job_max + 5:
            overqualification_penalty = 10
        elif resume_yoe >= job_max + 3:
            overqualification_penalty = 5
    
    candidate_tz = structured_resume.get("timezone_overlap", [])
    job_location = job.get("location", "")
    timezone_bonus = 0
    if candidate_tz:
        if any(tz in ["EST", "ET", "Eastern"] for tz in candidate_tz):
            if any(loc in job_location for loc in ["NY", "NYC", "Boston", "MA", "FL", "GA", "Atlanta"]):
                timezone_bonus = 3
        if any(tz in ["PST", "PT", "Pacific"] for tz in candidate_tz):
            if any(loc in job_location for loc in ["CA", "SF", "LA", "Seattle", "WA"]):
                timezone_bonus = 3
    
    # scoring formula - base similarity gives 0-60 pts
    score = base_similarity * 60
    
    # add green flag bonus (capped at 25)
    max_green_bonus = 25
    green_total = sum(green_flag_points(sim) for _, _, sim in green_matches)
    score += min(green_total, max_green_bonus)
    
    # subtract red flag penalty (capped at 35)
    max_red_penalty = 35
    red_total = sum(red_flag_penalty(sim) for _, _, sim in red_matches)
    score -= min(red_total, max_red_penalty)
    
    # small bonus if candidate worked at companies the job considers ideal
    if company_overlap and base_similarity >= 0.45:
        score += 5
    
    # penalize experience gaps and overqualification
    score -= yoe_gap * 3
    score -= overqualification_penalty
    score += timezone_bonus
    score = max(0, min(100, score))
    
    explanation_parts = []
    if green_matches:
        explanation_parts.append(f"✓ Matches {len(green_matches)} key criteria: {', '.join([g[0][:50] for g in green_matches[:3]])}")
    if company_overlap:
        explanation_parts.append("✓ Background aligns with ideal candidate profile")
    if timezone_bonus > 0:
        explanation_parts.append("✓ Good timezone overlap")
    if red_matches:
        explanation_parts.append(f"⚠ Potential concerns: {len(red_matches)} flags triggered")
    if yoe_gap > 0:
        explanation_parts.append(f"⚠ Experience gap: {yoe_gap} years under minimum")
    if overqualification_penalty > 0:
        explanation_parts.append(f"⚠ May be overqualified for this role")
    
    explanation = " | ".join(explanation_parts) if explanation_parts else "Good general match based on skills and experience"
    
    return score, [g[0] for g in green_matches], [r[0] for r in red_matches], explanation


def match_resume_to_jobs(resume_text: str) -> tuple[list[JobMatch], MatchMetadata]:
    """Main pipeline - runs all 4 phases and returns ranked matches."""
    start_time = time.time()
    
    print("\n[Phase 1] Extracting resume structure...")
    structured_resume = extract_resume_structure(resume_text)

    # print statemtent if wanna print structured resume output
    # try:
    #     print("[DEBUG] structured_resume:\n" + json.dumps(structured_resume, indent=2))
    # except Exception:
    #     print("[DEBUG] structured_resume (non-serializable)\n", structured_resume)
    
    resume_yoe = structured_resume.get("total_yoe")
    seniority = structured_resume.get("seniority", "unknown")
    print(f"  Candidate: {seniority}, {resume_yoe} YOE")
    
    print("[Phase 2] Vector recall...")
    recalled = recall_top_jobs(resume_text, structured_resume, TOP_N_RECALL)
    jobs_recalled = len(recalled)
    print(f"  {jobs_recalled} candidates retrieved")
    
    print("[Phase 3] Constraint filtering...")
    filtered_jobs = []
    for job_idx, base_score in recalled:
        job = jobs_data[job_idx]
        passes, _ = passes_constraints(job, structured_resume)
        if passes:
            filtered_jobs.append((job_idx, base_score))
    jobs_after_filter = len(filtered_jobs)
    print(f"  {jobs_after_filter}/{jobs_recalled} passed constraints")
    
    print("[Phase 4] Flag scoring...")
    resume_chunks = build_resume_evidence_chunks(structured_resume)
    if resume_chunks:
        chunk_embeddings = get_embeddings_batch(resume_chunks)
        chunk_embeddings = normalize_vectors(chunk_embeddings)
    else:
        chunk_embeddings = np.array([])
    
    skills = structured_resume.get("skills", {})
    all_skills = skills.get("languages", []) + skills.get("frameworks", []) + skills.get("infra", [])
    
    scored_jobs = []
    for job_idx, base_sim in filtered_jobs:
        job = jobs_data[job_idx]
        score, green_flags, red_flags, explanation = score_job_with_flags(
            job, base_sim, resume_chunks, chunk_embeddings, structured_resume
        )
        
        if score >= MATCH_SCORE_THRESHOLD:
            job_text = build_job_text(job).lower()
            matching_skills = [s for s in all_skills if s.lower() in job_text]
            job_yoe = job.get("YOE", "")
            exp_alignment = f"{seniority.title()} with {resume_yoe or 0} years | Job requires {job_yoe}"
            
            scored_jobs.append(JobMatch(
                job_id=job["job_id"],
                title=job["title"],
                company=job["company_name"],
                match_score=round(score, 1),
                explanation=explanation,
                matching_skills=matching_skills[:10],
                experience_alignment=exp_alignment,
                location=job.get("location"),
                salary_range=f"${job.get('salary_min', 0):,} - ${job.get('salary_max', 0):,}",
                job_category=job.get("job_category"),
                green_flags_matched=green_flags[:5],
                red_flags_triggered=red_flags[:3]
            ))
    
    scored_jobs.sort(key=lambda x: x.match_score, reverse=True)
    top_matches = scored_jobs[:MAX_RESULTS]
    
    processing_time = int((time.time() - start_time) * 1000)
    print(f"  {len(top_matches)} matches returned in {processing_time}ms\n")
    
    metadata = MatchMetadata(
        retrieval_method="faiss_embedding_similarity",
        reranking_method="semantic_flag_evidence_matching",
        processing_time_ms=processing_time,
        jobs_recalled=jobs_recalled,
        jobs_after_filter=jobs_after_filter,
        resume_yoe=resume_yoe
    )
    
    return top_matches, metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    load_and_index_jobs()
    print("\nService ready!\n")
    
    yield


app = FastAPI(
    title="Resume Matching Service",
    description="AI-powered resume to job matching",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "jobs_indexed": len(jobs_data)}


@app.post("/match", response_model=MatchResponse)
async def match_resume(request: MatchRequest):
    if not request.resume.content or len(request.resume.content.strip()) < 50:
        raise HTTPException(status_code=400, detail="Resume content too short")
    
    try:
        matches, metadata = match_resume_to_jobs(request.resume.content)
        return MatchResponse(matches=matches, metadata=metadata)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
