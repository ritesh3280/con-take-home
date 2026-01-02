# AI Engineer Take-Home: Resume Matching System

Build an AI-powered system that matches resumes to relevant jobs from 300 job descriptions.

KEY FOCUS: Applied AI Engineering skikks 

**Time: 3 hours** | **Focus:** System design + future improvements, not perfection

---

## The Challenge

You have **300 job descriptions** and need to match them to an uploaded resume. resumes. Design a system that efficiently retrieves and ranks jobs. We expect this to take ~3–4 hours. Please don’t spend more than that — we value tradeoff decisions over completeness

**What we're evaluating:**

- System design & architecture
- AI/ML engineering skills
- Thoughtful tradeoffs (accuracy vs speed vs cost)
- Code quality & documentation
- Honest reflection on your approach

---

## Quick Start

### 1. Setup

```bash
git clone <repo-url>
cd onsite-AI-eng
npm install
```

### 2. Explore the Data

```bash
# View jobs dataset (300 jobs): data/jobs.json

# View test resumes (20 samples in folders): data/sample-resumes/
```

### 3. Build Your Service

**Create your implementation in `candidate-service`**

Your service must:

- Run on `http://localhost:8000`
- Implement the `/match` endpoint
- Accept resume text and return top-k matching jobs

**Any language/framework:** Python, TypeScript, Go, Rust, etc.

### 4. Test Your System

```bash
# Terminal 1: Start your service
cd candidate-service/your-implementation
python main.py  # or npm start, cargo run, etc.

# Terminal 2: Start the test UI
npm run dev
```

Visit `http://localhost:3000` to test with resume uploads.

---

## Dataset

### Jobs (`data/jobs.json`)

300 job descriptions from early-stage startups and tech companies:

```typescript
{
  job_id: string              // UUID
  title: string               // e.g., "Senior Backend Engineer"
  company_name: string        // Mostly YC startups (Auctor, Harper, Weave, etc.)
  responsibilities: string    // HTML formatted description
  requirements: string        // Tech requirements and qualifications
  job_category: string        // Backend, Frontend, ML/AI, Data, DevOps, Mobile
  location: string            // 60% SF, 20% NY, 10% other US, 10% remote
  yoe_min: number             // Minimum years of experience
  salary_min/max: number      // Salary range
  // ... more fields (see types/index.ts)
}
```

**Characteristics:**

- 7 categories across Backend, Frontend, Full-Stack, ML/AI, Data, DevOps, Mobile
- Salary range: $80K-$250K
- Experience levels: Junior (0-2), Mid (3-5), Senior (6-8), Staff+ (9+)
- Realistic variation: detailed and vague descriptions, different company contexts

### Test Resumes (`data/sample-resumes/`)

20 realistic resumes organized by type:

- `new-grads/` - Recent grads (bootcamp, CS degree, research)
- `experienced/` - 4-8 years (backend, frontend, ML, DevOps)
- `international/` - Non-US candidates (Peru, India, China, UK, Brazil)
- `less-impressive/` - Job hoppers, minimal experience, career switchers

Use these to test and demonstrate your system. You may also add your files

---

## API Requirements

Your service must:

1. **Run at:** `http://localhost:8000`
2. **Expose a `/match` endpoint** (POST)
3. **Accept:** Resume text content
4. **Return:** A ranked list of matching jobs with scores

**Important:** Only return jobs that are good matches. If there are no good matches, return an empty list. Cap the results at a reasonable threshold (e.g., 10-15 jobs max).

You decide the exact request/response format. The frontend will send resume content as plain text from .txt files.

---

## Submission

Submit a GitHub repo (or zip) with:

### 1. Code (Required)

Your complete implementation in `candidate-service/your-name/`:

- Main service file (`main.py`, `server.ts`, etc.)
- Dependencies file (`requirements.txt`, `package.json`, etc.)
- Any config files needed

### 3. Video Walkthrough (Required)

5-10 min Loom showing:

- Demo on a test resume
- Code walkthrough (explain key parts)
- Your reasoning and design decisions
- What you'd do differently
- How you'd improve with more time

---

## Constraints

- **Must run locally** (we'll test on our machines)
- **Any language/framework** (your choice)
- **Any LLM APIs** (OpenAI, Anthropic, Cohere, local models, etc.)
- **No third-party matching services** (build it yourself)
- **No AutoML solutions** (show your engineering)

---

## Tips

1. **Start simple** - Basic pipeline first, iterate later
2. **Test early** - Use the UI (`localhost:3000`) as you build
3. **Document as you go** - Write down design decisions
4. **Be honest** - Explain failures candidly
5. **Show your thinking** - We value reasoning over perfection

---

## Technical Details

### Project Structure

```
onsite-AI-eng/
├── data/
│   ├── jobs.json                  # 300 jobs
│   └── sample-resumes/            # 20 test resumes
├── src/                           # Next.js frontend (for testing)
├── candidate-service/             # Your implementation goes here
│   ├── main.py               # Your service
│   ├── README.md             # Setup & architecture
│   └── EVALUATION.md         # Reflection
└── README.md                     # This file
```

### Architecture Overview

**Frontend (localhost:3000):**

- Next.js UI for testing your service
- Upload resumes and see results
- Proxies requests to your backend

**Your Service (localhost:8000):**

- Implements `/match` endpoint
- Your choice of language/framework
- Returns up to top-k matching jobs

**Data Flow:**

1. User uploads resume at localhost:3000
2. Frontend sends to localhost:8000/match
3. Your service processes and returns matches
4. Frontend displays results

### Port Configuration

- **Port 3000:** Next.js frontend (we provide)
- **Port 8000:** Your matching service (you build)

The frontend proxies to your service, so you can use any language.

---

## Good Luck! 🚀

We're excited to see your approach. Remember: we're evaluating your **AI engineering thinking** -- show us how you'd build a production system.

**Focus on core functionality** - Document what you'd do with more time.
