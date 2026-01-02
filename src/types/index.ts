// Job schema based on actual recruiting platform
export interface Job {
  job_id: string;
  title: string;
  company_name: string;
  company_id: string;
  responsibilities: string; // HTML formatted
  requirements: string;
  job_category: string; // e.g., "Full-Stack", "Backend", "ML/AI", "Frontend"
  location: string;
  work_location_type: "In-person" | "Remote" | "Hybrid";
  yoe_min: number;
  YOE: string; // Human readable, e.g., "2+ Years", "5-8 Years"
  salary_min: number;
  salary_max: number;
  salary_currency: string;
  equity_min?: number;
  equity_max?: number;
  employment_type: "Full-Time" | "Part-Time" | "Contract";
  benefits?: string[];
  h1b_sponsorship: boolean;
  status: "active" | "closed";
  created_at?: string;
}

// Resume data structure
export interface Resume {
  content: string; // Plain text content from .txt files
  filename: string;
  format: "txt"; // Always "txt" since all resumes are pre-extracted text files
}

// Match request from frontend to API
export interface MatchRequest {
  resume: Resume;
}

// Individual job match result
export interface JobMatch {
  job_id: string;
  title: string;
  company: string;
  match_score: number; // 0-100 or similar
  explanation: string;
  matching_skills: string[];
  experience_alignment: string;
  // Candidates can extend this with additional fields
  [key: string]: any;
}

// Match response from candidate's service
export interface MatchResponse {
  matches: JobMatch[];
  metadata: {
    retrieval_method: string;
    reranking_method: string;
    processing_time_ms: number;
    // Candidates can add more metadata fields
    [key: string]: any;
  };
}

// Ground truth for evaluation
export interface ExpectedMatch {
  resume_file: string;
  expected_job_ids: string[]; // Top 5 expected matches
  notes?: string;
}

// Error response
export interface ErrorResponse {
  error: string;
  message: string;
  details?: any;
}
