'use client';

import { useState } from 'react';
import { Briefcase, MapPin, DollarSign, ChevronDown, ChevronUp, CheckCircle2 } from 'lucide-react';
import { JobMatch } from '@/types';
import { formatScore, stripHtml, truncate } from '@/lib/utils';

interface JobCardProps {
  match: JobMatch;
  rank: number;
}

export default function JobCard({ match, rank }: JobCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="rounded-lg border border-border bg-card p-6 shadow-sm transition-shadow hover:shadow-md">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-sm font-bold text-primary">
              #{rank}
            </span>
            <h3 className="text-xl font-semibold">{match.title}</h3>
          </div>
          <p className="mt-1 text-muted-foreground">{match.company}</p>
        </div>
        <div className="flex flex-col items-end">
          <div className="rounded-full bg-green-100 px-3 py-1 text-sm font-semibold text-green-700">
            {formatScore(match.match_score)} Match
          </div>
        </div>
      </div>

      {/* Explanation */}
      <div className="mt-4 rounded-md bg-blue-50 p-4">
        <p className="text-sm text-blue-900">{match.explanation}</p>
      </div>

      {/* Matching Skills */}
      {match.matching_skills && match.matching_skills.length > 0 && (
        <div className="mt-4">
          <p className="mb-2 text-sm font-medium text-muted-foreground">Matching Skills:</p>
          <div className="flex flex-wrap gap-2">
            {match.matching_skills.map((skill, index) => (
              <span
                key={index}
                className="inline-flex items-center space-x-1 rounded-full bg-green-100 px-3 py-1 text-sm text-green-700"
              >
                <CheckCircle2 className="h-3 w-3" />
                <span>{skill}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Experience Alignment */}
      {match.experience_alignment && (
        <div className="mt-3 text-sm text-muted-foreground">
          <span className="font-medium">Experience:</span> {match.experience_alignment}
        </div>
      )}

      {/* Additional metadata */}
      <div className="mt-4 flex flex-wrap gap-4 text-sm text-muted-foreground">
        {match.location && (
          <div className="flex items-center space-x-1">
            <MapPin className="h-4 w-4" />
            <span>{match.location}</span>
          </div>
        )}
        {match.salary_range && (
          <div className="flex items-center space-x-1">
            <DollarSign className="h-4 w-4" />
            <span>{match.salary_range}</span>
          </div>
        )}
        {match.job_category && (
          <div className="flex items-center space-x-1">
            <Briefcase className="h-4 w-4" />
            <span>{match.job_category}</span>
          </div>
        )}
      </div>

      {/* Expandable Details */}
      {(match.responsibilities || match.requirements) && (
        <div className="mt-4">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex w-full items-center justify-between rounded-md border border-border bg-accent px-4 py-2 text-sm font-medium hover:bg-accent/80"
          >
            <span>View Job Details</span>
            {isExpanded ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </button>

          {isExpanded && (
            <div className="mt-3 space-y-4 rounded-md border border-border bg-muted/30 p-4 text-sm">
              {match.responsibilities && (
                <div>
                  <h4 className="font-semibold">Responsibilities:</h4>
                  <p className="mt-1 text-muted-foreground whitespace-pre-wrap">
                    {truncate(stripHtml(match.responsibilities), 500)}
                  </p>
                </div>
              )}
              {match.requirements && (
                <div>
                  <h4 className="font-semibold">Requirements:</h4>
                  <p className="mt-1 text-muted-foreground whitespace-pre-wrap">
                    {truncate(match.requirements, 400)}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
