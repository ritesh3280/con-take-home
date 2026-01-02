'use client';

import { Trophy, AlertCircle } from 'lucide-react';
import { MatchResponse } from '@/types';
import JobCard from './JobCard';

interface MatchingResultsProps {
  results: MatchResponse;
}

export default function MatchingResults({ results }: MatchingResultsProps) {
  if (!results.matches || results.matches.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-12 text-center">
        <AlertCircle className="mx-auto h-12 w-12 text-muted-foreground" />
        <h3 className="mt-4 text-lg font-semibold">No Matches Found</h3>
        <p className="mt-2 text-sm text-muted-foreground">
          No suitable jobs were found for this resume. Try a different resume or check your
          matching service.
        </p>
      </div>
    );
  }

  return (
    <div className="w-full">
      {/* Header */}
      <div className="mb-6 text-center">
        <div className="inline-flex items-center space-x-2 rounded-full bg-yellow-100 px-4 py-2">
          <Trophy className="h-5 w-5 text-yellow-700" />
          <span className="font-semibold text-yellow-900">
            Top {results.matches.length} Matching Jobs
          </span>
        </div>
      </div>

      {/* Job Matches */}
      <div className="space-y-4">
        {results.matches.map((match, index) => (
          <JobCard key={match.job_id || index} match={match} rank={index + 1} />
        ))}
      </div>

      {/* Metadata Footer */}
      {results.metadata && (
        <div className="mt-8 rounded-lg border border-border bg-muted/30 p-4">
          <h4 className="mb-3 font-semibold">Matching Pipeline Details</h4>
          <div className="grid grid-cols-1 gap-3 text-sm md:grid-cols-3">
            {results.metadata.retrieval_method && (
              <div>
                <span className="font-medium text-muted-foreground">Retrieval Method:</span>
                <p className="mt-1">{results.metadata.retrieval_method}</p>
              </div>
            )}
            {results.metadata.reranking_method && (
              <div>
                <span className="font-medium text-muted-foreground">Reranking Method:</span>
                <p className="mt-1">{results.metadata.reranking_method}</p>
              </div>
            )}
            {results.metadata.processing_time_ms && (
              <div>
                <span className="font-medium text-muted-foreground">Processing Time:</span>
                <p className="mt-1">{results.metadata.processing_time_ms}ms</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
