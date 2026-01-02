'use client';

import { useState } from 'react';
import { AlertCircle, Sparkles } from 'lucide-react';
import ResumeUploader from '@/components/ResumeUploader';
import MatchingResults from '@/components/MatchingResults';
import LoadingSpinner from '@/components/LoadingSpinner';
import { MatchResponse, ErrorResponse } from '@/types';

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<MatchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSampleSelect = async (path: string, displayName: string) => {
    setIsLoading(true);
    setError(null);
    setResults(null);

    try {
      // Fetch sample resume content
      const sampleResponse = await fetch(`/api/sample-resumes?filename=${encodeURIComponent(path)}`);
      if (!sampleResponse.ok) {
        throw new Error('Failed to load sample resume');
      }
      const sampleData = await sampleResponse.json();
      const text = sampleData.content;

      // Call matching API
      const response = await fetch('/api/match', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          resume: {
            content: text,
            filename: displayName + '.txt',
            format: 'txt',
          },
        }),
      });

      if (!response.ok) {
        const errorData: ErrorResponse = await response.json();
        throw new Error(errorData.message || 'Failed to match resume');
      }

      const data: MatchResponse = await response.json();
      setResults(data);
    } catch (err) {
      console.error('Error matching resume:', err);
      setError(
        err instanceof Error ? err.message : 'An unexpected error occurred. Please try again.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto max-w-6xl px-4 py-12">
      {/* Header */}
      <div className="mb-12 text-center">
        <div className="mb-4 inline-flex items-center space-x-2 rounded-full bg-primary/10 px-4 py-2 text-sm font-medium text-primary">
          <Sparkles className="h-4 w-4" />
          <span>AI Engineer Take-Home Assignment</span>
        </div>
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
          Resume to Job Matching
        </h1>
        <p className="mt-4 text-lg text-muted-foreground">
          Build an AI-powered system that matches resumes to the best fitting jobs
        </p>
        <p className="mt-4 text-sm text-muted-foreground">
          See README.md for complete assignment details
        </p>
      </div>

      {/* Instructions */}
      <div className="mb-8 rounded-lg border border-blue-200 bg-blue-50 p-6">
        <h2 className="mb-3 flex items-center space-x-2 text-lg font-semibold text-blue-900">
          <AlertCircle className="h-5 w-5" />
          <span>Getting Started</span>
        </h2>
        <div className="space-y-2 text-sm text-blue-800">
          <p>
            <strong>1.</strong> Build your matching service according to the requirements in the README
          </p>
          <p>
            <strong>2.</strong> Start your service at{' '}
            <code className="rounded bg-blue-100 px-1 py-0.5 font-mono">
              http://localhost:8000
            </code>
          </p>
          <p>
            <strong>3.</strong> Select a sample resume below to test your implementation
          </p>
          <p className="mt-3 pt-3 border-t border-blue-200">
            <strong>Note:</strong> Make sure your service implements the API contract specified in the README
          </p>
        </div>
      </div>

      {/* Resume Selection Section */}
      <div className="mb-8">
        <ResumeUploader onSampleSelect={handleSampleSelect} isLoading={isLoading} />
      </div>

      {/* Loading State */}
      {isLoading && (
        <LoadingSpinner message="Finding the best matching jobs for this resume..." />
      )}

      {/* Error State */}
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-6">
          <div className="flex items-start space-x-3">
            <AlertCircle className="h-6 w-6 text-red-600" />
            <div>
              <h3 className="font-semibold text-red-900">Error</h3>
              <p className="mt-1 text-sm text-red-800">{error}</p>
              <p className="mt-3 text-sm text-red-700">
                <strong>Common issues:</strong>
              </p>
              <ul className="mt-1 list-inside list-disc space-y-1 text-sm text-red-700">
                <li>Your matching service is not running at http://localhost:8000</li>
                <li>Your service returned an error or invalid response format</li>
                <li>Network connectivity issue</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {results && !isLoading && <MatchingResults results={results} />}

      {/* Footer */}
      <footer className="mt-16 border-t border-border pt-8 text-center text-sm text-muted-foreground">
        <p>
          Built with Next.js, TypeScript, and Tailwind CSS
        </p>
        <p className="mt-2">
          For questions or issues, please refer to the README or contact your recruiter
        </p>
      </footer>
    </div>
  );
}
