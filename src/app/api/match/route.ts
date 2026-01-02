import { NextRequest, NextResponse } from 'next/server';
import { MatchRequest, MatchResponse, ErrorResponse } from '@/types';

const CANDIDATE_SERVICE_URL = process.env.CANDIDATE_SERVICE_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body: MatchRequest = await request.json();

    // Validate request
    if (!body.resume || !body.resume.content) {
      return NextResponse.json(
        {
          error: 'Bad Request',
          message: 'Resume content is required',
        } as ErrorResponse,
        { status: 400 }
      );
    }

    // Forward request to candidate's service
    const response = await fetch(`${CANDIDATE_SERVICE_URL}/match`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30000), // 30 second timeout
    });

    if (!response.ok) {
      const errorText = await response.text();
      let errorMessage = `Candidate service returned ${response.status}`;

      try {
        const errorJson = JSON.parse(errorText);
        errorMessage = errorJson.message || errorJson.error || errorMessage;
      } catch {
        if (errorText) {
          errorMessage = errorText.substring(0, 200);
        }
      }

      return NextResponse.json(
        {
          error: 'Service Error',
          message: `Matching service error: ${errorMessage}`,
          details: { status: response.status },
        } as ErrorResponse,
        { status: response.status }
      );
    }

    const data: MatchResponse = await response.json();

    // Validate response structure
    if (!data.matches || !Array.isArray(data.matches)) {
      return NextResponse.json(
        {
          error: 'Invalid Response',
          message: 'Candidate service returned invalid response format (missing matches array)',
        } as ErrorResponse,
        { status: 500 }
      );
    }

    if (!data.metadata) {
      return NextResponse.json(
        {
          error: 'Invalid Response',
          message: 'Candidate service returned invalid response format (missing metadata)',
        } as ErrorResponse,
        { status: 500 }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error('Error in match API:', error);

    if (error instanceof Error) {
      // Handle specific error types
      if (error.name === 'AbortError' || error.message.includes('timeout')) {
        return NextResponse.json(
          {
            error: 'Timeout',
            message: 'Matching service took too long to respond (>30s). Please optimize your implementation.',
          } as ErrorResponse,
          { status: 504 }
        );
      }

      if (error.message.includes('ECONNREFUSED') || error.message.includes('fetch failed')) {
        return NextResponse.json(
          {
            error: 'Service Unavailable',
            message: `Cannot connect to matching service at ${CANDIDATE_SERVICE_URL}. Make sure your service is running.`,
          } as ErrorResponse,
          { status: 503 }
        );
      }
    }

    return NextResponse.json(
      {
        error: 'Internal Server Error',
        message: 'An unexpected error occurred while processing your request',
        details: error instanceof Error ? error.message : 'Unknown error',
      } as ErrorResponse,
      { status: 500 }
    );
  }
}
