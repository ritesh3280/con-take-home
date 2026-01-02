'use client';

import { useState, useEffect } from 'react';
import { FileText } from 'lucide-react';

interface ResumeUploaderProps {
  onSampleSelect: (path: string, displayName: string) => void;
  isLoading: boolean;
}

interface SampleResume {
  category: string;
  filename: string;
  displayName: string;
  path: string;
}

export default function ResumeUploader({ onSampleSelect, isLoading }: ResumeUploaderProps) {
  const [selectedSample, setSelectedSample] = useState<string>('');
  const [sampleResumes, setSampleResumes] = useState<SampleResume[]>([]);
  const [loadingSamples, setLoadingSamples] = useState(true);

  useEffect(() => {
    const fetchSampleResumes = async () => {
      try {
        const response = await fetch('/api/sample-resumes');
        const data = await response.json();
        setSampleResumes(data.resumes || []);
      } catch (error) {
        console.error('Error fetching sample resumes:', error);
      } finally {
        setLoadingSamples(false);
      }
    };

    fetchSampleResumes();
  }, []);

  const handleSampleSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const path = e.target.value;
    setSelectedSample(path);
    if (path) {
      const sample = sampleResumes.find(r => r.path === path);
      if (sample) {
        onSampleSelect(path, sample.displayName);
      }
    }
  };

  const groupedResumes = sampleResumes.reduce((acc, resume) => {
    if (!acc[resume.category]) {
      acc[resume.category] = [];
    }
    acc[resume.category].push(resume);
    return acc;
  }, {} as Record<string, SampleResume[]>);

  const categoryLabels: Record<string, string> = {
    'experienced': 'Experienced Engineers',
    'new-grads': 'New Graduates',
    'less-impressive': 'Less Impressive',
    'remote': 'Remote / International',
    'trash-resumes': 'Trash Resumes'
  };

  return (
    <div className="w-full">
      <div className="rounded-lg border border-border bg-card p-6">
        <div className="flex items-start space-x-3 mb-4">
          <FileText className="h-6 w-6 text-primary mt-1" />
          <div className="flex-1">
            <h3 className="font-semibold text-lg mb-2">Select a Sample Resume</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Choose from {sampleResumes.length} sample resumes to test the matching service
            </p>

            {loadingSamples ? (
              <div className="text-sm text-muted-foreground">Loading sample resumes...</div>
            ) : (
              <select
                value={selectedSample}
                onChange={handleSampleSelect}
                disabled={isLoading}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <option value="">-- Select a resume --</option>
                {Object.entries(groupedResumes).map(([category, resumes]) => (
                  <optgroup key={category} label={categoryLabels[category] || category}>
                    {resumes.map(resume => (
                      <option key={resume.path} value={resume.path}>
                        {resume.displayName}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
