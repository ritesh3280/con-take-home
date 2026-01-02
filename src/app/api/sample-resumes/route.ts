import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export const dynamic = 'force-dynamic';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const filename = searchParams.get('filename');

    const dataDir = path.join(process.cwd(), 'data', 'sample-resumes');

    // If filename is provided, return the content of that file
    if (filename) {
      const filePath = path.join(dataDir, filename);

      // Security check: ensure the file is within the sample-resumes directory
      const normalizedPath = path.normalize(filePath);
      if (!normalizedPath.startsWith(dataDir)) {
        return NextResponse.json(
          { error: 'Invalid file path' },
          { status: 400 }
        );
      }

      if (!fs.existsSync(filePath)) {
        return NextResponse.json(
          { error: 'File not found' },
          { status: 404 }
        );
      }

      const content = fs.readFileSync(filePath, 'utf-8');
      return NextResponse.json({ content, filename });
    }

    // Otherwise, return list of all sample resumes
    const categories = ['experienced', 'new-grads', 'less-impressive', 'remote', 'trash-resumes'];
    const resumes: Array<{ category: string; filename: string; displayName: string; path: string }> = [];

    for (const category of categories) {
      const categoryPath = path.join(dataDir, category);
      if (fs.existsSync(categoryPath)) {
        const files = fs.readdirSync(categoryPath);
        files
          .filter(file => file.endsWith('.txt'))
          .forEach(file => {
            const displayName = file
              .replace('.txt', '')
              .split('-')
              .map(word => word.charAt(0).toUpperCase() + word.slice(1))
              .join(' ');

            resumes.push({
              category,
              filename: file,
              displayName,
              path: `${category}/${file}`
            });
          });
      }
    }

    return NextResponse.json({ resumes });
  } catch (error) {
    console.error('Error fetching sample resumes:', error);
    return NextResponse.json(
      { error: 'Failed to fetch sample resumes' },
      { status: 500 }
    );
  }
}
