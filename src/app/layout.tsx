import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'AI Engineer Take-Home | Resume to Job Matching',
  description: 'Build an AI-powered resume to job matching system',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <main className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
          {children}
        </main>
      </body>
    </html>
  );
}
