'use client'

import { useState } from 'react'

interface ScoreBreakdownProps {
  modalityScore: number
  semanticScore: number
  bm25Score: number
}

function scoreLabel(v: number): string {
  if (v >= 0.8) return 'Excellent'
  if (v >= 0.6) return 'Good'
  if (v >= 0.4) return 'Moderate'
  return 'Low'
}

function ScoreBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-[var(--text-muted)]">
        <span>{label}</span>
        <span className="font-medium text-[var(--text)]">{scoreLabel(value)}</span>
      </div>
      <div className="h-1.5 rounded-full bg-gray-100 overflow-hidden">
        <div
          className="h-full rounded-full bg-[var(--primary)] transition-all duration-500"
          style={{ width: `${Math.round(value * 100)}%` }}
        />
      </div>
    </div>
  )
}

export default function ScoreBreakdown({ modalityScore, semanticScore, bm25Score }: ScoreBreakdownProps) {
  const [open, setOpen] = useState(false)

  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="text-xs text-[var(--primary)] hover:underline focus:outline-none"
      >
        {open ? 'Hide' : 'Score breakdown'} {open ? '▲' : '▼'}
      </button>
      {open && (
        <div className="mt-3 space-y-3">
          <ScoreBar label="Specialisation match" value={modalityScore} />
          <ScoreBar label="Bio relevance" value={semanticScore} />
          <ScoreBar label="Keyword match" value={bm25Score} />
        </div>
      )}
    </div>
  )
}
