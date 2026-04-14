'use client'

import { useState } from 'react'
import FitBadge from './FitBadge'
import ScoreBreakdown from './ScoreBreakdown'
import type { TherapistResult } from '@/lib/types'

interface TherapistCardProps {
  therapist: TherapistResult
  rank: number
  queryId: string
  insuranceDisplay?: string
  onFeedback: (therapistId: string, rating: number) => void
}

const MEDALS = ['🥇', '🥈', '🥉']

const CHIP_COLORS = [
  'bg-[#EBF4F7] text-[#2E6E8A]',
  'bg-[#EDF7ED] text-[#2E6E46]',
  'bg-[#F3EEF8] text-[#5B3F82]',
  'bg-[#FEF3EB] text-[#8A4E2E]',
  'bg-[#EBF0F7] text-[#2E4E8A]',
]

function rankLabel(i: number): string {
  return i < 3 ? MEDALS[i] : `#${i + 1}`
}

export default function TherapistCard({
  therapist,
  rank,
  insuranceDisplay,
  onFeedback,
}: TherapistCardProps) {
  const [whyOpen, setWhyOpen] = useState(false)
  const [feedbackGiven, setFeedbackGiven] = useState<'good' | 'bad' | null>(null)

  const accentBorder =
    rank === 0 ? 'border-l-[3px] border-l-yellow-400' :
    rank === 1 ? 'border-l-[3px] border-l-gray-400' :
    rank === 2 ? 'border-l-[3px] border-l-amber-600' :
                 'border-l-[3px] border-l-[var(--primary)]'

  const formats = therapist.session_formats ?? []
  const isOnline = formats.some((f) => f.toLowerCase().includes('telehealth'))
  const formatTag = isOnline ? '💻 Online' : '🏢 In-person'

  const showInsuranceTag =
    insuranceDisplay &&
    insuranceDisplay !== 'Any' &&
    therapist.accepts_insurance?.some((ins) =>
      ins.toLowerCase().replace(/_/g, ' ').includes(insuranceDisplay.toLowerCase())
    )

  function handleFeedback(rating: number) {
    const type = rating === 5 ? 'good' : 'bad'
    setFeedbackGiven(type)
    onFeedback(therapist.therapist_id, rating)
  }

  return (
    <div className={`bg-white rounded-2xl shadow-sm hover:shadow-md transition-shadow duration-200 border border-[var(--border)] ${accentBorder} p-6`}>
      <div className="flex flex-col md:flex-row gap-6">
        {/* ── Left: main info ── */}
        <div className="flex-1 min-w-0 space-y-3">
          {/* Name + rank */}
          <div className="flex items-start gap-2">
            <span className="text-2xl leading-none mt-0.5" aria-hidden="true">
              {rankLabel(rank)}
            </span>
            <div>
              <h2 className="text-lg font-bold text-[var(--text)] leading-tight">
                {therapist.name}
                {therapist.credentials?.length > 0 && (
                  <span className="font-normal text-[var(--text-muted)] ml-1 text-base">
                    {therapist.credentials.join(', ')}
                  </span>
                )}
              </h2>

              {/* Location / format / fee */}
              <p className="text-sm text-[var(--text-muted)] mt-0.5">
                📍 {therapist.city || 'California (Telehealth)'} &nbsp;·&nbsp; {formatTag}
                &nbsp;·&nbsp; {therapist.fee_range || 'Contact for pricing'}
                {showInsuranceTag && (
                  <span className="ml-2 text-green-700 font-medium">
                    ✓ Accepts {insuranceDisplay}
                  </span>
                )}
              </p>
            </div>
          </div>

          {/* Approach chips */}
          {therapist.matched_modalities?.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              <span className="text-sm font-semibold text-[var(--text-muted)] mr-1 self-center">
                Approaches:
              </span>
              {therapist.matched_modalities.slice(0, 5).map((m, idx) => (
                <span
                  key={m}
                  className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${CHIP_COLORS[idx % CHIP_COLORS.length]}`}
                >
                  {m.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                </span>
              ))}
            </div>
          )}

          {/* Bio excerpt */}
          <p className="text-sm text-[var(--text)] italic leading-relaxed">
            &ldquo;{therapist.bio_excerpt}&rdquo;
          </p>

          {/* Why this match */}
          <div>
            <button
              onClick={() => setWhyOpen(!whyOpen)}
              className="text-sm text-[var(--primary)] font-medium hover:underline focus:outline-none"
            >
              {whyOpen ? 'Hide explanation ▲' : 'Why this match? ▼'}
            </button>
            {whyOpen && (
              <div className="mt-2 p-3 rounded-xl bg-[#F4F9FB] border border-[#C5DFE8] text-sm text-[var(--text)] leading-relaxed">
                {therapist.narrative_explanation || 'No explanation available.'}
              </div>
            )}
          </div>
        </div>

        {/* ── Right: fit, scores, actions ── */}
        <div className="flex flex-col gap-3 md:w-44 shrink-0">
          <FitBadge rank={rank} />

          <ScoreBreakdown
            modalityScore={therapist.modality_score}
            semanticScore={therapist.semantic_score}
            bm25Score={therapist.bm25_score}
          />

          {/* Feedback */}
          {feedbackGiven ? (
            <p className="text-xs text-[var(--text-muted)] text-center py-1">
              {feedbackGiven === 'good' ? '💙 Thanks for the feedback!' : '👍 Noted, thank you!'}
            </p>
          ) : (
            <div className="flex gap-2">
              <button
                onClick={() => handleFeedback(5)}
                className="flex-1 text-xs py-2 rounded-lg border border-green-300 text-green-700 bg-green-50 hover:bg-green-100 transition-colors"
              >
                This feels right
              </button>
              <button
                onClick={() => handleFeedback(1)}
                className="flex-1 text-xs py-2 rounded-lg border border-gray-200 text-gray-500 bg-gray-50 hover:bg-gray-100 transition-colors"
              >
                Not quite
              </button>
            </div>
          )}

          {/* View profile */}
          <a
            href={therapist.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="block text-center text-sm font-semibold py-2.5 px-4 rounded-xl bg-[var(--primary)] text-white hover:bg-[var(--primary-dark)] transition-colors"
          >
            View Profile →
          </a>
        </div>
      </div>
    </div>
  )
}
