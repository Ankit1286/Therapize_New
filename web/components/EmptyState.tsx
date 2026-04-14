'use client'

const EXAMPLE_PROMPTS = [
  "I've been feeling overwhelmed at work and struggling to sleep",
  "Going through a divorce and need support with anxiety",
  "I lost someone close to me and I'm having trouble coping",
  "I have ADHD and want help with focus and daily routines",
  "Dealing with social anxiety and low self-esteem",
  "Recovering from a difficult relationship and rebuilding trust",
]

interface EmptyStateProps {
  onSelect: (prompt: string) => void
}

export default function EmptyState({ onSelect }: EmptyStateProps) {
  return (
    <div className="py-10 space-y-8">
      {/* Tagline */}
      <div className="text-center space-y-2">
        <p className="text-[var(--text-muted)] text-base">
          Not sure where to start? Try one of these:
        </p>
      </div>

      {/* Example prompt chips */}
      <div className="flex flex-wrap gap-2 justify-center">
        {EXAMPLE_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            onClick={() => onSelect(prompt)}
            className="px-4 py-2 rounded-full text-sm border border-[var(--border)] bg-white/70 text-[var(--text)] hover:border-[var(--primary)] hover:bg-white hover:text-[var(--primary)] transition-all duration-150 shadow-sm backdrop-blur-sm"
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* Reassurance copy */}
      <div className="max-w-lg mx-auto text-center space-y-3 pt-4">
        <div className="flex justify-center gap-8 text-sm text-[var(--text-muted)]">
          <span>🔒 Private &amp; confidential</span>
          <span>🤝 No account needed</span>
          <span>💙 Judgement-free</span>
        </div>
      </div>
    </div>
  )
}
