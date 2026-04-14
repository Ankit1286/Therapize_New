'use client'

interface SearchBarProps {
  value: string
  onChange: (v: string) => void
  onSearch: () => void
  loading: boolean
}

export default function SearchBar({ value, onChange, onSearch, loading }: SearchBarProps) {
  function handleKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      onSearch()
    }
  }

  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium text-[var(--text-muted)]">
        Describe what you&apos;re going through in your own words — the more detail you share, the
        better your matches will be.
      </label>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKey}
        rows={4}
        placeholder="e.g. I've been having panic attacks at work and trouble sleeping. I've been going through a difficult divorce and need someone who understands anxiety and major life transitions."
        className="w-full px-4 py-3 rounded-xl border border-[var(--border)] bg-white text-[var(--text)] text-sm placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--primary)] focus:border-transparent resize-none shadow-sm"
      />
      <div className="flex items-center justify-between">
        <button
          onClick={onSearch}
          disabled={loading}
          className="px-8 py-2.5 rounded-xl bg-gradient-to-r from-[#4A90A4] to-[#3A7A8E] text-white font-semibold text-sm hover:-translate-y-0.5 hover:shadow-md active:translate-y-0 disabled:opacity-60 disabled:cursor-not-allowed transition-all duration-150 shadow-sm"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              Searching…
            </span>
          ) : (
            'Search'
          )}
        </button>
        <span className="text-xs text-[var(--text-muted)] hidden sm:block">
          Press Ctrl+Enter to search
        </span>
      </div>
    </div>
  )
}
