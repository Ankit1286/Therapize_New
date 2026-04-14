'use client'

import { useEffect, useState } from 'react'
import { v4 as uuidv4 } from 'uuid'
import SearchBar from '@/components/SearchBar'
import FilterSidebar, { type Filters } from '@/components/FilterSidebar'
import TherapistCard from '@/components/TherapistCard'
import ConcernsBanner from '@/components/ConcernsBanner'
import RelaxationNote from '@/components/RelaxationNote'
import SkeletonCard from '@/components/SkeletonCard'
import { search, fetchStats, submitFeedback } from '@/lib/api'
import EmptyState from '@/components/EmptyState'
import type { SearchResponse } from '@/lib/types'

const SESSION_ID = uuidv4()

const DEFAULT_FILTERS: Filters = {
  sessionFormat: 'telehealth',
  city: '',
  insurance: '',
  maxBudget: 0,
  gender: '',
  language: '',
  ageGroup: '',
}

const INSURANCE_LABELS: Record<string, string> = {
  blue_shield: 'Blue Shield',
  blue_cross: 'Blue Cross',
  aetna: 'Aetna',
  united_healthcare: 'United Healthcare',
  cigna: 'Cigna',
  kaiser: 'Kaiser',
  magellan: 'Magellan',
  sliding_scale: 'Sliding Scale',
  self_pay: 'Self Pay',
}

export default function HomePage() {
  const [query, setQuery] = useState('')
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS)
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState<SearchResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [totalTherapists, setTotalTherapists] = useState<number>(0)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)

  useEffect(() => {
    fetchStats().then((s) => setTotalTherapists(s.total_therapists))
  }, [])

  async function handleSearch() {
    if (!query.trim() && !filters.city) {
      setError('Please enter a search query or select your city.')
      return
    }
    setError(null)
    setLoading(true)
    setData(null)

    const questionnaire: Record<string, string | number> = {}
    if (filters.city) questionnaire.city = filters.city
    if (filters.insurance) questionnaire.insurance = filters.insurance
    if (filters.maxBudget > 0) questionnaire.max_budget_per_session = filters.maxBudget
    if (filters.sessionFormat !== 'any') questionnaire.session_format = filters.sessionFormat
    if (filters.gender) questionnaire.preferred_gender = filters.gender
    if (filters.language) questionnaire.preferred_language = filters.language
    if (filters.ageGroup) questionnaire.age_group = filters.ageGroup

    try {
      const result = await search({
        free_text: query,
        questionnaire,
        session_id: SESSION_ID,
      })
      setData(result)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Search failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  function handleFeedback(queryId: string, therapistId: string, rating: number, rank: number) {
    submitFeedback({
      query_id: queryId,
      therapist_id: therapistId,
      rating,
      rank_position: rank + 1,
      event_type: 'explicit',
    })
  }

  const insuranceDisplay = filters.insurance ? INSURANCE_LABELS[filters.insurance] : undefined

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b border-[var(--border)] bg-white/80 backdrop-blur-sm sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
          <div>
            <span className="text-xl font-bold text-[var(--primary)]">💙 Therapize</span>
          </div>
          {/* Mobile filter toggle */}
          <button
            className="md:hidden text-sm font-medium text-[var(--primary)] border border-[var(--primary)] px-3 py-1.5 rounded-lg"
            onClick={() => setMobileSidebarOpen(true)}
          >
            Filters
          </button>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 flex gap-8 items-start">
        {/* Sidebar */}
        <FilterSidebar
          filters={filters}
          onChange={setFilters}
          mobileOpen={mobileSidebarOpen}
          onMobileClose={() => setMobileSidebarOpen(false)}
        />

        {/* Main content */}
        <main className="flex-1 min-w-0 space-y-6">
          <div className="rounded-2xl bg-gradient-to-br from-[#EBF6F9] via-[#F5F0EB] to-[#FAF7F4] p-8 border border-[var(--border)]">
            <h1 className="text-3xl font-bold text-[var(--text)] mb-4">
              Find a therapist who gets you
            </h1>
            <SearchBar
              value={query}
              onChange={setQuery}
              onSearch={handleSearch}
              loading={loading}
            />
          </div>

          {error && (
            <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm">
              {error}
            </div>
          )}

          {/* Empty state — shown before any search */}
          {!loading && !data && !error && (
            <EmptyState onSelect={setQuery} />
          )}

          {/* Loading skeletons */}
          {loading && (
            <div className="space-y-4">
              {[0, 1, 2].map((i) => (
                <SkeletonCard key={i} />
              ))}
            </div>
          )}

          {/* Results */}
          {data && !loading && (
            <div className="space-y-5">
              {/* Result summary */}
              <p className="text-sm text-[var(--text-muted)]">
                Showing the{' '}
                <strong className="text-[var(--text)]">{data.results.length} best matches</strong>{' '}
                from{' '}
                <strong className="text-[var(--text)]">
                  {totalTherapists ? totalTherapists.toLocaleString() : '1,000+'}
                </strong>{' '}
                therapists across California.
              </p>

              {/* Filter relaxation note */}
              {data.filter_relaxation_note && (
                <RelaxationNote message={data.filter_relaxation_note} />
              )}

              {/* Concerns banner */}
              {data.extracted_intent?.emotional_concerns?.length > 0 && (
                <ConcernsBanner concerns={data.extracted_intent.emotional_concerns} />
              )}

              {/* Therapist cards */}
              {data.results.map((therapist, i) => (
                <div key={therapist.therapist_id} className="card-enter" style={{ animationDelay: `${i * 60}ms` }}>
                  <TherapistCard
                    therapist={therapist}
                    rank={i}
                    queryId={data.query_id}
                    insuranceDisplay={insuranceDisplay}
                    onFeedback={(id, rating) => handleFeedback(data.query_id, id, rating, i)}
                  />
                </div>
              ))}

              {data.results.length === 0 && (
                <div className="text-center py-16 text-[var(--text-muted)]">
                  <p className="text-lg font-medium mb-2">No therapists found</p>
                  <p className="text-sm">Try broadening your filters or changing your search terms.</p>
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
