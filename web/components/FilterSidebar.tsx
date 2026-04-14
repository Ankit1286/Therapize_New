'use client'

import { useEffect, useState } from 'react'
import { fetchCities, fetchLanguages } from '@/lib/api'

export interface Filters {
  sessionFormat: 'any' | 'in_person' | 'telehealth'
  city: string
  insurance: string
  maxBudget: number
  gender: string
  language: string
  ageGroup: string
}

const DEFAULT_FILTERS: Filters = {
  sessionFormat: 'telehealth',
  city: '',
  insurance: '',
  maxBudget: 0,
  gender: '',
  language: '',
  ageGroup: '',
}

const INSURANCE_OPTIONS = [
  { label: 'Any', value: '' },
  { label: 'Blue Shield', value: 'blue_shield' },
  { label: 'Blue Cross', value: 'blue_cross' },
  { label: 'Aetna', value: 'aetna' },
  { label: 'United Healthcare', value: 'united_healthcare' },
  { label: 'Cigna', value: 'cigna' },
  { label: 'Kaiser', value: 'kaiser' },
  { label: 'Magellan', value: 'magellan' },
  { label: 'Sliding Scale', value: 'sliding_scale' },
  { label: 'Self Pay', value: 'self_pay' },
]

const GENDER_OPTIONS = [
  { label: 'Any', value: '' },
  { label: 'Female', value: 'female' },
  { label: 'Male', value: 'male' },
  { label: 'Non-binary', value: 'non_binary' },
]

const AGE_OPTIONS = [
  { label: 'Any', value: '' },
  { label: 'An adult', value: 'adult' },
  { label: 'A teenager', value: 'adolescent' },
  { label: 'A child', value: 'child' },
]

interface Props {
  filters: Filters
  onChange: (f: Filters) => void
  mobileOpen: boolean
  onMobileClose: () => void
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <label className="block text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide mb-1">
      {children}
    </label>
  )
}

function Select({
  value,
  onChange,
  options,
}: {
  value: string
  onChange: (v: string) => void
  options: { label: string; value: string }[]
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full px-3 py-2 rounded-lg border border-[var(--border)] bg-white text-sm text-[var(--text)] focus:outline-none focus:ring-2 focus:ring-[var(--primary)]"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  )
}

export default function FilterSidebar({ filters, onChange, mobileOpen, onMobileClose }: Props) {
  const [cities, setCities] = useState<string[]>([])
  const [languages, setLanguages] = useState<string[]>([])

  useEffect(() => {
    fetchCities().then((c) => setCities(c))
    fetchLanguages().then((l) => setLanguages(l))
  }, [])

  function set<K extends keyof Filters>(key: K, val: Filters[K]) {
    onChange({ ...filters, [key]: val })
  }

  function reset() {
    onChange(DEFAULT_FILTERS)
  }

  const inner = (
    <div className="space-y-5 p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-bold text-[var(--text)]">Your Preferences</h2>
        <button onClick={onMobileClose} className="md:hidden text-gray-400 hover:text-gray-600 text-xl leading-none">
          ✕
        </button>
      </div>
      <p className="text-xs text-[var(--text-muted)]">Only fill in filters that are must-haves for you.</p>

      {/* Session format */}
      <div>
        <Label>Session format</Label>
        <div className="flex gap-1">
          {(['any', 'in_person', 'telehealth'] as const).map((v) => {
            const label = v === 'any' ? 'Any' : v === 'in_person' ? 'In person' : 'Online'
            return (
              <button
                key={v}
                onClick={() => set('sessionFormat', v)}
                className={`flex-1 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
                  filters.sessionFormat === v
                    ? 'bg-[var(--primary)] text-white border-[var(--primary)]'
                    : 'bg-white text-[var(--text-muted)] border-[var(--border)] hover:border-[var(--primary)]'
                }`}
              >
                {label}
              </button>
            )
          })}
        </div>
      </div>

      {/* City — only when not telehealth */}
      {filters.sessionFormat !== 'telehealth' && (
        <div>
          <Label>City</Label>
          <Select
            value={filters.city}
            onChange={(v) => set('city', v)}
            options={[{ label: '(any)', value: '' }, ...cities.map((c) => ({ label: c, value: c }))]}
          />
        </div>
      )}

      {/* Insurance */}
      <div>
        <Label>Insurance / payment</Label>
        <Select
          value={filters.insurance}
          onChange={(v) => set('insurance', v)}
          options={INSURANCE_OPTIONS}
        />
        <p className="text-xs text-[var(--text-muted)] mt-1">We&apos;ll prioritise therapists who accept your plan.</p>
      </div>

      {/* Budget */}
      <div>
        <Label>
          Max budget per session —{' '}
          <span className="text-[var(--primary)] font-semibold">
            {filters.maxBudget === 0 ? 'No limit' : `$${filters.maxBudget}`}
          </span>
        </Label>
        <input
          type="range"
          min={0}
          max={500}
          step={10}
          value={filters.maxBudget}
          onChange={(e) => set('maxBudget', Number(e.target.value))}
          className="w-full accent-[var(--primary)]"
        />
        <div className="flex justify-between text-xs text-[var(--text-muted)]">
          <span>$0</span>
          <span>$500</span>
        </div>
      </div>

      <div className="border-t-2 border-[var(--border)] pt-1" />

      {/* Gender */}
      <div>
        <Label>Therapist gender</Label>
        <Select
          value={filters.gender}
          onChange={(v) => set('gender', v)}
          options={GENDER_OPTIONS}
        />
      </div>

      {/* Language */}
      <div>
        <Label>Preferred language</Label>
        <Select
          value={filters.language}
          onChange={(v) => set('language', v)}
          options={[{ label: 'Any', value: '' }, ...languages.map((l) => ({ label: l, value: l }))]}
        />
      </div>

      {/* Age group */}
      <div>
        <Label>Therapy is for</Label>
        <Select
          value={filters.ageGroup}
          onChange={(v) => set('ageGroup', v)}
          options={AGE_OPTIONS}
        />
      </div>

      <button
        onClick={reset}
        className="text-xs text-[var(--text-muted)] hover:text-[var(--primary)] underline focus:outline-none"
      >
        Reset filters
      </button>
    </div>
  )

  return (
    <>
      {/* Desktop sticky sidebar */}
      <aside className="hidden md:block w-72 shrink-0 sticky top-6 self-start max-h-[calc(100vh-3rem)] overflow-y-auto rounded-2xl bg-[var(--sidebar-bg)] border border-[var(--border)] border-t-4 border-t-[var(--primary)]">
        {inner}
      </aside>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 flex md:hidden">
          <div className="absolute inset-0 bg-black/40" onClick={onMobileClose} />
          <aside className="relative ml-auto w-80 max-w-full h-full overflow-y-auto bg-[var(--sidebar-bg)] shadow-xl">
            {inner}
          </aside>
        </div>
      )}
    </>
  )
}
