import type { SearchRequest, SearchResponse, FeedbackPayload } from './types'

const API_BASE =
  (process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '') + '/api/v1'

export async function fetchCities(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/cities`, { next: { revalidate: 300 } })
  if (!res.ok) return []
  return res.json()
}

export async function fetchLanguages(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/languages`, { next: { revalidate: 300 } })
  if (!res.ok) return []
  return res.json()
}

export async function fetchEthnicities(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/ethnicities`, { next: { revalidate: 300 } })
  if (!res.ok) return []
  return res.json()
}

export async function fetchStats(): Promise<{ total_therapists: number }> {
  const res = await fetch(`${API_BASE}/stats`, { next: { revalidate: 3600 } })
  if (!res.ok) return { total_therapists: 0 }
  return res.json()
}

export async function search(req: SearchRequest): Promise<SearchResponse> {
  const res = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `Search failed (${res.status})`)
  }
  return res.json()
}

export async function submitFeedback(payload: FeedbackPayload): Promise<void> {
  try {
    await fetch(`${API_BASE}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  } catch {
    // fire-and-forget — never interrupt the user
  }
}
