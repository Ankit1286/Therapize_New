export interface UserQuestionnaire {
  city?: string
  insurance?: string
  session_format?: string
  max_budget_per_session?: number
  preferred_gender?: string
  preferred_language?: string
  preferred_ethnicity?: string
  age_group?: string
}

export interface SearchRequest {
  free_text: string
  questionnaire: UserQuestionnaire
  session_id: string
}

export interface TherapistResult {
  therapist_id: string
  name: string
  credentials: string[]
  city?: string
  session_formats: string[]
  accepts_insurance: string[]
  fee_range?: string
  matched_modalities: string[]
  bio_excerpt: string
  source_url: string
  composite_score: number
  modality_score: number
  semantic_score: number
  bm25_score: number
  rating_score: number
  match_explanation: string
  narrative_explanation: string
}

export interface SearchResponse {
  query_id: string
  results: TherapistResult[]
  total_candidates: number
  filtered_count: number
  extracted_intent: {
    emotional_concerns: string[]
    query_summary: string
  }
  filter_relaxation_note: string
  latency_ms: number
}

export interface FeedbackPayload {
  query_id: string
  therapist_id: string
  rating: number
  rank_position?: number
  event_type?: string
}
