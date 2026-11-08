const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || res.statusText)
  }
  return res.json() as Promise<T>
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || res.statusText)
  }
  return res.json() as Promise<T>
}

export type BiasOverview = {
  by_trial_block: { trial_block: number; bad_deck_rate: number }[]
  after_loss_bad_deck_rate: number
  after_win_bad_deck_rate: number
  late_bad_deck_rate: number
  early_bad_deck_rate: number
  overall_bad_deck_rate: number
  high_risk_situations: {
    situation: string
    n_trials: number
    bad_deck_rate: number
  }[]
}

export type Participant = {
  participant_id: string
  cluster: number
  tier: string
  switch_rate: number
  bad_deck_rate: number
  late_bad_deck_rate: number
  late_advantageous_rate: number
  mean_net_outcome: number
  final_cumulative: number
  loss_sensitivity: number
  deck_prefs: Record<string, number>
}

export type ParticipantDetail = Participant & {
  similar_players: Participant[]
  bias: {
    participant_id: string
    mistake_count: number
    bad_deck_rate: number
    late_bad_deck_rate: number
    timeline: {
      trial_number: number
      deck_choice: number
      is_suboptimal: number
      net_outcome: number
      after_loss: number
      cumulative_before: number
    }[]
    mistake_contexts: Record<string, number>[]
    top_error_situations: {
      situation: string
      n_trials: number
      bad_deck_rate: number
    }[]
  }
}

export type PredictResponse = {
  model: string
  deck_model: string
  switch_model: string
  predicted_deck: number
  predicted_deck_suboptimal: boolean
  deck_probabilities: {
    deck: number
    probability: number
    is_suboptimal: boolean
  }[]
  switch_probability: number
  predicted_switch: boolean
  explanation: string
}

export type TierSummary = {
  tier: string
  cluster: number
  n_participants: number
  late_advantageous_rate: number
  late_bad_deck_rate: number
  mean_net_outcome: number
}

export type MetricRow = {
  accuracy: number
  f1: number
  mean_kl?: number
  accuracy_std?: number
  f1_std?: number
  mean_kl_std?: number
  n_folds?: number
}

export type ModelMetrics = {
  deck: Record<string, MetricRow>
  switch: Record<string, MetricRow>
  loss?: string
  cv?: {
    method: string
    n_folds: number
    n_participants: number
    n_trials_evaluated: number
    note: string
  }
  n_train_trials: number
  n_test_trials: number
  situation_features: string[]
  best_models?: {
    deck: string
    switch: string
    selection_rule?: string
    deck_metrics?: MetricRow
    switch_metrics?: MetricRow
  }
}

export const api = {
  biasOverview: () => get<BiasOverview>('/bias/overview'),
  participants: (tier?: string) =>
    get<Participant[]>(tier ? `/participants?tier=${tier}` : '/participants'),
  participant: (id: string) => get<ParticipantDetail>(`/participants/${id}`),
  tiers: () => get<{ tiers: TierSummary[] }>('/tiers'),
  metrics: () => get<ModelMetrics>('/models/metrics'),
  predict: (body: Record<string, unknown>) =>
    post<PredictResponse>('/predict', body),
}

export function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

export function labelSituation(name: string): string {
  return name.replaceAll('_', ' ')
}
