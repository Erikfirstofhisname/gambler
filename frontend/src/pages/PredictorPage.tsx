import { useState, type FormEvent } from 'react'
import { api, type PredictResponse, pct } from '../api'

const initial = {
  trial_number: 60,
  previous_choice: 2,
  previous_net_outcome: -150,
  cumulative_net_outcome_before_trial: -400,
  recent_loss_count_5: 3,
  switched_deck_lag: 1,
  model: 'best' as const,
}

export function PredictorPage() {
  const [form, setForm] = useState(initial)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const res = await api.predict(form)
      setResult(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  function update<K extends keyof typeof form>(key: K, value: string) {
    setForm((prev) => ({
      ...prev,
      [key]:
        key === 'model'
          ? (value as typeof form.model)
          : (Number(value) as (typeof form)[K]),
    }))
  }

  return (
    <div className="grid">
      <section className="hero">
        <h1>Live situation predictor</h1>
        <p>
          Enter the current trial context. The model returns next-deck probabilities,
          switch likelihood, and whether the predicted choice is suboptimal.
        </p>
      </section>

      <div className="grid grid-2">
        <section className="panel">
          <h2>Situation features</h2>
          <p className="sub">Defaults mirror a late-session loss streak.</p>
          <form className="grid" onSubmit={onSubmit}>
            <div className="form-grid">
              <label>
                Trial number
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={form.trial_number}
                  onChange={(e) => update('trial_number', e.target.value)}
                />
              </label>
              <label>
                Previous choice
                <input
                  type="number"
                  min={0}
                  max={4}
                  value={form.previous_choice}
                  onChange={(e) => update('previous_choice', e.target.value)}
                />
              </label>
              <label>
                Previous net outcome
                <input
                  type="number"
                  value={form.previous_net_outcome}
                  onChange={(e) => update('previous_net_outcome', e.target.value)}
                />
              </label>
              <label>
                Cumulative before trial
                <input
                  type="number"
                  value={form.cumulative_net_outcome_before_trial}
                  onChange={(e) =>
                    update('cumulative_net_outcome_before_trial', e.target.value)
                  }
                />
              </label>
              <label>
                Recent loss count (5)
                <input
                  type="number"
                  min={0}
                  max={5}
                  value={form.recent_loss_count_5}
                  onChange={(e) => update('recent_loss_count_5', e.target.value)}
                />
              </label>
              <label>
                Switched last trial
                <select
                  value={form.switched_deck_lag}
                  onChange={(e) => update('switched_deck_lag', e.target.value)}
                >
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </label>
              <label>
                Model
                <select
                  value={form.model}
                  onChange={(e) => update('model', e.target.value)}
                >
                  <option value="best">Best (auto by held-out metrics)</option>
                  <option value="svm">SVM (RBF)</option>
                  <option value="pnn">Probabilistic neural net (PNN)</option>
                  <option value="kl_pnn">KL probabilistic net</option>
                </select>
              </label>
            </div>
            <button className="primary" type="submit" disabled={loading}>
              {loading ? 'Predicting…' : 'Predict behavior'}
            </button>
          </form>
          {error && <p className="error">{error}</p>}
        </section>

        <section className="panel">
          <h2>Prediction</h2>
          <p className="sub">Deck 1–2 = suboptimal; 3–4 = advantageous.</p>
          {!result && <p className="muted">Run a prediction to see results.</p>}
          {result && (
            <>
              <p className="muted" style={{ marginTop: 0 }}>
                Serving deck with <strong>{result.deck_model}</strong>, switch with{' '}
                <strong>{result.switch_model}</strong>
                {result.model === 'best' ? ' (auto-selected winners)' : ''}.
              </p>
              <div className="stat-row" style={{ gridTemplateColumns: '1fr 1fr' }}>
                <div className={`stat ${result.predicted_deck_suboptimal ? 'danger' : 'good'}`}>
                  <div className="label">Predicted deck</div>
                  <div className="value">{result.predicted_deck}</div>
                </div>
                <div className="stat warn">
                  <div className="label">Switch probability</div>
                  <div className="value">{pct(result.switch_probability)}</div>
                </div>
              </div>
              <div
                className={`callout ${result.predicted_deck_suboptimal ? 'bad' : 'ok'}`}
              >
                <strong>
                  {result.predicted_deck_suboptimal
                    ? 'Predicted choice is suboptimal'
                    : 'Predicted choice is advantageous'}
                </strong>
                <p style={{ margin: '8px 0 0' }}>{result.explanation}</p>
              </div>
              <div className="list" style={{ marginTop: 16 }}>
                {result.deck_probabilities.map((d) => (
                  <div className="list-item" key={d.deck}>
                    <div>
                      <strong>Deck {d.deck}</strong>
                      <div className="muted">
                        {d.is_suboptimal ? 'Disadvantageous' : 'Advantageous'}
                      </div>
                    </div>
                    <span className={`badge ${d.is_suboptimal ? 'bad' : 'ok'}`}>
                      {pct(d.probability)}
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  )
}
