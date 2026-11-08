import { useEffect, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api, type BiasOverview, type ModelMetrics, labelSituation, pct } from '../api'

export function BiasOverviewPage() {
  const [bias, setBias] = useState<BiasOverview | null>(null)
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.biasOverview(), api.metrics()])
      .then(([b, m]) => {
        setBias(b)
        setMetrics(m)
      })
      .catch((err: Error) => setError(err.message))
  }, [])

  if (error) return <p className="error">{error}</p>
  if (!bias || !metrics) return <p className="muted">Loading bias overview…</p>

  const blockData = bias.by_trial_block.map((row) => ({
    block: `T${(row.trial_block - 1) * 20 + 1}–${row.trial_block * 20}`,
    badDeckRate: Number((row.bad_deck_rate * 100).toFixed(1)),
  }))

  return (
    <div className="grid">
      <section className="hero">
        <h1>When and why players act suboptimally</h1>
        <p>
          Disadvantageous decks (1–2) are flagged as mistakes. Aggregate curves show
          how bias shifts across the session and after wins versus losses.
        </p>
      </section>

      <div className="stat-row">
        <div className="stat danger">
          <div className="label">Overall bad-deck rate</div>
          <div className="value">{pct(bias.overall_bad_deck_rate)}</div>
        </div>
        <div className="stat warn">
          <div className="label">After a loss</div>
          <div className="value">{pct(bias.after_loss_bad_deck_rate)}</div>
        </div>
        <div className="stat good">
          <div className="label">After a win</div>
          <div className="value">{pct(bias.after_win_bad_deck_rate)}</div>
        </div>
        <div className="stat">
          <div className="label">Late trials (51–100)</div>
          <div className="value">{pct(bias.late_bad_deck_rate)}</div>
        </div>
      </div>

      <div className="grid grid-2">
        <section className="panel">
          <h2>Bad-deck rate by trial block</h2>
          <p className="sub">
            Late-block mistakes matter most — players should have learned by then.
          </p>
          <div className="chart-box">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={blockData}>
                <CartesianGrid stroke="rgba(232,240,236,0.08)" vertical={false} />
                <XAxis dataKey="block" stroke="#9bb0a6" />
                <YAxis stroke="#9bb0a6" unit="%" />
                <Tooltip
                  contentStyle={{
                    background: '#162420',
                    border: '1px solid rgba(232,240,236,0.12)',
                    borderRadius: 12,
                  }}
                  formatter={(value) => [`${value}%`, 'Bad deck rate']}
                />
                <Bar dataKey="badDeckRate" fill="#e07a5f" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="panel">
          <h2>High-risk situations</h2>
          <p className="sub">Situations ranked by suboptimal deck selection rate.</p>
          <div className="list">
            {bias.high_risk_situations.map((s) => (
              <div className="list-item" key={s.situation}>
                <div>
                  <strong>{labelSituation(s.situation)}</strong>
                  <div className="muted">{s.n_trials} trials</div>
                </div>
                <span className="badge bad">{pct(s.bad_deck_rate)}</span>
              </div>
            ))}
          </div>
        </section>
      </div>

      <section className="panel">
        <h2>Model transparency</h2>
        <p className="sub">
          {metrics.cv
            ? `${metrics.cv.n_folds}-fold leave-one-participant-out CV (${metrics.cv.n_trials_evaluated} trials each used once for testing). `
            : null}
          Auto-picks best model by mean CV accuracy → F1 → lower KL. Winners: deck{' '}
          <strong>{metrics.best_models?.deck ?? '—'}</strong>, switch{' '}
          <strong>{metrics.best_models?.switch ?? '—'}</strong>.
        </p>
        <table>
          <thead>
            <tr>
              <th>Target</th>
              <th>Model</th>
              <th>CV Accuracy</th>
              <th>CV F1</th>
              <th>CV Mean KL</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(metrics.deck).map(([name, m]) => (
              <tr key={`deck-${name}`}>
                <td>Next deck</td>
                <td>{name.replaceAll('_', ' ')}</td>
                <td>
                  {pct(m.accuracy)}
                  {m.accuracy_std != null ? ` ± ${pct(m.accuracy_std)}` : ''}
                </td>
                <td>
                  {pct(m.f1)}
                  {m.f1_std != null ? ` ± ${pct(m.f1_std)}` : ''}
                </td>
                <td>
                  {m.mean_kl?.toFixed(3) ?? '—'}
                  {m.mean_kl_std != null ? ` ± ${m.mean_kl_std.toFixed(3)}` : ''}
                </td>
                <td>
                  {metrics.best_models?.deck === name ? (
                    <span className="badge ok">best</span>
                  ) : null}
                </td>
              </tr>
            ))}
            {Object.entries(metrics.switch).map(([name, m]) => (
              <tr key={`switch-${name}`}>
                <td>Switch deck</td>
                <td>{name.replaceAll('_', ' ')}</td>
                <td>
                  {pct(m.accuracy)}
                  {m.accuracy_std != null ? ` ± ${pct(m.accuracy_std)}` : ''}
                </td>
                <td>
                  {pct(m.f1)}
                  {m.f1_std != null ? ` ± ${pct(m.f1_std)}` : ''}
                </td>
                <td>
                  {m.mean_kl?.toFixed(3) ?? '—'}
                  {m.mean_kl_std != null ? ` ± ${m.mean_kl_std.toFixed(3)}` : ''}
                </td>
                <td>
                  {metrics.best_models?.switch === name ? (
                    <span className="badge ok">best</span>
                  ) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}
