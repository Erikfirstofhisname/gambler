import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type Participant, type ParticipantDetail, labelSituation, pct } from '../api'

export function PlayerProfilePage() {
  const [list, setList] = useState<Participant[]>([])
  const [selected, setSelected] = useState('')
  const [detail, setDetail] = useState<ParticipantDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .participants()
      .then((rows) => {
        setList(rows)
        if (rows.length) setSelected(rows[0].participant_id)
      })
      .catch((err: Error) => setError(err.message))
  }, [])

  useEffect(() => {
    if (!selected) return
    api
      .participant(selected)
      .then(setDetail)
      .catch((err: Error) => setError(err.message))
  }, [selected])

  if (error) return <p className="error">{error}</p>
  if (!detail) return <p className="muted">Loading player profile…</p>

  return (
    <div className="grid">
      <section className="hero">
        <h1>Player profile</h1>
        <p>
          GMM tier, personal mistake timeline, and the situations where this person
          most often chooses disadvantageous decks.
        </p>
      </section>

      <section className="panel">
        <label>
          Participant
          <select value={selected} onChange={(e) => setSelected(e.target.value)}>
            {list.map((p) => (
              <option key={p.participant_id} value={p.participant_id}>
                {p.participant_id} · Tier {p.tier}
              </option>
            ))}
          </select>
        </label>
      </section>

      <div className="stat-row">
        <div className="stat">
          <div className="label">Tier</div>
          <div className="value">
            <span className={`badge ${detail.tier}`}>{detail.tier}</span>
          </div>
        </div>
        <div className="stat danger">
          <div className="label">Mistakes</div>
          <div className="value">{detail.bias.mistake_count}</div>
        </div>
        <div className="stat warn">
          <div className="label">Bad-deck rate</div>
          <div className="value">{pct(detail.bad_deck_rate)}</div>
        </div>
        <div className="stat">
          <div className="label">Late bad-deck rate</div>
          <div className="value">{pct(detail.late_bad_deck_rate)}</div>
        </div>
      </div>

      <div className="grid grid-2">
        <section className="panel">
          <h2>Mistake timeline</h2>
          <p className="sub">
            Each cell is a trial. Coral marks disadvantageous deck choices.
          </p>
          <div className="timeline">
            {detail.bias.timeline.map((t) => (
              <div
                key={t.trial_number}
                className={`tick ${t.is_suboptimal ? 'bad' : ''}`}
                title={`Trial ${t.trial_number}: deck ${t.deck_choice}${
                  t.is_suboptimal ? ' (suboptimal)' : ''
                }`}
              />
            ))}
          </div>
        </section>

        <section className="panel">
          <h2>Where they err</h2>
          <p className="sub">Top personal situations by suboptimal rate.</p>
          <div className="list">
            {detail.bias.top_error_situations.map((s) => (
              <div className="list-item" key={s.situation}>
                <div>
                  <strong>{labelSituation(s.situation)}</strong>
                  <div className="muted">{s.n_trials} trials</div>
                </div>
                <span className="badge bad">{pct(s.bad_deck_rate)}</span>
              </div>
            ))}
            {!detail.bias.top_error_situations.length && (
              <p className="muted">Not enough situation samples.</p>
            )}
          </div>
        </section>
      </div>

      <section className="panel">
        <h2>Similar players (same tier)</h2>
        <p className="sub">Closest late mistake rates within Tier {detail.tier}.</p>
        <table>
          <thead>
            <tr>
              <th>Player</th>
              <th>Late bad rate</th>
              <th>Mean net</th>
              <th>Loss sensitivity</th>
            </tr>
          </thead>
          <tbody>
            {detail.similar_players.map((p) => (
              <tr key={p.participant_id}>
                <td>
                  <Link to="#" onClick={() => setSelected(p.participant_id)}>
                    {p.participant_id}
                  </Link>
                </td>
                <td>{pct(p.late_bad_deck_rate)}</td>
                <td>{p.mean_net_outcome.toFixed(1)}</td>
                <td>{pct(p.loss_sensitivity)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}
