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
import { api, type Participant, type TierSummary, pct } from '../api'

export function TierBoardPage() {
  const [tiers, setTiers] = useState<TierSummary[]>([])
  const [players, setPlayers] = useState<Participant[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.tiers(), api.participants()])
      .then(([t, p]) => {
        setTiers(t.tiers)
        setPlayers(p)
      })
      .catch((err: Error) => setError(err.message))
  }, [])

  if (error) return <p className="error">{error}</p>
  if (!tiers.length) return <p className="muted">Loading tiers…</p>

  const chartData = tiers.map((t) => ({
    tier: `Tier ${t.tier}`,
    lateMistake: Number((t.late_bad_deck_rate * 100).toFixed(1)),
    lateGood: Number((t.late_advantageous_rate * 100).toFixed(1)),
  }))

  return (
    <div className="grid">
      <section className="hero">
        <h1>GMM tier board</h1>
        <p>
          Players are clustered with a Gaussian mixture on behavior features, then
          ranked A→C by late-session advantageous deck rate.
        </p>
      </section>

      <div className="grid grid-3">
        {tiers.map((t) => (
          <section className="panel" key={t.tier}>
            <h2>
              Tier <span className={`badge ${t.tier}`}>{t.tier}</span>
            </h2>
            <p className="sub">
              Cluster {t.cluster} · {t.n_participants} players
            </p>
            <div className="list">
              <div className="list-item">
                <span>Late advantageous</span>
                <strong>{pct(t.late_advantageous_rate)}</strong>
              </div>
              <div className="list-item">
                <span>Late mistake rate</span>
                <strong>{pct(t.late_bad_deck_rate)}</strong>
              </div>
              <div className="list-item">
                <span>Mean net outcome</span>
                <strong>{t.mean_net_outcome.toFixed(1)}</strong>
              </div>
            </div>
          </section>
        ))}
      </div>

      <section className="panel">
        <h2>Tier comparison</h2>
        <p className="sub">Late-session learning quality by GMM tier.</p>
        <div className="chart-box">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid stroke="rgba(232,240,236,0.08)" vertical={false} />
              <XAxis dataKey="tier" stroke="#9bb0a6" />
              <YAxis stroke="#9bb0a6" unit="%" />
              <Tooltip
                contentStyle={{
                  background: '#162420',
                  border: '1px solid rgba(232,240,236,0.12)',
                  borderRadius: 12,
                }}
              />
              <Bar dataKey="lateGood" name="Advantageous" fill="#6fbf8f" radius={[8, 8, 0, 0]} />
              <Bar dataKey="lateMistake" name="Mistakes" fill="#e07a5f" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="panel">
        <h2>All players</h2>
        <p className="sub">Sorted by tier, then late advantageous rate.</p>
        <table>
          <thead>
            <tr>
              <th>Player</th>
              <th>Tier</th>
              <th>Late good</th>
              <th>Late bad</th>
              <th>Switch rate</th>
              <th>Final cumulative</th>
            </tr>
          </thead>
          <tbody>
            {players.map((p) => (
              <tr key={p.participant_id}>
                <td>{p.participant_id}</td>
                <td>
                  <span className={`badge ${p.tier}`}>{p.tier}</span>
                </td>
                <td>{pct(p.late_advantageous_rate)}</td>
                <td>{pct(p.late_bad_deck_rate)}</td>
                <td>{pct(p.switch_rate)}</td>
                <td>{p.final_cumulative.toFixed(0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}
