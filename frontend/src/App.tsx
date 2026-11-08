import { NavLink, Route, Routes } from 'react-router-dom'
import { BiasOverviewPage } from './pages/BiasOverviewPage'
import { PlayerProfilePage } from './pages/PlayerProfilePage'
import { PredictorPage } from './pages/PredictorPage'
import { TierBoardPage } from './pages/TierBoardPage'

function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <strong>Gambler Lens</strong>
          <span>IGT suboptimal-behavior recommender</span>
        </div>
        <nav className="nav">
          <NavLink to="/" end>
            Mistake & bias
          </NavLink>
          <NavLink to="/players">Player profile</NavLink>
          <NavLink to="/predict">Live predictor</NavLink>
          <NavLink to="/tiers">Tier board</NavLink>
        </nav>
      </header>

      <Routes>
        <Route path="/" element={<BiasOverviewPage />} />
        <Route path="/players" element={<PlayerProfilePage />} />
        <Route path="/predict" element={<PredictorPage />} />
        <Route path="/tiers" element={<TierBoardPage />} />
      </Routes>
    </div>
  )
}

export default App
