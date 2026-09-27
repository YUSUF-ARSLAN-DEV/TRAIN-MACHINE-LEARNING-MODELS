import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { fetchModels } from './api'
import StatusBadge from './components/StatusBadge.jsx'

export default function App() {
  const [models, setModels] = useState([])
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchModels()
      .then(setModels)
      .catch((err) => setError(err.message))
  }, [])

  return (
    <div className="layout">
      <aside className="sidebar">
        <NavLink to="/" className="brand" end>
          <span className="brand-mark">ML</span>
          <span className="brand-text">
            Train ML Models
            <small>fine-tuning showcase</small>
          </span>
        </NavLink>

        <nav className="nav-list">
          {models.map((model) => (
            <NavLink
              key={model.id}
              to={`/models/${model.id}`}
              className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
            >
              <span className="nav-item-name">{model.shortName}</span>
              <StatusBadge status={model.status} compact />
            </NavLink>
          ))}
          {error && <p className="nav-error">Couldn't reach API: {error}</p>}
        </nav>

        <div className="sidebar-footer">
          Independent learning project — not affiliated with any coursework or employer.
        </div>
      </aside>

      <main className="content">
        <Outlet context={{ models }} />
      </main>
    </div>
  )
}
