export default function ScoreBar({ label, score, highlight = false }) {
  const pct = Math.round(score * 1000) / 10
  return (
    <div className={`score-bar${highlight ? ' highlight' : ''}`}>
      <div className="score-bar-label">
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div className="score-bar-track">
        <div className="score-bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}
