const LABELS = {
  ready: 'Live',
  planned: 'Coming soon',
}

export default function StatusBadge({ status, compact = false }) {
  const label = LABELS[status] ?? status
  return (
    <span className={`status-badge status-${status}${compact ? ' compact' : ''}`}>
      <span className="status-dot" />
      {!compact && label}
    </span>
  )
}
