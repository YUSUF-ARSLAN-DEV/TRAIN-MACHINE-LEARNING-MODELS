import { useState } from 'react'
import { predictBert } from '../../api'
import ScoreBar from '../../components/ScoreBar.jsx'

const EXAMPLES = [
  'My package never showed up and I want my money back',
  "I need to change the shipping address on my order, I moved",
  'How long until my order gets delivered',
  'I want to speak to a real person about my account',
]

export default function BertPanel() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    try {
      const data = await predictBert(text.trim())
      setResult(data)
    } catch (err) {
      setError(err.message)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="panel">
      <form className="panel-form" onSubmit={handleSubmit}>
        <label htmlFor="bert-text">Customer message</label>
        <textarea
          id="bert-text"
          rows={4}
          placeholder="Type a customer support message…"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <div className="example-chips">
          {EXAMPLES.map((example) => (
            <button
              type="button"
              key={example}
              className="chip"
              onClick={() => setText(example)}
            >
              {example}
            </button>
          ))}
        </div>

        <button type="submit" className="primary-button" disabled={loading || !text.trim()}>
          {loading ? 'Classifying…' : 'Classify intent'}
        </button>
      </form>

      {error && (
        <div className="panel-error">
          <strong>Couldn't get a prediction.</strong>
          <span>{error}</span>
        </div>
      )}

      {result && (
        <div className="panel-result">
          <div className="result-headline">
            <span className="result-label">Predicted intent</span>
            <span className="result-value">{result.intent.replaceAll('_', ' ')}</span>
          </div>
          <h3>Top 5 predictions</h3>
          {result.top_k.map((item, i) => (
            <ScoreBar
              key={item.label}
              label={item.label.replaceAll('_', ' ')}
              score={item.score}
              highlight={i === 0}
            />
          ))}
        </div>
      )}
    </div>
  )
}
