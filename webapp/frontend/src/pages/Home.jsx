import { Link, useOutletContext } from 'react-router-dom'
import StatusBadge from '../components/StatusBadge.jsx'

export default function Home() {
  const { models } = useOutletContext()
  const readyCount = models.filter((m) => m.status === 'ready').length

  return (
    <div className="home">
      <section className="hero">
        <h1>Fine-tuning showcase</h1>
        <p>
          Five projects tracking a self-directed ML fine-tuning track — computer vision, NLP,
          object detection, LLM adaptation, and multi-task serving. Pick a model below to try it
          live.
        </p>
        {models.length > 0 && (
          <p className="hero-meta">
            {readyCount} of {models.length} models live
          </p>
        )}
      </section>

      <section className="card-grid">
        {models.map((model) => (
          <Link key={model.id} to={`/models/${model.id}`} className="model-card">
            <div className="model-card-top">
              <span className="model-card-short">{model.shortName}</span>
              <StatusBadge status={model.status} />
            </div>
            <h2>{model.name}</h2>
            <p>{model.description}</p>
            <ul className="stack-list">
              {model.stack.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </Link>
        ))}
      </section>
    </div>
  )
}
