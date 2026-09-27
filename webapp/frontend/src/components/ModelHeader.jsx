import StatusBadge from './StatusBadge.jsx'

export default function ModelHeader({ model }) {
  return (
    <header className="model-header">
      <div className="model-header-top">
        <h1>{model.name}</h1>
        <StatusBadge status={model.status} />
      </div>
      <p className="model-description">{model.description}</p>
      <ul className="stack-list">
        {model.stack.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </header>
  )
}
