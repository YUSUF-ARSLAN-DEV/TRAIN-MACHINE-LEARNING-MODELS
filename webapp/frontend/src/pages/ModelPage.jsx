import { useParams, useOutletContext } from 'react-router-dom'
import ModelHeader from '../components/ModelHeader.jsx'
import BertPanel from './panels/BertPanel.jsx'
import ResnetPanel from './panels/ResnetPanel.jsx'
import ComingSoonPanel from './panels/ComingSoonPanel.jsx'

const PANELS = {
  'bert-intent': BertPanel,
  'resnet18-eurosat': ResnetPanel,
}

export default function ModelPage() {
  const { modelId } = useParams()
  const { models } = useOutletContext()
  const model = models.find((m) => m.id === modelId)

  if (!model) {
    return <p className="loading">Loading model…</p>
  }

  const Panel = model.status === 'ready' ? PANELS[model.id] : ComingSoonPanel

  return (
    <div className="model-page">
      <ModelHeader model={model} />
      {Panel ? <Panel model={model} /> : <ComingSoonPanel model={model} />}
    </div>
  )
}
