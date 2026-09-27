import { useCallback, useRef, useState } from 'react'
import { predictResnet } from '../../api'
import ScoreBar from '../../components/ScoreBar.jsx'

export default function ResnetPanel() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [notTrained, setNotTrained] = useState(false)
  const [loading, setLoading] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef(null)

  const chooseFile = useCallback((selected) => {
    if (!selected || !selected.type.startsWith('image/')) return
    setFile(selected)
    setPreviewUrl(URL.createObjectURL(selected))
    setResult(null)
    setError(null)
    setNotTrained(false)
  }, [])

  async function handleSubmit() {
    if (!file) return
    setLoading(true)
    setError(null)
    setNotTrained(false)
    try {
      const data = await predictResnet(file)
      setResult(data)
    } catch (err) {
      if (err.status === 503) {
        setNotTrained(true)
      } else {
        setError(err.message)
      }
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="panel">
      <div
        className={`dropzone${dragActive ? ' active' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragActive(true)
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragActive(false)
          chooseFile(e.dataTransfer.files?.[0])
        }}
        onClick={() => inputRef.current?.click()}
      >
        {previewUrl ? (
          <img src={previewUrl} alt="Upload preview" className="preview-image" />
        ) : (
          <>
            <p>Drag a satellite image tile here, or click to browse</p>
            <p className="dropzone-hint">64×64+ px land-use imagery works best (EuroSAT-style)</p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => chooseFile(e.target.files?.[0])}
        />
      </div>

      <button
        type="button"
        className="primary-button"
        disabled={!file || loading}
        onClick={handleSubmit}
      >
        {loading ? 'Classifying…' : 'Classify image'}
      </button>

      {notTrained && (
        <div className="panel-notice">
          <strong>Model weights not available yet.</strong>
          <span>
            This project's training script hasn't saved a checkpoint to disk yet — the UI is
            wired up and will work as soon as trained weights are exported.
          </span>
        </div>
      )}

      {error && (
        <div className="panel-error">
          <strong>Couldn't get a prediction.</strong>
          <span>{error}</span>
        </div>
      )}

      {result && (
        <div className="panel-result">
          <div className="result-headline">
            <span className="result-label">Predicted class</span>
            <span className="result-value">{result.label}</span>
          </div>
          <h3>Top predictions</h3>
          {result.top_k.map((item, i) => (
            <ScoreBar key={item.label} label={item.label} score={item.score} highlight={i === 0} />
          ))}
        </div>
      )}
    </div>
  )
}
