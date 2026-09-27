const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, options)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ?? detail
    } catch {
      // response wasn't JSON, keep statusText
    }
    const error = new Error(detail)
    error.status = res.status
    throw error
  }
  return res.json()
}

export function fetchModels() {
  return request('/api/models')
}

export function fetchModel(modelId) {
  return request(`/api/models/${modelId}`)
}

export function predictBert(text) {
  return request('/api/bert/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
}

export function predictResnet(file) {
  const formData = new FormData()
  formData.append('file', file)
  return request('/api/resnet/predict', {
    method: 'POST',
    body: formData,
  })
}
