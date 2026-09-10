const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '')

export const recognitionService = {
  async recognizeImage(file) {
    const body = new FormData()
    body.append('file', file)
    const response = await fetch(`${API_BASE_URL}/api/recognize`, { method: 'POST', body })
    const payload = await response.json().catch(() => ({}))
    if (!response.ok) throw new Error(payload.error?.message || 'Unable to process this image.')
    return payload
  },
}
