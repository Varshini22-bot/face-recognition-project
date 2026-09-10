const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export async function getEvaluation() {
  const response = await fetch(`${API_BASE_URL}/api/evaluation`)
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(payload.message || 'Unable to load evaluation data.')
  return payload
}
