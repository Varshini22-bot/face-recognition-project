const DEPLOYED_API_BASE_URL = 'https://prediction-titled-advantage-yrs.trycloudflare.com'

function getApiBaseUrl() {
  const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim().replace(/\/+$/, '')
  const isLocalBaseUrl = /^(https?:\/\/)?(localhost|127\.0\.0\.1)(:\d+)?$/i.test(configuredBaseUrl || '')

  // Use the billing-free deployed FastAPI service when Vercel preview variables
  // are missing or still point at a stale deployment.
  if (configuredBaseUrl && !isLocalBaseUrl) return configuredBaseUrl
  if (import.meta.env.PROD) return DEPLOYED_API_BASE_URL
  return 'http://127.0.0.1:8000'
}

function buildUrl(path) {
  const normalizedPath = `/${path}`.replace(/\/+/g, '/').replace(/^\/api\//, '/')
  const baseUrl = getApiBaseUrl()

  if (baseUrl === '/api') return `${baseUrl}${normalizedPath}`
  return `${baseUrl}${normalizedPath.startsWith('/api/') ? normalizedPath : `/api${normalizedPath}`}`
}

async function parseResponse(response, fallbackMessage) {
  const payload = await response.json().catch(() => null)
  if (response.ok) return payload

  const apiError = payload?.error
  const message = apiError?.message || payload?.message || payload?.detail
  if (response.status === 413) throw new Error('The uploaded image is too large.')
  const error = new Error(message || fallbackMessage)
  if (apiError?.code) error.code = apiError.code
  error.status = response.status
  throw error
}

export async function apiRequest(path, options = {}, fallbackMessage = 'The API request failed.') {
  try {
    const response = await fetch(buildUrl(path), options)
    return await parseResponse(response, fallbackMessage)
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error('Unable to reach the VisionID API. Check the backend connection and try again.')
    }
    throw error
  }
}

export { buildUrl }

export function multipartBody(fields) {
  const body = new FormData()
  for (const [key, value] of Object.entries(fields)) body.append(key, value)
  return body
}

export function jsonHeaders() {
  return { 'Content-Type': 'application/json' }
}

export default apiRequest

