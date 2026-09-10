const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options)
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(payload.error?.message || 'People service unavailable.')
  return payload
}

export const peopleService = {
  getPeople: () => request('/api/people'),
  registerPerson: (name, file) => {
    const body = new FormData()
    body.append('name', name)
    body.append('file', file)
    return request('/api/people/register', { method: 'POST', body })
  },
  deletePerson: (id) => request(`/api/people/${id}`, { method: 'DELETE' }),
}
