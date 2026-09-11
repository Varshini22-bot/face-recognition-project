import { apiRequest, multipartBody } from './api'

export const peopleService = {
  getPeople: () => apiRequest('/people', {}, 'People service unavailable.'),
  registerPerson: (name, file) => apiRequest(
    '/people/register',
    { method: 'POST', body: multipartBody({ name, file }) },
    'Unable to register this person.',
  ),
  deletePerson: (id) => apiRequest(`/people/${id}`, { method: 'DELETE' }, 'Unable to delete this person.'),
}
