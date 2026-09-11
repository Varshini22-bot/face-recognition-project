import { apiRequest } from './api'

export function getEvaluation() {
  return apiRequest('/evaluation', {}, 'Unable to load evaluation data.')
}
