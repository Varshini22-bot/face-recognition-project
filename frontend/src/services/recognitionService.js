import { apiRequest, multipartBody } from './api'

export const recognitionService = {
  recognizeImage: (file) => apiRequest(
    '/recognize',
    { method: 'POST', body: multipartBody({ file }) },
    'Unable to process this image.',
  ),
}
