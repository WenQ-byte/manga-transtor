const BASE = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    let detail = `请求失败 (${res.status})`
    try {
      const data = await res.json()
      if (data.detail) detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)
    } catch {
      /* ignore */
    }
    const err = new Error(detail)
    err.status = res.status
    throw err
  }
  const contentType = res.headers.get('content-type') || ''
  if (contentType.includes('application/json')) return res.json()
  return res
}

export function createTranslateTask(file, sourceLang, targetLang) {
  const form = new FormData()
  form.append('file', file)
  const query = new URLSearchParams({ source_lang: sourceLang, target_lang: targetLang })
  return request(`/translate?${query}`, { method: 'POST', body: form })
}

export function createBatchTranslateTask(files, sourceLang, targetLang) {
  const form = new FormData()
  files.forEach((file) => form.append('files[]', file))
  form.append('source_lang', sourceLang)
  form.append('target_lang', targetLang)
  return request('/translate/batch', { method: 'POST', body: form })
}

export function getBatchTaskStatus(taskIds) {
  return request('/translate/batch/status', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_ids: taskIds }),
  })
}

export async function downloadBatchZip(taskIds) {
  const response = await request('/translate/batch/zip', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_ids: taskIds }),
  })
  return response.blob()
}

export function getTaskStatus(taskId) {
  return request(`/translate/${taskId}/status`)
}

export function getTaskResultUrl(taskId) {
  return `${BASE}/translate/${taskId}/result`
}

export function retryTranslateTask(taskId) {
  return request(`/translate/${taskId}/retry`, { method: 'POST' })
}

export async function downloadTaskResult(taskId) {
  const response = await request(`/translate/${taskId}/result`)
  const disposition = response.headers.get('content-disposition') || ''
  const encoded = disposition.match(/filename\*=utf-8''([^;]+)/i)?.[1]
  const plain = disposition.match(/filename="?([^";]+)"?/i)?.[1]
  let filename = `translated_${taskId}.png`
  try {
    filename = encoded ? decodeURIComponent(encoded) : (plain || filename)
  } catch {
    filename = plain || filename
  }
  return { blob: await response.blob(), filename }
}

export function deleteTask(taskId) {
  return request(`/translate/${taskId}`, { method: 'DELETE' })
}

export function listGlossary(params = {}) {
  const qs = new URLSearchParams()
  if (params.lang) qs.append('lang', params.lang)
  if (params.search) qs.append('search', params.search)
  const suffix = qs.toString() ? `?${qs}` : ''
  return request(`/glossary${suffix}`)
}

export function createGlossary(item) {
  return request('/glossary', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(item),
  })
}

export function updateGlossary(item) {
  return request(`/glossary/${item.id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(item),
  })
}

export function deleteGlossary(id) {
  return request(`/glossary/${id}`, { method: 'DELETE' })
}

export function importGlossary(jsonText) {
  return request('/glossary/import', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: jsonText,
  })
}

export function formatBytes(bytes) {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let i = 0
  let val = bytes
  while (val >= 1024 && i < units.length - 1) {
    val /= 1024
    i++
  }
  return `${val.toFixed(val >= 10 ? 0 : 1)} ${units[i]}`
}

export function formatDuration(ms) {
  if (!ms) return '—'
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}
