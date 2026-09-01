import { useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  AlertTriangle, ArrowDownToLine, ArrowRight, ArrowUp, Check, ChevronDown, Clipboard,
  Clock3, FileImage, Image as ImageIcon, Loader2, Paperclip, RefreshCw, Sparkles, X,
} from 'lucide-react'
import {
  createBatchTranslateTask, createTranslateTask, downloadBatchZip, formatBytes,
  formatDuration, getBatchTaskStatus, getTaskResultUrl,
} from '../api'
import { useToast } from './Toast'

const ALLOWED = '.jpg,.jpeg,.png,.webp,.bmp'
const MAX_MB = 10
const MAX_BATCH_FILES = 100
const MAX_BATCH_MB = 500
const STORAGE_KEY = 'manga-translator-current-batches-v1'
const TERMINAL_STATUSES = new Set(['completed', 'failed'])
const LANGUAGE_OPTIONS = [
  { value: 'auto', label: '自动检测' }, { value: 'zh', label: '中文' },
  { value: 'ja', label: '日语' }, { value: 'en', label: '英语' },
]
const TARGET_OPTIONS = LANGUAGE_OPTIONS.filter((item) => item.value !== 'auto')
const PIPELINE_STEPS = [
  { end: 15, label: '检测文字区域' }, { end: 40, label: '识别原文' },
  { end: 55, label: '修复原图' }, { end: 85, label: '翻译文本' },
  { end: 101, label: '渲染译文' },
]

function languageLabel(code) {
  return { auto: '自动检测', zh: '中文', ja: '日语', en: '英语' }[code] || code || '未知语言'
}

function pipelineStage(progress) {
  return PIPELINE_STEPS.find((step) => progress < step.end)?.label || '渲染译文'
}

function validateFile(file) {
  if (!file) return '请选择图片文件'
  const extension = (file.name.split('.').pop() || '').toLowerCase()
  if (!ALLOWED.includes(`.${extension}`)) return `不支持 .${extension} 格式，仅支持 JPG / PNG / WebP / BMP`
  if (file.size > MAX_MB * 1024 * 1024) return `文件超过 ${MAX_MB}MB 限制，请压缩后重试`
  if (file.size === 0) return '文件为空'
  return ''
}

function loadStoredBatches() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
    if (!Array.isArray(stored)) return []
    return stored.map((batch) => ({
      ...batch,
      items: (batch.items || []).map((item) => ({ ...item, file: null, previewUrl: '' })),
    }))
  } catch {
    return []
  }
}

function persistableBatches(batches) {
  return batches.slice(-12).map((batch) => ({
    id: batch.id, createdAt: batch.createdAt, sourceLang: batch.sourceLang, targetLang: batch.targetLang,
    items: batch.items.map((item) => ({
      task_id: item.task_id, filename: item.filename, index: item.index, status: item.status,
      progress: item.progress || 0, error: item.error || '', text_count: item.text_count || 0,
      duration_ms: item.duration_ms || 0, source_lang: item.source_lang || batch.sourceLang,
      target_lang: item.target_lang || batch.targetLang,
      detected_source_lang: item.detected_source_lang || null,
      translation_backends: item.translation_backends || [], ocr_backend: item.ocr_backend || '',
      render_font: item.render_font || '',
    })),
  }))
}

function batchStats(batch) {
  const completed = batch.items.filter((item) => item.status === 'completed').length
  const failed = batch.items.filter((item) => item.status === 'failed').length
  const active = batch.items.length - completed - failed
  const progress = batch.items.length
    ? Math.round(batch.items.reduce((sum, item) => sum + (item.progress || 0), 0) / batch.items.length)
    : 0
  return { completed, failed, active, progress }
}

function triggerDownload(url, filename) {
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename || ''
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
}

export default function TranslatePanel() {
  const notify = useToast()
  const [batches, setBatches] = useState(loadStoredBatches)
  const [files, setFiles] = useState([])
  const [sourceLang, setSourceLang] = useState('auto')
  const [targetLang, setTargetLang] = useState('zh')
  const [fileError, setFileError] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [zipLoadingId, setZipLoadingId] = useState('')
  const [retryingTaskId, setRetryingTaskId] = useState('')
  const inputRef = useRef(null)
  const endRef = useRef(null)
  const mountedRef = useRef(true)
  const batchesRef = useRef(batches)
  const pollTimerRef = useRef(null)
  const pollInFlightRef = useRef(false)
  const pollErrorNotifiedRef = useRef(false)
  const submitLockRef = useRef(false)
  const objectUrlsRef = useRef(new Set())
  const totalBytes = useMemo(() => files.reduce((sum, file) => sum + file.size, 0), [files])
  const languageInvalid = sourceLang !== 'auto' && sourceLang === targetLang

  useEffect(() => {
    batchesRef.current = batches
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(persistableBatches(batches)))
    } catch {
      // 本地存储不可用时不影响任务本身和轮询。
    }
  }, [batches])

  useEffect(() => {
    mountedRef.current = true
    schedulePoll(0)
    return () => {
      mountedRef.current = false
      if (pollTimerRef.current) clearTimeout(pollTimerRef.current)
      objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url))
      objectUrlsRef.current.clear()
    }
  }, [])

  useEffect(() => {
    const onPaste = (event) => {
      const pasted = Array.from(event.clipboardData?.items || [])
        .filter((item) => item.kind === 'file' && item.type.startsWith('image/'))
        .map((item, index) => {
          const blob = item.getAsFile()
          if (!blob) return null
          const extension = blob.type.split('/')[1]?.replace('jpeg', 'jpg') || 'png'
          return new File([blob], `剪贴板图片-${Date.now()}-${index + 1}.${extension}`, {
            type: blob.type, lastModified: Date.now(),
          })
        }).filter(Boolean)
      if (!pasted.length) return
      event.preventDefault()
      if (addFiles(pasted)) notify(`已粘贴 ${pasted.length} 张图片`, 'success')
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [files])

  function activeTaskIds() {
    return batchesRef.current.flatMap((batch) =>
      batch.items.filter((item) => !TERMINAL_STATUSES.has(item.status)).map((item) => item.task_id),
    )
  }

  function schedulePoll(delay = 900) {
    if (!mountedRef.current) return
    if (pollTimerRef.current) clearTimeout(pollTimerRef.current)
    if (!activeTaskIds().length) return
    pollTimerRef.current = setTimeout(runPoll, delay)
  }

  async function runPoll() {
    if (!mountedRef.current || pollInFlightRef.current) return
    const ids = activeTaskIds()
    if (!ids.length) return
    pollInFlightRef.current = true
    let nextDelay = 900
    try {
      const response = await getBatchTaskStatus(ids)
      if (!mountedRef.current) return
      pollErrorNotifiedRef.current = false
      const updates = new Map(response.items.map((item) => [item.task_id, item]))
      setBatches((current) => current.map((batch) => ({
        ...batch,
        items: batch.items.map((item) => {
          const update = updates.get(item.task_id)
          return update ? { ...item, ...update, file: item.file, previewUrl: item.previewUrl } : item
        }),
      })))
    } catch (error) {
      nextDelay = 2500
      if (mountedRef.current && !pollErrorNotifiedRef.current) {
        pollErrorNotifiedRef.current = true
        notify(`任务状态暂时无法更新：${error.message}`, 'error')
      }
    } finally {
      pollInFlightRef.current = false
      if (mountedRef.current) schedulePoll(nextDelay)
    }
  }

  function addFiles(selected) {
    const incoming = Array.from(selected || [])
    if (!incoming.length) return false
    const error = incoming.map(validateFile).find(Boolean)
    if (error) { setFileError(error); return false }
    const next = [...files, ...incoming]
    if (next.length > MAX_BATCH_FILES) { setFileError(`每批最多选择 ${MAX_BATCH_FILES} 张图片`); return false }
    if (next.reduce((sum, file) => sum + file.size, 0) > MAX_BATCH_MB * 1024 * 1024) {
      setFileError(`每批图片总大小不能超过 ${MAX_BATCH_MB}MB`); return false
    }
    setFiles(next)
    setFileError('')
    return true
  }

  function handleFileSelect(event) {
    addFiles(event.target.files)
    event.target.value = ''
  }

  async function submitBatch() {
    if (!files.length || submitting || submitLockRef.current) return
    if (languageInvalid) { setFileError('源语言和目标语言不能相同'); return }
    submitLockRef.current = true
    setSubmitting(true)
    setFileError('')
    const selectedFiles = [...files]
    try {
      let responseItems
      if (selectedFiles.length === 1) {
        const response = await createTranslateTask(selectedFiles[0], sourceLang, targetLang)
        responseItems = [{ task_id: response.task_id, filename: selectedFiles[0].name, index: 1 }]
      } else {
        responseItems = (await createBatchTranslateTask(selectedFiles, sourceLang, targetLang)).items
      }
      if (!mountedRef.current) return
      const items = responseItems.map((item, index) => {
        const file = selectedFiles[item.index ? item.index - 1 : index]
        const previewUrl = URL.createObjectURL(file)
        objectUrlsRef.current.add(previewUrl)
        return {
          ...item, filename: item.filename || file.name, status: 'queued', progress: 0, error: '',
          source_lang: sourceLang, target_lang: targetLang, file, previewUrl,
        }
      })
      const batch = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        createdAt: new Date().toISOString(), sourceLang, targetLang, items,
      }
      setBatches((current) => [...current, batch])
      batchesRef.current = [...batchesRef.current, batch]
      setFiles([])
      notify(`已提交 ${items.length} 张图片`, 'success')
      requestAnimationFrame(() => endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }))
      schedulePoll(0)
    } catch (error) {
      if (mountedRef.current) { setFileError(error.message); notify(`提交失败：${error.message}`, 'error') }
    } finally {
      submitLockRef.current = false
      if (mountedRef.current) setSubmitting(false)
    }
  }

  async function retryItem(batchId, taskId) {
    const batch = batchesRef.current.find((candidate) => candidate.id === batchId)
    const item = batch?.items.find((candidate) => candidate.task_id === taskId)
    if (!batch || !item?.file || retryingTaskId) return
    setRetryingTaskId(taskId)
    try {
      const response = await createTranslateTask(item.file, batch.sourceLang, batch.targetLang)
      if (!mountedRef.current) return
      const replaceTask = (candidate) => candidate.id !== batchId ? candidate : {
        ...candidate,
        items: candidate.items.map((currentItem) => currentItem.task_id !== taskId ? currentItem : {
          ...currentItem, task_id: response.task_id, status: 'queued', progress: 0, error: '',
          text_count: 0, duration_ms: 0,
        }),
      }
      setBatches((current) => current.map(replaceTask))
      batchesRef.current = batchesRef.current.map(replaceTask)
      notify(`已重新提交 ${item.filename}`, 'success')
      schedulePoll(0)
    } catch (error) {
      if (mountedRef.current) notify(`重新翻译失败：${error.message}`, 'error')
    } finally {
      if (mountedRef.current) setRetryingTaskId('')
    }
  }

  async function downloadZip(batch) {
    setZipLoadingId(batch.id)
    try {
      const blob = await downloadBatchZip(batch.items.map((item) => item.task_id))
      if (!mountedRef.current) return
      const url = URL.createObjectURL(blob)
      triggerDownload(url, `翻译结果-${batch.id.slice(0, 8)}.zip`)
      setTimeout(() => URL.revokeObjectURL(url), 0)
    } catch (error) {
      if (mountedRef.current) notify(`ZIP 下载失败：${error.message}`, 'error')
    } finally {
      if (mountedRef.current) setZipLoadingId('')
    }
  }

  return (
    <div className="relative min-h-[calc(100vh-4rem)]">
      <section className="mx-auto w-full max-w-5xl px-4 pb-[330px] pt-10 sm:px-6 md:pb-[270px]">
        {!batches.length ? <EmptyConversation /> : null}
        <AnimatePresence initial={false}>
          {batches.map((batch) => (
            <BatchMessage key={batch.id} batch={batch} zipLoading={zipLoadingId === batch.id}
              retryingTaskId={retryingTaskId} onDownloadZip={() => downloadZip(batch)}
              onRetry={(taskId) => retryItem(batch.id, taskId)} />
          ))}
        </AnimatePresence>
        <div ref={endRef} />
      </section>
      <Composer files={files} sourceLang={sourceLang} targetLang={targetLang} totalBytes={totalBytes}
        fileError={fileError} isDragging={isDragging} submitting={submitting}
        languageInvalid={languageInvalid} inputRef={inputRef} onSourceLangChange={setSourceLang}
        onTargetLangChange={setTargetLang} onFileSelect={handleFileSelect}
        onRemoveFile={(index) => { setFiles((current) => current.filter((_, i) => i !== index)); setFileError('') }}
        onDraggingChange={setIsDragging} onDrop={(event) => {
          event.preventDefault(); setIsDragging(false); addFiles(event.dataTransfer.files)
        }} onSubmit={submitBatch} />
    </div>
  )
}

function EmptyConversation() {
  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
      className="flex min-h-[46vh] flex-col items-center justify-center text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-line-strong bg-surface/80 shadow-[0_0_40px_-14px_rgba(255,255,255,0.35)]">
        <Sparkles size={24} className="text-accent" />
      </div>
      <h1 className="mt-6 font-display text-3xl font-semibold tracking-tight text-ink-100 sm:text-4xl">今天要翻译哪几页漫画？</h1>
      <p className="mt-3 max-w-xl text-sm leading-6 text-ink-500 sm:text-base">
        从底部选择或直接粘贴多张图片。每张图片会独立显示进度，完成一张就能立即预览和下载。
      </p>
      <div className="mt-7 flex flex-wrap justify-center gap-2 text-xs text-ink-500">
        {['多图批量提交', '剪贴板粘贴', '逐张实时预览'].map((text) =>
          <span key={text} className="rounded-full border border-line bg-surface/60 px-3 py-1.5">{text}</span>)}
      </div>
    </motion.div>
  )
}

function BatchMessage({ batch, zipLoading, retryingTaskId, onDownloadZip, onRetry }) {
  const stats = batchStats(batch)
  return (
    <motion.article initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} className="mb-14 space-y-7">
      <div className="ml-auto max-w-3xl">
        <div className="rounded-2xl rounded-br-md border border-line-strong bg-surface-2/90 p-4 sm:p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="font-medium text-ink-100">翻译这 {batch.items.length} 张图片</p>
            <p className="text-xs text-ink-500">{languageLabel(batch.sourceLang)} <ArrowRight size={12} className="mx-1 inline" /> {languageLabel(batch.targetLang)}</p>
          </div>
          <div className="mt-4 grid grid-cols-3 gap-2 sm:grid-cols-5 md:grid-cols-6">
            {batch.items.map((item) => <OriginalThumb key={item.task_id} item={item} />)}
          </div>
        </div>
        <p className="mt-2 text-right font-mono text-[10px] text-ink-600">
          {new Date(batch.createdAt).toLocaleString('zh-CN', { hour12: false })}
        </p>
      </div>
      <div className="flex gap-3 sm:gap-4">
        <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-line-strong bg-surface text-accent"><Sparkles size={15} /></div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="font-display text-lg font-semibold text-ink-100">{stats.active ? '正在处理翻译任务' : '本批任务已处理完成'}</h2>
              <p className="mt-1 text-sm text-ink-500">已完成 {stats.completed} 张 · 失败 {stats.failed} 张 · {stats.active ? `待处理 ${stats.active} 张` : '全部结束'}</p>
            </div>
            {stats.completed > 0 ? (
              <button type="button" disabled={zipLoading} onClick={onDownloadZip}
                className="inline-flex items-center gap-2 rounded-xl border border-line-strong bg-surface-2 px-3.5 py-2 text-sm text-ink-200 transition hover:border-accent/35 hover:text-ink-100 disabled:cursor-wait disabled:opacity-60">
                {zipLoading ? <Loader2 size={15} className="animate-spin" /> : <ArrowDownToLine size={15} />} 下载本批 ZIP
              </button>
            ) : null}
          </div>
          <div className="mt-4 h-1 overflow-hidden rounded-full bg-surface-2">
            <motion.div className="h-full rounded-full bg-gradient-to-r from-ink-500 to-accent"
              animate={{ width: `${stats.progress}%` }} transition={{ type: 'spring', stiffness: 120, damping: 24 }} />
          </div>
          <div className="mt-5 space-y-4">
            {batch.items.map((item) => <TaskCard key={item.task_id} item={item}
              retrying={retryingTaskId === item.task_id} onRetry={() => onRetry(item.task_id)} />)}
          </div>
        </div>
      </div>
    </motion.article>
  )
}

function OriginalThumb({ item }) {
  return (
    <div className="group relative aspect-[4/5] overflow-hidden rounded-lg border border-line bg-bg">
      {item.previewUrl ? <img src={item.previewUrl} alt={item.filename} className="h-full w-full object-cover" /> : (
        <div className="flex h-full flex-col items-center justify-center gap-2 px-2 text-center text-ink-600">
          <ImageIcon size={20} /><span className="line-clamp-2 text-[9px]">刷新后原图不再缓存</span>
        </div>
      )}
      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/90 to-transparent px-2 pb-1.5 pt-6"><p className="truncate text-[10px] text-white/80">{item.filename}</p></div>
    </div>
  )
}

function TaskCard({ item, retrying, onRetry }) {
  const progress = Math.max(0, Math.min(100, item.progress || 0))
  const status = item.status === 'processing' ? 'processing' : item.status
  const statusMeta = {
    queued: { label: '等待中', icon: Clock3, color: 'text-ink-400' },
    processing: { label: '处理中', icon: Loader2, color: 'text-accent' },
    completed: { label: '已完成', icon: Check, color: 'text-ok' },
    failed: { label: '失败', icon: AlertTriangle, color: 'text-danger' },
  }[status] || { label: status || '等待中', icon: Clock3, color: 'text-ink-400' }
  const StatusIcon = statusMeta.icon
  const resultUrl = status === 'completed' ? getTaskResultUrl(item.task_id) : ''
  return (
    <motion.section layout className="overflow-hidden rounded-2xl border border-line bg-surface/70">
      <div className="p-4 sm:p-5">
        <div className="flex items-start gap-3">
          <div className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-surface-2 ${statusMeta.color}`}>
            <StatusIcon size={16} className={status === 'processing' ? 'animate-spin' : ''} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="max-w-full truncate text-sm font-medium text-ink-100">{item.filename}</p>
              <span className={`font-mono text-xs ${statusMeta.color}`}>{statusMeta.label}</span>
            </div>
            {status === 'queued' ? <p className="mt-1 text-xs text-ink-500">任务已进入队列，等待前序图片处理</p> : null}
            {status === 'processing' ? <p className="mt-1 text-xs text-ink-500">{pipelineStage(progress)} · {progress}%</p> : null}
            {status === 'completed' ? <p className="mt-1 text-xs text-ink-500">识别 {item.text_count || 0} 处文字{item.duration_ms ? ` · 耗时 ${formatDuration(item.duration_ms)}` : ''}{item.translation_backends?.length ? ` · ${item.translation_backends.join(' / ')}` : ''}</p> : null}
            {status === 'failed' ? <p className="mt-2 rounded-lg border border-danger/20 bg-danger/5 px-3 py-2 text-xs leading-5 text-danger">{item.error || '翻译失败，请重新提交这张图片'}</p> : null}
          </div>
        </div>
        {(status === 'queued' || status === 'processing') ? (
          <div className="mt-4 h-1 overflow-hidden rounded-full bg-surface-2"><motion.div className="h-full rounded-full bg-accent" animate={{ width: `${progress}%` }} transition={{ type: 'spring', stiffness: 120, damping: 24 }} /></div>
        ) : null}
      </div>
      {status === 'completed' ? (
        <div className="border-t border-line bg-bg/40 p-3 sm:p-4"><div className="overflow-hidden rounded-xl border border-line bg-surface-2"><img src={resultUrl} alt={`${item.filename} 翻译结果`} className="max-h-[720px] w-full object-contain" /></div></div>
      ) : null}
      {(status === 'completed' || status === 'failed') ? (
        <div className="flex flex-wrap items-center gap-2 border-t border-line px-4 py-3 sm:px-5">
          {status === 'completed' ? <button type="button" onClick={() => triggerDownload(resultUrl, `translated_${item.filename}`)} className="inline-flex items-center gap-2 rounded-lg border border-line-strong bg-surface-2 px-3 py-2 text-xs text-ink-200 transition hover:text-ink-100"><ArrowDownToLine size={14} />下载图片</button> : null}
          <button type="button" disabled={!item.file || retrying} onClick={onRetry}
            title={item.file ? '使用原图重新创建翻译任务' : '页面刷新后需重新选择原图才能重新翻译'}
            className="inline-flex items-center gap-2 rounded-lg border border-line-strong px-3 py-2 text-xs text-ink-400 transition hover:bg-surface-2 hover:text-ink-100 disabled:cursor-not-allowed disabled:opacity-40">
            {retrying ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}重新翻译
          </button>
          {!item.file ? <span className="text-[11px] text-ink-600">刷新后重新翻译需再次选择原图</span> : null}
        </div>
      ) : null}
    </motion.section>
  )
}

function Composer({ files, sourceLang, targetLang, totalBytes, fileError, isDragging, submitting,
  languageInvalid, inputRef, onSourceLangChange, onTargetLangChange, onFileSelect,
  onRemoveFile, onDraggingChange, onDrop, onSubmit }) {
  return (
    <div className="fixed inset-x-0 bottom-0 z-40 border-t border-line bg-bg/88 px-3 pb-4 pt-3 backdrop-blur-2xl sm:px-6 sm:pb-6">
      <div className={`mx-auto max-w-4xl rounded-2xl border bg-surface/95 p-3 shadow-[0_-18px_60px_-35px_rgba(255,255,255,0.28)] transition sm:p-4 ${isDragging ? 'border-accent/60 ring-4 ring-accent/10' : 'border-line-strong'}`}
        onDragOver={(event) => { event.preventDefault(); onDraggingChange(true) }}
        onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) onDraggingChange(false) }} onDrop={onDrop}>
        <input ref={inputRef} type="file" accept={ALLOWED} multiple className="hidden" onChange={onFileSelect} />
        {files.length ? <div className="mb-3 flex max-h-24 gap-2 overflow-x-auto pb-1">
          {files.map((file, index) => <PendingFile key={`${file.name}-${file.size}-${file.lastModified}-${index}`} file={file} onRemove={() => onRemoveFile(index)} />)}
        </div> : null}
        <div className="flex items-center gap-2">
          <button type="button" onClick={() => inputRef.current?.click()} className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-line text-ink-400 transition hover:bg-surface-2 hover:text-ink-100" aria-label="选择图片" title="选择多张图片"><Paperclip size={18} /></button>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm text-ink-200">{files.length ? `已选择 ${files.length} 张图片 · ${formatBytes(totalBytes)}` : '选择、拖入或粘贴漫画图片'}</p>
            <p className="mt-0.5 hidden items-center gap-1.5 text-[11px] text-ink-600 sm:flex"><Clipboard size={11} /> 支持 JPG / PNG / WebP / BMP，单张不超过 10MB</p>
          </div>
          <button type="button" disabled={!files.length || submitting || languageInvalid} onClick={onSubmit}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent text-bg transition hover:bg-accent-strong disabled:cursor-not-allowed disabled:bg-surface-2 disabled:text-ink-600" aria-label="提交翻译" title="提交翻译">
            {submitting ? <Loader2 size={18} className="animate-spin" /> : <ArrowUp size={19} />}
          </button>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-line pt-3">
          <CompactLangSelect value={sourceLang} options={LANGUAGE_OPTIONS} onChange={onSourceLangChange} label="源语言" />
          <ArrowRight size={14} className="text-ink-600" />
          <CompactLangSelect value={targetLang} options={TARGET_OPTIONS} onChange={onTargetLangChange} label="目标语言" />
          {languageInvalid ? <span className="text-xs text-danger">源语言和目标语言不能相同</span> : null}
          {fileError ? <span className="flex min-w-0 items-center gap-1.5 text-xs text-danger" role="alert"><AlertTriangle size={13} className="shrink-0" /><span className="truncate">{fileError}</span></span> : null}
        </div>
      </div>
      <p className="mx-auto mt-2 max-w-4xl text-center text-[10px] text-ink-600">翻译任务按顺序执行，关闭或刷新页面后仍会继续处理</p>
    </div>
  )
}

function CompactLangSelect({ value, options, onChange, label }) {
  return (
    <label className="relative"><span className="sr-only">{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)} className="appearance-none rounded-lg border border-line bg-surface-2 py-1.5 pl-3 pr-8 text-xs text-ink-300 outline-none transition focus:border-accent/40">
        {options.map((option) => <option key={option.value} value={option.value} className="bg-surface-2">{option.label}</option>)}
      </select>
      <ChevronDown size={12} className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-ink-500" />
    </label>
  )
}

function PendingFile({ file, onRemove }) {
  const [previewUrl, setPreviewUrl] = useState('')
  useEffect(() => {
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])
  return (
    <div className="relative h-20 w-16 shrink-0 overflow-hidden rounded-lg border border-line bg-surface-2">
      {previewUrl ? <img src={previewUrl} alt={file.name} className="h-full w-full object-cover" /> : <FileImage size={18} />}
      <button type="button" onClick={onRemove} className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center rounded-full bg-black/75 text-white transition hover:bg-danger" aria-label={`移除 ${file.name}`}><X size={11} /></button>
      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/90 to-transparent px-1 pb-1 pt-4"><p className="truncate text-[8px] text-white/80">{file.name}</p></div>
    </div>
  )
}
