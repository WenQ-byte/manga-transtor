import { useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  AlertTriangle, ArrowDownToLine, ArrowRight, ArrowUp, Check, ChevronDown,
  Clock3, FileImage, Image as ImageIcon, Languages, LayoutGrid, Loader2, Maximize2,
  PanelLeftClose, PanelLeftOpen, Paperclip, Pencil, RefreshCw, Search,
  SquarePen, Trash2, X,
} from 'lucide-react'
import {
  createBatchTranslateTask, createTranslateTask, deleteTask, downloadBatchZip, downloadTaskResult,
  formatBytes, formatDuration, getBatchTaskStatus, getTaskResultUrl, retryTranslateTask,
} from '../api'
import { useToast } from './Toast'
import GlossaryPanel from './GlossaryPanel'
import { ComicXMark, ComicXWordmark } from './ComicXBrand'

const ALLOWED = '.jpg,.jpeg,.png,.webp,.bmp'
const MAX_MB = 10
const MAX_BATCH_FILES = 100
const MAX_BATCH_MB = 500
const STORAGE_KEY = 'manga-translator-current-batches-v1'
const NEW_TASK_ID = '__new__'
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
    id: batch.id, name: batchName(batch), createdAt: batch.createdAt,
    sourceLang: batch.sourceLang, targetLang: batch.targetLang,
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

function batchName(batch) {
  if (batch.name) return batch.name
  const first = batch.items[0]?.filename || '图片翻译'
  return batch.items.length > 1 ? `${first} 等 ${batch.items.length} 张` : first
}

function itemFilename(item, index = 0) {
  return item?.filename && item.filename !== '未知文件' ? item.filename : `图片 ${index + 1}`
}

function batchStatus(batch) {
  const stats = batchStats(batch)
  if (stats.active) return { label: '处理中', color: 'text-accent' }
  if (stats.failed === batch.items.length) return { label: '全部失败', color: 'text-danger' }
  if (stats.failed) return { label: '部分失败', color: 'text-amber-400' }
  return { label: '已完成', color: 'text-ok' }
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
  const [activeBatchId, setActiveBatchId] = useState(() => {
    const stored = loadStoredBatches()
    return stored[stored.length - 1]?.id || ''
  })
  const [workspaceView, setWorkspaceView] = useState('translate')
  const [files, setFiles] = useState([])
  const [sourceLang, setSourceLang] = useState('auto')
  const [targetLang, setTargetLang] = useState('zh')
  const [fileError, setFileError] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [zipLoadingId, setZipLoadingId] = useState('')
  const [downloadTaskId, setDownloadTaskId] = useState('')
  const [retryingTaskId, setRetryingTaskId] = useState('')
  const [lightbox, setLightbox] = useState(null)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
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
  const activeBatch = batches.find((batch) => batch.id === activeBatchId) || null

  useEffect(() => {
    if (!activeBatchId && batches.length) setActiveBatchId(batches[batches.length - 1].id)
    if (activeBatchId && activeBatchId !== NEW_TASK_ID && !batches.some((batch) => batch.id === activeBatchId)) {
      setActiveBatchId(batches[batches.length - 1]?.id || '')
    }
  }, [activeBatchId, batches])

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
          if (!update) return item
          const filename = update.filename && update.filename !== '未知文件'
            ? update.filename
            : item.filename
          return { ...item, ...update, filename, file: item.file, previewUrl: item.previewUrl }
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
        name: items.length > 1 ? `${items[0].filename} 等 ${items.length} 张` : items[0].filename,
      }
      setBatches((current) => [...current, batch])
      batchesRef.current = [...batchesRef.current, batch]
      setActiveBatchId(batch.id)
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

  async function downloadSingle(item) {
    if (downloadTaskId) return
    setDownloadTaskId(item.task_id)
    try {
      const { blob, filename } = await downloadTaskResult(item.task_id)
      if (!mountedRef.current) return
      const url = URL.createObjectURL(blob)
      triggerDownload(url, filename)
      setTimeout(() => URL.revokeObjectURL(url), 0)
    } catch (error) {
      if (mountedRef.current) notify(`图片下载失败：${error.message}`, 'error')
    } finally {
      if (mountedRef.current) setDownloadTaskId('')
    }
  }

  function renameBatch(batchId, name) {
    const value = name.trim().slice(0, 80)
    if (!value) return
    const rename = (batch) => batch.id === batchId ? { ...batch, name: value } : batch
    setBatches((current) => current.map(rename))
    batchesRef.current = batchesRef.current.map(rename)
    notify('任务名称已更新', 'success')
  }

  async function retryItem(batchId, taskId) {
    if (retryingTaskId) return
    setRetryingTaskId(taskId)
    try {
      const response = await retryTranslateTask(taskId)
      if (!mountedRef.current) return
      const replace = (batch) => batch.id !== batchId ? batch : {
        ...batch,
        items: batch.items.map((item) => item.task_id !== taskId ? item : {
          ...item, task_id: response.task_id, status: 'queued', progress: 0,
          error: '', text_count: 0, duration_ms: 0,
        }),
      }
      setBatches((current) => current.map(replace))
      batchesRef.current = batchesRef.current.map(replace)
      notify('失败图片已重新加入翻译队列', 'success')
      schedulePoll(0)
    } catch (error) {
      if (mountedRef.current) notify(`重新翻译失败：${error.message}`, 'error')
    } finally {
      if (mountedRef.current) setRetryingTaskId('')
    }
  }

  async function deleteBatch(batchId) {
    const batch = batches.find((item) => item.id === batchId)
    if (!batch) return
    if (!window.confirm('确定删除这个翻译任务吗？删除后无法恢复。')) return
    try {
      await Promise.all(batch.items.map((item) => deleteTask(item.task_id)))
    } catch (error) {
      if (mountedRef.current) notify(`删除失败：${error.message}`, 'error')
      return
    }
    setBatches((current) => current.filter((item) => item.id !== batchId))
    batchesRef.current = batchesRef.current.filter((item) => item.id !== batchId)
    if (activeBatchId === batchId) setActiveBatchId('')
    notify('任务已删除', 'success')
  }

  return (
    <div className="relative flex min-h-screen">
      <TaskSidebar batches={batches} activeBatchId={activeBatchId} workspaceView={workspaceView}
        collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed((value) => !value)}
        onSelect={(id) => { setActiveBatchId(id); setWorkspaceView('translate') }}
        onNew={() => { setActiveBatchId(NEW_TASK_ID); setWorkspaceView('translate') }}
        onDelete={deleteBatch} onRename={renameBatch}
        onGlossary={() => setWorkspaceView('glossary')} />
      <div className={`min-w-0 flex-1 transition-[margin] duration-300 ${sidebarCollapsed ? 'md:ml-16' : 'md:ml-72'}`}>
        {workspaceView === 'glossary' ? (
          <motion.section key="glossary" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="mx-auto w-full max-w-6xl px-4 pb-24 pt-10 sm:px-8">
            <button type="button" onClick={() => setWorkspaceView('translate')} className="mb-6 inline-flex items-center gap-2 rounded-lg border border-line px-3 py-2 text-xs text-ink-400 md:hidden"><Languages size={14} />返回翻译任务</button>
            <GlossaryPanel />
          </motion.section>
        ) : (
          <>
            <section className="mx-auto w-full max-w-5xl px-4 pb-[330px] pt-10 sm:px-6 md:pb-[270px]">
              {!activeBatch ? <EmptyConversation /> : (
            <AnimatePresence mode="wait" initial={false}>
              <BatchMessage key={activeBatch.id} batch={activeBatch}
                zipLoading={zipLoadingId === activeBatch.id}
                downloadingTaskId={downloadTaskId}
                retryingTaskId={retryingTaskId}
                onDownloadZip={() => downloadZip(activeBatch)}
                onDownloadSingle={downloadSingle}
                onRetry={(taskId) => retryItem(activeBatch.id, taskId)}
                onView={(item) => setLightbox({
                  url: getTaskResultUrl(item.task_id), filename: itemFilename(item, activeBatch.items.indexOf(item)),
                })} />
            </AnimatePresence>
              )}
              <div ref={endRef} />
            </section>
            <Composer files={files} sourceLang={sourceLang} targetLang={targetLang} totalBytes={totalBytes}
              fileError={fileError} isDragging={isDragging} submitting={submitting}
              languageInvalid={languageInvalid} inputRef={inputRef} collapsed={sidebarCollapsed}
              onSourceLangChange={setSourceLang}
              onTargetLangChange={setTargetLang} onFileSelect={handleFileSelect}
              onRemoveFile={(index) => { setFiles((current) => current.filter((_, i) => i !== index)); setFileError('') }}
              onDraggingChange={setIsDragging} onDrop={(event) => {
                event.preventDefault(); setIsDragging(false); addFiles(event.dataTransfer.files)
              }} onSubmit={submitBatch} />
          </>
        )}
      </div>
      <ImageLightbox value={lightbox} onClose={() => setLightbox(null)} />
    </div>
  )
}

function TaskSidebar({ batches, activeBatchId, workspaceView, collapsed, onToggle, onSelect, onNew, onDelete, onRename, onGlossary }) {
  const [search, setSearch] = useState('')
  const [renamingId, setRenamingId] = useState('')
  const [renameDraft, setRenameDraft] = useState('')
  const searchInputRef = useRef(null)
  const query = search.trim().toLowerCase()
  const filtered = query
    ? batches.filter((batch) => batchName(batch).toLowerCase().includes(query))
    : batches

  function startRename(batch) {
    setRenamingId(batch.id)
    setRenameDraft(batchName(batch))
  }

  function commitRename() {
    if (renamingId && renameDraft.trim()) onRename(renamingId, renameDraft)
    setRenamingId('')
    setRenameDraft('')
  }

  function railSearch() {
    onToggle()
    requestAnimationFrame(() => searchInputRef.current?.focus())
  }

  function railGlossary() {
    onToggle()
    requestAnimationFrame(onGlossary)
  }

  if (collapsed) {
    return (
      <aside className="fixed bottom-0 left-0 top-0 z-30 hidden w-16 flex-col items-center border-r border-line bg-bg/92 py-3 backdrop-blur-xl md:flex" aria-label="侧边导航（收起）">
        <button type="button" onClick={onToggle}
          className="flex h-10 w-10 items-center justify-center rounded-xl text-ink-400 transition hover:bg-surface-2 hover:text-ink-100" aria-label="展开侧边栏" title="展开侧边栏">
          <PanelLeftOpen size={18} />
        </button>
        <div className="mt-2 flex w-full flex-col items-center gap-1">
          <button type="button" onClick={onNew}
            className="flex h-10 w-10 items-center justify-center rounded-xl text-ink-400 transition hover:bg-surface-2 hover:text-ink-100" aria-label="新建翻译任务" title="新建翻译任务">
            <SquarePen size={18} />
          </button>
          <button type="button" onClick={railSearch}
            className="flex h-10 w-10 items-center justify-center rounded-xl text-ink-400 transition hover:bg-surface-2 hover:text-ink-100" aria-label="搜索翻译任务" title="搜索翻译任务">
            <Search size={18} />
          </button>
          <button type="button" onClick={railGlossary}
            className={`flex h-10 w-10 items-center justify-center rounded-xl transition ${workspaceView === 'glossary' ? 'bg-surface-2 text-ink-100' : 'text-ink-400 hover:bg-surface-2 hover:text-ink-100'}`} aria-label="专有名词库" title="专有名词库">
            <LayoutGrid size={18} />
          </button>
        </div>
      </aside>
    )
  }

  return (
    <aside className="fixed bottom-0 left-0 top-0 z-30 hidden w-72 flex-col border-r border-line bg-bg/92 backdrop-blur-xl md:flex" aria-label="侧边导航">
      <div className="flex items-center justify-between border-b border-line px-4 py-4">
        <div className="flex items-center gap-3">
          <ComicXMark rounded="rounded-lg" className="h-7 w-7" />
          <span className="font-display text-lg font-semibold tracking-tight text-ink-100">ComicX</span>
        </div>
        <button type="button" onClick={onToggle}
          className="flex h-9 w-9 items-center justify-center rounded-xl border border-line text-ink-400 transition hover:bg-surface-2 hover:text-ink-100" aria-label="收起侧边栏" title="收起侧边栏">
          <PanelLeftClose size={16} />
        </button>
      </div>

      <div className="border-b border-line p-3">
        <button type="button" onClick={onNew}
          className="flex w-full items-center gap-2.5 rounded-xl bg-surface-2 px-3.5 py-2.5 text-sm text-ink-100 transition hover:bg-surface-2/70">
          <SquarePen size={16} className="text-ink-300" /> 新建翻译任务
        </button>
      </div>

      <nav className="min-h-0 flex-1 overflow-y-auto px-3 py-4" aria-label="侧边导航">
        <div className="mb-2 px-2 text-xs font-medium text-ink-500">最近</div>
        <div className="mb-2 flex items-center gap-2 rounded-xl border border-line px-3 py-2">
          <Search size={14} className="shrink-0 text-ink-500" />
          <input ref={searchInputRef} value={search} onChange={(event) => setSearch(event.target.value)}
            placeholder="搜索对话内容" aria-label="搜索任务"
            className="min-w-0 flex-1 bg-transparent text-sm text-ink-100 outline-none placeholder:text-ink-600" />
          {search ? <button type="button" onClick={() => setSearch('')} aria-label="清空搜索"><X size={14} className="text-ink-500 hover:text-ink-300" /></button> : null}
        </div>
        <div className="space-y-1">
          {[...filtered].reverse().map((batch) => {
            const status = batchStatus(batch)
            const active = workspaceView === 'translate' && batch.id === activeBatchId
            return (
              <div key={batch.id} onClick={() => { if (renamingId !== batch.id) onSelect(batch.id) }}
                className={`group relative w-full cursor-pointer rounded-xl border px-3 py-2.5 text-left transition ${active ? 'border-accent/25 bg-surface-2' : 'border-transparent hover:border-line hover:bg-surface/60'}`}>
                {renamingId === batch.id ? (
                  <input value={renameDraft} onChange={(event) => setRenameDraft(event.target.value)} maxLength={80} autoFocus
                    onClick={(event) => event.stopPropagation()}
                    onKeyDown={(event) => { if (event.key === 'Enter') commitRename(); if (event.key === 'Escape') { setRenamingId(''); setRenameDraft('') } }}
                    onBlur={commitRename}
                    className="w-full min-w-0 rounded-lg border border-accent/30 bg-surface-2 px-2 py-1 text-sm text-ink-100 outline-none" aria-label="任务名称" />
                ) : (
                  <p className={`truncate pr-14 text-sm ${active ? 'text-ink-100' : 'text-ink-300'}`}>{batchName(batch)}</p>
                )}
                <div className="mt-1 flex items-center justify-between gap-2 text-[10px]">
                  <span className="text-ink-500">{new Date(batch.createdAt).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false })}</span>
                  <span className={status.color}>{status.label}</span>
                </div>
                {renamingId !== batch.id ? (
                  <div className="absolute right-1.5 top-2 flex gap-0.5 opacity-0 transition group-hover:opacity-100 focus-within:opacity-100">
                    <button type="button" onClick={(event) => { event.stopPropagation(); startRename(batch) }}
                      className="flex h-6 w-6 items-center justify-center rounded-md text-ink-500 transition hover:bg-surface-2 hover:text-ink-100"
                      aria-label={`重命名 ${batchName(batch)}`} title="重命名">
                      <Pencil size={13} />
                    </button>
                    <button type="button" onClick={(event) => { event.stopPropagation(); onDelete(batch.id) }}
                      className="flex h-6 w-6 items-center justify-center rounded-md text-ink-500 transition hover:bg-danger/10 hover:text-danger"
                      aria-label={`删除 ${batchName(batch)}`} title="删除任务">
                      <Trash2 size={13} />
                    </button>
                  </div>
                ) : null}
              </div>
            )
          })}
          {!filtered.length ? <div className="px-2 py-6 text-center text-xs leading-5 text-ink-500">{query ? '没有匹配的任务' : '当前会话还没有翻译任务。'}</div> : null}
        </div>
      </nav>

      <div className="border-t border-line p-3">
        <button type="button" onClick={onGlossary}
          className={`flex w-full items-center gap-2.5 rounded-xl px-3.5 py-2.5 text-sm transition ${workspaceView === 'glossary' ? 'bg-surface-2 text-ink-100' : 'text-ink-400 hover:bg-surface/60 hover:text-ink-200'}`}>
          <LayoutGrid size={16} /> 专有名词库
        </button>
      </div>
    </aside>
  )
}

function ImageLightbox({ value, onClose }) {
  useEffect(() => {
    if (!value) return undefined
    const close = (event) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', close)
    return () => window.removeEventListener('keydown', close)
  }, [value, onClose])
  return (
    <AnimatePresence>
      {value ? (
        <motion.div className="fixed inset-0 z-[80] flex items-center justify-center bg-black/85 p-4 backdrop-blur-sm"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} role="dialog" aria-modal="true" aria-label={`${value.filename} 大图预览`}>
          <button type="button" onClick={onClose} className="absolute right-5 top-5 flex h-10 w-10 items-center justify-center rounded-full bg-black/70 text-white" aria-label="关闭大图"><X size={20} /></button>
          <motion.img src={value.url} alt={`${value.filename} 翻译结果大图`} className="max-h-[92vh] max-w-[96vw] object-contain"
            initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.97 }} onClick={(event) => event.stopPropagation()} />
        </motion.div>
      ) : null}
    </AnimatePresence>
  )
}

function EmptyConversation() {
  return (
    <motion.div initial={{ opacity: 0, y: 14, scale: 0.96, filter: 'blur(10px)' }}
      animate={{ opacity: 1, y: 0, scale: 1, filter: 'blur(0px)' }}
      transition={{ type: 'spring', stiffness: 120, damping: 20 }}
      className="flex min-h-[46vh] flex-col items-center justify-center text-center">
      <ComicXWordmark glow className="h-36 w-auto sm:h-48" />
    </motion.div>
  )
}

function BatchMessage({ batch, zipLoading, downloadingTaskId, retryingTaskId, onDownloadZip, onDownloadSingle, onRetry, onView }) {
  const stats = batchStats(batch)
  const overall = batchStatus(batch)
  return (
    <motion.article initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} className="mb-14 space-y-7">
      <div className="ml-auto max-w-3xl">
        <div className="rounded-2xl rounded-br-md border border-line-strong bg-surface-2/90 p-4 sm:p-5">
          <p className="font-medium text-ink-100">已提交 {batch.items.length} 张图片</p>
          <div className="mt-4 grid grid-cols-3 gap-2 sm:grid-cols-5 md:grid-cols-6">
            {batch.items.map((item, index) => <OriginalThumb key={item.task_id} item={item} displayIndex={index} />)}
          </div>
        </div>
      </div>
      <div className="flex gap-3 sm:gap-4">
        <div className="mt-1 flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded-xl border border-line-strong bg-surface"><ComicXMark rounded="rounded-none" className="h-full w-full" /></div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center justify-end gap-3">
            {!stats.active && stats.completed > 0 ? (
              <button type="button" disabled={zipLoading} onClick={onDownloadZip}
                className="inline-flex items-center gap-2 rounded-xl border border-line-strong bg-surface-2 px-3.5 py-2 text-sm text-ink-200 transition hover:border-accent/35 hover:text-ink-100 disabled:cursor-wait disabled:opacity-60">
                {zipLoading ? <Loader2 size={15} className="animate-spin" /> : <ArrowDownToLine size={15} />} 下载本批 ZIP
              </button>
            ) : null}
          </div>
          <div className="space-y-4">
            {batch.items.map((item, index) => <TaskCard key={item.task_id} item={item} displayIndex={index + 1}
              downloading={downloadingTaskId === item.task_id}
              retrying={retryingTaskId === item.task_id}
              onDownload={() => onDownloadSingle(item)} onRetry={() => onRetry(item.task_id)}
              onView={() => onView(item)} />)}
          </div>
          {!stats.active ? <p className={`mt-4 text-sm ${overall.color}`}>全部任务已结束：成功 {stats.completed} 张，失败 {stats.failed} 张（{overall.label}）。</p> : null}
        </div>
      </div>
    </motion.article>
  )
}

function OriginalThumb({ item, displayIndex }) {
  const filename = itemFilename(item, displayIndex)
  return (
    <div className="group relative aspect-[4/5] overflow-hidden rounded-lg border border-line bg-bg">
      {item.previewUrl ? <img src={item.previewUrl} alt={filename} className="h-full w-full object-cover" /> : (
        <div className="flex h-full items-center justify-center text-ink-500">
          <ImageIcon size={20} />
        </div>
      )}
      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/90 to-transparent px-2 pb-1.5 pt-6"><p className="truncate text-[10px] text-white/80">{filename}</p></div>
    </div>
  )
}

function TaskCard({ item, displayIndex, downloading, retrying, onDownload, onRetry, onView }) {
  const [previewFailed, setPreviewFailed] = useState(false)
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
  const filename = itemFilename(item, displayIndex - 1)
  return (
    <motion.section layout className="overflow-hidden rounded-2xl border border-line bg-surface/70">
      <div className="p-4 sm:p-5">
        <div className="flex items-start gap-3">
          <div className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-surface-2 ${statusMeta.color}`}>
            <StatusIcon size={16} className={status === 'processing' ? 'animate-spin' : ''} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="max-w-full truncate text-sm font-medium text-ink-100">{filename}</p>
              <span className={`font-mono text-xs ${statusMeta.color}`}>{statusMeta.label}</span>
            </div>
            {status === 'completed' ? <p className="mt-1 text-xs text-ink-500">识别 {item.text_count || 0} 处文字{item.duration_ms ? ` · 耗时 ${formatDuration(item.duration_ms)}` : ''}{item.translation_backends?.length ? ` · ${item.translation_backends.join(' / ')}` : ''}</p> : null}
            {status === 'failed' ? <p className="mt-2 rounded-lg border border-danger/20 bg-danger/5 px-3 py-2 text-xs leading-5 text-danger">{item.error || '翻译失败，请重新提交这张图片'}</p> : null}
          </div>
        </div>
        {status === 'processing' ? (
          <div className="relative mt-4 overflow-hidden rounded-2xl border border-line bg-surface-2/50 p-4">
            <div className="grid-dots pointer-events-none absolute inset-0 opacity-50" />
            <p className="relative text-sm text-ink-300">正在翻译 · {pipelineStage(progress)}</p>
            <div className="relative flex h-36 items-center justify-center">
              <span className="font-mono text-3xl font-medium tabular-nums text-ink-200">{progress}%</span>
            </div>
          </div>
        ) : null}
      </div>
      {status === 'completed' ? (
        <div className="border-t border-line bg-bg/40 p-3 sm:p-4">
          {previewFailed ? (
            <div className="flex min-h-36 items-center justify-center rounded-xl border border-danger/20 bg-danger/5 px-4 text-center text-sm text-danger">结果图片加载失败，请稍后重试或直接下载。</div>
          ) : (
            <button type="button" onClick={onView} className="group relative block w-full overflow-hidden rounded-xl border border-line bg-surface-2" aria-label={`查看 ${filename} 大图`}>
              <img src={resultUrl} alt={`${filename} 翻译结果`} loading="lazy" decoding="async"
                onError={() => setPreviewFailed(true)} className="max-h-[720px] w-full object-contain" />
              <span className="absolute right-3 top-3 flex items-center gap-1.5 rounded-lg bg-black/70 px-2.5 py-1.5 text-xs text-white opacity-0 transition group-hover:opacity-100"><Maximize2 size={13} />查看大图</span>
            </button>
          )}
        </div>
      ) : null}
      {status === 'failed' ? (
        <div className="flex flex-wrap items-center gap-2 border-t border-line px-4 py-3 sm:px-5">
          <button type="button" disabled={retrying} onClick={onRetry}
            className="inline-flex items-center gap-2 rounded-lg border border-line-strong bg-surface-2 px-3 py-2 text-xs text-ink-200 transition hover:text-ink-100 disabled:cursor-wait disabled:opacity-60">
            {retrying ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}重新翻译
          </button>
        </div>
      ) : null}
      {status === 'completed' ? (
        <div className="flex flex-wrap items-center gap-2 border-t border-line px-4 py-3 sm:px-5">
          <button type="button" disabled={downloading} onClick={onDownload}
            className="inline-flex items-center gap-2 rounded-lg border border-line-strong bg-surface-2 px-3 py-2 text-xs text-ink-200 transition hover:text-ink-100 disabled:cursor-wait disabled:opacity-60">
            {downloading ? <Loader2 size={14} className="animate-spin" /> : <ArrowDownToLine size={14} />}下载图片
          </button>
        </div>
      ) : null}
    </motion.section>
  )
}

function Composer({ files, sourceLang, targetLang, totalBytes, fileError, isDragging, submitting,
  languageInvalid, inputRef, collapsed, onSourceLangChange, onTargetLangChange, onFileSelect,
  onRemoveFile, onDraggingChange, onDrop, onSubmit }) {
  return (
    <div className={`fixed inset-x-0 bottom-0 z-40 border-t border-line bg-bg/88 px-3 pb-4 pt-3 backdrop-blur-2xl transition-[left] duration-300 sm:px-6 sm:pb-6 ${collapsed ? 'md:left-16' : 'md:left-72'}`}>
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
          </div>
          <button type="button" disabled={!files.length || submitting || languageInvalid} onClick={onSubmit}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent text-bg transition hover:bg-accent-strong disabled:cursor-not-allowed disabled:bg-surface-2 disabled:text-ink-500" aria-label="提交翻译" title="提交翻译">
            {submitting ? <Loader2 size={18} className="animate-spin" /> : <ArrowUp size={19} />}
          </button>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-line pt-3">
          <CompactLangSelect value={sourceLang} options={LANGUAGE_OPTIONS} onChange={onSourceLangChange} label="源语言" />
          <ArrowRight size={14} className="text-ink-500" />
          <CompactLangSelect value={targetLang} options={TARGET_OPTIONS} onChange={onTargetLangChange} label="目标语言" />
          {languageInvalid ? <span className="text-xs text-danger">源语言和目标语言不能相同</span> : null}
          {fileError ? <span className="flex min-w-0 items-center gap-1.5 text-xs text-danger" role="alert"><AlertTriangle size={13} className="shrink-0" /><span className="truncate">{fileError}</span></span> : null}
        </div>
      </div>
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
      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/90 to-transparent px-1 pb-1 pt-4"><p className="truncate text-[10px] text-white/80">{file.name}</p></div>
    </div>
  )
}
