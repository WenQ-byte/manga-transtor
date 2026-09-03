import { memo, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion, useDragControls } from 'framer-motion'
import {
  AlertTriangle, ArrowDownToLine, ArrowRight, ArrowUp, Check, ChevronDown,
  Clock3, FileImage, Image as ImageIcon, Languages, LayoutGrid, Loader2, Maximize2,
  PanelLeftClose, PanelLeftOpen, Paperclip, Pencil, RefreshCw, Search,
  Eraser, SquarePen, Trash2, Undo2, X,
} from 'lucide-react'
import {
  createBatchTranslateTask, createTaskTextBox, createTranslateTask, deleteTask, deleteTaskTextBox, downloadBatchZip, downloadTaskResult,
  editTaskRegion, eraseTask, formatBytes, formatDuration, getBatchTaskStatus, getTaskCleanedUrl, getTaskRegions, getTaskResultUrl, previewTaskRegionFont, restoreTaskTextRegion, retryTranslateTask, undoEraseTask,
} from '../api'
import { useToast } from './Toast'
import GlossaryPanel from './GlossaryPanel'
import { ComicXMark, ComicXWordmark } from './ComicXBrand'
import SpecularButton from './SpecularButton'
import { SPECULAR_PRIMARY, SPECULAR_SECONDARY } from './specularPresets'
import { collectDroppedImageFiles, isImageFile, sortImageFiles } from '../utils/folderFiles'

const ALLOWED = '.jpg,.jpeg,.png,.webp,.bmp'
const MAX_MB = 10
const MAX_BATCH_FILES = 100
const MAX_BATCH_MB = 500
const STORAGE_KEY = 'manga-translator-current-batches-v1'
const NEW_TASK_ID = '__new__'
const REGION_MIN_SIZE = 28
const REGION_MIN_VISIBLE = 18
const REGION_MAX_SIZE_RATIO = 0.96
const REGION_SYNC_DELAY_MS = 420
const NEON_BRUSH_COLOR = '#c084fc'
const NEON_BRUSH_FILL = 'rgba(192, 132, 252, 0.2)'
const NEON_REGION_FILL = 'rgba(192, 132, 252, 0.10)'
const SUMMARY_MAX = 6
const PENDING_MAX = 8
const TERMINAL_STATUSES = new Set(['completed', 'failed'])
const LANGUAGE_OPTIONS = [
  { value: 'auto', label: '自动检测' }, { value: 'zh', label: '中文' },
  { value: 'ja', label: '日语' }, { value: 'en', label: '英语' },
]
const TARGET_OPTIONS = LANGUAGE_OPTIONS.filter((item) => item.value !== 'auto')
const POLISH_OPTIONS = [
  { value: 'natural', label: '自然' }, { value: 'colloquial', label: '口语化' },
  { value: 'passionate', label: '热血' }, { value: 'funny', label: '搞笑' },
  { value: 'formal', label: '正式' }, { value: 'custom', label: '自定义' },
]
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

function regionBox(region) {
  const box = Array.isArray(region?.box) ? region.box : []
  const pts = box.filter((p) => Array.isArray(p) && p.length >= 2)
  if (!pts.length) return { x: 0, y: 0, width: 1, height: 1 }
  const xs = pts.map((point) => Number(point[0]) || 0)
  const ys = pts.map((point) => Number(point[1]) || 0)
  const x = Math.min(...xs); const y = Math.min(...ys)
  return { x, y, width: Math.max(1, Math.max(...xs) - x), height: Math.max(1, Math.max(...ys) - y) }
}

function validateFile(file) {
  if (!file) return '请选择图片文件'
  const extension = (file.name.split('.').pop() || '').toLowerCase()
  if (!ALLOWED.includes(`.${extension}`)) return `不支持 .${extension} 格式，仅支持 JPG / PNG / WebP / BMP`
  if (file.size > MAX_MB * 1024 * 1024) return `文件超过 ${MAX_MB}MB 限制，请压缩后重试`
  if (file.size === 0) return '文件为空'
  return ''
}

function localTextColor(region) {
  return Array.isArray(region?.color) && region.color.length === 3
    ? `#${region.color.map((part) => Number(part).toString(16).padStart(2, '0')).join('')}`
    : '#000000'
}

function localTextSize(text, box, baseSize, vertical) {
  const lines = text.split(/\r?\n/)
  const longest = Math.max(1, ...lines.map((line) => line.length))
  const fitted = vertical
    ? Math.min(baseSize, box.width * 0.78, box.height / longest * 0.88)
    : Math.min(baseSize, box.height / Math.max(1, lines.length * 1.25) * 0.86, box.width / longest * 1.35)
  return Math.max(8, Math.floor(fitted || 8))
}

function regionLayoutBox(region, active = false) {
  return regionBox(region)
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
      result_revision: item.result_revision || 0,
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
  const [polishEnabled, setPolishEnabled] = useState(false)
  const [polishStyle, setPolishStyle] = useState('natural')
  const [customPrompt, setCustomPrompt] = useState('')
  const [fileError, setFileError] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [zipLoadingId, setZipLoadingId] = useState('')
  const [downloadTaskId, setDownloadTaskId] = useState('')
  const [retryingTaskId, setRetryingTaskId] = useState('')
  const [lightbox, setLightbox] = useState(null)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const inputRef = useRef(null)
  const folderInputRef = useRef(null)
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

  function startNewTask() {
    setActiveBatchId(NEW_TASK_ID)
    setWorkspaceView('translate')
    setPolishEnabled(false)
    setPolishStyle('natural')
    setCustomPrompt('')
    setFileError('')
  }

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

  function addFiles(selected, sortIncoming = false) {
    const incoming = sortIncoming ? sortImageFiles(Array.from(selected || [])) : Array.from(selected || [])
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

  async function handleFolderSelect(event) {
    const selected = sortImageFiles(Array.from(event.target.files || []).filter(isImageFile))
    event.target.value = ''
    await importFolderFiles(selected)
  }

  async function importFolderFiles(selected) {
    if (!selected.length) {
      setFileError('文件夹中没有可用图片（支持 JPG、PNG、WebP、BMP）')
      return
    }
    setFileError(`正在读取文件夹中的 ${selected.length} 张图片…`)
    const readable = []
    let unreadable = 0
    let invalid = 0
    for (const file of selected) {
      if (validateFile(file)) {
        invalid += 1
        continue
      }
      try {
        if (typeof createImageBitmap === 'function') {
          const bitmap = await createImageBitmap(file)
          bitmap.close()
        }
        readable.push(file)
      } catch {
        unreadable += 1
      }
    }
    if (!readable.length) {
      setFileError('文件夹中的图片均无法读取，请检查文件是否损坏')
      return
    }
    const added = addFiles(readable, true)
    if (added && (unreadable || invalid)) setFileError(`已导入 ${readable.length} 张图片，忽略 ${unreadable + invalid} 张无法读取或不符合限制的图片`)
    else if (added) notify(`已按文件夹顺序导入 ${readable.length} 张图片`, 'success')
  }

  async function handleDrop(event) {
    event.preventDefault()
    setIsDragging(false)
    setFileError('')
    const result = await collectDroppedImageFiles(event.dataTransfer)
    if (result.hasDirectory) await importFolderFiles(result.files)
    else addFiles(result.files)
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
        const response = await createTranslateTask(selectedFiles[0], sourceLang, targetLang, polishEnabled ? polishStyle : '', polishEnabled && polishStyle === 'custom' ? customPrompt : '')
        responseItems = [{ task_id: response.task_id, filename: selectedFiles[0].name, index: 1 }]
      } else {
        responseItems = (await createBatchTranslateTask(selectedFiles, sourceLang, targetLang, polishEnabled ? polishStyle : '', polishEnabled && polishStyle === 'custom' ? customPrompt : '')).items
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

  function refreshEditedResult(taskId) {
    const revision = Date.now()
    const refresh = (batch) => ({
      ...batch,
      items: batch.items.map((item) => item.task_id === taskId ? { ...item, result_revision: revision } : item),
    })
    setBatches((current) => current.map(refresh))
    batchesRef.current = batchesRef.current.map(refresh)
    setLightbox((current) => current?.taskId === taskId ? { ...current, revision } : current)
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
        onNew={startNewTask}
        onDelete={deleteBatch} onRename={renameBatch}
        onGlossary={() => setWorkspaceView('glossary')} />
      <div className={`min-w-0 flex-1 bg-[#171717] transition-[margin] duration-300 ${sidebarCollapsed ? 'md:ml-16' : 'md:ml-72'}`}>
        <AnimatePresence mode="wait" initial={false}>
        {workspaceView === 'glossary' ? (
          <motion.section key="glossary" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.22, ease: 'easeOut' }}
            className="mx-auto w-full max-w-6xl px-4 pb-24 pt-10 sm:px-8">
            <button type="button" onClick={() => setWorkspaceView('translate')} className="mb-6 inline-flex items-center gap-2 rounded-lg border border-line px-3 py-2 text-xs text-ink-400 md:hidden"><Languages size={14} />返回翻译任务</button>
            <GlossaryPanel />
          </motion.section>
        ) : (
          <motion.div key="translate" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.18, ease: 'easeOut' }}>
            <section className="mx-auto w-full max-w-5xl px-4 pb-[330px] pt-10 sm:px-6 md:pb-[270px]">
              <AnimatePresence mode="wait" initial={false}>
                {!activeBatch ? (
                  <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
                    <EmptyConversation />
                  </motion.div>
                ) : (
                  <BatchMessage key={activeBatch.id} batch={activeBatch}
                    zipLoading={zipLoadingId === activeBatch.id}
                    downloadingTaskId={downloadTaskId}
                    retryingTaskId={retryingTaskId}
                    onDownloadZip={() => downloadZip(activeBatch)}
                    onDownloadSingle={downloadSingle}
                    onRetry={(taskId) => retryItem(activeBatch.id, taskId)}
                    onView={(item) => setLightbox({
                      taskId: item.task_id, item, editMode: false, revision: item.result_revision || 0,
                      url: getTaskResultUrl(item.task_id), cleanedUrl: getTaskCleanedUrl(item.task_id), filename: itemFilename(item, activeBatch.items.indexOf(item)),
                    })} />
                )}
              </AnimatePresence>
              <div ref={endRef} />
            </section>
            <Composer files={files} sourceLang={sourceLang} targetLang={targetLang} totalBytes={totalBytes}
              polishEnabled={polishEnabled} polishStyle={polishStyle} customPrompt={customPrompt}
              fileError={fileError} isDragging={isDragging} submitting={submitting}
              languageInvalid={languageInvalid} inputRef={inputRef} collapsed={sidebarCollapsed}
              onSourceLangChange={setSourceLang}
              onTargetLangChange={setTargetLang} onPolishEnabledChange={setPolishEnabled}
              onPolishStyleChange={setPolishStyle} onCustomPromptChange={setCustomPrompt}
              onFileSelect={handleFileSelect} folderInputRef={folderInputRef} onFolderSelect={handleFolderSelect}
              onRemoveFile={(index) => { setFiles((current) => current.filter((_, i) => i !== index)); setFileError('') }}
              onDraggingChange={setIsDragging} onDrop={handleDrop} onSubmit={submitBatch} />
          </motion.div>
        )}
        </AnimatePresence>
      </div>
      <ImageLightbox value={lightbox} onClose={(event) => { if (!event || event.currentTarget?.tagName === 'BUTTON') setLightbox(null) }} onUpdated={refreshEditedResult}
        downloading={downloadTaskId === lightbox?.taskId} onDownload={() => lightbox?.item && downloadSingle(lightbox.item)} />
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
      <aside className="fixed bottom-0 left-0 top-0 z-30 hidden w-16 flex-col items-center border-r border-line bg-[#1d1a1f]/95 py-3 backdrop-blur-xl md:flex" aria-label="侧边导航（收起）">
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
    <aside className="fixed bottom-0 left-0 top-0 z-30 hidden w-72 flex-col border-r border-line bg-[#211e22]/95 backdrop-blur-xl md:flex" aria-label="侧边导航">
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
          className="flex w-full items-center gap-2.5 rounded-xl bg-[#2b282c] px-3.5 py-2.5 text-sm text-ink-100 transition hover:bg-[#353137]">
          <SquarePen size={16} className="text-ink-300" /> 新建翻译任务
        </button>
      </div>

      <nav className="min-h-0 flex-1 overflow-y-auto px-3 py-4" aria-label="侧边导航">
        <button type="button" onClick={onGlossary}
          className={`mb-4 flex w-full items-center gap-2.5 rounded-xl border px-3 py-2.5 text-sm transition ${workspaceView === 'glossary' ? 'border-white/15 bg-[#302d31] text-ink-100' : 'border-transparent text-ink-400 hover:border-line hover:bg-[#29262a] hover:text-ink-200'}`}>
          <LayoutGrid size={16} /> 专有名词库
        </button>
        <div className="mb-2 px-2 text-xs font-medium text-ink-500">最近</div>
        <div className="mb-2 flex items-center gap-2 rounded-xl border border-line px-3 py-2">
          <Search size={14} className="shrink-0 text-ink-500" />
          <input ref={searchInputRef} value={search} onChange={(event) => setSearch(event.target.value)}
            placeholder="搜索翻译任务" aria-label="搜索任务"
            className="min-w-0 flex-1 bg-transparent text-sm text-ink-100 outline-none placeholder:text-ink-600" />
          {search ? <button type="button" onClick={() => setSearch('')} aria-label="清空搜索"><X size={14} className="text-ink-500 hover:text-ink-300" /></button> : null}
        </div>
        <div className="space-y-1">
          {[...filtered].reverse().map((batch) => {
            const status = batchStatus(batch)
            const active = workspaceView === 'translate' && batch.id === activeBatchId
            return (
              <div key={batch.id} onClick={() => { if (renamingId !== batch.id) onSelect(batch.id) }}
                 className={`group relative w-full cursor-pointer rounded-xl border px-3 py-2.5 text-left transition ${active ? 'border-white/15 bg-[#302d31]' : 'border-transparent hover:border-line hover:bg-[#29262a]'}`}>
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

    </aside>
  )
}

function PlannedRegionText({ region, text, color, fontFamily, fontWeight, box, baseBox = null, dragging = false }) {
  const plan = region.render_layout
  const renderBox = region.render_box ? regionBox({ box: region.render_box }) : box
  const design = baseBox || box
  const sx = box.width / Math.max(1, design.width)
  const sy = box.height / Math.max(1, design.height)
  const boxCenterX = box.x + box.width / 2
  const boxCenterY = box.y + box.height / 2
  const designCenterX = design.x + design.width / 2
  const designCenterY = design.y + design.height / 2
  const wrapTransform = `translate(${boxCenterX} ${boxCenterY}) scale(${sx} ${sy}) translate(${-designCenterX} ${-designCenterY})`
  const fontSize = Number(plan.font_size || region.font_size || region.default_font_size || 24)
  const strokeWidth = Number(plan.stroke_width || Math.max(1, fontSize / 14))
  const common = { fill: color, stroke: color === '#ffffff' ? '#000000' : '#ffffff', strokeWidth, paintOrder: 'stroke', fontFamily, fontSize, fontWeight }
  const groupStyle = { transformBox: 'view-box', transition: dragging ? 'none' : 'transform 150ms ease-out' }
  if (plan.direction === 'v') {
    return <g transform={wrapTransform} style={groupStyle}>
      <g transform={`translate(${renderBox.x} ${renderBox.y})`}>
        {(plan.columns || []).flatMap((column, columnIndex) => Array.from(column.text || '').map((character, characterIndex) => (
          <text key={`${columnIndex}-${characterIndex}`} x={Number(column.center_x || 0)} y={Number(column.top || 0) + characterIndex * Number(plan.char_height || fontSize * 1.15)} textAnchor="middle" dominantBaseline="hanging" style={common}>{character}</text>
        )))}
      </g>
    </g>
  }
  return <g transform={wrapTransform} style={groupStyle}>
    <g transform={`translate(${renderBox.x} ${renderBox.y})`}>
      {(plan.lines || [text]).map((line, index) => <text key={index} x={Number(plan.center_x || 0)} y={Number(plan.top || 0) + index * (Number(plan.line_height || fontSize * 1.4) + Number(plan.spacing || 0))} textAnchor="middle" dominantBaseline="hanging" style={common}>{line}</text>)}
    </g>
  </g>
}

const LiveTextLayer = memo(function LiveTextLayer({ regions, selected, draft, fontSize, fontFamily, fontWeight, defaultTextColor, textColor, fontOptions, imageSize, draggingIndex = null, baseBoxRef = null, hidden = false, className = '' }) {
  return <svg className={`pointer-events-none absolute inset-0 h-full w-full ${className || 'z-[5]'}`} viewBox={`0 0 ${imageSize.width} ${imageSize.height}`} preserveAspectRatio="none" style={hidden ? { visibility: 'hidden' } : undefined} aria-label="实时文字图层">
    {regions.map((region) => {
      const active = selected?.index === region.index
      const dragging = draggingIndex === region.index
      const text = active ? draft : (region.translated || '')
      if (!text.trim()) return null
      const box = regionBox(region)
      const baseBox = dragging && baseBoxRef ? baseBoxRef.current.get(region.index) : null
      const vertical = region.direction === 'v' || box.height > box.width * 1.55
      const size = localTextSize(text, box, active ? fontSize : (region.font_size || region.default_font_size || 24), vertical)
      const option = fontOptions.find((item) => item.value === (active ? fontFamily : region.font_family))
      const color = active ? (defaultTextColor ? '#000000' : textColor) : localTextColor(region)
      // 拖动或等待服务端保存时使用当前文本框实时排版，避免旧布局坐标瞬移。
      if (region.render_layout && !active && !dragging) {
        return <PlannedRegionText key={region.id || region.index} region={region} text={text} color={color} fontFamily={option?.css_family || 'Microsoft YaHei'} fontWeight={active ? fontWeight : (region.font_weight || 400)} box={box} baseBox={baseBox} dragging={dragging} />
      }
      return <foreignObject key={region.id || region.index} x={box.x} y={box.y} width={box.width} height={box.height}>
        <div xmlns="http://www.w3.org/1999/xhtml" style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', padding: `${Math.max(2, size * 0.12)}px`, color, fontFamily: option?.css_family || 'Microsoft YaHei', fontSize: `${size}px`, fontWeight: active ? fontWeight : (region.font_weight || 400), lineHeight: 1.15, textAlign: 'center', whiteSpace: 'pre-wrap', wordBreak: 'break-word', writingMode: vertical ? 'vertical-rl' : 'horizontal-tb' }}>
          {text}
        </div>
      </foreignObject>
    })}
  </svg>
})

function ImageLightbox({ value, onClose, onUpdated, onDownload, downloading }) {
  const [regions, setRegions] = useState([])
  const [fontOptions, setFontOptions] = useState([])
  const [imageSize, setImageSize] = useState({ width: 1, height: 1 })
  const [editMode, setEditMode] = useState(false)
  const [regionsLoading, setRegionsLoading] = useState(false)
  const [selected, setSelected] = useState(null)
  const [draft, setDraft] = useState('')
  const [fontSizeInput, setFontSizeInput] = useState('24')
  const [fontFamily, setFontFamily] = useState('')
  const [fontWeight, setFontWeight] = useState(400)
  const [fontPanelOpen, setFontPanelOpen] = useState(false)
  const [fontBeforePanel, setFontBeforePanel] = useState({ family: '', weight: 400 })
  const [fontPreviewUrl, setFontPreviewUrl] = useState('')
  const [fontPreviewLoading, setFontPreviewLoading] = useState(false)
  const [textColor, setTextColor] = useState('#000000')
  const [defaultTextColor, setDefaultTextColor] = useState(true)
  const [colorPreviewActive, setColorPreviewActive] = useState(false)
  const [colorPanelOpen, setColorPanelOpen] = useState(false)
  const fontSize = Math.max(8, Math.min(160, Number(fontSizeInput) || 8))
  const selectedFontOption = fontOptions.find((font) => font.value === fontFamily)
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [restoring, setRestoring] = useState(false)
  const [error, setError] = useState('')
  const [cacheBust, setCacheBust] = useState(0)
  const [eraseMode, setEraseMode] = useState(false)
  const [brushSize, setBrushSize] = useState(32)
  const [eraseBusy, setEraseBusy] = useState(false)
  const [hasEraseHistory, setHasEraseHistory] = useState(false)
  const [eraseHasMask, setEraseHasMask] = useState(false)
  const [eraseCursor, setEraseCursor] = useState(null)
  const [panelOffset, setPanelOffset] = useState({ x: 0, y: 0 })
  const [canvasZoom, setCanvasZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [spacePressed, setSpacePressed] = useState(false)
  const lightboxRef = useRef(null)
  const viewportRef = useRef(null)
  const spacePressedRef = useRef(false)
  const panRef = useRef(null)
  const panMovedRef = useRef(false)
  const clickRef = useRef(null)
  const imageCanvasRef = useRef(null)
  const eraseCanvasRef = useRef(null)
  const eraseDrawingRef = useRef(false)
  const regionInteractionRef = useRef(null)
  const [draggingIndex, setDraggingIndex] = useState(null)
  const baseBoxRef = useRef(new Map())
  const regionsRef = useRef(regions)
  const imageSizeRef = useRef(imageSize)
  const valueRef = useRef(value)
  regionsRef.current = regions
  imageSizeRef.current = imageSize
  valueRef.current = value
  const [panActive, setPanActive] = useState(false)
  const [imgLoading, setImgLoading] = useState(true)
  const [imgError, setImgError] = useState(false)
  const [openToken, setOpenToken] = useState(0)
  const [displaySrc, setDisplaySrc] = useState('')
  const dragControls = useDragControls()
  const fontPreviewRequest = useRef(0)
  const fontPreviewTimer = useRef(null)
  const regionSyncTimer = useRef(null)
  const regionSyncRequest = useRef(0)
  const imgLoadTimer = useRef(null)
  const imgLoadedRef = useRef(false)
  const styleSnapshotRef = useRef(null)
  const editLoadRequest = useRef(0)
  const imageSrc = fontPreviewUrl || displaySrc || value?.url || ''
  const previewText = (draft.trim() || '漫画对白预览').slice(0, 80)

  function selectRegion(region) {
    setSelected({ ...region }); setDraft(region.translated || '')
    setFontSizeInput(String(region.font_size || region.default_font_size || 24)); setFontFamily(region.font_family || '')
    setFontWeight(region.font_weight ?? 400); setDefaultTextColor(!region.color); setColorPreviewActive(false); setColorPanelOpen(false)
    setTextColor(region.color ? `#${region.color.map((part) => Number(part).toString(16).padStart(2, '0')).join('')}` : '#000000'); setError('')
  }

  function updateRegionBox(index, box) {
    const next = { box: [[box.x, box.y], [box.x + box.width, box.y], [box.x + box.width, box.y + box.height], [box.x, box.y + box.height]] }
    setRegions((items) => items.map((item) => item.index === index ? { ...item, ...next } : item))
    setSelected((item) => item?.index === index ? { ...item, ...next } : item)
  }

  function scheduleRegionSync(overrides = {}) {
    if (!selected) return
    if (regionSyncTimer.current) clearTimeout(regionSyncTimer.current)
    const requestId = regionSyncRequest.current + 1
    regionSyncRequest.current = requestId
    const index = selected.index
    const box = overrides.box || regionBox(selected)
    const nextTranslated = overrides.translated ?? draft
    const nextFontSize = overrides.fontSize ?? fontSize
    const nextFontFamily = overrides.fontFamily ?? fontFamily
    const nextFontWeight = overrides.fontWeight ?? fontWeight
    const nextColor = overrides.color ?? (defaultTextColor ? '' : textColor)
    regionSyncTimer.current = setTimeout(async () => {
      setSaving(true); setError('')
      try {
        const response = await editTaskRegion(value.taskId, index, nextTranslated, { fontSize: nextFontSize, fontFamily: nextFontFamily, fontWeight: nextFontWeight, color: nextColor, ...box })
        if (regionSyncRequest.current !== requestId) return
        const colorMatch = nextColor.match(/^#([0-9a-f]{6})$/i)
        const parsedColor = colorMatch ? colorMatch[1].match(/../g).map((part) => parseInt(part, 16)) : null
        setRegions((items) => items.map((item) => item.index === index ? { ...item, translated: response.translated || nextTranslated.trim(), font_size: response.font_size, default_font_size: response.default_font_size || item.default_font_size, font_family: response.font_family ?? nextFontFamily, font_weight: response.font_weight ?? nextFontWeight, color: parsedColor, box: response.box || item.box, render_box: response.render_box || item.render_box, render_layout: response.render_layout || item.render_layout } : item))
        setSelected((item) => item?.index === index ? { ...item, translated: response.translated || nextTranslated.trim(), font_size: response.font_size, default_font_size: response.default_font_size || item.default_font_size, font_family: response.font_family ?? nextFontFamily, font_weight: response.font_weight ?? nextFontWeight, color: parsedColor, box: response.box || item.box, render_box: response.render_box || item.render_box, render_layout: response.render_layout || item.render_layout } : item)
        setError(response.warning || '')
        setCacheBust(Date.now()); onUpdated(value.taskId)
      } catch (err) {
        if (regionSyncRequest.current === requestId) setError(`同步文本失败，当前图片未改变：${err.message}`)
      } finally {
        if (regionSyncRequest.current === requestId) setSaving(false)
      }
    }, REGION_SYNC_DELAY_MS)
  }

  async function persistRegionBox(index, box, before) {
    const current = regionsRef.current.find((item) => item.index === index)
    const currentValue = valueRef.current
    if (!current || !currentValue) return
    if (regionSyncTimer.current) clearTimeout(regionSyncTimer.current)
    regionSyncRequest.current += 1
    setSaving(true)
  try {
      const response = await editTaskRegion(currentValue.taskId, index, current.translated || '', { moveOnly: true, ...box })
      // move_only 会同时变换同一气泡的所有行，重新读取服务端状态，避免只更新当前行造成闪回。
      try {
        const refreshed = await getTaskRegions(currentValue.taskId)
        setRegions(refreshed.regions || [])
        setImageSize({ width: refreshed.width || 1, height: refreshed.height || 1 })
        setHasEraseHistory(Boolean(refreshed.has_erase_history))
        const refreshedSelected = (refreshed.regions || []).find((item) => item.index === index)
        if (refreshedSelected) selectRegion(refreshedSelected)
      } catch {
        // 保存已经成功，读取失败时仍采用本次响应保持当前行位置。
        setRegions((items) => items.map((item) => item.index === index ? { ...item, box: response.box || item.box, render_box: response.render_box || item.render_box, render_layout: response.render_layout || item.render_layout, font_size: response.font_size, default_font_size: response.default_font_size || item.default_font_size } : item))
      }
      setCacheBust(Date.now()); setError(response.warning || ''); onUpdated(currentValue.taskId)
    } catch (err) { updateRegionBox(index, before); setError(`文本框更新失败，已恢复原位置：${err.message}`) }
    finally { baseBoxRef.current.delete(index); setDraggingIndex(null); setSaving(false) }
  }

  async function addTextBox() {
    if (regionsLoading || !imageSize.width || !imageSize.height) return
    const width = Math.min(320, Math.max(120, Math.round(imageSize.width * 0.28)))
    const height = Math.min(180, Math.max(72, Math.round(imageSize.height * 0.16)))
    try {
      const created = await createTaskTextBox(value.taskId, { x: (imageSize.width - width) / 2, y: (imageSize.height - height) / 2, width, height })
      setRegions((items) => [...items, created])
      selectRegion(created)
    } catch (err) { setError(`添加文本框失败：${err.message}`) }
  }

  async function deleteSelectedTextBox(event) {
    event.preventDefault(); event.stopPropagation()
    if (!selected || deleting || saving) return
    if (!window.confirm('确定删除当前文本框吗？')) return
    const deletedIndex = selected.index
    setDeleting(true); setError('')
    try {
      await deleteTaskTextBox(value.taskId, deletedIndex)
      const response = await getTaskRegions(value.taskId)
      const revision = Date.now()
      setSelected(null); setRegions(response.regions || []); setImageSize({ width: response.width || 1, height: response.height || 1 })
      setFontPreviewUrl((current) => { if (current) URL.revokeObjectURL(current); return '' })
      setCacheBust(revision); onUpdated(value.taskId)
    } catch (err) {
      setError(`删除文本框失败，原结果未改变：${err.message}`)
    } finally {
      setDeleting(false)
    }
  }

  async function restoreSelectedTextBox() {
    if (!selected || restoring || saving || deleting) return
    if (!window.confirm('确定恢复该文本框区域的原图吗？这会移除当前错误识别的文本框。')) return
    setRestoring(true); setError('')
    try {
      await restoreTaskTextRegion(value.taskId, selected.index)
      const response = await getTaskRegions(value.taskId)
      const revision = Date.now()
      setSelected(null); setRegions(response.regions || []); setImageSize({ width: response.width || 1, height: response.height || 1 }); setHasEraseHistory(Boolean(response.has_erase_history))
      setFontPreviewUrl((current) => { if (current) URL.revokeObjectURL(current); return '' })
      setCacheBust(revision); onUpdated(value.taskId)
    } catch (err) { setError(`恢复原图失败：${err.message}`) }
    finally { setRestoring(false) }
  }

  async function beginEdit() {
    if (editMode || regionsLoading) return
    const requestId = editLoadRequest.current + 1
    editLoadRequest.current = requestId
    setEditMode(true); setRegionsLoading(true); setError('')
    try {
      const response = await getTaskRegions(value.taskId)
      if (editLoadRequest.current !== requestId) return
      const nextRegions = response.regions || []
      setRegions(nextRegions); setFontOptions(response.font_options || [])
      setImageSize({ width: response.width || 1, height: response.height || 1 }); setRegionsLoading(false)
    } catch (err) {
      if (editLoadRequest.current !== requestId) return
      setEditMode(false); setRegionsLoading(false); setError(`文本区域加载失败：${err.message}`)
    }
  }

  function toggleEditMode() {
    if (!editMode) { beginEdit(); return }
    if (eraseMode) { clearEraseMask(); setEraseMode(false) }
    setSelected(null); setEditMode(false); setCanvasZoom(1)
    recenterView()
  }

  function recenterView() {
    setCanvasZoom(1)
    setPan({ x: 0, y: 0 })
  }

  function interactiveTarget(event) {
    const target = event.target
    return target && (target.closest ? target.closest('button, input, select, a, textarea') : false)
  }

  function startCanvasPan(event) {
    if (!editMode || eraseMode || event.button !== 0) return
    if (interactiveTarget(event)) return
    event.preventDefault()
    if (spacePressedRef.current) {
      panRef.current = { x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y, nx: pan.x, ny: pan.y, zoom: canvasZoom }
      panMovedRef.current = false
      setPanActive(true)
      event.currentTarget.setPointerCapture?.(event.pointerId)
    } else {
      clickRef.current = { x: event.clientX, y: event.clientY }
    }
  }

  function moveCanvasPan(event) {
    const p = panRef.current
    if (!p) return
    event.preventDefault()
    const dx = event.clientX - p.x
    const dy = event.clientY - p.y
    if (Math.hypot(dx, dy) > 4) panMovedRef.current = true
    const nx = p.panX + dx
    const ny = p.panY + dy
    p.nx = nx
    p.ny = ny
    const canvas = imageCanvasRef.current
    if (canvas) canvas.style.transform = `translate(${nx}px, ${ny}px) scale(${p.zoom})`
  }

  function endCanvasPan(event) {
    const p = panRef.current
    if (p) {
      panRef.current = null
      setPanActive(false)
      setPan({ x: p.nx, y: p.ny })
      return
    }
    const c = clickRef.current
    clickRef.current = null
    if (c && !eraseMode && event && Math.hypot(event.clientX - c.x, event.clientY - c.y) <= 4) setSelected(null)
  }

  function zoomCanvas(event) {
    if (!editMode || eraseMode) return
    event.preventDefault()
    const viewport = viewportRef.current
    if (!viewport) return
    const rect = viewport.getBoundingClientRect()
    const relX = (event.clientX - rect.left) - rect.width / 2
    const relY = (event.clientY - rect.top) - rect.height / 2
    const z = canvasZoom
    const nextZoom = Math.max(0.2, Math.min(16, z * (event.deltaY < 0 ? 1.12 : 0.9)))
    if (nextZoom === z) return
    const factor = nextZoom / z
    setCanvasZoom(nextZoom)
    setPan({ x: relX - factor * (relX - pan.x), y: relY - factor * (relY - pan.y) })
  }

  function clearEraseMask() {
    const canvas = eraseCanvasRef.current
    if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)
    setEraseHasMask(false)
  }

  function enterEraseMode() {
    if (!editMode || regionsLoading || eraseBusy) return
    clearEraseMask(); setSelected(null); setEraseMode(true); setError('')
  }

  function leaveEraseMode() {
    if (eraseBusy) return
    clearEraseMask(); setEraseCursor(null); setEraseMode(false)
  }

  function erasePoint(event) {
    const canvas = eraseCanvasRef.current
    const holder = imageCanvasRef.current
    if (!canvas || !holder) return
    const rect = holder.getBoundingClientRect()
    return { x: (event.clientX - rect.left) * imageSize.width / Math.max(1, rect.width), y: (event.clientY - rect.top) * imageSize.height / Math.max(1, rect.height) }
  }

  function startErase(event) {
    if (!eraseMode || eraseBusy) return
    event.preventDefault(); event.stopPropagation()
    const canvas = eraseCanvasRef.current; const point = erasePoint(event)
    if (!canvas || !point) return
    const ctx = canvas.getContext('2d'); ctx.strokeStyle = NEON_BRUSH_COLOR; ctx.fillStyle = NEON_BRUSH_COLOR; ctx.lineWidth = brushSize; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; setEraseCursor(point)
    ctx.beginPath(); ctx.moveTo(point.x, point.y); ctx.lineTo(point.x + 0.01, point.y + 0.01); ctx.stroke()
    setEraseHasMask(true); eraseDrawingRef.current = true; event.currentTarget.setPointerCapture?.(event.pointerId)
  }

  function moveErase(event) {
    event.preventDefault()
    const canvas = eraseCanvasRef.current; const point = erasePoint(event)
    if (!canvas || !point) return
    setEraseCursor(point)
    if (!eraseDrawingRef.current) return
    const ctx = canvas.getContext('2d'); ctx.lineTo(point.x, point.y); ctx.stroke(); setEraseHasMask(true)
  }

  function endErase() {
    eraseDrawingRef.current = false
  }

  async function confirmErase() {
    const canvas = eraseCanvasRef.current
    if (!canvas || !eraseHasMask || eraseBusy) return
    setEraseBusy(true); setError('')
    try {
      const blob = await new Promise((resolve, reject) => canvas.toBlob((value) => value ? resolve(value) : reject(new Error('无法生成擦除 mask')), 'image/png'))
      await eraseTask(value.taskId, blob)
      setEraseMode(false); clearEraseMask(); setEraseCursor(null); setHasEraseHistory(true); setCacheBust(Date.now()); onUpdated(value.taskId)
    } catch (err) {
      setError(`背景修复失败，原图片未改变：${err.message}`)
    } finally { setEraseBusy(false) }
  }

  async function undoErase() {
    if (eraseBusy) return
    setEraseBusy(true); setError('')
    try {
      const response = await undoEraseTask(value.taskId)
      setHasEraseHistory(Boolean(response.has_erase_history)); setCacheBust(Date.now()); onUpdated(value.taskId)
    } catch (err) { setError(`撤销擦除失败，当前图片未改变：${err.message}`) }
    finally { setEraseBusy(false) }
  }

  function startRegionInteraction(event, region, handle = '') {
    event.preventDefault(); event.stopPropagation(); selectRegion(region)
    const canvas = imageCanvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect(); const start = regionBox(region)
    const pointer = (clientX, clientY) => ({ x: (clientX - rect.left) * imageSize.width / Math.max(1, rect.width), y: (clientY - rect.top) * imageSize.height / Math.max(1, rect.height) })
    regionInteractionRef.current = { index: region.index, handle, start, startPointer: pointer(event.clientX, event.clientY), before: start, pointer, pointerId: event.pointerId, dragging: false }
    baseBoxRef.current.set(region.index, { x: start.x, y: start.y, width: start.width, height: start.height })
    event.currentTarget.setPointerCapture?.(event.pointerId)
  }

  useEffect(() => {
    const move = (event) => {
      const interaction = regionInteractionRef.current
      if (!interaction) return
      if (interaction.pointerId != null && event.pointerId != null && interaction.pointerId !== event.pointerId) return
      event.preventDefault(); const now = interaction.pointer(event.clientX, event.clientY)
      const dx = now.x - interaction.startPointer.x; const dy = now.y - interaction.startPointer.y; const { start, handle } = interaction
      if (!interaction.dragging && Math.hypot(dx, dy) > 3) { interaction.dragging = true; setDraggingIndex(interaction.index) }
      const next = { ...start }
      if (!handle) {
        const size = imageSizeRef.current
        next.x = Math.max(-start.width + REGION_MIN_VISIBLE, Math.min(size.width - REGION_MIN_VISIBLE, start.x + dx))
        next.y = Math.max(-start.height + REGION_MIN_VISIBLE, Math.min(size.height - REGION_MIN_VISIBLE, start.y + dy))
      } else {
        const size = imageSizeRef.current
        if (handle.includes('e')) next.width = Math.max(REGION_MIN_SIZE, Math.min(size.width * REGION_MAX_SIZE_RATIO - start.x, start.width + dx))
        if (handle.includes('s')) next.height = Math.max(REGION_MIN_SIZE, Math.min(size.height * REGION_MAX_SIZE_RATIO - start.y, start.height + dy))
        if (handle.includes('w')) { const x = Math.max(0, Math.min(start.x + start.width - REGION_MIN_SIZE, start.x + dx)); next.x = x; next.width = start.width + start.x - x }
        if (handle.includes('n')) { const y = Math.max(0, Math.min(start.y + start.height - REGION_MIN_SIZE, start.y + dy)); next.y = y; next.height = start.height + start.y - y }
      }
      updateRegionBox(interaction.index, next); interaction.last = next
    }
    const up = (event) => {
      const interaction = regionInteractionRef.current
      if (!interaction) return
      if (interaction.pointerId != null && event?.pointerId != null && interaction.pointerId !== event.pointerId) return
      regionInteractionRef.current = null
      if (interaction.last && JSON.stringify(interaction.last) !== JSON.stringify(interaction.before)) {
        persistRegionBox(interaction.index, interaction.last, interaction.before)
      } else {
        baseBoxRef.current.delete(interaction.index)
        setDraggingIndex(null)
      }
    }
    window.addEventListener('pointermove', move, { passive: false }); window.addEventListener('pointerup', up)
    window.addEventListener('pointercancel', up)
    return () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); window.removeEventListener('pointercancel', up) }
  }, [])

  useEffect(() => {
    if (!value) return undefined
    let active = true
    const nextEditMode = Boolean(value.editMode)
    if (regionSyncTimer.current) clearTimeout(regionSyncTimer.current)
    regionSyncRequest.current += 1
    setRegions([]); setFontOptions([]); setImageSize({ width: 1, height: 1 }); setEditMode(nextEditMode); setRegionsLoading(nextEditMode); setSelected(null); setEraseMode(false); setEraseHasMask(false); setEraseCursor(null); setHasEraseHistory(false); setEraseBusy(false); setPanelOffset({ x: 0, y: 0 }); setCanvasZoom(1); setPan({ x: 0, y: 0 }); setFontPanelOpen(false); setFontBeforePanel({ family: '', weight: 400 }); setFontWeight(400); setFontPreviewUrl(''); setFontPreviewLoading(false); setDefaultTextColor(true); setColorPreviewActive(false); setColorPanelOpen(false); setError(''); setCacheBust(value.revision || 0); setImgLoading(false); setImgError(false); setOpenToken((token) => token + 1); setDisplaySrc(`${value.url}${value.revision ? `?v=${value.revision}` : ''}`)
    if (!nextEditMode) return () => { active = false }
    getTaskRegions(value.taskId).then((response) => {
      if (active) {
        const nextRegions = response.regions || []
        setRegions(nextRegions); setFontOptions(response.font_options || []); setImageSize({ width: response.width || 1, height: response.height || 1 }); setHasEraseHistory(Boolean(response.has_erase_history)); setRegionsLoading(false)
      }
    }).catch((err) => { if (active) { setRegionsLoading(false); setError(`文本区域加载失败：${err.message}`) } })
    return () => { active = false }
  }, [value?.taskId])

  useEffect(() => {
    if (!value) return undefined
    const raf = requestAnimationFrame(() => recenterView())
    return () => cancelAnimationFrame(raf)
  }, [value?.taskId, imageSize.width, imageSize.height])

  useEffect(() => {
    if (!value || !value.url) return undefined
    const target = `${value.url}${cacheBust ? `?v=${cacheBust}` : ''}`
    if (!target || target === displaySrc) return undefined
    const preload = new Image()
    preload.onload = () => setDisplaySrc(target)
    preload.onerror = () => setDisplaySrc(target)
    preload.src = target
    return undefined
  }, [cacheBust, value?.url, value?.taskId])

  useEffect(() => {
    if (!value) return undefined
    setImgLoading(false)
    imgLoadedRef.current = false
    imgLoadTimer.current = window.setTimeout(() => { if (!imgLoadedRef.current) setImgLoading(true) }, 400)
    return () => { if (imgLoadTimer.current) { clearTimeout(imgLoadTimer.current); imgLoadTimer.current = null } }
  }, [openToken])

  useEffect(() => {
    const canvas = eraseCanvasRef.current
    if (!canvas || !imageSize.width || !imageSize.height) return
    canvas.width = imageSize.width; canvas.height = imageSize.height
  }, [imageSize, eraseMode])

  useEffect(() => () => { if (fontPreviewUrl) URL.revokeObjectURL(fontPreviewUrl) }, [fontPreviewUrl])

  useEffect(() => {
    if (!value) return undefined
    const close = (event) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', close)
    return () => window.removeEventListener('keydown', close)
  }, [value, onClose])

  useEffect(() => {
    const keydown = (event) => {
      const target = event.target
      if (target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))) return
      if (event.code !== 'Space') return
      event.preventDefault(); spacePressedRef.current = true; setSpacePressed(true)
    }
    const keyup = (event) => {
      if (event.code !== 'Space') return
      spacePressedRef.current = false; setSpacePressed(false); panRef.current = null
    }
    const blur = () => { spacePressedRef.current = false; setSpacePressed(false); panRef.current = null }
    window.addEventListener('keydown', keydown); window.addEventListener('keyup', keyup); window.addEventListener('blur', blur)
    return () => { window.removeEventListener('keydown', keydown); window.removeEventListener('keyup', keyup); window.removeEventListener('blur', blur) }
  }, [])

  async function saveSelectedRegion() {
    if (!selected || saving || !draft.trim()) return
    if (fontPreviewTimer.current) { clearTimeout(fontPreviewTimer.current); fontPreviewTimer.current = null }
    setSaving(true)
    setError('')
    try {
      const response = await editTaskRegion(value.taskId, selected.index, draft, {
        fontSize,
        fontFamily,
        fontWeight,
        color: defaultTextColor ? '' : textColor,
        ...regionBox(selected),
      })
      const revision = Date.now()
      const parsedColor = defaultTextColor
        ? null
        : textColor.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i)?.slice(1).map((part) => parseInt(part, 16))
      setRegions((items) => items.map((item) => item.index === selected.index ? {
        ...item,
        translated: draft.trim(),
        font_size: response.font_size,
        default_font_size: response.default_font_size || item.default_font_size,
        font_family: fontFamily,
        font_weight: response.font_weight || fontWeight,
        color: parsedColor,
      } : item))
      setSelected(null)
      setColorPanelOpen(false)
      setError(response.warning || '')
      setCacheBust(revision)
      onUpdated(value.taskId)
    } catch (err) {
      setError(`保存失败：${err.message}`)
    } finally {
      setSaving(false)
    }
  }

  function openFontPanel() {
    setFontBeforePanel({ family: fontFamily, weight: fontWeight })
    setFontPanelOpen(true)
    setColorPanelOpen(false)
    setError('')
  }

  function closeFontPanel(saveDraft) {
    if (!saveDraft) { setFontFamily(fontBeforePanel.family); setFontWeight(fontBeforePanel.weight) }
    setFontPanelOpen(false)
    setFontPreviewLoading(false)
    setFontPreviewUrl((current) => { if (current) URL.revokeObjectURL(current); return '' })
  }

  async function previewFont(fontFamilyValue, fontWeightValue) {
    const requestId = fontPreviewRequest.current + 1
    fontPreviewRequest.current = requestId
    setFontPreviewLoading(true)
    try {
      const response = await previewTaskRegionFont(value.taskId, selected.index, fontFamilyValue, fontWeightValue)
      const nextUrl = URL.createObjectURL(await response.blob())
      if (fontPreviewRequest.current !== requestId) { URL.revokeObjectURL(nextUrl); return }
      setFontPreviewUrl((current) => { if (current) URL.revokeObjectURL(current); return nextUrl })
    } catch (err) {
      if (fontPreviewRequest.current === requestId) setError(`字体预览失败：${err.message}`)
    } finally {
      if (fontPreviewRequest.current === requestId) setFontPreviewLoading(false)
    }
  }

  function selectFont(font) {
    setFontPreviewUrl((current) => { if (current) URL.revokeObjectURL(current); return '' })
    setFontPreviewLoading(false)
    setImgError(false)
    setFontFamily(font.value)
    setFontPanelOpen(false)
    scheduleRegionSync({ fontFamily: font.value })
  }

  function selectFontWeight(value) {
    const nextWeight = Number(value)
    setFontWeight(nextWeight)
    scheduleRegionSync({ fontWeight: nextWeight })
  }

  function selectTextColor(color, useDefault = false) {
    setTextColor(color)
    setDefaultTextColor(useDefault)
    setColorPreviewActive(!useDefault)
    scheduleRegionSync({ color: useDefault ? '' : color })
  }

  useEffect(() => {
    if (!selected) { styleSnapshotRef.current = null; return }
    const snapshot = `${selected.index}:${textColor}:${defaultTextColor}`
    const previous = styleSnapshotRef.current
    styleSnapshotRef.current = snapshot
    if (previous && previous.split(':')[0] === String(selected.index) && previous !== snapshot) {
      scheduleRegionSync({ color: defaultTextColor ? '' : textColor })
    }
  }, [selected?.index, textColor, defaultTextColor])

  return (
    <>
      {value ? (
        <motion.div ref={lightboxRef} className="fixed inset-0 z-[80] flex items-center justify-center bg-black/85 p-3 backdrop-blur-sm"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} role="dialog" aria-modal="true" aria-label={`${value.filename} 大图预览`}>
          <div className="flex max-h-[98vh] max-w-[99vw] flex-col overflow-hidden rounded-xl border border-white/10 bg-black/75 shadow-2xl" onClick={(event) => event.stopPropagation()}>
            <div className="flex h-12 shrink-0 items-center justify-between gap-4 border-b border-white/10 px-3">
              <p className="min-w-0 truncate text-sm text-white/70">{value.filename}</p>
              <div className="flex shrink-0 items-center gap-2">
                {!editMode ? <button type="button" onClick={toggleEditMode} disabled={regionsLoading} className="flex h-9 w-9 items-center justify-center rounded-lg border border-accent/40 bg-accent/10 text-accent hover:bg-accent/20 disabled:opacity-50" title="开始编辑" aria-label="开始编辑"><Pencil size={16} /></button> : null}
                {editMode ? <button type="button" onClick={toggleEditMode} className={`flex h-9 w-9 items-center justify-center rounded-lg border ${eraseMode ? 'border-accent/50 bg-accent/20 text-accent' : 'border-accent/40 bg-accent/10 text-accent'}`} title={eraseMode ? '退出编辑模式' : '退出编辑模式'} aria-label="退出编辑模式"><Pencil size={16} /></button> : null}
                {editMode && !eraseMode && hasEraseHistory ? <button type="button" onClick={undoErase} disabled={eraseBusy} className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/15 bg-white/5 text-white/80 hover:bg-white/10 disabled:opacity-50" title="撤销最近一次擦除" aria-label="撤销最近一次擦除"><Undo2 size={16} /></button> : null}
                <button type="button" disabled={downloading} onClick={onDownload} className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/15 bg-white/5 text-white hover:bg-white/10 disabled:opacity-50" title="下载" aria-label="下载">
                  {downloading ? <Loader2 size={16} className="animate-spin" /> : <ArrowDownToLine size={16} />}
                </button>
                <button type="button" onClick={onClose} className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/15 bg-white/5 text-white hover:bg-white/10" aria-label="关闭大图"><X size={18} /></button>
              </div>
            </div>
            <div ref={viewportRef} className={`relative min-h-0 overflow-hidden flex items-center justify-center ${editMode && !eraseMode ? (spacePressed ? 'cursor-grabbing' : 'cursor-grab') : ''}`} onWheel={zoomCanvas} onPointerDown={startCanvasPan} onPointerMove={moveCanvasPan} onPointerUp={endCanvasPan} onPointerCancel={endCanvasPan}>
              {editMode && regionsLoading ? <div className="pointer-events-auto absolute inset-0 z-30 flex min-h-24 items-center justify-center bg-black/35"><div className="inline-flex items-center gap-2 rounded-xl border border-white/20 bg-black/75 px-4 py-2 text-sm text-white/80 shadow-xl"><Loader2 size={15} className="animate-spin" />正在加载文本框</div></div> : null}
              {imgLoading ? (
                <motion.div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <div className="flex flex-col items-center gap-5 rounded-3xl border border-white/10 bg-black/60 px-10 py-9 shadow-2xl backdrop-blur-xl">
                    <div className="relative h-14 w-14">
                      <div className="absolute inset-0 rounded-full border-2 border-white/10" />
                      <div className="absolute inset-0 animate-spin rounded-full border-2 border-transparent border-t-accent border-r-accent/40" style={{ animationDuration: '0.9s' }} />
                      <ImageIcon size={20} className="absolute inset-0 m-auto text-white/70" />
                    </div>
                    <div className="text-center">
                      <p className="text-sm font-medium text-ink-100">正在加载预览</p>
                      <p className="mt-1 text-xs text-ink-500">大图加载中，请稍候</p>
                    </div>
                    <div className="flex gap-1.5">
                      {[0, 1, 2].map((i) => (
                        <motion.span key={i} className="h-1.5 w-1.5 rounded-full bg-accent/70"
                          animate={{ opacity: [0.3, 1, 0.3] }}
                          transition={{ duration: 1.1, repeat: Infinity, delay: i * 0.18 }} />
                      ))}
                    </div>
                  </div>
                </motion.div>
              ) : null}
              {imgError && !imgLoading ? <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center"><div className="rounded-xl border border-red-400/30 bg-red-500/10 px-4 py-2 text-sm text-red-200">预览图加载失败，请稍后重试或直接下载</div></div> : null}
              <div ref={imageCanvasRef} className="relative" style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${canvasZoom})`, transformOrigin: 'center', willChange: 'transform' }}>
                <img key={value.taskId} src={imageSrc} alt={`${value.filename} 翻译结果大图`} className="block max-h-[calc(98vh-48px)] max-w-[99vw] select-none object-contain"
                  onLoad={() => { imgLoadedRef.current = true; if (imgLoadTimer.current) { clearTimeout(imgLoadTimer.current); imgLoadTimer.current = null }; setImgLoading(false) }}
                  onError={() => {
                    if (fontPreviewUrl) {
                      URL.revokeObjectURL(fontPreviewUrl)
                      setFontPreviewUrl('')
                      setImgError(false)
                      return
                    }
                    if (imgLoadTimer.current) { clearTimeout(imgLoadTimer.current); imgLoadTimer.current = null }
                    setImgLoading(false); setImgError(true)
                  }} />
                {editMode && value.cleanedUrl ? <img src={`${value.cleanedUrl}${cacheBust ? `?v=${cacheBust}` : ''}`} alt="" className="pointer-events-none absolute inset-0 z-[2] block h-full w-full object-contain" /> : null}
                {editMode && imageSize.width > 1 ? <LiveTextLayer regions={regions} selected={selected} draft={draft} fontSize={fontSize} fontFamily={fontFamily} fontWeight={fontWeight} defaultTextColor={defaultTextColor} textColor={textColor} fontOptions={fontOptions} imageSize={imageSize} draggingIndex={draggingIndex} baseBoxRef={baseBoxRef} hidden={panActive} className={eraseMode ? 'z-40' : 'z-[5]'} /> : null}
                {eraseMode ? <canvas ref={eraseCanvasRef} className="absolute inset-0 z-30 h-full w-full cursor-none touch-none opacity-45" onPointerDown={startErase} onPointerMove={moveErase} onPointerEnter={moveErase} onPointerLeave={() => { if (!eraseDrawingRef.current) setEraseCursor(null) }} onPointerUp={endErase} onPointerCancel={endErase} aria-label="擦除区域画布" /> : null}
                {eraseMode && eraseCursor ? <span className="pointer-events-none absolute z-40 rounded-full" style={{ width: brushSize * (imageCanvasRef.current?.getBoundingClientRect().width || imageSize.width) / Math.max(1, imageSize.width), height: brushSize * (imageCanvasRef.current?.getBoundingClientRect().width || imageSize.width) / Math.max(1, imageSize.width), left: `${eraseCursor.x / imageSize.width * 100}%`, top: `${eraseCursor.y / imageSize.height * 100}%`, transform: 'translate(-50%, -50%)', border: `2px solid ${NEON_BRUSH_COLOR}`, backgroundColor: NEON_BRUSH_FILL }} /> : null}
                {editMode ? <svg className={`absolute inset-0 z-10 h-full w-full touch-none ${eraseMode ? 'pointer-events-none' : ''}`} viewBox={`0 0 ${imageSize.width} ${imageSize.height}`} preserveAspectRatio="none" aria-label="文本框编辑画布">
                  {regions.map((region) => { const box = regionBox(region); const active = selected?.index === region.index; return <rect key={region.id || region.index} x={box.x} y={box.y} width={box.width} height={box.height} rx={6} fill={active ? 'rgba(129,140,248,0.22)' : NEON_REGION_FILL} stroke={active ? '#a5b4fc' : NEON_BRUSH_COLOR} strokeWidth={Math.max(2, imageSize.width / 360)} vectorEffect="non-scaling-stroke" className="cursor-move" onPointerDown={(event) => startRegionInteraction(event, region)} /> })}
                </svg> : null}
                {editMode && selected ? (() => {
                  const box = regionBox(selected)
                  const handles = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w']
                  const canvasRect = imageCanvasRef.current?.getBoundingClientRect()
                  const displayScale = canvasRect ? Math.min(canvasRect.width / Math.max(1, imageSize.width), canvasRect.height / Math.max(1, imageSize.height)) : 1
                  const displayedMin = Math.min(box.width, box.height) * displayScale
                  const handleSize = Math.max(7, Math.min(11, Math.round(displayedMin * 0.16)))
                  const handleOffset = Math.round(handleSize / 2)
                  const deleteSize = Math.max(22, Math.min(30, Math.round(displayedMin * 0.42)))
                  return <div className="pointer-events-none absolute z-20 touch-none" style={{ left: `${box.x / imageSize.width * 100}%`, top: `${box.y / imageSize.height * 100}%`, width: `${box.width / imageSize.width * 100}%`, height: `${box.height / imageSize.height * 100}%` }}>
                    <div className="pointer-events-auto absolute inset-0 cursor-move" onPointerDown={(event) => startRegionInteraction(event, selected)} />
                    {handles.map((handle) => <span key={handle} onPointerDown={(event) => startRegionInteraction(event, selected, handle)} style={{ width: handleSize, height: handleSize, top: handle.includes('n') ? -handleOffset : handle.includes('s') ? 'auto' : '50%', bottom: handle.includes('s') ? -handleOffset : 'auto', left: handle.includes('w') ? -handleOffset : handle.includes('e') ? 'auto' : '50%', right: handle.includes('e') ? -handleOffset : 'auto', transform: handle.includes('n') || handle.includes('s') ? (handle === 'n' || handle === 's' ? 'translateX(-50%)' : undefined) : 'translate(-50%, -50%)' }} className={`pointer-events-auto absolute rounded-sm border border-white bg-accent shadow ${handle === 'n' || handle === 's' ? 'cursor-ns-resize' : handle === 'e' || handle === 'w' ? 'cursor-ew-resize' : handle === 'ne' || handle === 'sw' ? 'cursor-nesw-resize' : 'cursor-nwse-resize'}`} />)}
                    <button type="button" disabled={deleting} onPointerDown={(event) => { event.preventDefault(); event.stopPropagation() }} onClick={deleteSelectedTextBox} style={{ width: deleteSize, height: deleteSize, left: '50%', top: -(deleteSize + 8), transform: 'translateX(-50%)' }} className="pointer-events-auto absolute flex items-center justify-center rounded-full border border-red-300/70 bg-red-600 text-white shadow-lg transition hover:bg-red-500 disabled:cursor-wait disabled:opacity-60" aria-label="删除当前文本框" title="删除当前文本框">{deleting ? <Loader2 size={Math.max(12, Math.round(deleteSize * 0.48))} className="animate-spin" /> : <Trash2 size={Math.max(12, Math.round(deleteSize * 0.48))} />}</button>
                  </div>
                })() : null}
              </div>
              {eraseMode ? <div className="pointer-events-auto absolute bottom-4 left-1/2 z-50 flex -translate-x-1/2 items-center gap-3 rounded-xl border border-white/15 bg-black/80 px-3 py-2 text-sm text-white shadow-xl backdrop-blur">
                <label className="flex items-center gap-2 whitespace-nowrap">画笔 <input type="range" min="8" max="120" step="2" value={brushSize} onChange={(event) => { setBrushSize(Number(event.target.value)); setEraseCursor({ x: imageSize.width / 2, y: imageSize.height / 2 }) }} className="w-28 accent-accent" /> <span className="w-8 text-right text-xs text-white/70">{brushSize}px</span></label>
                <button type="button" onClick={clearEraseMask} disabled={eraseBusy} className="rounded-lg px-2.5 py-1.5 text-white/70 hover:bg-white/10 disabled:opacity-50">清空</button>
                <button type="button" onClick={leaveEraseMode} disabled={eraseBusy} className="rounded-lg px-2.5 py-1.5 text-white/70 hover:bg-white/10 disabled:opacity-50">取消</button>
                <button type="button" onClick={confirmErase} disabled={!eraseHasMask || eraseBusy} className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-bg disabled:cursor-not-allowed disabled:opacity-50">{eraseBusy ? <Loader2 size={14} className="animate-spin" /> : <Check size={14} />}确认修复</button>
              </div> : null}
            </div>
          </div>
          {editMode && !selected && error ? <div className="absolute bottom-5 left-1/2 -translate-x-1/2 rounded-lg bg-danger/90 px-4 py-2 text-sm text-white">{error}</div> : null}
          {editMode && !selected && !eraseMode ? <div className="pointer-events-none absolute inset-3 flex items-end justify-center">
            <motion.div drag dragListener={false} dragControls={dragControls} dragConstraints={lightboxRef} dragElastic={0} dragMomentum={false}
              style={{ x: panelOffset.x, y: panelOffset.y, width: 'clamp(280px, 28vw, 380px)', height: 'clamp(220px, 30vh, 320px)', minWidth: 'min(280px, calc(100vw - 24px))', minHeight: '220px', maxWidth: 'calc(100vw - 24px)', maxHeight: 'calc(100vh - 24px)', resize: 'both', overflow: 'auto' }}
              onDragEnd={(_, info) => setPanelOffset((current) => ({ x: current.x + info.offset.x, y: current.y + info.offset.y }))}
              className="edit-panel glass-panel pointer-events-auto relative z-[60] flex max-w-[calc(100vw-24px)] flex-col overflow-hidden rounded-2xl text-left" onPointerDown={(event) => event.stopPropagation()} onClick={(event) => event.stopPropagation()}>
              <div onPointerDownCapture={(event) => { event.preventDefault(); event.stopPropagation(); event.currentTarget.setPointerCapture?.(event.pointerId); dragControls.start(event) }} className="pointer-events-auto absolute left-1/2 top-2 z-[100] h-1.5 w-12 -translate-x-1/2 cursor-move rounded-full bg-ink-500/70 touch-none select-none" aria-label="拖动编辑窗口" />
              <div onPointerDown={(event) => dragControls.start(event)} className="flex cursor-move touch-none items-center justify-between border-b border-line px-5 py-3 select-none">
                <p className="text-sm font-medium text-ink-100">编辑文本框</p><span className="text-xs text-ink-500">请先选择文本框</span>
              </div>
              <div className="flex min-h-0 flex-1 flex-col p-5 pt-4">
                <div className="flex min-h-0 flex-1 items-center justify-center rounded-xl border border-line bg-surface-2 px-4 py-3 text-center text-base text-ink-400" aria-label="文本预览提示">请选择文本框</div>
                <div className="mt-3 flex items-center gap-1"><button type="button" onClick={addTextBox} className="flex h-9 w-9 items-center justify-center rounded-lg text-accent hover:bg-accent/10" title="添加文字" aria-label="添加文字"><SquarePen size={16} /></button><button type="button" onClick={enterEraseMode} disabled={regionsLoading || eraseBusy} className="flex h-9 w-9 items-center justify-center rounded-lg text-accent hover:bg-accent/10 disabled:opacity-50" title="擦除背景" aria-label="擦除背景"><Eraser size={16} /></button></div>
              </div>
            </motion.div>
          </div> : null}
          {editMode && selected && !eraseMode ? <div className="pointer-events-none absolute inset-3 flex items-end justify-center">
            <motion.div drag dragListener={false} dragControls={dragControls} dragConstraints={lightboxRef} dragElastic={0} dragMomentum={false}
              style={{ x: panelOffset.x, y: panelOffset.y, width: '380px', height: '320px', minWidth: '300px', minHeight: '220px', maxWidth: 'calc(100vw - 24px)', maxHeight: 'calc(100vh - 24px)', resize: 'both', overflow: 'auto' }}
              onDragEnd={(_, info) => setPanelOffset((current) => ({ x: current.x + info.offset.x, y: current.y + info.offset.y }))}
              className="edit-panel glass-panel pointer-events-auto relative z-[60] flex max-w-[calc(100vw-24px)] flex-col overflow-hidden rounded-2xl text-left" onPointerDown={(event) => event.stopPropagation()} onClick={(event) => event.stopPropagation()}>
              <div onPointerDownCapture={(event) => { event.preventDefault(); event.stopPropagation(); event.currentTarget.setPointerCapture?.(event.pointerId); dragControls.start(event) }} className="pointer-events-auto absolute left-1/2 top-2 z-[100] h-1.5 w-12 -translate-x-1/2 cursor-move rounded-full bg-ink-500/70 touch-none select-none" aria-label="拖动编辑窗口" />
              <div onPointerDown={(event) => dragControls.start(event)} className="flex cursor-move touch-none items-center justify-between border-b border-line px-5 py-3 select-none">
                <p className="text-sm font-medium text-ink-100">编辑当前文本框完整译文</p><span className="text-xs text-ink-500">拖动此处移动 · 支持换行</span>
              </div>
              <div className="flex min-h-0 flex-1 flex-col p-5 pt-4">
                <textarea value={draft} onChange={(event) => { const next = event.target.value; setDraft(next); scheduleRegionSync({ translated: next }) }} rows={6} autoFocus style={{ fontSize: `${fontSize}px`, color: colorPreviewActive && !defaultTextColor ? textColor : '#f5f5f5', fontFamily: selectedFontOption?.css_family || undefined, fontWeight }} className="max-h-[60vh] min-h-0 flex-1 resize-y overflow-y-auto rounded-xl border border-line bg-surface-2 px-4 py-3 text-base leading-7 text-ink-100 outline-none focus:border-accent/50" aria-label="编辑完整译文" />
                <div className="mt-3 rounded-xl border border-line bg-surface-2 px-3 py-2">
                  <div className="flex items-center justify-between text-xs text-ink-300"><span>文字粗细</span><span className="text-ink-200">{fontWeight}</span></div>
                  <input type="range" min="100" max="900" step="1" value={fontWeight} onChange={(event) => selectFontWeight(event.target.value)} className="mt-2 w-full accent-accent" aria-label="调整文字粗细" />
                </div>
                <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-3">
                  <label className="text-xs text-ink-300">字号<input type="text" inputMode="numeric" pattern="[0-9]*" value={fontSizeInput} onChange={(event) => { const next = event.target.value; if (/^\d*$/.test(next)) { setFontSizeInput(next); if (next) scheduleRegionSync({ fontSize: Math.max(8, Math.min(160, Number(next))) }) } }} onBlur={() => setFontSizeInput(String(fontSize))} className="mt-1 h-9 w-full rounded-lg border border-line bg-surface-2 px-2 text-sm text-ink-100 outline-none focus:border-accent/50" /></label>
                  <label className="text-xs text-ink-300">字体<button type="button" onClick={openFontPanel} className="mt-1 flex h-9 w-full items-center justify-between rounded-lg border border-line bg-surface-2 px-3 text-sm text-ink-100 hover:border-accent/40"><span className="truncate">{selectedFontOption?.label || '默认漫画'}</span><ArrowRight size={14} className="shrink-0 text-ink-500" /></button></label>
                  <label className="text-xs text-ink-300">文字颜色<button type="button" onClick={() => setColorPanelOpen(true)} className="mt-1 flex h-9 w-full items-center justify-between rounded-lg border border-line bg-surface-2 px-3 text-sm text-ink-100 hover:border-accent/40"><span className="flex items-center gap-2"><span className="h-4 w-4 rounded-full border border-white/30" style={{ backgroundColor: defaultTextColor ? '#000000' : textColor }} />{defaultTextColor ? '默认' : textColor.toUpperCase()}</span><ArrowRight size={14} className="text-ink-500" /></button></label>
                </div>
                {selected?.default_font_size && (fontSize > selected.default_font_size * 1.5 || fontSize < Math.max(8, selected.default_font_size * 0.5)) ? <p className="mt-1 text-xs text-amber-400">字号明显偏大或偏小，确认后将恢复为原始字号</p> : null}
                {error ? <p className="mt-1 text-xs text-danger">{error}</p> : null}
                <div className="mt-3 flex items-center justify-between gap-2"><div className="flex items-center gap-1"><button type="button" onClick={addTextBox} className="flex h-9 w-9 items-center justify-center rounded-lg text-accent hover:bg-accent/10" title="添加文字" aria-label="添加文字"><SquarePen size={16} /></button><button type="button" onClick={enterEraseMode} disabled={regionsLoading || eraseBusy} className="flex h-9 w-9 items-center justify-center rounded-lg text-accent hover:bg-accent/10 disabled:opacity-50" title="擦除背景" aria-label="擦除背景"><Eraser size={16} /></button><button type="button" onClick={restoreSelectedTextBox} disabled={restoring || saving || deleting} className="flex h-9 w-9 items-center justify-center rounded-lg text-ink-300 hover:bg-surface-2 disabled:opacity-50" title="恢复原图" aria-label="恢复原图">{restoring ? <Loader2 size={16} className="animate-spin" /> : <Undo2 size={16} />}</button></div><div className="flex gap-2"><button type="button" onClick={() => setSelected(null)} className="rounded-lg px-4 py-2 text-sm text-ink-400 hover:bg-surface-2">取消</button>{saving ? <span className="px-4 py-2 text-xs text-ink-500">同步中…</span> : null}</div></div>
              </div>
              {fontPanelOpen ? (
                <div className="absolute inset-0 z-20 flex min-h-0 min-w-0 flex-col rounded-2xl bg-surface px-4 pb-4 pt-5 sm:px-5 sm:pb-5">
                  <div className="flex shrink-0 items-center justify-between border-b border-line pb-3">
                    <button type="button" onClick={() => closeFontPanel(false)} className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-sm text-ink-300 hover:bg-surface-2"><ArrowRight size={14} className="rotate-180" />返回</button>
                    <span className="text-sm text-ink-100">选择字体</span>
                    <span className="w-14" />
                  </div>
                  <div className="min-h-0 min-w-0 flex-1 overflow-y-auto py-3">
                    <p className="mb-3 text-xs text-ink-500">点击字体即可在漫画预览中查看效果</p>
                    <div className="grid min-w-0 grid-cols-1 gap-2 sm:grid-cols-2">
                      {fontOptions.map((font) => <button key={font.value || 'default'} type="button" onClick={() => selectFont(font)} className={`min-w-0 rounded-xl border p-3 text-left transition ${fontFamily === font.value ? 'border-accent bg-accent/10 ring-1 ring-accent/30' : 'border-line bg-surface-2 hover:border-accent/50'}`}>
                        <div className="flex items-center justify-between gap-2"><span className="truncate text-sm text-ink-100">{font.label}</span><span className="shrink-0 rounded-full bg-white/5 px-2 py-0.5 text-[10px] text-ink-500">{font.category || '漫画'}</span></div>
                        <div className="mt-2 max-h-14 overflow-hidden break-words text-sm leading-6 text-ink-200" style={{ fontFamily: font.css_family || undefined, fontWeight: font.weight || 400 }}>{previewText}</div>
                      </button>)}
                    </div>
                  </div>
                  {fontPreviewLoading ? <div className="flex shrink-0 items-center gap-2 py-1 text-xs text-ink-500"><Loader2 size={13} className="animate-spin" />正在更新漫画预览</div> : null}
                  <div className="flex shrink-0 justify-end gap-2 border-t border-line pt-3"><button type="button" onClick={() => closeFontPanel(false)} className="rounded-xl px-4 py-2 text-sm text-ink-400 hover:bg-surface-2">取消</button><button type="button" onClick={() => closeFontPanel(true)} className="rounded-xl bg-accent px-4 py-2 text-sm text-bg">确认</button></div>
                </div>
              ) : colorPanelOpen ? (
                <div className="absolute inset-0 z-20 flex flex-col rounded-2xl bg-surface px-5 pb-5 pt-7">
                  <div className="flex items-center justify-between border-b border-line pb-3">
                    <button type="button" onClick={() => setColorPanelOpen(false)} className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-sm text-ink-300 hover:bg-surface-2"><ArrowRight size={14} className="rotate-180" />返回</button>
                    <span className="text-sm text-ink-100">选择文字颜色</span>
                    <span className="w-14" />
                  </div>
                  <div className="min-h-0 min-w-0 flex-1 overflow-y-auto py-4">
                    <div className="grid grid-cols-3 gap-3 sm:grid-cols-6">
                      {['#000000', '#ffffff', '#dc2626', '#2563eb', '#eab308', '#16a34a'].map((color) => <button key={color} type="button" onClick={() => { setTextColor(color); setDefaultTextColor(false); setColorPreviewActive(true) }} className={`aspect-square min-h-12 w-full rounded-xl border-2 transition ${!defaultTextColor && textColor === color ? 'border-accent scale-105' : 'border-white/20 hover:border-white/50'}`} style={{ backgroundColor: color }} aria-label={`选择颜色 ${color}`} />)}
                    </div>
                    <label className="flex items-center justify-between rounded-xl border border-line bg-surface-2 px-4 py-3 text-sm text-ink-300">自定义颜色<input type="color" value={textColor} onChange={(event) => { setTextColor(event.target.value); setDefaultTextColor(false); setColorPreviewActive(true) }} className="h-9 w-14 cursor-pointer border-0 bg-transparent p-0" /></label>
                    <button type="button" onClick={() => { setTextColor('#000000'); setDefaultTextColor(true); setColorPreviewActive(false) }} className={`rounded-xl border px-4 py-3 text-sm transition ${defaultTextColor ? 'border-accent/50 bg-accent/10 text-accent' : 'border-line text-ink-300 hover:bg-surface-2'}`}>恢复默认文字颜色</button>
                  </div>
                  <button type="button" onClick={() => setColorPanelOpen(false)} className="rounded-xl bg-accent px-4 py-2.5 text-sm text-bg">完成</button>
                </div>
              ) : null}
            </motion.div>
          </div> : null}
        </motion.div>
      ) : null}
    </>
  )
}

function EmptyConversation() {
  return (
    <motion.div initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="flex min-h-[46vh] flex-col items-center justify-center text-center">
      <ComicXWordmark />
    </motion.div>
  )
}

function BatchMessage({ batch, zipLoading, downloadingTaskId, retryingTaskId, onDownloadZip, onDownloadSingle, onRetry, onView }) {
  const stats = batchStats(batch)
  const overall = batchStatus(batch)
  return (
    <motion.article initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} className="mb-14 space-y-7">
      <div className="ml-auto w-fit max-w-full">
        <div className="rounded-2xl rounded-br-md border border-white/10 bg-[#242124] p-4 sm:p-5">
          <p className="font-medium text-ink-100">已提交 {batch.items.length} 张图片</p>
          <div className="mt-4 flex max-w-full flex-wrap gap-2">
            {batch.items.slice(0, SUMMARY_MAX).map((item, index) => <div key={item.task_id} className="w-16 shrink-0 sm:w-20"><OriginalThumb item={item} displayIndex={index} /></div>)}
            {batch.items.length > SUMMARY_MAX ? <div className="flex aspect-[4/5] w-16 shrink-0 items-center justify-center rounded-lg border border-dashed border-white/15 bg-white/[0.03] text-xs font-medium text-ink-400 sm:w-20">+{batch.items.length - SUMMARY_MAX}</div> : null}
          </div>
        </div>
      </div>
      <div className="flex gap-3 sm:gap-4">
      <div className="mt-1 flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded-xl border border-white/10 bg-[#242124]"><ComicXMark rounded="rounded-none" className="h-full w-full" /></div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center justify-end gap-3">
            {!stats.active && stats.completed > 0 ? (
              <SpecularButton {...SPECULAR_SECONDARY} size="sm" disabled={zipLoading} onClick={onDownloadZip}
                className="text-sm disabled:cursor-wait">
                {zipLoading ? <Loader2 size={15} className="animate-spin" /> : <ArrowDownToLine size={15} />} 下载本批 ZIP
              </SpecularButton>
            ) : null}
          </div>
          <div className="space-y-4">
            {batch.items.map((item, index) => <TaskCard key={item.task_id} item={item} displayIndex={index + 1}
              downloading={downloadingTaskId === item.task_id}
              retrying={retryingTaskId === item.task_id}
              onDownload={() => onDownloadSingle(item)} onRetry={() => onRetry(item.task_id)}
              onView={() => onView(item, false)} />)}
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
  const [resultLoading, setResultLoading] = useState(true)
  const progress = Math.max(0, Math.min(100, item.progress || 0))
  const status = item.status === 'processing' ? 'processing' : item.status
  const statusMeta = {
    queued: { label: '等待中', icon: Clock3, color: 'text-ink-400' },
    processing: { label: '处理中', icon: Loader2, color: 'text-accent' },
    completed: { label: '已完成', icon: Check, color: 'text-ok' },
    failed: { label: '失败', icon: AlertTriangle, color: 'text-danger' },
  }[status] || { label: status || '等待中', icon: Clock3, color: 'text-ink-400' }
  const StatusIcon = statusMeta.icon
  const resultUrl = status === 'completed' ? `${getTaskResultUrl(item.task_id)}?v=${item.result_revision || 0}` : ''
  const filename = itemFilename(item, displayIndex - 1)
  useEffect(() => { setPreviewFailed(false); setResultLoading(true) }, [resultUrl])
  return (
    <motion.section layout className="overflow-hidden rounded-2xl border border-white/10 bg-[#202020]">
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
        <div className="p-3 sm:p-4">
          {previewFailed ? (
            <div className="flex min-h-36 items-center justify-center rounded-xl border border-danger/20 bg-danger/5 px-4 text-center text-sm text-danger">结果图片加载失败，请稍后重试或直接下载。</div>
          ) : (
            <div className="group relative block w-full overflow-hidden rounded-xl border border-line bg-[#202020]">
              {resultLoading ? <div className="flex min-h-36 items-center justify-center text-ink-500"><Loader2 size={20} className="animate-spin" /></div> : null}
              <img src={resultUrl} alt={`${filename} 翻译结果`} decoding="async" style={{ display: resultLoading ? 'none' : 'block' }}
                onLoad={() => setResultLoading(false)} onError={() => { setPreviewFailed(true); setResultLoading(false) }} onClick={onView} className="max-h-[720px] w-full cursor-zoom-in object-contain" />
              <div className="absolute right-3 top-3 flex gap-2 opacity-0 transition group-hover:opacity-100 focus-within:opacity-100">
                <button type="button" onClick={(event) => { event.stopPropagation(); onView() }} className="flex items-center gap-1.5 rounded-lg bg-black/75 px-2.5 py-1.5 text-xs text-white hover:bg-black/90"><Maximize2 size={13} />查看大图</button>
              </div>
            </div>
          )}
        </div>
      ) : null}
      {status === 'failed' ? (
        <div className="flex flex-wrap items-center gap-2 px-4 py-3 sm:px-5">
          <SpecularButton {...SPECULAR_SECONDARY} size="sm" disabled={retrying} onClick={onRetry}
            className="text-xs disabled:cursor-wait">
            {retrying ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}重新翻译
          </SpecularButton>
        </div>
      ) : null}
      {status === 'completed' ? (
        <div className="flex flex-wrap items-center gap-2 px-4 py-3 sm:px-5">
                <SpecularButton {...SPECULAR_SECONDARY} size="sm" disabled={downloading} onClick={onDownload}
            className="text-xs disabled:cursor-wait">
            {downloading ? <Loader2 size={14} className="animate-spin" /> : <ArrowDownToLine size={14} />}下载图片
                </SpecularButton>
        </div>
      ) : null}
    </motion.section>
  )
}

function Composer({ files, sourceLang, targetLang, totalBytes, polishEnabled, polishStyle, customPrompt, fileError, isDragging, submitting,
  languageInvalid, inputRef, folderInputRef, collapsed, onSourceLangChange, onTargetLangChange, onPolishEnabledChange, onPolishStyleChange, onCustomPromptChange, onFileSelect, onFolderSelect,
  onRemoveFile, onDraggingChange, onDrop, onSubmit }) {
  return (
    <div className={`fixed inset-x-0 bottom-0 z-40 px-3 pb-4 pt-2 transition-[left] duration-300 sm:px-6 sm:pb-6 ${collapsed ? 'md:left-16' : 'md:left-72'}`}>
      <div className={`glass-panel mx-auto max-w-4xl rounded-2xl p-3 transition sm:p-4 ${isDragging ? 'border-accent/60 ring-4 ring-accent/10' : ''}`}
        onDragOver={(event) => { event.preventDefault(); onDraggingChange(true) }}
        onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) onDraggingChange(false) }} onDrop={onDrop}>
        <input ref={inputRef} type="file" accept={ALLOWED} multiple className="hidden" onChange={onFileSelect} />
        <input ref={folderInputRef} type="file" webkitdirectory="" directory="" multiple className="hidden" onChange={onFolderSelect} />
        {files.length ? <div className="mb-3 flex max-w-full flex-wrap gap-2">
          {files.slice(0, PENDING_MAX).map((file, index) => <PendingFile key={`${file.name}-${file.size}-${file.lastModified}-${index}`} file={file} onRemove={() => onRemoveFile(index)} />)}
          {files.length > PENDING_MAX ? <div className="flex h-20 w-16 items-center justify-center rounded-lg border border-dashed border-white/15 bg-white/[0.03] text-xs font-medium text-ink-400">+{files.length - PENDING_MAX}</div> : null}
        </div> : null}
        <div className="flex items-center gap-2">
          {/* 文件夹也可通过拖入上传区域导入；文件夹选择按钮使用隐藏 input。 */}
          <button type="button" onClick={() => inputRef.current?.click()} className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-white/10 text-ink-300 transition hover:bg-white/10 hover:text-ink-100" aria-label="选择图片" title="选择多张图片"><Paperclip size={18} /></button>
          <button type="button" onClick={() => folderInputRef.current?.click()} className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-white/10 text-ink-300 transition hover:bg-white/10 hover:text-ink-100" aria-label="选择文件夹" title="选择文件夹"><LayoutGrid size={17} /></button>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm text-ink-200">{files.length ? `已选择 ${files.length} 张图片 · ${formatBytes(totalBytes)}` : '选择、拖入或粘贴漫画图片'}</p>
          </div>
          <SpecularButton {...SPECULAR_PRIMARY} size="sm" icon disabled={!files.length || submitting || languageInvalid} onClick={onSubmit}
            className="h-10 w-10 shrink-0" aria-label="提交翻译" title="提交翻译">
            {submitting ? <Loader2 size={18} className="animate-spin" /> : <ArrowUp size={19} />}
          </SpecularButton>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-white/10 pt-3">
          <CompactLangSelect value={sourceLang} options={LANGUAGE_OPTIONS} onChange={onSourceLangChange} label="源语言" />
          <ArrowRight size={14} className="text-ink-500" />
          <CompactLangSelect value={targetLang} options={TARGET_OPTIONS} onChange={onTargetLangChange} label="目标语言" />
          <label className="ml-1 inline-flex items-center gap-1.5 text-xs text-ink-400">
            <input type="checkbox" checked={polishEnabled} onChange={(event) => onPolishEnabledChange(event.target.checked)} className="accent-accent" /> AI 润色
          </label>
          {polishEnabled ? <CompactLangSelect value={polishStyle} options={POLISH_OPTIONS} onChange={onPolishStyleChange} label="润色风格" /> : null}
          {polishEnabled && polishStyle === 'custom' ? <input value={customPrompt} onChange={(event) => onCustomPromptChange(event.target.value)} maxLength={1000} placeholder="自定义润色提示词" className="min-w-40 flex-1 rounded-lg border border-line bg-surface-2 px-3 py-1.5 text-xs text-ink-200 outline-none focus:border-accent/40" /> : null}
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
