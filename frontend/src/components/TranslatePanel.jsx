import { Fragment, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  UploadCloud,
  FileImage,
  X,
  ArrowRight,
  Wand2,
  Download,
  RotateCcw,
  RefreshCw,
  AlertTriangle,
  Check,
  Loader2,
  ChevronDown,
} from 'lucide-react'
import {
  createTranslateTask,
  createBatchTranslateTask,
  getTaskStatus,
  getBatchTaskStatus,
  getTaskResultUrl,
  downloadBatchZip,
  deleteTask,
  formatBytes,
  formatDuration,
} from '../api'
import { useToast } from './Toast'
import SpecularButton from './SpecularButton'
import GlassSurface from './GlassSurface'
import { SPECULAR_PRIMARY, SPECULAR_SECONDARY } from './specularPresets'

const STEPS = [
  { key: 'detect', label: '检测区域', weight: 0.15 },
  { key: 'ocr', label: '识别文字', weight: 0.25 },
  { key: 'inpaint', label: '修复图像', weight: 0.15 },
  { key: 'translate', label: '翻译', weight: 0.3 },
  { key: 'render', label: '渲染译文', weight: 0.15 },
]

const ALLOWED = '.jpg,.jpeg,.png,.webp,.bmp'
const MAX_MB = 10
const MAX_BATCH_FILES = 10
const MAX_BATCH_MB = 50

function stepIndexFromProgress(progress) {
  let acc = 0
  for (let i = 0; i < STEPS.length; i++) {
    acc += STEPS[i].weight * 100
    if (progress < acc) return i
  }
  return STEPS.length - 1
}

function validateFile(f) {
  if (!f) return '请选择图片文件'
  const ext = (f.name.split('.').pop() || '').toLowerCase()
  if (!ALLOWED.includes(`.${ext}`)) return `不支持 .${ext} 格式，仅支持 JPG / PNG / WebP / BMP`
  if (f.size > MAX_MB * 1024 * 1024) return `文件超过 ${MAX_MB}MB 限制，请压缩后重试`
  if (f.size === 0) return '文件为空'
  return ''
}

function LangSelect({ id, label, value, options, onChange }) {
  return (
    <label className="block">
      <span className="mb-2 block font-mono text-[10px] uppercase tracking-[0.25em] text-ink-500">
        {label}
      </span>
      <div className="relative">
        <GlassSurface
          width="100%"
          height="100%"
          borderRadius={10}
          borderWidth={0.07}
          brightness={55}
          opacity={0.9}
          blur={12}
          backgroundOpacity={0.06}
          saturation={1.3}
          className="absolute inset-0"
        />
        <select
          id={id}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="relative z-10 w-full appearance-none bg-transparent px-4 py-3 pr-10 font-medium text-ink-100 outline-none"
        >
          {options.map((o) => (
            <option key={o.value} value={o.value} className="bg-surface-2">
              {o.label}
            </option>
          ))}
        </select>
        <ChevronDown
          size={16}
          className="pointer-events-none absolute right-3 top-1/2 z-10 -translate-y-1/2 text-ink-500"
        />
      </div>
    </label>
  )
}

const panelMotion = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -16 },
  transition: { type: 'spring', stiffness: 200, damping: 24 },
}

export default function TranslatePanel() {
  const notify = useToast()

  const [phase, setPhase] = useState('idle') // idle | submitting | running | result | error
  const [files, setFiles] = useState([])
  const [filePreviewUrl, setFilePreviewUrl] = useState('')
  const [fileError, setFileError] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [sourceLang, setSourceLang] = useState('ja')
  const [targetLang, setTargetLang] = useState('zh')

  const [taskId, setTaskId] = useState(null)
  const [taskIds, setTaskIds] = useState([])
  const [progress, setProgress] = useState(0)
  const [batchStatus, setBatchStatus] = useState(null)
  const [zipLoading, setZipLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')

  const inputRef = useRef(null)
  const pollRef = useRef(null)
  const pollTokenRef = useRef(0)
  const activeRef = useRef(true)
  const file = files[0] || null

  useEffect(() => {
    activeRef.current = true
    return () => {
      activeRef.current = false
      stopPolling()
    }
  }, [])
  useEffect(() => {
    if (!file) {
      setFilePreviewUrl('')
      return undefined
    }
    const url = URL.createObjectURL(file)
    setFilePreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const currentStep = useMemo(
    () => (phase === 'running' ? stepIndexFromProgress(progress) : -1),
    [phase, progress],
  )

  function stopPolling() {
    pollTokenRef.current += 1
    if (pollRef.current) {
      clearTimeout(pollRef.current)
      pollRef.current = null
    }
  }

  function addFiles(selected) {
    const incoming = Array.from(selected || [])
    if (!incoming.length) return false
    const next = [...files, ...incoming]
    const error = incoming.map(validateFile).find(Boolean)
    if (error) {
      setFileError(error)
      return false
    }
    if (next.length > MAX_BATCH_FILES) {
      setFileError(`批量翻译最多选择 ${MAX_BATCH_FILES} 张图片`)
      return false
    }
    const totalBytes = next.reduce((sum, item) => sum + item.size, 0)
    if (totalBytes > MAX_BATCH_MB * 1024 * 1024) {
      setFileError(`批量图片总大小不能超过 ${MAX_BATCH_MB}MB`)
      return false
    }
    setFileError('')
    setFiles(next)
    return true
  }

  function onFileSelect(e) {
    addFiles(e.target.files)
    e.target.value = ''
  }

  function onDrop(e) {
    e.preventDefault()
    setIsDragging(false)
    addFiles(e.dataTransfer.files)
  }

  function removeFile(index) {
    setFiles((current) => current.filter((_, itemIndex) => itemIndex !== index))
    setFileError('')
  }

  async function startTranslate() {
    if (!files.length) return
    setPhase('submitting')
    setFileError('')
    setErrorMessage('')
    setResult(null)
    setBatchStatus(null)
    setProgress(0)
    try {
      if (files.length === 1) {
        const data = await createTranslateTask(files[0], sourceLang, targetLang)
        if (!activeRef.current) return
        setTaskId(data.task_id)
        setTaskIds([data.task_id])
        setPhase('running')
        startPolling([data.task_id], false)
      } else {
        const data = await createBatchTranslateTask(files, sourceLang, targetLang)
        if (!activeRef.current) return
        const ids = data.items.map((item) => item.task_id)
        setTaskIds(ids)
        setBatchStatus({
          total: data.total,
          completed: 0,
          processing: data.total,
          failed: 0,
          progress: 0,
          items: data.items.map((item) => ({ ...item, progress: 0 })),
        })
        setPhase('running')
        startPolling(ids, true)
      }
    } catch (err) {
      if (!activeRef.current) return
      setPhase('error')
      setErrorMessage(err.message)
      notify(err.message, 'error')
    }
  }

  function startPolling(ids, isBatch) {
    stopPolling()
    const token = pollTokenRef.current
    const tick = async () => {
      const done = isBatch ? await pollBatchStatus(ids, token) : await pollStatus(ids[0], token)
      if (activeRef.current && token === pollTokenRef.current && !done) {
        pollRef.current = setTimeout(tick, 800)
      }
    }
    tick()
  }

  async function pollStatus(id, token) {
    try {
      const status = await getTaskStatus(id)
      if (!activeRef.current || token !== pollTokenRef.current) return true
      setProgress((current) => Math.max(current, status.progress || 0))
      if (status.status === 'completed') {
        setResult({
          resultUrl: getTaskResultUrl(id),
          originalUrl: filePreviewUrl,
          textCount: status.text_count,
          durationMs: status.duration_ms,
        })
        setPhase('result')
        notify('翻译完成', 'success')
        return true
      } else if (status.status === 'failed') {
        setErrorMessage(status.error || '翻译失败，请重试')
        setPhase('error')
        notify(status.error || '翻译失败', 'error')
        return true
      }
    } catch (err) {
      if (!activeRef.current || token !== pollTokenRef.current) return true
      setErrorMessage(err.message)
      setPhase('error')
      notify('获取进度失败：' + err.message, 'error')
      return true
    }
    return false
  }

  async function pollBatchStatus(ids, token) {
    try {
      const status = await getBatchTaskStatus(ids)
      if (!activeRef.current || token !== pollTokenRef.current) return true
      setBatchStatus(status)
      setProgress(status.progress || 0)
      if (status.completed + status.failed === status.total) {
        setPhase('batch-result')
        notify(
          status.failed ? `批量处理结束，${status.completed} 张成功、${status.failed} 张失败` : '批量翻译完成',
          status.failed ? 'error' : 'success',
        )
        return true
      }
    } catch (err) {
      if (!activeRef.current || token !== pollTokenRef.current) return true
      setErrorMessage(err.message)
      setPhase('error')
      notify('获取批量进度失败：' + err.message, 'error')
      return true
    }
    return false
  }

  function cancelTask() {
    stopPolling()
    const ids = taskIds.length ? taskIds : taskId ? [taskId] : []
    Promise.allSettled(ids.map((id) => deleteTask(id))).catch(() => {})
    resetAll()
  }

  async function downloadZip() {
    setZipLoading(true)
    try {
      const blob = await downloadBatchZip(taskIds)
      if (!activeRef.current) return
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'translated_images.zip'
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      notify('ZIP 下载失败：' + err.message, 'error')
    } finally {
      if (activeRef.current) setZipLoading(false)
    }
  }

  function resetAll() {
    stopPolling()
    setTaskId(null)
    setTaskIds([])
    setProgress(0)
    setBatchStatus(null)
    setZipLoading(false)
    setErrorMessage('')
    setResult(null)
    setPhase('idle')
  }

  function startNewTranslation() {
    resetAll()
    setFiles([])
    setFileError('')
  }

  const isBlurHint = (errorMessage || '').includes('模糊')

  return (
    <div className="card relative mx-auto max-w-[760px] bg-surface/40 p-8 md:p-10 backdrop-blur-sm">
      <AnimatePresence mode="wait">
        {phase === 'idle' && (
          <motion.div key="idle" {...panelMotion}>
            <StepHeading index="01" title="上传漫画图片" />
            <Dropzone
              files={files}
              previewUrl={filePreviewUrl}
              isDragging={isDragging}
              setIsDragging={setIsDragging}
              onDrop={onDrop}
              onSelect={onFileSelect}
              onRemove={removeFile}
              inputRef={inputRef}
            />
            {fileError && (
              <motion.p
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-3 flex items-center gap-2 text-sm text-danger"
                role="alert"
              >
                <AlertTriangle size={15} />
                {fileError}
              </motion.p>
            )}

            <StepHeading index="02" title="选择翻译方向" className="mt-9" />
            <div className="grid grid-cols-[1fr_auto_1fr] items-end gap-4">
              <LangSelect
                id="source-lang"
                label="Source"
                value={sourceLang}
                onChange={setSourceLang}
                options={[
                  { value: 'ja', label: '日语' },
                  { value: 'en', label: '英语' },
                ]}
              />
              <div className="flex h-[50px] items-center justify-center text-ink-500">
                <ArrowRight size={18} />
              </div>
              <LangSelect
                id="target-lang"
                label="Target"
                value={targetLang}
                onChange={setTargetLang}
                options={[{ value: 'zh', label: '中文' }]}
              />
            </div>

            <div className="mt-9 flex justify-center">
              <SpecularButton {...SPECULAR_PRIMARY} disabled={!files.length} onClick={startTranslate}>
                <Wand2 size={18} />
                {files.length > 1 ? `开始批量翻译（${files.length} 张）` : '开始翻译'}
              </SpecularButton>
            </div>
          </motion.div>
        )}

        {phase === 'submitting' && (
          <motion.div key="submitting" {...panelMotion} className="py-10 text-center">
            <Loader2 size={28} className="mx-auto animate-spin text-accent" />
            <p className="mt-4 text-ink-200">正在提交任务…</p>
          </motion.div>
        )}

        {phase === 'running' && (
          <motion.div key="running" {...panelMotion}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Loader2 size={20} className="animate-spin text-accent" />
                <h2 className="font-display text-xl font-semibold">
                  {taskIds.length > 1 ? '正在批量翻译漫画' : '正在翻译漫画'}
                </h2>
              </div>
              <span className="font-mono text-sm text-ink-400">{progress}%</span>
            </div>
            <p className="mt-1 truncate text-sm text-ink-500">
              {taskIds.length > 1 ? `${taskIds.length} 张图片独立处理中` : file?.name}
            </p>

            <div className="mt-8 h-1.5 overflow-hidden rounded-full bg-surface-2">
              <motion.div
                className="h-full rounded-full bg-gradient-to-r from-accent-strong to-accent"
                animate={{ width: `${progress}%` }}
                transition={{ type: 'spring', stiffness: 120, damping: 24 }}
              />
            </div>

            {taskIds.length > 1 && batchStatus ? (
              <>
                <p className="mt-4 text-sm text-ink-500">
                  已完成 {batchStatus.completed} 张 · 失败 {batchStatus.failed} 张 · 处理中 {batchStatus.processing} 张
                </p>
                <BatchItems items={batchStatus.items} />
              </>
            ) : (
              <Pipeline currentStep={currentStep} progress={progress} />
            )}

            <div className="mt-8 flex justify-center">
              <SpecularButton {...SPECULAR_SECONDARY} onClick={cancelTask}>
                取消翻译
              </SpecularButton>
            </div>
          </motion.div>
        )}

        {phase === 'batch-result' && batchStatus && (
          <motion.div key="batch-result" {...panelMotion}>
            <div className="flex flex-wrap items-end justify-between gap-4">
              <div>
                <h2 className="font-display text-xl font-semibold text-ink-100">批量处理完成</h2>
                <p className="mt-1 text-sm text-ink-500">
                  成功 {batchStatus.completed} 张 · 失败 {batchStatus.failed} 张
                </p>
              </div>
              {batchStatus.completed > 0 && (
                <SpecularButton {...SPECULAR_PRIMARY} size="md" disabled={zipLoading} onClick={downloadZip}>
                  {zipLoading ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
                  下载 ZIP
                </SpecularButton>
              )}
            </div>
            <BatchItems items={batchStatus.items} />
            <div className="mt-7 flex justify-center">
              <SpecularButton {...SPECULAR_SECONDARY} onClick={startNewTranslation}>
                <RotateCcw size={16} />
                继续翻译
              </SpecularButton>
            </div>
          </motion.div>
        )}

        {phase === 'result' && result && (
          <motion.div key="result" {...panelMotion}>
            <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
              <div>
                <h2 className="font-display text-xl font-semibold text-ink-100">翻译完成</h2>
                <p className="mt-1 text-sm text-ink-500">
                  共识别 {result.textCount} 处文字
                  {result.durationMs ? ` · 耗时 ${formatDuration(result.durationMs)}` : ''}
                </p>
              </div>
              <SpecularButton
                {...SPECULAR_PRIMARY}
                size="md"
                onClick={() => {
                  const a = document.createElement('a')
                  a.href = result.resultUrl
                  a.download = ''
                  document.body.appendChild(a)
                  a.click()
                  a.remove()
                }}
              >
                <Download size={16} />
                下载图片
              </SpecularButton>
            </div>

            <div className="relative flex w-full items-center justify-center overflow-hidden rounded-xl border border-line bg-surface-2">
              <img
                src={result.resultUrl}
                alt="翻译结果"
                className="w-full object-contain"
                draggable={false}
              />
            </div>

            <div className="mt-6 flex justify-center">
              <SpecularButton {...SPECULAR_SECONDARY} onClick={startNewTranslation}>
                <RotateCcw size={16} />
                翻译下一张
              </SpecularButton>
            </div>
          </motion.div>
        )}

        {phase === 'error' && (
          <motion.div key="error" {...panelMotion} className="py-8 text-center">
            <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full border border-danger/30 bg-danger/10 text-danger">
              <AlertTriangle size={26} />
            </div>
            <h2 className="mt-5 font-display text-xl font-semibold">翻译失败</h2>
            <p className="mx-auto mt-3 max-w-md text-sm leading-relaxed text-ink-400">{errorMessage}</p>
            {isBlurHint && (
              <p className="mt-2 text-sm text-ink-500">建议：尝试更换更清晰、对比度更高的图片。</p>
            )}
            <div className="mt-8 flex justify-center gap-3">
              <SpecularButton {...SPECULAR_PRIMARY} onClick={startNewTranslation}>
                <RefreshCw size={16} />
                重新上传
              </SpecularButton>
              <SpecularButton {...SPECULAR_SECONDARY} disabled={!files.length} onClick={startTranslate}>
                重试
              </SpecularButton>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function StepHeading({ index, title, className = '' }) {
  return (
    <div className={`mb-5 flex items-center gap-3 ${className}`}>
      <span className="flex h-7 w-7 items-center justify-center rounded-md border border-accent/30 bg-accent/10 font-mono text-xs font-medium text-accent">
        {index}
      </span>
      <h2 className="font-display text-lg font-medium text-ink-100">{title}</h2>
    </div>
  )
}

function Dropzone({ files, previewUrl, isDragging, setIsDragging, onDrop, onSelect, onRemove, inputRef }) {
  return (
    <div
      className={`relative cursor-pointer overflow-hidden rounded-xl border-2 border-dashed transition-all duration-200 ${
        isDragging ? 'border-accent ring-glow' : 'border-line-strong hover:border-accent/40'
      }`}
      onDragOver={(e) => {
        e.preventDefault()
        setIsDragging(true)
      }}
      onDragLeave={(e) => {
        e.preventDefault()
        setIsDragging(false)
      }}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          inputRef.current?.click()
        }
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ALLOWED}
        multiple
        className="hidden"
        onChange={onSelect}
      />

      <GlassSurface
        width="100%"
        height="100%"
        borderRadius={12}
        borderWidth={0.07}
        brightness={55}
        opacity={0.9}
        blur={12}
        backgroundOpacity={0.06}
        saturation={1.3}
        displace={3}
      >
        {!files.length ? (
          <div className="flex flex-col items-center justify-center px-6 py-14 text-center">
            <motion.div
              animate={isDragging ? { y: -6, scale: 1.05 } : { y: 0, scale: 1 }}
              className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-line bg-surface text-accent"
            >
              <UploadCloud size={26} />
            </motion.div>
            <p className="text-[15px] font-medium text-ink-100">点击上传，或拖拽多张图片到此处</p>
            <p className="mt-2 text-[13px] text-ink-500">
              支持 JPG / PNG / WebP / BMP · 单张 10MB · 最多 10 张 / 共 50MB
            </p>
          </div>
        ) : files.length === 1 ? (
          <FileRow
            file={files[0]}
            previewUrl={previewUrl}
            onRemove={(event) => {
              event.stopPropagation()
              onRemove(0)
            }}
          />
        ) : (
          <div className="px-5 py-4 text-left">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-sm font-medium text-ink-200">已选择 {files.length} 张图片</span>
              <span className="font-mono text-xs text-ink-500">
                {formatBytes(files.reduce((sum, file) => sum + file.size, 0))}
              </span>
            </div>
            <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
              {files.map((file, index) => (
                <div
                  key={`${file.name}-${file.size}-${file.lastModified}-${index}`}
                  className="flex items-center gap-3 rounded-lg border border-line bg-surface/45 px-3 py-2.5"
                >
                  <FileImage size={17} className="shrink-0 text-accent" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm text-ink-100">{file.name}</p>
                    <p className="font-mono text-[10px] text-ink-500">{formatBytes(file.size)}</p>
                  </div>
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation()
                      onRemove(index)
                    }}
                    className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-ink-500 transition-colors hover:bg-surface hover:text-danger"
                    aria-label={`移除 ${file.name}`}
                  >
                    <X size={16} />
                  </button>
                </div>
              ))}
            </div>
            <p className="mt-3 text-center text-xs text-ink-500">点击空白处可继续添加图片</p>
          </div>
        )}
      </GlassSurface>
    </div>
  )
}

function FileRow({ file, previewUrl, onRemove }) {
  return (
    <div className="flex items-center gap-5 px-6 py-6">
      <div className="h-20 w-20 shrink-0 overflow-hidden rounded-lg border border-line bg-surface-2">
        <img src={previewUrl} alt="待翻译图片预览" className="h-full w-full object-cover" />
      </div>
      <div className="min-w-0 flex-1 text-left">
        <p className="truncate text-[15px] font-medium text-ink-100">{file.name}</p>
        <p className="mt-1 text-[13px] text-ink-500">{formatBytes(file.size)}</p>
      </div>
      <button
        type="button"
        onClick={onRemove}
        className="flex h-9 w-9 items-center justify-center rounded-lg text-ink-500 transition-colors hover:bg-surface hover:text-danger"
        aria-label="移除图片"
      >
        <X size={18} />
      </button>
    </div>
  )
}

function BatchItems({ items }) {
  const statusLabel = {
    queued: '等待中',
    processing: '处理中',
    completed: '已完成',
    failed: '失败',
  }
  return (
    <div className="mt-7 space-y-2">
      {items.map((item) => (
        <div key={item.task_id} className="rounded-lg border border-line bg-surface/35 px-4 py-3">
          <div className="flex items-center gap-3">
            {item.status === 'completed' ? (
              <Check size={16} className="shrink-0 text-accent" />
            ) : item.status === 'failed' ? (
              <AlertTriangle size={16} className="shrink-0 text-danger" />
            ) : (
              <Loader2 size={16} className="shrink-0 animate-spin text-accent" />
            )}
            <span className="min-w-0 flex-1 truncate text-sm text-ink-100">{item.filename}</span>
            <span className={`font-mono text-xs ${item.status === 'failed' ? 'text-danger' : 'text-ink-500'}`}>
              {statusLabel[item.status] || item.status} · {item.progress || 0}%
            </span>
          </div>
          <div className="mt-2 h-1 overflow-hidden rounded-full bg-surface-2">
            <div
              className={`h-full rounded-full ${item.status === 'failed' ? 'bg-danger' : 'bg-accent'}`}
              style={{ width: `${item.progress || 0}%` }}
            />
          </div>
          {item.error && <p className="mt-2 text-xs text-danger">{item.error}</p>}
        </div>
      ))}
    </div>
  )
}

function Pipeline({ currentStep, progress }) {
  return (
    <div className="mt-8">
      <div className="flex items-center">
        {STEPS.map((s, i) => {
          const done = i < currentStep
          const active = i === currentStep
          return (
            <Fragment key={s.key}>
              {i > 0 && (
                <div className="relative h-px flex-1 overflow-hidden bg-line-strong">
                  <motion.div
                    className="absolute inset-y-0 left-0 bg-accent"
                    initial={false}
                    animate={{ width: i <= currentStep ? '100%' : '0%' }}
                    transition={{ duration: 0.5 }}
                  />
                </div>
              )}
              <div className="flex flex-col items-center gap-2">
                <motion.span
                  animate={
                    active
                      ? { boxShadow: ['0 0 0 0 rgba(255,255,255,0.4)', '0 0 0 8px rgba(255,255,255,0)'] }
                      : {}
                  }
                  transition={{ duration: 1.6, repeat: active ? Infinity : 0, ease: 'easeOut' }}
                  className={`flex h-9 w-9 items-center justify-center rounded-full border text-xs font-medium transition-colors ${
                    done
                      ? 'border-accent bg-accent text-bg'
                      : active
                        ? 'border-accent bg-accent/15 text-accent'
                        : 'border-line-strong bg-surface-2 text-ink-600'
                  }`}
                >
                  {done ? <Check size={16} /> : <span className="font-mono">{i + 1}</span>}
                </motion.span>
                <span
                  className={`font-mono text-[10px] uppercase tracking-wider ${
                    active ? 'text-accent' : done ? 'text-ink-200' : 'text-ink-600'
                  }`}
                >
                  {s.label}
                </span>
              </div>
            </Fragment>
          )
        })}
      </div>
    </div>
  )
}
