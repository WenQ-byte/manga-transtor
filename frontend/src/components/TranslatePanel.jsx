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
  getTaskStatus,
  getTaskResultUrl,
  deleteTask,
  formatBytes,
  formatDuration,
} from '../api'
import { useToast } from './Toast'
import SpecularButton from './SpecularButton'
import { SPECULAR_PRIMARY, SPECULAR_SECONDARY } from './specularPresets'

const STEPS = [
  { key: 'detect', label: '检测区域', weight: 0.15 },
  { key: 'ocr', label: '识别文字', weight: 0.25 },
  { key: 'translate', label: '翻译', weight: 0.3 },
  { key: 'inpaint', label: '修复图像', weight: 0.15 },
  { key: 'render', label: '渲染译文', weight: 0.15 },
]

const ALLOWED = '.jpg,.jpeg,.png,.webp,.bmp'
const MAX_MB = 10

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
  if (!f.type.startsWith('image/')) return '请上传图片文件'
  return ''
}

function LangSelect({ id, label, value, options, onChange }) {
  return (
    <label className="block">
      <span className="mb-2 block font-mono text-[10px] uppercase tracking-[0.25em] text-ink-500">
        {label}
      </span>
      <div className="relative">
        <select
          id={id}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="input-base appearance-none px-4 py-3 pr-10 font-medium"
        >
          {options.map((o) => (
            <option key={o.value} value={o.value} className="bg-surface-2">
              {o.label}
            </option>
          ))}
        </select>
        <ChevronDown
          size={16}
          className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-ink-500"
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
  const [file, setFile] = useState(null)
  const [filePreviewUrl, setFilePreviewUrl] = useState('')
  const [fileError, setFileError] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [sourceLang, setSourceLang] = useState('ja')
  const [targetLang, setTargetLang] = useState('zh')

  const [taskId, setTaskId] = useState(null)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')

  const inputRef = useRef(null)
  const pollRef = useRef(null)
  const fileRef = useRef(null)

  useEffect(() => () => stopPolling(), [])
  useEffect(() => () => {
    if (filePreviewUrl) URL.revokeObjectURL(filePreviewUrl)
  }, [filePreviewUrl])

  const currentStep = useMemo(
    () => (phase === 'running' ? stepIndexFromProgress(progress) : -1),
    [phase, progress],
  )

  function stopPolling() {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  function setFile0(f) {
    const err = validateFile(f)
    setFileError(err)
    if (err) {
      setFile(null)
      setFilePreviewUrl('')
      return false
    }
    fileRef.current = f
    setFile(f)
    setFilePreviewUrl(URL.createObjectURL(f))
    return true
  }

  function onFileSelect(e) {
    const f = e.target.files?.[0]
    if (f) setFile0(f)
    e.target.value = ''
  }

  function onDrop(e) {
    e.preventDefault()
    setIsDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) setFile0(f)
  }

  function clearFile() {
    setFile(null)
    setFilePreviewUrl('')
    setFileError('')
    fileRef.current = null
  }

  function startTranslate() {
    if (!file) return
    const f = file
    setPhase('submitting')
    setFileError('')
    setErrorMessage('')
    setResult(null)
    setProgress(0)

    createTranslateTask(f, sourceLang, targetLang)
      .then((data) => {
        setTaskId(data.task_id)
        setPhase('running')
        startPolling(data.task_id)
      })
      .catch((err) => {
        setPhase('error')
        setErrorMessage(err.message)
        notify(err.message, 'error')
      })
  }

  function startPolling(id) {
    stopPolling()
    const tick = () => pollStatus(id)
    pollRef.current = setInterval(tick, 800)
    tick()
  }

  async function pollStatus(id) {
    try {
      const status = await getTaskStatus(id)
      setProgress(Math.max(progress, status.progress || 0))
      if (status.status === 'completed') {
        stopPolling()
        setResult({
          resultUrl: getTaskResultUrl(id),
          originalUrl: filePreviewUrl,
          textCount: status.text_count,
          durationMs: status.duration_ms,
        })
        setPhase('result')
        notify('翻译完成', 'success')
      } else if (status.status === 'failed') {
        stopPolling()
        setErrorMessage(status.error || '翻译失败，请重试')
        setPhase('error')
        notify(status.error || '翻译失败', 'error')
      }
    } catch (err) {
      stopPolling()
      setErrorMessage(err.message)
      setPhase('error')
      notify('获取进度失败：' + err.message, 'error')
    }
  }

  function cancelTask() {
    stopPolling()
    if (taskId) deleteTask(taskId).catch(() => {})
    resetAll()
  }

  function resetAll() {
    stopPolling()
    setTaskId(null)
    setProgress(0)
    setErrorMessage('')
    setResult(null)
    setPhase('idle')
  }

  const isBlurHint = (errorMessage || '').includes('模糊')

  return (
    <div className="card relative mx-auto max-w-[760px] p-8 md:p-10">
      <AnimatePresence mode="wait">
        {phase === 'idle' && (
          <motion.div key="idle" {...panelMotion}>
            <StepHeading index="01" title="上传漫画图片" />
            <Dropzone
              file={file}
              previewUrl={filePreviewUrl}
              isDragging={isDragging}
              setIsDragging={setIsDragging}
              onDrop={onDrop}
              onSelect={onFileSelect}
              onClear={clearFile}
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
              <SpecularButton {...SPECULAR_PRIMARY} disabled={!file} onClick={startTranslate}>
                <Wand2 size={18} />
                开始翻译
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
                <h2 className="font-display text-xl font-semibold">正在翻译漫画</h2>
              </div>
              <span className="font-mono text-sm text-ink-400">{progress}%</span>
            </div>
            <p className="mt-1 truncate text-sm text-ink-500">{file?.name}</p>

            <div className="mt-8 h-1.5 overflow-hidden rounded-full bg-surface-2">
              <motion.div
                className="h-full rounded-full bg-gradient-to-r from-accent-strong to-accent"
                animate={{ width: `${progress}%` }}
                transition={{ type: 'spring', stiffness: 120, damping: 24 }}
              />
            </div>

            <Pipeline currentStep={currentStep} progress={progress} />

            <div className="mt-8 flex justify-center">
              <SpecularButton {...SPECULAR_SECONDARY} onClick={cancelTask}>
                取消翻译
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
              <SpecularButton {...SPECULAR_SECONDARY} onClick={resetAll}>
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
              <SpecularButton {...SPECULAR_PRIMARY} onClick={resetAll}>
                <RefreshCw size={16} />
                重新上传
              </SpecularButton>
              <SpecularButton {...SPECULAR_SECONDARY} disabled={!file} onClick={startTranslate}>
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

function Dropzone({ file, previewUrl, isDragging, setIsDragging, onDrop, onSelect, onClear, inputRef }) {
  return (
    <div
      className={`relative cursor-pointer overflow-hidden rounded-xl border-2 border-dashed transition-all duration-200 ${
        isDragging
          ? 'border-accent bg-accent/[0.06] ring-glow'
          : 'border-line-strong bg-surface-2/40 hover:border-accent/40 hover:bg-surface-2/70'
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
        className="hidden"
        onChange={onSelect}
      />

      {!file ? (
        <div className="flex flex-col items-center justify-center px-6 py-14 text-center">
          <motion.div
            animate={isDragging ? { y: -6, scale: 1.05 } : { y: 0, scale: 1 }}
            className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-line bg-surface text-accent"
          >
            <UploadCloud size={26} />
          </motion.div>
          <p className="text-[15px] font-medium text-ink-100">点击上传，或拖拽图片到此处</p>
          <p className="mt-2 text-[13px] text-ink-500">支持 JPG / PNG / WebP / BMP · 单张不超过 10MB</p>
        </div>
      ) : (
        <div className="flex items-center gap-5 px-6 py-6">
          <div className="h-20 w-20 shrink-0 overflow-hidden rounded-lg border border-line bg-surface-2">
            <img src={previewUrl} alt="待翻译图片预览" className="h-full w-full object-cover" />
          </div>
          <div className="min-w-0 flex-1 text-left">
            <p className="truncate text-[15px] font-medium text-ink-100">{file.name}</p>
            <p className="mt-1 text-[13px] text-ink-500">{formatBytes(file.size)}</p>
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation()
              onClear()
            }}
            className="flex h-9 w-9 items-center justify-center rounded-lg text-ink-500 transition-colors hover:bg-surface hover:text-danger"
            aria-label="移除图片"
          >
            <X size={18} />
          </button>
        </div>
      )}
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
