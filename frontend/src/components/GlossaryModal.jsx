import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { X, Loader2 } from 'lucide-react'
import { createGlossary, updateGlossary } from '../api'
import { useToast } from './Toast'

export default function GlossaryModal({ open, item, onClose, onSaved }) {
  const notify = useToast()
  const [form, setForm] = useState({ source: '', target: '', lang: 'ja', note: '' })
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (open) {
      setForm(
        item
          ? { id: item.id, source: item.source, target: item.target, lang: item.lang, note: item.note || '' }
          : { id: null, source: '', target: '', lang: 'ja', note: '' },
      )
      setError('')
    }
  }, [open, item])

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose()
    }
    if (open) window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  async function save() {
    if (!form.source.trim()) {
      setError('source')
      return
    }
    if (!form.target.trim()) {
      setError('target')
      return
    }
    setSaving(true)
    try {
      if (form.id) {
        await updateGlossary(form)
        notify('词条已更新', 'success')
      } else {
        await createGlossary(form)
        notify('词条已添加', 'success')
      }
      onSaved()
      onClose()
    } catch (err) {
      notify(err.message, 'error')
    } finally {
      setSaving(false)
    }
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[200] flex items-center justify-center bg-bg/70 px-4 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) onClose()
          }}
        >
          <motion.div
            className="w-full max-w-[480px] rounded-2xl border border-line bg-surface shadow-[0_32px_80px_-24px_rgba(0,0,0,0.8)]"
            initial={{ opacity: 0, y: 20, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 16, scale: 0.97 }}
            transition={{ type: 'spring', stiffness: 360, damping: 30 }}
            role="dialog"
            aria-modal="true"
            aria-label={form.id ? '编辑词条' : '新增词条'}
          >
            <div className="flex items-center justify-between border-b border-line px-6 py-5">
              <h3 className="font-display text-lg font-semibold text-ink-100">
                {form.id ? '编辑词条' : '新增词条'}
              </h3>
              <button
                onClick={onClose}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-500 transition-colors hover:bg-surface-2 hover:text-ink-100"
                aria-label="关闭"
              >
                <X size={18} />
              </button>
            </div>

            <div className="space-y-5 px-6 py-6">
              <Field label="源词" required>
                <input
                  value={form.source}
                  onChange={(e) => {
                    setForm((f) => ({ ...f, source: e.target.value }))
                    setError('')
                  }}
                  className={`input-base px-4 py-3 ${error === 'source' ? '!border-danger/60' : ''}`}
                  placeholder="例如：ナルト"
                  autoFocus
                />
                {error === 'source' && <ErrorHint>源词不能为空</ErrorHint>}
                <Helper>原始语言的专有名词，如角色名、作品名</Helper>
              </Field>

              <Field label="译词" required>
                <input
                  value={form.target}
                  onChange={(e) => {
                    setForm((f) => ({ ...f, target: e.target.value }))
                    setError('')
                  }}
                  className={`input-base px-4 py-3 ${error === 'target' ? '!border-danger/60' : ''}`}
                  placeholder="例如：鸣人"
                />
                {error === 'target' && <ErrorHint>译词不能为空</ErrorHint>}
              </Field>

              <Field label="源语言">
                <div className="relative">
                  <select
                    value={form.lang}
                    onChange={(e) => setForm((f) => ({ ...f, lang: e.target.value }))}
                    className="input-base appearance-none px-4 py-3 pr-10"
                  >
                    <option value="ja" className="bg-surface-2">日语</option>
                    <option value="en" className="bg-surface-2">英语</option>
                    <option value="zh" className="bg-surface-2">中文</option>
                  </select>
                </div>
              </Field>

              <Field label="备注">
                <input
                  value={form.note}
                  onChange={(e) => setForm((f) => ({ ...f, note: e.target.value }))}
                  className="input-base px-4 py-3"
                  maxLength={200}
                  placeholder="可选，例如：火影忍者主角"
                />
              </Field>
            </div>

            <div className="flex justify-end gap-3 border-t border-line px-6 py-5">
              <button
                onClick={onClose}
                className="rounded-lg border border-line px-5 py-2.5 text-sm text-ink-300 transition-colors hover:text-ink-100"
              >
                取消
              </button>
              <button
                onClick={save}
                disabled={saving}
                className="flex items-center gap-2 rounded-lg bg-gradient-to-r from-accent-strong to-accent px-5 py-2.5 text-sm font-medium text-bg transition-transform hover:scale-[1.03] disabled:opacity-60"
              >
                {saving && <Loader2 size={15} className="animate-spin" />}
                {saving ? '保存中…' : '保存'}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

function Field({ label, required, children }) {
  return (
    <div>
      <label className="mb-2 block text-sm font-medium text-ink-200">
        {label}
        {required && <span className="ml-1 text-danger">*</span>}
      </label>
      {children}
    </div>
  )
}

function ErrorHint({ children }) {
  return <p className="mt-1.5 text-[13px] text-danger">{children}</p>
}

function Helper({ children }) {
  return <p className="mt-1.5 text-[13px] text-ink-500">{children}</p>
}
