import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Plus, Search, FileJson, FileDown, ArrowRight, Pencil, Trash2, BookMarked } from 'lucide-react'
import { listGlossary, deleteGlossary, importGlossary } from '../api'
import { useToast } from './Toast'
import GlossaryModal from './GlossaryModal'
import SpecularButton from './SpecularButton'
import { SPECULAR_PRIMARY, SPECULAR_SECONDARY } from './specularPresets'

const LANG = { ja: '日', en: '英', zh: '中' }

const TEMPLATE = [
  { source: 'ナルト', target: '鸣人', lang: 'ja', target_lang: 'zh', note: '火影忍者主角' },
  { source: 'ルフィ', target: '路飞', lang: 'ja', target_lang: 'zh', note: '海贼王主角' },
]

const listMotion = {
  hidden: {},
  show: { transition: { staggerChildren: 0.04 } },
}

const itemMotion = {
  hidden: { opacity: 0, y: 8 },
  show: { opacity: 1, y: 0, transition: { duration: 0.28, ease: 'easeOut' } },
}

let glossaryCache = null

export default function GlossaryPanel({ onCountChange }) {
  const notify = useToast()
  const [items, setItems] = useState(() => glossaryCache?.items ?? [])
  const [total, setTotal] = useState(() => glossaryCache?.total ?? 0)
  const [loading, setLoading] = useState(() => glossaryCache == null)
  const [search, setSearch] = useState('')
  const [showEditor, setShowEditor] = useState(false)
  const [editing, setEditing] = useState(null)

  const importRef = useRef(null)
  const searchTimer = useRef(null)

  useEffect(() => {
    if (!glossaryCache) loadList()
    onCountChange?.(glossaryCache?.total ?? 0)
  }, [])

  async function loadList(term) {
    const isRefresh = glossaryCache != null
    if (!isRefresh) setLoading(true)
    try {
      const data = await listGlossary({ search: term === undefined ? search : term })
      if (!term) glossaryCache = { items: data.items, total: data.total }
      setItems(data.items)
      setTotal(data.total)
      onCountChange?.(data.total)
    } catch (err) {
      notify(err.message, 'error')
    } finally {
      setLoading(false)
    }
  }

  function onSearchChange(value) {
    setSearch(value)
    clearTimeout(searchTimer.current)
    searchTimer.current = setTimeout(() => loadList(value), 250)
  }

  function openEditor(item = null) {
    setEditing(item)
    setShowEditor(true)
  }

  async function onDelete(item) {
    if (!window.confirm(`确定删除词条「${item.source} → ${item.target}」吗？`)) return
    try {
      await deleteGlossary(item.id)
      notify('词条已删除', 'success')
      loadList()
    } catch (err) {
      notify(err.message, 'error')
    }
  }

  function downloadTemplate() {
    const blob = new Blob([JSON.stringify(TEMPLATE, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = '专有名词模板.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  async function onImportFile(e) {
    const f = e.target.files?.[0]
    if (!f) return
    try {
      const text = await f.text()
      const result = await importGlossary(text)
      const msgs = []
      if (result.imported > 0) msgs.push(`成功导入 ${result.imported} 条`)
      if (result.skipped > 0) msgs.push(`跳过重复 ${result.skipped} 条`)
      notify(msgs.join('，') || '没有导入任何词条', result.errors.length ? 'error' : 'success')
      if (result.errors.length > 0) notify(result.errors[0], 'error')
      loadList()
    } catch (err) {
      notify(err.message, 'error')
    } finally {
      e.target.value = ''
    }
  }

  return (
    <div className="mx-auto max-w-[880px]">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-wrap items-end justify-between gap-4 pb-10"
      >
        <div>
          <p className="eyebrow mb-3">Glossary</p>
          <h1 className="font-display text-3xl font-semibold tracking-tight text-ink-100">专有名词管理</h1>
          <p className="mt-3 max-w-xl text-[15px] leading-relaxed text-ink-400">
            为翻译添加统一译法，角色名、作品名等专有名词将保持一致。可手动录入或批量导入。
          </p>
        </div>
        <SpecularButton {...SPECULAR_PRIMARY} size="md" onClick={() => openEditor()}>
          <Plus size={17} />
          新增词条
        </SpecularButton>
      </motion.div>

      <div className="card mb-5 flex flex-wrap items-center justify-between gap-4 p-4">
        <div className="flex min-w-[220px] flex-1 items-center gap-3 text-ink-500">
          <Search size={18} />
          <input
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            className="min-h-[40px] flex-1 bg-transparent text-[15px] text-ink-100 outline-none placeholder:text-ink-500"
            type="search"
            placeholder="搜索源词或译词…"
            aria-label="搜索专有名词"
          />
        </div>
        <div className="flex items-center gap-2">
          <input
            ref={importRef}
            id="import-file"
            type="file"
            accept=".json"
            className="hidden"
            onChange={onImportFile}
          />
          <SpecularButton {...SPECULAR_SECONDARY} size="sm" onClick={() => importRef.current?.click()}>
            <FileJson size={16} />
            导入 JSON
          </SpecularButton>
          <SpecularButton {...SPECULAR_SECONDARY} size="sm" onClick={downloadTemplate}>
            <FileDown size={16} />
            模板
          </SpecularButton>
        </div>
      </div>

      {loading && items.length === 0 ? (
        <div className="space-y-3">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="card h-[76px] animate-pulse bg-surface-2/40" />
          ))}
        </div>
      ) : items.length === 0 ? (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="card flex flex-col items-center justify-center px-6 py-16 text-center"
        >
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-line bg-surface-2 text-ink-500">
            <BookMarked size={26} />
          </div>
          <h3 className="mt-5 font-display text-lg font-medium text-ink-100">
            {search ? '未找到匹配词条' : '还没有专有名词词条'}
          </h3>
          <p className="mt-2 text-sm text-ink-500">
            {search ? '换个关键词试试' : '点击「新增词条」添加你的第一个专有名词'}
          </p>
        </motion.div>
      ) : (
        <motion.div variants={listMotion} initial="hidden" animate="show" className="space-y-3">
          {loading && <div className="py-1 text-center font-mono text-xs text-ink-500">加载中…</div>}
          <AnimatePresence initial={false}>
            {items.map((item) => (
              <motion.div
                key={item.id}
                variants={itemMotion}
                className="card group flex items-center justify-between gap-4 p-5 transition-colors hover:border-accent/25"
              >
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2.5">
                    <span className="truncate font-medium text-ink-100">{item.source}</span>
                    <span className="text-ink-500">
                      <ArrowRight size={15} />
                    </span>
                    <span className="truncate font-medium text-accent">{item.target}</span>
                  </div>
                  {item.note && <p className="mt-1.5 truncate text-[13px] text-ink-500">{item.note}</p>}
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <span className="mr-1 rounded-md border border-line bg-surface-2 px-2 py-0.5 font-mono text-[11px] text-ink-400">
                    {LANG[item.lang] || item.lang}→{LANG[item.target_lang || 'zh'] || item.target_lang}
                  </span>
                  <button
                    onClick={() => openEditor(item)}
                    className="flex h-9 w-9 items-center justify-center rounded-lg text-ink-500 transition-colors hover:bg-surface-2 hover:text-accent"
                    aria-label="编辑词条"
                    title="编辑"
                  >
                    <Pencil size={16} />
                  </button>
                  <button
                    onClick={() => onDelete(item)}
                    className="flex h-9 w-9 items-center justify-center rounded-lg text-ink-500 transition-colors hover:bg-danger/10 hover:text-danger"
                    aria-label="删除词条"
                    title="删除"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </motion.div>
      )}

      <div className="mt-6 text-center font-mono text-xs text-ink-500">共 {total} 条词条</div>

      <GlossaryModal
        open={showEditor}
        item={editing}
        onClose={() => setShowEditor(false)}
        onSaved={loadList}
      />
    </div>
  )
}
