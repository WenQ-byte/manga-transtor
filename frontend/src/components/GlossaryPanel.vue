<template>
  <div class="glossary-panel">
    <div class="glossary-hero">
      <div>
        <h1 class="page-title">专有名词管理</h1>
        <p class="page-sub">
          为翻译添加统一译法，角色名、作品名等专有名词将保持一致。可手动录入或批量导入。
        </p>
      </div>
      <button class="btn btn-primary" @click="openEditor()">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>
        新增词条
      </button>
    </div>

    <!-- 搜索与筛选 -->
    <div class="glossary-toolbar card">
      <div class="toolbar-row">
        <div class="search-box">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m21 21-4-4"/></svg>
          <input
            v-model="search"
            class="search-input"
            type="search"
            placeholder="搜索源词或译词…"
            aria-label="搜索专有名词"
            @input="loadList"
          />
        </div>

        <div class="toolbar-actions">
          <label class="btn btn-secondary import-btn" for="import-file">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5"/><path d="M12 3v12"/></svg>
            导入 JSON
          </label>
          <input ref="importInput" id="import-file" type="file" accept=".json" class="hidden-input" @change="onImportFile" />
          <button class="btn btn-secondary" @click="downloadTemplate">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/></svg>
            模板
          </button>
        </div>
      </div>
    </div>

    <!-- 列表 -->
    <div v-if="loading" class="glossary-list">
      <div v-for="i in 4" :key="i" class="skeleton skeleton-row"></div>
    </div>

    <div v-else-if="items.length === 0" class="empty-state card">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>
        <path d="M9 7h7M9 11h5"/>
      </svg>
      <h3>{{ search ? '未找到匹配词条' : '还没有专有名词词条' }}</h3>
      <p>{{ search ? '换个关键词试试' : '点击「新增词条」添加你的第一个专有名词' }}</p>
    </div>

    <div v-else class="glossary-list">
      <div class="glossary-item card" v-for="item in items" :key="item.id">
        <div class="item-main">
          <div class="item-word">
            <strong class="item-source">{{ item.source }}</strong>
            <span class="item-arrow" aria-hidden="true">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="m13 6 6 6-6 6"/></svg>
            </span>
            <strong class="item-target">{{ item.target }}</strong>
          </div>
          <p v-if="item.note" class="item-note">{{ item.note }}</p>
        </div>
        <div class="item-actions">
          <span class="lang-tag">{{ langName(item.lang) }}</span>
          <button class="icon-btn" @click="openEditor(item)" aria-label="编辑词条" title="编辑">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/></svg>
          </button>
          <button class="icon-btn danger" @click="onDelete(item)" aria-label="删除词条" title="删除">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
          </button>
        </div>
      </div>
    </div>

    <!-- 总数 -->
    <div class="glossary-footer">
      <span>共 {{ total }} 条词条</span>
    </div>

    <!-- 编辑弹窗 -->
    <div v-if="showEditor" class="modal-overlay" @click.self="closeEditor">
      <div class="modal" role="dialog" aria-modal="true" :aria-label="editorForm.id ? '编辑词条' : '新增词条'">
        <div class="modal-header">
          <h3>{{ editorForm.id ? '编辑词条' : '新增词条' }}</h3>
          <button class="icon-btn" @click="closeEditor" aria-label="关闭">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"/></svg>
          </button>
        </div>

        <div class="modal-body">
          <div class="form-group">
            <label class="form-label" for="ed-source">源词 <span class="required">*</span></label>
            <input id="ed-source" v-model="editorForm.source" class="form-input" :class="{ 'is-error': editorError === 'source' }" placeholder="例如：ナルト" @input="editorError = ''" />
            <p v-if="editorError === 'source'" class="field-error">源词不能为空</p>
            <p class="helper-text">原始语言的专有名词，如角色名、作品名</p>
          </div>

          <div class="form-group">
            <label class="form-label" for="ed-target">译词 <span class="required">*</span></label>
            <input id="ed-target" v-model="editorForm.target" class="form-input" :class="{ 'is-error': editorError === 'target' }" placeholder="例如：鸣人" @input="editorError = ''" />
            <p v-if="editorError === 'target'" class="field-error">译词不能为空</p>
          </div>

          <div class="form-group">
            <label class="form-label" for="ed-lang">源语言</label>
            <select id="ed-lang" v-model="editorForm.lang" class="form-select">
              <option value="ja">日语</option>
              <option value="en">英语</option>
              <option value="zh">中文</option>
            </select>
          </div>

          <div class="form-group">
            <label class="form-label" for="ed-note">备注</label>
            <input id="ed-note" v-model="editorForm.note" class="form-input" maxlength="200" placeholder="可选，例如：火影忍者主角" />
          </div>
        </div>

        <div class="modal-footer">
          <button class="btn btn-secondary" @click="closeEditor">取消</button>
          <button class="btn btn-primary" :disabled="saving" @click="saveItem">
            {{ saving ? '保存中…' : '保存' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import {
  listGlossary,
  createGlossary,
  updateGlossary,
  deleteGlossary,
  importGlossary,
} from '../api'

const emit = defineEmits(['notify', 'glossary-updated'])

const items = ref([])
const total = ref(0)
const loading = ref(false)
const saving = ref(false)
const search = ref('')
const importInput = ref(null)

const showEditor = ref(false)
const editorForm = ref({ id: null, source: '', target: '', lang: 'ja', note: '' })
const editorError = ref('')

const TEMPLATE = [
  { source: 'ナルト', target: '鸣人', lang: 'ja', note: '火影忍者主角' },
  { source: 'ルフィ', target: '路飞', lang: 'ja', note: '海贼王主角' },
]

function langName(lang) {
  return { ja: '日', en: '英', zh: '中' }[lang] || lang
}

async function loadList() {
  loading.value = true
  try {
    const data = await listGlossary({ search: search.value })
    items.value = data.items
    total.value = data.total
    emit('glossary-updated')
  } catch (err) {
    emit('notify', err.message, 'error')
  } finally {
    loading.value = false
  }
}

function openEditor(item = null) {
  if (item) {
    editorForm.value = { id: item.id, source: item.source, target: item.target, lang: item.lang, note: item.note || '' }
  } else {
    editorForm.value = { id: null, source: '', target: '', lang: 'ja', note: '' }
  }
  editorError.value = ''
  showEditor.value = true
}

function closeEditor() {
  if (saving.value) return
  showEditor.value = false
}

async function saveItem() {
  const f = editorForm.value
  if (!f.source.trim()) {
    editorError.value = 'source'
    return
  }
  if (!f.target.trim()) {
    editorError.value = 'target'
    return
  }
  saving.value = true
  try {
    if (f.id) {
      await updateGlossary(f)
      emit('notify', '词条已更新', 'success')
    } else {
      await createGlossary(f)
      emit('notify', '词条已添加', 'success')
    }
    showEditor.value = false
    loadList()
  } catch (err) {
    emit('notify', err.message, 'error')
  } finally {
    saving.value = false
  }
}

async function onDelete(item) {
  if (!window.confirm(`确定删除词条「${item.source} → ${item.target}」吗？`)) return
  try {
    await deleteGlossary(item.id)
    emit('notify', '词条已删除', 'success')
    loadList()
  } catch (err) {
    emit('notify', err.message, 'error')
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
    emit('notify', msgs.join('，') || '没有导入任何词条', result.errors.length ? 'error' : 'success')
    if (result.errors.length > 0) {
      emit('notify', result.errors[0], 'error')
    }
    loadList()
  } catch (err) {
    emit('notify', err.message, 'error')
  } finally {
    e.target.value = ''
  }
}

onMounted(loadList)
</script>

<style scoped>
.glossary-panel {
  max-width: 860px;
  margin: 0 auto;
}

.glossary-hero {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: var(--space-4);
  margin-bottom: var(--space-6);
  flex-wrap: wrap;
  animation: slide-up var(--duration-base) var(--ease);
}

.page-title {
  font-size: 30px;
  letter-spacing: -0.5px;
}

.page-sub {
  color: var(--ink-500);
  margin-top: var(--space-2);
  max-width: 560px;
  font-size: 15px;
}

.glossary-toolbar {
  padding: var(--space-4);
  margin-bottom: var(--space-5);
}

.toolbar-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
  flex-wrap: wrap;
}

.search-box {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex: 1;
  min-width: 200px;
  color: var(--ink-500);
}

.search-input {
  border: none;
  background: transparent;
  font-size: 15px;
  color: var(--ink-900);
  padding: 8px 0;
  min-height: 40px;
  flex: 1;
  outline: none;
}

.search-input::placeholder {
  color: var(--ink-300);
}

.toolbar-actions {
  display: flex;
  gap: var(--space-2);
  align-items: center;
}

.hidden-input {
  display: none;
}

.import-btn {
  position: relative;
  cursor: pointer;
}

.glossary-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.glossary-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
  padding: var(--space-4) var(--space-5);
  transition: box-shadow var(--duration-fast) var(--ease), transform var(--duration-fast) var(--ease);
  animation: fade-in var(--duration-base) var(--ease);
}

.glossary-item:hover {
  box-shadow: var(--shadow-lg);
  transform: translateY(-1px);
}

.item-main {
  min-width: 0;
}

.item-word {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.item-source {
  font-size: 17px;
  color: var(--ink-900);
}

.item-arrow {
  color: var(--ink-300);
  display: inline-flex;
}

.item-target {
  font-size: 17px;
  color: var(--brand-600);
}

.item-note {
  font-size: 13px;
  color: var(--ink-500);
  margin-top: 4px;
}

.item-actions {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex-shrink: 0;
}

.lang-tag {
  font-size: 12px;
  font-weight: 600;
  color: var(--ink-500);
  background: var(--surface-2);
  padding: 3px 8px;
  border-radius: var(--radius-full);
  margin-right: var(--space-2);
}

.icon-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: var(--radius-md);
  border: none;
  background: transparent;
  color: var(--ink-500);
  transition: all var(--duration-fast) var(--ease);
}

.icon-btn:hover {
  background: var(--surface-2);
  color: var(--ink-900);
}

.icon-btn.danger:hover {
  background: #fef2f2;
  color: var(--danger-500);
}

.empty-state {
  text-align: center;
  padding: var(--space-7) var(--space-5);
  color: var(--ink-500);
  animation: fade-in var(--duration-base) var(--ease);
}

.empty-state h3 {
  color: var(--ink-700);
  margin-top: var(--space-3);
  font-size: 18px;
}

.empty-state p {
  font-size: 14px;
  margin-top: var(--space-2);
}

.skeleton-row {
  height: 76px;
}

.glossary-footer {
  text-align: center;
  color: var(--ink-300);
  font-size: 13px;
  margin-top: var(--space-5);
}

/* 弹窗 */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(30, 27, 46, 0.5);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 200;
  padding: var(--space-4);
  animation: fade-in var(--duration-fast) var(--ease);
}

.modal {
  background: var(--surface-0);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-lg);
  width: 100%;
  max-width: 480px;
  animation: scale-in var(--duration-base) var(--ease-spring);
  max-height: 90vh;
  overflow-y: auto;
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-5) var(--space-6) var(--space-3);
}

.modal-header h3 {
  font-size: 19px;
}

.modal-body {
  padding: 0 var(--space-6);
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-3);
  padding: var(--space-4) var(--space-6) var(--space-6);
}

.required {
  color: var(--danger-500);
}

@media (max-width: 640px) {
  .glossary-hero {
    flex-direction: column;
    align-items: flex-start;
  }
  .toolbar-row {
    flex-direction: column;
    align-items: stretch;
  }
  .toolbar-actions {
    justify-content: flex-start;
  }
}
</style>
