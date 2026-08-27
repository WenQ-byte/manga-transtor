<template>
  <div class="translate-panel card">
    <div class="panel-main">
      <!-- 步骤 1-2：上传与配置 -->
      <div class="config-section" v-if="!taskRunning && !result && !errorMessage">
        <div class="section-title">
          <span class="step-chip">1</span>
          <h2>上传漫画图片</h2>
        </div>

        <div
          class="drop-zone"
          :class="{ 'is-dragging': isDragging, 'has-file': !!file }"
          @dragover.prevent="isDragging = true"
          @dragleave.prevent="isDragging = false"
          @drop.prevent="onDrop"
          @click="triggerFileInput"
          role="button"
          :aria-label="'上传图片，支持格式 ' + allowedExts"
          tabindex="0"
          @keydown.enter.prevent="triggerFileInput"
          @keydown.space.prevent="triggerFileInput"
        >
          <input
            ref="fileInput"
            type="file"
            accept=".jpg,.jpeg,.png,.webp,.bmp"
            class="hidden-input"
            @change="onFileSelect"
          />

          <template v-if="!file">
            <div class="drop-icon" aria-hidden="true">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <path d="m17 8-5-5-5 5"/>
                <path d="M12 3v12"/>
              </svg>
            </div>
            <p class="drop-title">点击上传，或拖拽图片到此处</p>
            <p class="drop-hint">支持 JPG / PNG / WebP / BMP，单张不超过 10MB</p>
          </template>

          <template v-else>
            <div class="file-preview">
              <img :src="filePreviewUrl" alt="待翻译图片预览" />
              <div class="file-meta">
                <strong class="file-name">{{ file.name }}</strong>
                <span class="file-size">{{ formatBytes(file.size) }}</span>
              </div>
              <button
                class="btn btn-ghost remove-btn"
                type="button"
                @click.stop="clearFile"
                aria-label="移除图片"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"/></svg>
              </button>
            </div>
          </template>
        </div>

        <p v-if="fileError" class="field-error" role="alert">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 8v5"/><path d="M12 16h.01"/></svg>
          {{ fileError }}
        </p>

        <div class="section-title spaced">
          <span class="step-chip">2</span>
          <h2>选择翻译语言</h2>
        </div>

        <div class="lang-row">
          <div class="form-group lang-group">
            <label class="form-label" for="source-lang">源语言</label>
            <select id="source-lang" v-model="sourceLang" class="form-select">
              <option value="ja">日语</option>
              <option value="en">英语</option>
            </select>
          </div>

          <div class="lang-swap" aria-hidden="true">
            <button class="swap-btn" type="button" @click="swapLangs" aria-label="交换语言方向" tabindex="-1">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M7 16V4m0 0L3 8m4-4 4 4"/>
                <path d="M17 8v12m0 0 4-4m-4 4-4-4"/>
              </svg>
            </button>
          </div>

          <div class="form-group lang-group">
            <label class="form-label" for="target-lang">目标语言</label>
            <select id="target-lang" v-model="targetLang" class="form-select">
              <option value="zh">中文</option>
            </select>
          </div>
        </div>

        <div class="action-row">
          <button
            class="btn btn-primary btn-translate"
            :disabled="!file || !canSubmit"
            @click="startTranslate"
          >
            <svg v-if="!submitting" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="m12 3 5 5-5 5M5 8h12M7 14l-2 7"/>
            </svg>
            <span v-if="submitting">正在提交…</span>
            <span v-else>开始翻译</span>
          </button>
        </div>
      </div>

      <!-- 步骤 3：翻译进度 -->
      <div class="progress-section" v-else-if="taskRunning">
        <div class="progress-header">
          <span class="spinner" aria-hidden="true"></span>
          <h2>正在翻译漫画…</h2>
        </div>
        <p class="progress-file">{{ file.name }}</p>

        <div class="progress-track" role="progressbar" :aria-valuenow="overallProgress" aria-valuemin="0" aria-valuemax="100" aria-label="翻译进度">
          <div class="progress-fill" :style="{ width: overallProgress + '%' }"></div>
        </div>
        <div class="progress-percent">{{ overallProgress }}%</div>

        <ol class="step-list">
          <li
            v-for="step in stepList"
            :key="step.key"
            class="step-item"
            :class="{ done: step.done, active: step.active }"
          >
            <span class="step-dot" aria-hidden="true">
              <svg v-if="step.done" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
            </span>
            <span class="step-label">{{ step.label }}</span>
            <span class="step-percent" v-if="step.active">{{ step.percent }}%</span>
          </li>
        </ol>

        <button class="btn btn-ghost btn-cancel" @click="cancelTask">
          取消翻译
        </button>
      </div>

      <!-- 结果展示 -->
      <div class="result-section" v-else-if="result">
        <div class="result-header">
          <div>
            <h2>翻译完成</h2>
            <p class="result-meta">
              共识别 {{ result.textCount || 0 }} 处文字
              <template v-if="result.durationMs"> · 耗时 {{ formatDuration(result.durationMs) }}</template>
            </p>
          </div>
          <a class="btn btn-primary" :href="result.resultUrl" download :aria-label="'下载翻译结果 ' + file.name">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/></svg>
            下载图片
          </a>
        </div>

        <div class="compare-grid">
          <figure class="compare-item">
            <figcaption class="compare-label">原图</figcaption>
            <div class="image-frame">
              <img :src="result.originalUrl" :alt="'原图 ' + file.name" loading="lazy" />
            </div>
          </figure>
          <figure class="compare-item">
            <figcaption class="compare-label compare-label-result">
              <span class="result-tag">译后</span>
            </figcaption>
            <div class="image-frame">
              <img :src="result.resultUrl" :alt="'翻译结果 ' + file.name" loading="lazy" />
            </div>
          </figure>
        </div>

        <div class="result-actions">
          <button class="btn btn-secondary" @click="resetAll">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/></svg>
            翻译下一张
          </button>
        </div>
      </div>

      <!-- 失败态 -->
      <div class="error-section" v-else-if="errorMessage">
        <div class="error-icon" aria-hidden="true">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 8v5"/><path d="M12 16h.01"/></svg>
        </div>
        <h2>翻译失败</h2>
        <p class="error-text">{{ errorMessage }}</p>
        <p v-if="isBlurHint" class="error-hint">建议：尝试更换更清晰、对比度更高的图片。</p>
        <div class="error-actions">
          <button class="btn btn-primary" @click="resetAll">重新上传</button>
          <button class="btn btn-secondary" @click="retryTask">重试</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import {
  createTranslateTask,
  getTaskStatus,
  getTaskResultUrl,
  deleteTask,
  formatBytes,
  formatDuration,
} from '../api'

const emit = defineEmits(['notify'])

const STEPS = [
  { key: 'detect', label: '检测文本区域' },
  { key: 'ocr', label: '识别文字' },
  { key: 'translate', label: '翻译内容' },
  { key: 'inpaint', label: '修复图像' },
  { key: 'render', label: '渲染译文' },
]
const STEP_ORDER = STEPS.map((s) => s.key)

const allowedExts = '.jpg,.jpeg,.png,.webp,.bmp'
const MAX_MB = 10

const fileInput = ref(null)
const file = ref(null)
const filePreviewUrl = ref('')
const fileError = ref('')
const sourceLang = ref('ja')
const targetLang = ref('zh')
const isDragging = ref(false)

const taskId = ref(null)
const taskRunning = ref(false)
const submitting = ref(false)
const currentStep = ref('')
const stepProgress = ref({})
const errorMessage = ref('')
const result = ref(null)

let pollTimer = null

const canSubmit = computed(() => !!file.value && !fileError.value)

function validateFile(f) {
  if (!f) return '请选择图片文件'
  const ext = (f.name.split('.').pop() || '').toLowerCase()
  if (!allowedExts.includes(`.${ext}`)) return `不支持 .${ext} 格式，仅支持 JPG / PNG / WebP / BMP`
  if (f.size > MAX_MB * 1024 * 1024) return `文件超过 ${MAX_MB}MB 限制，请压缩后重试`
  if (f.size === 0) return '文件为空'
  if (!f.type.startsWith('image/')) return '请上传图片文件'
  return ''
}

function setFile(f) {
  const err = validateFile(f)
  fileError.value = err
  if (err) {
    file.value = null
    filePreviewUrl.value = ''
    return false
  }
  file.value = f
  filePreviewUrl.value = URL.createObjectURL(f)
  return true
}

function onFileSelect(e) {
  const f = e.target.files?.[0]
  if (f) setFile(f)
  e.target.value = ''
}

function onDrop(e) {
  isDragging.value = false
  const f = e.dataTransfer.files?.[0]
  if (f) setFile(f)
}

function triggerFileInput() {
  fileInput.value?.click()
}

function clearFile() {
  file.value = null
  filePreviewUrl.value = ''
  fileError.value = ''
}

function swapLangs() {
  if (sourceLang.value === 'zh') return
  const tmp = sourceLang.value
  sourceLang.value = targetLang.value === 'zh' ? (tmp === 'ja' ? 'en' : 'ja') : targetLang.value
  targetLang.value = tmp === 'zh' ? (targetLang.value === 'ja' ? 'zh' : 'zh') : tmp
}

const stepList = computed(() => {
  const currentIdx = STEP_ORDER.indexOf(currentStep.value)
  return STEPS.map((s, i) => ({
    ...s,
    done: i < currentIdx || (i === currentIdx && stepProgress.value[s.key] >= 100),
    active: i === currentIdx && stepProgress.value[s.key] < 100,
    percent: stepProgress.value[s.key] || 0,
  }))
})

const overallProgress = computed(() => {
  if (!currentStep.value) return 0
  const idx = STEP_ORDER.indexOf(currentStep.value)
  const doneWeight = idx / STEP_ORDER.length
  const curWeight = (stepProgress.value[currentStep.value] || 0) / 100 / STEP_ORDER.length
  return Math.min(99, Math.round((doneWeight + curWeight) * 100))
})

function startTranslate() {
  if (!file.value) return
  const f = file.value
  submitting.value = true
  fileError.value = ''
  errorMessage.value = ''
  result.value = null

  createTranslateTask(f, sourceLang.value, targetLang.value)
    .then((data) => {
      taskId.value = data.task_id
      taskRunning.value = true
      submitting.value = false
      currentStep.value = 'detect'
      stepProgress.value = {}
      startPolling()
    })
    .catch((err) => {
      submitting.value = false
      errorMessage.value = err.message
      emit('notify', err.message, 'error')
    })
}

function startPolling() {
  stopPolling()
  pollTimer = setInterval(pollStatus, 800)
  pollStatus()
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

async function pollStatus() {
  if (!taskId.value) return
  try {
    const status = await getTaskStatus(taskId.value)
    currentStep.value = status.step || currentStep.value
    stepProgress.value[currentStep.value] = Math.max(stepProgress.value[currentStep.value] || 0, status.progress)
    // 计算步骤内进度
    if (status.step) {
      const idx = STEP_ORDER.indexOf(status.step)
      if (idx > 0) {
        const prevDone = idx
        const stepBase = (prevDone / STEP_ORDER.length) * 100
        const stepShare = ((status.progress - stepBase) / (100 / STEP_ORDER.length)) * 100
        stepProgress.value[status.step] = Math.min(100, Math.max(0, Math.round(stepShare)))
      }
    }

    if (status.status === 'completed') {
      stopPolling()
      taskRunning.value = false
      result.value = {
        resultUrl: getTaskResultUrl(taskId.value),
        originalUrl: filePreviewUrl.value,
        textCount: status.text_count,
        durationMs: status.duration_ms,
      }
      emit('notify', '翻译完成！', 'success')
    } else if (status.status === 'failed') {
      stopPolling()
      taskRunning.value = false
      errorMessage.value = status.error || '翻译失败，请重试'
      emit('notify', errorMessage.value, 'error')
    }
  } catch (err) {
    // 网络抖动时继续轮询，多次失败则提示
    stopPolling()
    taskRunning.value = false
    errorMessage.value = err.message
    emit('notify', '获取进度失败：' + err.message, 'error')
  }
}

const isBlurHint = computed(() => (errorMessage.value || '').includes('模糊'))

function cancelTask() {
  stopPolling()
  if (taskId.value) {
    deleteTask(taskId.value).catch(() => {})
  }
  taskRunning.value = false
  resetAll()
}

function retryTask() {
  resetAll()
  setTimeout(() => startTranslate(), 50)
}

function resetAll() {
  stopPolling()
  taskId.value = null
  taskRunning.value = false
  submitting.value = false
  currentStep.value = ''
  stepProgress.value = {}
  errorMessage.value = ''
  result.value = null
}

onUnmounted(() => {
  stopPolling()
  if (filePreviewUrl.value) URL.revokeObjectURL(filePreviewUrl.value)
})

defineExpose({ refresh: () => {} })
</script>

<style scoped>
.translate-panel {
  max-width: 860px;
  margin: 0 auto;
  padding: var(--space-6);
  animation: fade-in var(--duration-base) var(--ease);
}

.hidden-input {
  display: none;
}

.section-title {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  margin-bottom: var(--space-4);
}

.section-title.spaced {
  margin-top: var(--space-6);
}

.step-chip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: var(--radius-full);
  background: var(--brand-600);
  color: #fff;
  font-size: 14px;
  font-weight: 700;
  flex-shrink: 0;
}

.section-title h2 {
  font-size: 19px;
}

.drop-zone {
  border: 2px dashed var(--border);
  border-radius: var(--radius-lg);
  padding: var(--space-7) var(--space-5);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  cursor: pointer;
  transition: all var(--duration-base) var(--ease);
  background: var(--surface-1);
  min-height: 200px;
}

.drop-zone:hover,
.drop-zone:focus-visible {
  border-color: var(--brand-400);
  background: var(--brand-50);
}

.drop-zone.is-dragging {
  border-color: var(--brand-500);
  background: var(--brand-50);
  transform: scale(1.01);
}

.drop-icon {
  color: var(--brand-400);
  margin-bottom: var(--space-3);
}

.drop-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--ink-700);
}

.drop-hint {
  font-size: 13px;
  color: var(--ink-500);
  margin-top: var(--space-2);
}

/* 文件预览 */
.file-preview {
  display: flex;
  align-items: center;
  gap: var(--space-4);
  width: 100%;
  max-width: 480px;
}

.file-preview img {
  width: 88px;
  height: 88px;
  object-fit: cover;
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);
  border: 1px solid var(--border);
}

.file-meta {
  display: flex;
  flex-direction: column;
  text-align: left;
  flex: 1;
  min-width: 0;
}

.file-name {
  font-size: 15px;
  color: var(--ink-900);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-size {
  font-size: 13px;
  color: var(--ink-500);
}

.remove-btn {
  align-self: flex-start;
  flex-shrink: 0;
  min-height: 36px;
}

.lang-row {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: var(--space-3);
  align-items: end;
}

.lang-swap {
  display: flex;
  align-items: center;
  padding-bottom: 12px;
}

.swap-btn {
  border: 1px solid var(--border);
  background: var(--surface-0);
  width: 40px;
  height: 40px;
  border-radius: var(--radius-full);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--ink-500);
  transition: all var(--duration-fast) var(--ease);
}

.swap-btn:hover {
  color: var(--brand-600);
  border-color: var(--brand-400);
  transform: rotate(180deg);
}

.action-row {
  display: flex;
  justify-content: center;
  margin-top: var(--space-6);
}

.btn-translate {
  min-width: 200px;
  padding: 14px 32px;
  font-size: 16px;
}

/* 进度 */
.progress-section {
  padding: var(--space-4) var(--space-2);
  text-align: center;
}

.progress-header {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-3);
}

.progress-header h2 {
  font-size: 20px;
}

.progress-file {
  color: var(--ink-500);
  font-size: 14px;
  margin-top: var(--space-2);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.spinner {
  width: 26px;
  height: 26px;
  border: 3px solid var(--brand-200);
  border-top-color: var(--brand-600);
  border-radius: 50%;
  animation: spin 0.9s linear infinite;
  flex-shrink: 0;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.progress-track {
  margin-top: var(--space-6);
  height: 10px;
  background: var(--surface-2);
  border-radius: var(--radius-full);
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--brand-500), var(--brand-600));
  border-radius: var(--radius-full);
  transition: width var(--duration-base) var(--ease);
}

.progress-percent {
  margin-top: var(--space-2);
  font-size: 14px;
  font-weight: 700;
  color: var(--brand-600);
}

.step-list {
  list-style: none;
  margin-top: var(--space-6);
  display: grid;
  gap: var(--space-2);
  text-align: left;
  max-width: 480px;
  margin-left: auto;
  margin-right: auto;
}

.step-item {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: 10px 14px;
  border-radius: var(--radius-md);
  background: var(--surface-1);
  color: var(--ink-500);
  transition: all var(--duration-fast) var(--ease);
}

.step-item.active {
  background: var(--brand-50);
  color: var(--brand-700);
}

.step-item.done {
  background: #ecfdf5;
  color: var(--success-500);
}

.step-dot {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: var(--radius-full);
  background: var(--surface-2);
  flex-shrink: 0;
}

.step-item.active .step-dot {
  background: var(--brand-200);
  animation: pulse 1.5s var(--ease) infinite;
}

@keyframes pulse {
  0%, 100% {
    box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.3);
  }
  50% {
    box-shadow: 0 0 0 6px rgba(99, 102, 241, 0);
  }
}

.step-item.done .step-dot {
  background: var(--success-500);
  color: #fff;
}

.step-label {
  flex: 1;
  font-size: 14px;
  font-weight: 500;
}

.step-percent {
  font-size: 13px;
  font-weight: 600;
}

.btn-cancel {
  margin-top: var(--space-6);
}

/* 结果 */
.result-section {
  padding: var(--space-2);
}

.result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
  flex-wrap: wrap;
  margin-bottom: var(--space-5);
}

.result-header h2 {
  font-size: 22px;
}

.result-meta {
  color: var(--ink-500);
  font-size: 14px;
  margin-top: 4px;
}

.compare-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-4);
}

.compare-label {
  font-size: 14px;
  font-weight: 600;
  color: var(--ink-500);
  margin-bottom: var(--space-2);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.result-tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  font-weight: 600;
  color: var(--brand-700);
  background: var(--brand-100);
  padding: 2px 8px;
  border-radius: var(--radius-full);
}

.image-frame {
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  overflow: hidden;
  background: var(--surface-1);
  aspect-ratio: 3 / 4;
  display: flex;
  align-items: center;
  justify-content: center;
}

.image-frame img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.result-actions {
  display: flex;
  justify-content: center;
  margin-top: var(--space-6);
}

/* 错误 */
.error-section {
  text-align: center;
  padding: var(--space-5) var(--space-2);
}

.error-icon {
  color: var(--danger-500);
  margin-bottom: var(--space-3);
}

.error-section h2 {
  font-size: 22px;
}

.error-text {
  color: var(--ink-700);
  margin-top: var(--space-3);
  max-width: 460px;
  margin-left: auto;
  margin-right: auto;
}

.error-hint {
  color: var(--ink-500);
  font-size: 14px;
  margin-top: var(--space-2);
}

.error-actions {
  display: flex;
  justify-content: center;
  gap: var(--space-3);
  margin-top: var(--space-6);
}

@media (max-width: 640px) {
  .compare-grid {
    grid-template-columns: 1fr;
  }
  .lang-row {
    grid-template-columns: 1fr;
  }
  .lang-swap {
    justify-content: center;
    padding-bottom: 0;
  }
}
</style>
