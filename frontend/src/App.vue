<template>
  <div class="app">
    <!-- 头部 -->
    <header class="site-header">
      <div class="container header-inner">
        <a class="brand" href="#" @click.prevent="activeTab = 'translate'">
          <span class="brand-mark" aria-hidden="true">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M3 5h13v10H3z" fill="currentColor" opacity="0.2"/>
              <path d="M3 19h8M16 4l5 14-3.2-2.8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M6 9h7M9.5 5v4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            </svg>
          </span>
          <span class="brand-text">
            <strong>漫译</strong>
            <span class="brand-sub">漫画多语言翻译</span>
          </span>
        </a>

        <nav class="main-nav" aria-label="主导航">
          <button
            class="nav-item"
            :class="{ active: activeTab === 'translate' }"
            @click="activeTab = 'translate'"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <rect x="3" y="3" width="18" height="18" rx="3"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <path d="m21 15-5-5L5 21"/>
            </svg>
            翻译工具
          </button>
          <button
            class="nav-item"
            :class="{ active: activeTab === 'glossary' }"
            @click="activeTab = 'glossary'"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>
            </svg>
            专有名词
            <span class="nav-badge" v-if="glossaryCount > 0">{{ glossaryCount }}</span>
          </button>
        </nav>
      </div>
    </header>

    <main class="site-main container">
      <!-- 翻译页 -->
      <section v-if="activeTab === 'translate'" class="translate-page" aria-label="翻译工具">
        <div class="hero">
          <h1 class="hero-title">让漫画翻译，一键搞定</h1>
          <p class="hero-sub">
            上传漫画图片，智能识别气泡文字并翻译，保持原版排版不改变
          </p>
          <div class="hero-badges">
            <span class="badge">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg>
              保持原排版
            </span>
            <span class="badge">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z"/><path d="m9 12 2 2 4-4"/></svg>
              日语 / 英语 → 中文
            </span>
            <span class="badge">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 7V4h16v3"/><path d="M9 20h6"/><path d="M12 4v16"/></svg>
              专有名词统一译法
            </span>
          </div>
        </div>

        <TranslatePanel
          @notify="notify"
          @glossary-updated="refreshGlossaryCount"
        />
      </section>

      <!-- 专有名词页 -->
      <section v-else-if="activeTab === 'glossary'" class="glossary-page" aria-label="专有名词管理">
        <GlossaryPanel
          :key="glossaryRefreshKey"
          @notify="notify"
          @glossary-updated="refreshGlossaryCount"
        />
      </section>
    </main>

    <footer class="site-footer">
      <div class="container">
        <p>漫译 · 漫画多语言智能翻译系统</p>
        <p class="footer-sub">支持 日语 / 英语 → 中文，专有名词自定义词典</p>
      </div>
    </footer>

    <!-- Toast -->
    <div class="toast-container" aria-live="polite">
      <transition-group name="toast">
        <div v-for="t in toasts" :key="t.id" class="toast" :class="`toast-${t.type}`">
          <svg v-if="t.type === 'success'" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>
          <svg v-else-if="t.type === 'error'" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 8v5"/><path d="M12 16h.01"/></svg>
          <svg v-else width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/></svg>
          <span>{{ t.message }}</span>
        </div>
      </transition-group>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import TranslatePanel from './components/TranslatePanel.vue'
import GlossaryPanel from './components/GlossaryPanel.vue'
import { listGlossary } from './api'

const activeTab = ref('translate')
const toasts = ref([])
const glossaryCount = ref(0)
const glossaryRefreshKey = ref(0)

let toastId = 0

function notify(message, type = 'info', duration = 3500) {
  const id = ++toastId
  toasts.value.push({ id, message, type })
  setTimeout(() => {
    const idx = toasts.value.findIndex((t) => t.id === id)
    if (idx !== -1) toasts.value.splice(idx, 1)
  }, duration)
}

async function refreshGlossaryCount() {
  try {
    const data = await listGlossary()
    glossaryCount.value = data.total
  } catch {
    glossaryCount.value = 0
  }
}

onMounted(refreshGlossaryCount)
</script>

<style scoped>
.site-header {
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(246, 246, 250, 0.85);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
}

.header-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
  padding-top: var(--space-4);
  padding-bottom: var(--space-4);
}

.brand {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  color: var(--ink-900);
}

.brand-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 42px;
  height: 42px;
  border-radius: var(--radius-md);
  background: linear-gradient(135deg, var(--brand-500), var(--brand-700));
  color: #fff;
  box-shadow: var(--shadow-glow);
}

.brand-text {
  display: flex;
  flex-direction: column;
  line-height: 1.2;
}

.brand-text strong {
  font-size: 19px;
  letter-spacing: 0.5px;
}

.brand-sub {
  font-size: 12px;
  color: var(--ink-500);
}

.main-nav {
  display: flex;
  gap: var(--space-2);
}

.nav-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: none;
  background: transparent;
  color: var(--ink-500);
  font-size: 14px;
  font-weight: 600;
  padding: 10px 14px;
  border-radius: var(--radius-md);
  transition: all var(--duration-fast) var(--ease);
  position: relative;
}

.nav-item:hover {
  background: var(--surface-0);
  color: var(--ink-900);
}

.nav-item.active {
  background: var(--surface-0);
  color: var(--brand-600);
  box-shadow: var(--shadow-sm);
}

.nav-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 18px;
  height: 18px;
  padding: 0 5px;
  border-radius: var(--radius-full);
  background: var(--brand-100);
  color: var(--brand-700);
  font-size: 11px;
}

.site-main {
  padding-top: var(--space-7);
  padding-bottom: var(--space-8);
  min-height: calc(100vh - 220px);
}

.hero {
  text-align: center;
  margin-bottom: var(--space-7);
  animation: slide-up var(--duration-slow) var(--ease);
}

.hero-title {
  font-size: clamp(28px, 4vw, 40px);
  letter-spacing: -0.5px;
  background: linear-gradient(120deg, var(--ink-900), var(--brand-700));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.hero-sub {
  color: var(--ink-500);
  font-size: 17px;
  margin-top: var(--space-3);
  max-width: 560px;
  margin-left: auto;
  margin-right: auto;
}

.hero-badges {
  display: flex;
  justify-content: center;
  gap: var(--space-3);
  margin-top: var(--space-5);
  flex-wrap: wrap;
}

.site-footer {
  border-top: 1px solid var(--border);
  padding: var(--space-6) 0;
  text-align: center;
  color: var(--ink-500);
  font-size: 14px;
}

.footer-sub {
  font-size: 12px;
  color: var(--ink-300);
  margin-top: 4px;
}

/* 响应式 */
@media (max-width: 640px) {
  .hero-badges {
    flex-direction: column;
    align-items: center;
  }
  .header-inner {
    flex-direction: column;
    gap: var(--space-3);
  }
  .brand-sub {
    display: none;
  }
}
</style>
