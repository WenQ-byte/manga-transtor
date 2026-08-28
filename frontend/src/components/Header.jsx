import { motion } from 'framer-motion'
import { Languages, BookMarked } from 'lucide-react'

function BrandMark() {
  return (
    <span className="relative flex h-9 w-9 items-center justify-center rounded-[11px] bg-surface-2 ring-1 ring-white/15 shadow-[0_0_20px_-6px_rgba(255,255,255,0.12)]">
      <svg width="20" height="20" viewBox="0 0 32 32" fill="none" aria-hidden="true">
        <g stroke="#ffffff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12.5 7 C10.8 7.2 10.2 8.8 10.5 10.5 C10.8 12 12.2 12.8 14 12.5" />
          <path d="M19.5 7 C21.2 7.2 21.8 8.8 21.5 10.5 C21.2 12 19.8 12.8 18 12.5" />
          <path d="M5.5 16.5 L26.5 16.5" />
          <path d="M13.5 19 C11.8 21.5 11.2 24.5 13 26.8 C14.8 28.8 18 28.2 19.5 25.8" />
          <path d="M18.5 19 C20.2 21.5 20.8 24.5 19 26.8 C17.2 28.8 14 28.2 12.5 25.8" />
        </g>
        <rect x="14.6" y="15.1" width="2.8" height="2.8" rx="0.6" fill="#ffffff" />
      </svg>
    </span>
  )
}

export default function Header({ active, onChange, glossaryCount }) {
  const items = [
    { key: 'translate', label: '翻译工具', icon: Languages },
    { key: 'glossary', label: '专有名词', icon: BookMarked, badge: glossaryCount },
  ]

  return (
    <header className="sticky top-0 z-50 border-b border-line bg-bg/70 backdrop-blur-xl">
      <div className="container-main flex h-16 items-center justify-between">
        <button
          className="group flex items-center gap-3"
          onClick={() => onChange('translate')}
          aria-label="漫译首页"
        >
          <BrandMark />
          <span className="flex flex-col items-start leading-none">
            <span className="font-display text-[17px] font-semibold tracking-tight text-ink-100">
              漫译
            </span>
            <span className="font-mono text-[9px] uppercase tracking-[0.3em] text-ink-500 transition-colors group-hover:text-ink-400">
              Comic Translator
            </span>
          </span>
        </button>

        <nav className="relative flex items-center gap-1" aria-label="主导航">
          {items.map((item) => {
            const Icon = item.icon
            const isActive = active === item.key
            return (
              <button
                key={item.key}
                onClick={() => onChange(item.key)}
                className="relative flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors"
              >
                {isActive && (
                  <motion.span
                    layoutId="nav-active"
                    className="absolute inset-0 rounded-lg border border-line-strong bg-surface-2"
                    transition={{ type: 'spring', stiffness: 400, damping: 32 }}
                  />
                )}
                <Icon
                  size={16}
                  className={`relative z-10 ${isActive ? 'text-accent' : 'text-ink-400'}`}
                />
                <span
                  className={`relative z-10 ${isActive ? 'text-ink-100' : 'text-ink-400 hover:text-ink-200'}`}
                >
                  {item.label}
                </span>
                {item.badge > 0 && (
                  <span className="relative z-10 flex h-[18px] min-w-[18px] items-center justify-center rounded-full bg-accent/15 px-1.5 font-mono text-[10px] font-medium text-accent">
                    {item.badge}
                  </span>
                )}
              </button>
            )
          })}
        </nav>
      </div>
    </header>
  )
}
