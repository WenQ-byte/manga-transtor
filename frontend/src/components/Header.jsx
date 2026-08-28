import { motion } from 'framer-motion'
import { Languages, BookMarked } from 'lucide-react'

function BrandMark() {
  return (
    <span className="relative flex h-9 w-9 items-center justify-center rounded-[10px] bg-gradient-to-br from-accent to-accent-strong shadow-[0_0_20px_-4px_rgba(34,211,238,0.5)]">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3h11A2.5 2.5 0 0 1 20 5.5v8a2.5 2.5 0 0 1-2.5 2.5H9l-4 4v-4h-.5A2.5 2.5 0 0 1 4 13.5v-8z" fill="rgba(6,10,16,0.35)" />
        <path d="M8 8h8M8 11.5h5" stroke="#061018" strokeWidth="1.8" strokeLinecap="round" />
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
