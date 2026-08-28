import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { CheckCircle2, AlertCircle, Info } from 'lucide-react'

const ToastContext = createContext(() => {})

const STYLES = {
  success: { icon: CheckCircle2, color: 'text-ok' },
  error: { icon: AlertCircle, color: 'text-danger' },
  info: { icon: Info, color: 'text-accent' },
}

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const notify = useCallback((message, type = 'info', duration = 3800) => {
    const id = Date.now() + Math.random()
    setToasts((prev) => [...prev, { id, message, type }])
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id))
    }, duration)
  }, [])

  const value = useMemo(() => notify, [notify])

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="fixed bottom-6 right-6 z-[300] flex flex-col items-end gap-3 pointer-events-none">
        <AnimatePresence>
          {toasts.map((t) => {
            const cfg = STYLES[t.type] || STYLES.info
            const Icon = cfg.icon
            return (
              <motion.div
                key={t.id}
                layout
                initial={{ opacity: 0, x: 40, scale: 0.95 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                exit={{ opacity: 0, x: 24, scale: 0.95 }}
                transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                className="pointer-events-auto flex items-center gap-3 rounded-xl border border-line bg-surface-2/90 px-4 py-3 shadow-[0_16px_48px_-12px_rgba(0,0,0,0.7)] backdrop-blur-xl"
              >
                <Icon size={18} className={`shrink-0 ${cfg.color}`} />
                <span className="text-sm text-ink-100">{t.message}</span>
              </motion.div>
            )
          })}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  )
}

export function useToast() {
  return useContext(ToastContext)
}
