import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ToastProvider } from './components/Toast'
import Background from './components/Background'
import Header from './components/Header'
import TranslatePanel from './components/TranslatePanel'
import GlossaryPanel from './components/GlossaryPanel'
import { listGlossary } from './api'

export default function App() {
  const [active, setActive] = useState('translate')
  const [glossaryCount, setGlossaryCount] = useState(0)

  useEffect(() => {
    listGlossary()
      .then((data) => setGlossaryCount(data.total))
      .catch(() => {})
  }, [])

  return (
    <ToastProvider>
      <div className="relative min-h-screen">
        <Background />
        <Header active={active} onChange={setActive} glossaryCount={glossaryCount} />

        <main className="relative z-10">
          <AnimatePresence mode="wait">
            {active === 'translate' ? (
              <motion.div
                key="translate"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
              >
                <TranslatePanel />
              </motion.div>
            ) : (
              <motion.div
                key="glossary"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="container-main pt-14 pb-24"
              >
                <GlossaryPanel onCountChange={setGlossaryCount} />
              </motion.div>
            )}
          </AnimatePresence>
        </main>

        {active === 'glossary' ? (
          <footer className="relative z-10 border-t border-line py-8">
            <div className="container-main text-center text-sm text-ink-500">
              漫译 · 漫画多语言智能翻译系统
            </div>
          </footer>
        ) : null}
      </div>
    </ToastProvider>
  )
}
