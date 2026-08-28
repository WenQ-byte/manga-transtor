import { useCallback, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { MoveHorizontal } from 'lucide-react'

export default function CompareSlider({ before, after, beforeLabel = '原图', afterLabel = '译后' }) {
  const [pos, setPos] = useState(50)
  const [dragging, setDragging] = useState(false)
  const ref = useRef(null)

  const updateFromClientX = useCallback((clientX) => {
    const el = ref.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const p = ((clientX - rect.left) / rect.width) * 100
    setPos(Math.max(0, Math.min(100, p)))
  }, [])

  const onPointerDown = (e) => {
    e.preventDefault()
    setDragging(true)
    updateFromClientX(e.clientX)
  }

  const onPointerMove = (e) => {
    if (!dragging) return
    updateFromClientX(e.clientX)
  }

  const stopDrag = () => setDragging(false)

  return (
    <div
      ref={ref}
      className="relative w-full cursor-ew-resize select-none overflow-hidden rounded-xl border border-line bg-surface-2"
      style={{ touchAction: 'none' }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={stopDrag}
      onPointerLeave={stopDrag}
      role="slider"
      aria-label="前后对比"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pos)}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'ArrowLeft') setPos((p) => Math.max(0, p - 2))
        if (e.key === 'ArrowRight') setPos((p) => Math.min(100, p + 2))
      }}
    >
      <div className="relative flex aspect-[3/4] max-h-[68vh] w-full items-center justify-center">
        <img src={before} alt={beforeLabel} className="absolute inset-0 h-full w-full object-contain" draggable={false} />

        <div className="absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
          <img src={after} alt={afterLabel} className="absolute inset-0 h-full w-full object-contain" draggable={false} />
        </div>

        <span className="absolute left-4 top-3 rounded-md border border-line bg-bg/70 px-2.5 py-1 font-mono text-[11px] uppercase tracking-widest text-ink-200 backdrop-blur">
          {beforeLabel}
        </span>
        <span className="absolute right-4 top-3 rounded-md border border-accent/30 bg-accent/10 px-2.5 py-1 font-mono text-[11px] uppercase tracking-widest text-accent backdrop-blur">
          {afterLabel}
        </span>

        <div className="absolute inset-y-0" style={{ left: `${pos}%` }}>
          <div className="absolute inset-y-0 -translate-x-1/2 w-px bg-accent/70" />
          <motion.div
            className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2"
            animate={{ scale: dragging ? 1.12 : 1 }}
            transition={{ type: 'spring', stiffness: 400, damping: 25 }}
          >
            <span className="ring-glow flex h-9 w-9 items-center justify-center rounded-full bg-bg/90 text-accent backdrop-blur">
              <MoveHorizontal size={16} />
            </span>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
