import { motion } from 'framer-motion'
import { ScanText, ArrowRightLeft, BookMarked } from 'lucide-react'
import MaskedHeading from './MaskedHeading'
import heroFill from '../assets/hero-fill.svg'

const BADGES = [
  { icon: ScanText, label: '保持原排版' },
  { icon: ArrowRightLeft, label: '日语 / 英语 → 中文' },
  { icon: BookMarked, label: '专有名词统一译法' },
]

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.09, delayChildren: 0.05 } },
}

const item = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 120, damping: 18 } },
}

export default function Hero() {
  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="relative mx-auto max-w-3xl pt-16 pb-14 text-center"
    >
      <motion.p variants={item} className="eyebrow mb-6">
        Comic Translation System
      </motion.p>

      <MaskedHeading
        text="跨越语言 读懂每一格"
        tag="h1"
        src={heroFill}
        align="center"
        weight={600}
        tracking={-0.01}
        lineHeight={1.12}
        textScale={0.125}
        fillScale={1.22}
        parallax={22}
        drift={16}
        brightness={1.05}
        saturation={1.15}
        reveal="rise"
        duration={1.15}
        stagger={0.1}
        trigger="view"
      />

      <motion.p variants={item} className="mx-auto mt-8 max-w-xl text-[17px] leading-relaxed text-ink-400">
        上传漫画图片，智能识别气泡文字并翻译，在保留原始排版与画面细节的同时，输出地道译文。
      </motion.p>

      <motion.div variants={item} className="mt-10 flex flex-wrap items-center justify-center gap-3">
        {BADGES.map((b) => {
          const Icon = b.icon
          return (
            <span
              key={b.label}
              className="flex items-center gap-2 rounded-full border border-line bg-surface/70 px-4 py-2 text-[13px] text-ink-200 backdrop-blur"
            >
              <Icon size={15} className="text-accent" />
              {b.label}
            </span>
          )
        })}
      </motion.div>
    </motion.div>
  )
}
