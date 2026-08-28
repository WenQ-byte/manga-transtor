import { motion } from 'framer-motion'
import ShinyText from './ShinyText'

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
        <ShinyText
          text="Comic Translation System"
          color="#6b7280"
          shineColor="#ffffff"
          speed={3}
          spread={120}
          delay={1.5}
        />
      </motion.p>

      <h1
        className="mx-auto flex flex-col items-center"
        style={{
          fontSize: 'clamp(2.5rem, 9vw, 6.5rem)',
          fontWeight: 800,
          lineHeight: 1.15,
          letterSpacing: '-0.02em',
        }}
      >
        <ShinyText
          text="跨越语言"
          color="#6b7280"
          shineColor="#ffffff"
          speed={3}
          spread={120}
          delay={1.5}
        />
        <ShinyText
          text="读懂每一格"
          color="#6b7280"
          shineColor="#ffffff"
          speed={3}
          spread={120}
          delay={1.5}
        />
      </h1>
    </motion.div>
  )
}
