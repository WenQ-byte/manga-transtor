import { useCallback, useEffect, useRef } from 'react'
import './BorderGlow.css'

function parseHSL(value) {
  const match = String(value).match(/[\d.]+/g)
  if (!match || match.length < 3) return { h: 40, s: 80, l: 80 }
  return { h: Number(match[0]), s: Number(match[1]), l: Number(match[2]) }
}

function glowVariables(color, intensity) {
  const { h, s, l } = parseHSL(color)
  const values = [100, 60, 50, 40, 30, 20, 10]
  const keys = ['', '-60', '-50', '-40', '-30', '-20', '-10']
  return Object.fromEntries(values.map((opacity, index) => [
    `--glow-color${keys[index]}`,
    `hsl(${h}deg ${s}% ${l}% / ${Math.min(opacity * intensity, 100)}%)`,
  ]))
}

function gradientVariables(colors) {
  const positions = ['80% 55%', '69% 34%', '8% 6%', '41% 38%', '86% 85%', '82% 18%', '51% 4%']
  const map = [0, 1, 2, 0, 1, 2, 1]
  const vars = {}
  positions.forEach((position, index) => {
    vars[`--gradient-${index + 1}`] = `radial-gradient(at ${position}, ${colors[map[index]] || colors[0]} 0, transparent 52%)`
  })
  vars['--gradient-base'] = `linear-gradient(${colors[0]} 0 100%)`
  return vars
}

export default function BorderGlow({
  children,
  as: Component = 'div',
  className = '',
  edgeSensitivity = 30,
  glowColor = '40 80 80',
  backgroundColor = '#120F17',
  borderRadius = 12,
  glowRadius = 40,
  glowIntensity = 1,
  coneSpread = 25,
  animated = false,
  colors = ['#c084fc', '#f472b6', '#38bdf8'],
  fillOpacity = 0.5,
  style,
  ...props
}) {
  const cardRef = useRef(null)

  const updatePointer = useCallback((event) => {
    const card = cardRef.current
    if (!card) return
    const rect = card.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    const cx = rect.width / 2
    const cy = rect.height / 2
    const dx = x - cx
    const dy = y - cy
    const edge = Math.min(Math.max(Math.min(Math.abs(dx) / Math.max(cx, 1), Math.abs(dy) / Math.max(cy, 1)), 0), 1)
    let angle = Math.atan2(dy, dx) * 180 / Math.PI + 90
    if (angle < 0) angle += 360
    card.style.setProperty('--edge-proximity', `${(edge * 100).toFixed(2)}`)
    card.style.setProperty('--cursor-angle', `${angle.toFixed(2)}deg`)
  }, [])

  const resetPointer = useCallback(() => {
    const card = cardRef.current
    if (!card) return
    card.style.setProperty('--edge-proximity', '0')
    card.style.setProperty('--cursor-angle', '0deg')
  }, [])

  useEffect(() => {
    if (!animated || !cardRef.current) return undefined
    const card = cardRef.current
    card.classList.add('border-glow-card--animated')
    return () => card.classList.remove('border-glow-card--animated')
  }, [animated])

  return (
    <Component
      ref={cardRef}
      onPointerMove={updatePointer}
      onPointerLeave={resetPointer}
      className={`border-glow-card ${className}`}
      style={{
        '--card-bg': backgroundColor,
        '--edge-sensitivity': edgeSensitivity,
        '--border-radius': `${borderRadius}px`,
        '--glow-padding': `${glowRadius}px`,
        '--cone-spread': `${coneSpread}deg`,
        '--fill-opacity': fillOpacity,
        ...glowVariables(glowColor, glowIntensity),
        ...gradientVariables(colors),
        ...style,
      }}
      {...props}
    >
      <span className="edge-light" aria-hidden="true" />
      <span className="border-glow-inner">{children}</span>
    </Component>
  )
}
