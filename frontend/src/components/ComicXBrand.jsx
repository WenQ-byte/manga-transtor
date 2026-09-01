export function ComicXMark({ className = '', rounded = 'rounded-lg' }) {
  return (
    <img src="/comicx-mark-transparent.png" alt="ComicX" draggable={false}
      className={`${rounded} shrink-0 select-none object-contain ${className}`} />
  )
}

export function ComicXWordmark({ className = '', glow = false }) {
  return (
    <img src="/comicx-wordmark-transparent.png" alt="ComicX" draggable={false}
      className={`select-none ${glow ? 'comicx-glow mix-blend-screen' : ''} ${className}`} />
  )
}
