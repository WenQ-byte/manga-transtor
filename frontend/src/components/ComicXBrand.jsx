export function ComicXMark({ className = '', rounded = 'rounded-lg' }) {
  return (
    <img src="/comicx-app-icon.png" alt="ComicX" draggable={false}
      className={`${rounded} shrink-0 select-none object-contain ${className}`} />
  )
}

const LETTERS = [
  { left: 2.07, width: 16.35 },
  { left: 20.03, width: 16.30 },
  { left: 37.89, width: 17.91 },
  { left: 57.37, width: 7.87 },
  { left: 66.53, width: 15.98 },
  { left: 84.16, width: 15.38 },
]

function clipFor({ left, width }) {
  return `inset(0 ${(100 - left - width).toFixed(2)}% 0 ${left.toFixed(2)}%)`
}

export function ComicXWordmark({ className = '', entrance = true }) {
  return (
    <span className={`comicx-wordmark ${className}`} aria-label="ComicX">
      {LETTERS.map((letter, index) => (
        <span
          key={index}
          className={`comicx-wordmark__letter${entrance ? ' comicx-wordmark__letter--animate' : ''}`}
          style={{
            clipPath: clipFor(letter),
            backgroundImage: 'url(/comicx-wordmark-transparent.png)',
            animationDelay: entrance ? `${index * 0.14}s` : undefined,
          }}
        />
      ))}
    </span>
  )
}
