export function ComicXMark({ className = '', rounded = 'rounded-lg' }) {
  return (
    <img src="/comicx-app-icon.png" alt="ComicX" draggable={false}
      className={`${rounded} shrink-0 select-none object-contain ${className}`} />
  )
}

export function ComicXWordmark({ className = '', shiny = false }) {
  return (
    <span className={`comicx-wordmark ${shiny ? 'comicx-wordmark--shiny' : ''} ${className}`}>
      <img src="/comicx-wordmark-transparent.png" alt="ComicX" draggable={false}
        className="comicx-wordmark__base select-none" />
      {shiny ? <span className="comicx-wordmark__shine" aria-hidden="true" /> : null}
    </span>
  )
}
