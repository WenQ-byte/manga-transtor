import AcidSquares from './AcidSquares'

export default function Background() {
  return (
    <div className="fixed inset-0 z-0 overflow-hidden">
      <AcidSquares
        color1="#5227FF"
        color2="#A855F7"
        color3="#FFFFFF"
        detail="medium"
        speed={0.7}
        waveDepth={1}
        zoom={1.3}
        density={10.0}
        glow={1.0}
        exposure={2700}
        spread={0.3}
        stepSize={0.002}
        colorShift={0}
        contrast={1}
        brightness={1.0}
        opacity={0.5}
        mouseInteraction
        mouseStrength={0.1}
        mouseRadius={0.35}
        blur={0}
        grain
        grainIntensity={0.05}
      />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(130%_95%_at_50%_0%,rgba(9,9,9,0)_45%,rgba(9,9,9,0.6)_100%)]" />
    </div>
  )
}
