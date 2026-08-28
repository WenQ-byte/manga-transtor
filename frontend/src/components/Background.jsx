export default function Background() {
  return (
    <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
      <div className="absolute inset-0 grid-dots [mask-image:radial-gradient(ellipse_70%_60%_at_50%_0%,black_0%,transparent_75%)] opacity-60" />

      <div className="absolute -top-40 left-1/2 h-[520px] w-[900px] -translate-x-1/2 rounded-full bg-accent/[0.07] blur-[140px]" />
      <div className="absolute top-1/3 -left-52 h-[420px] w-[420px] rounded-full bg-accent-strong/[0.05] blur-[120px]" />
      <div className="absolute bottom-0 -right-40 h-[460px] w-[520px] rounded-full bg-accent/[0.04] blur-[140px]" />

      <div className="absolute inset-0 noise opacity-[0.025]" />
    </div>
  )
}
