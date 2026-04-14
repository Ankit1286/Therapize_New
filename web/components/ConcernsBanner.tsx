interface ConcernsBannerProps {
  concerns: string[]
}

export default function ConcernsBanner({ concerns }: ConcernsBannerProps) {
  if (!concerns || concerns.length === 0) return null

  return (
    <div className="flex flex-wrap items-center gap-2 p-4 rounded-xl bg-[#EBF4F7] border border-[#C5DFE8]">
      <span className="text-[#1A5276] font-semibold text-sm mr-1">
        💙 Based on what you shared, we focused on therapists experienced with:
      </span>
      {concerns.map((concern) => (
        <span
          key={concern}
          className="px-3 py-1 rounded-full text-xs font-semibold bg-[#4A90A4] text-white"
        >
          {concern.charAt(0).toUpperCase() + concern.slice(1)}
        </span>
      ))}
    </div>
  )
}
