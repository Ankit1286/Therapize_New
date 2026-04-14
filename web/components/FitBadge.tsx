interface FitBadgeProps {
  rank: number
}

export default function FitBadge({ rank }: FitBadgeProps) {
  if (rank === 0) {
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold bg-green-100 text-green-800">
        <span className="w-2 h-2 rounded-full bg-green-500" />
        Strong fit
      </span>
    )
  }
  if (rank <= 2) {
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold bg-yellow-100 text-yellow-800">
        <span className="w-2 h-2 rounded-full bg-yellow-500" />
        Good fit
      </span>
    )
  }
  if (rank <= 5) {
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold bg-orange-100 text-orange-800">
        <span className="w-2 h-2 rounded-full bg-orange-400" />
        Decent fit
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold bg-gray-100 text-gray-600">
      <span className="w-2 h-2 rounded-full bg-gray-400" />
      Possible match
    </span>
  )
}
