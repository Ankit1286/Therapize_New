export default function SkeletonCard() {
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-[var(--border)] p-6 animate-pulse">
      <div className="flex gap-4">
        <div className="flex-1 space-y-3">
          <div className="h-5 bg-gray-200 rounded w-2/5" />
          <div className="h-4 bg-gray-100 rounded w-1/3" />
          <div className="flex gap-2 mt-2">
            <div className="h-6 bg-gray-100 rounded-full w-16" />
            <div className="h-6 bg-gray-100 rounded-full w-20" />
            <div className="h-6 bg-gray-100 rounded-full w-14" />
          </div>
          <div className="space-y-2 mt-3">
            <div className="h-3 bg-gray-100 rounded w-full" />
            <div className="h-3 bg-gray-100 rounded w-5/6" />
            <div className="h-3 bg-gray-100 rounded w-4/6" />
          </div>
        </div>
        <div className="w-32 space-y-3">
          <div className="h-8 bg-gray-200 rounded-full w-28" />
          <div className="h-9 bg-gray-100 rounded-lg w-full" />
          <div className="h-9 bg-gray-100 rounded-lg w-full" />
        </div>
      </div>
    </div>
  )
}
