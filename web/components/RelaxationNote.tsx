interface RelaxationNoteProps {
  message: string
}

export default function RelaxationNote({ message }: RelaxationNoteProps) {
  if (!message) return null
  return (
    <div className="flex gap-3 p-4 rounded-xl bg-blue-50 border border-blue-200 text-blue-800 text-sm">
      <span className="text-lg leading-none mt-0.5">ℹ️</span>
      <p>{message}</p>
    </div>
  )
}
