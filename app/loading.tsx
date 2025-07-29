export default function Loading() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
      <div className="text-center">
        <div className="w-16 h-16 border-4 border-amber-600/30 border-t-amber-600 rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-slate-300">Loading Vestige Capital...</p>
      </div>
    </div>
  )
}
