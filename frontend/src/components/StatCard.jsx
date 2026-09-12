import React from 'react'

export default function StatCard({ label, value, detail, icon: Icon, tone = '' }) {
  return (
    <article className={`stat-card ${tone}`}>
      <div className="stat-card-head"><span>{label}</span><Icon size={17} /></div>
      <strong className="stat-value">{value}</strong>
      <span className="stat-detail">{detail}</span>
    </article>
  )
}
