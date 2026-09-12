import React from 'react'
import { Activity, CircleHelp, LayoutDashboard, ScanFace, UsersRound } from 'lucide-react'

const links = [
  { label: 'Dashboard', icon: LayoutDashboard },
  { label: 'Recognition', icon: ScanFace },
  { label: 'People', icon: UsersRound },
  { label: 'Evaluation', icon: Activity },
]

export default function Navbar({ active, onNavigate }) {
  return (
    <header className="topbar">
      <a className="brand" href="#dashboard" aria-label="VisionID dashboard">
        <span className="brand-mark"><ScanFace size={19} strokeWidth={2.4} /></span>
        <span>vision<span className="brand-accent">id</span></span>
      </a>
      <nav className="main-nav" aria-label="Primary navigation">
        {links.map(({ label, icon: Icon }) => (
          <button
            className={`nav-link ${active === label ? 'is-active' : ''}`}
            key={label}
            onClick={() => onNavigate(label)}
            type="button"
          >
            <Icon size={16} />
            {label}
          </button>
        ))}
      </nav>
      <div className="system-status"><span className="status-dot" /> Local engine <CircleHelp size={15} /></div>
    </header>
  )
}
