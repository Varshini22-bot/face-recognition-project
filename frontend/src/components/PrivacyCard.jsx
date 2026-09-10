import { LockKeyhole } from 'lucide-react'

export default function PrivacyCard() {
  return (
    <aside className="privacy-card">
      <span className="privacy-icon"><LockKeyhole size={18} /></span>
      <div><strong>Privacy by design</strong><p>Biometric data stays within the configured recognition environment. No raw embeddings are displayed here.</p></div>
    </aside>
  )
}
