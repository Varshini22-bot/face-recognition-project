import React from 'react'
import { Camera, CircleDot, Fingerprint, ScanFace, ShieldCheck } from 'lucide-react'

const stages = [
  { label: 'Input', detail: 'Image or camera', icon: Camera },
  { label: 'Detect', detail: 'YuNet face scan', icon: ScanFace },
  { label: 'Embed', detail: 'ArcFace vector', icon: Fingerprint },
  { label: 'Compare', detail: 'Cosine similarity', icon: CircleDot },
  { label: 'Decide', detail: 'Known or unknown', icon: ShieldCheck },
]

export default function Pipeline({ activeStage, onStageChange }) {
  return (
    <div className="pipeline" aria-label="AI recognition pipeline">
      {stages.map(({ label, detail, icon: Icon }, index) => (
        <div className="pipeline-item" key={label}>
          <button
            className={`pipeline-step ${activeStage === index ? 'is-active' : ''}`}
            type="button"
            onClick={() => onStageChange(index)}
            aria-label={`${label}: ${detail}`}
          >
            <span className="pipeline-icon"><Icon size={18} /></span>
            <span><strong>{label}</strong><small>{detail}</small></span>
          </button>
          {index < stages.length - 1 && <span className="pipeline-line" aria-hidden="true" />}
        </div>
      ))}
    </div>
  )
}
