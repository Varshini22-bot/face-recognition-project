import React from 'react'
import { Check, FileImage, LockKeyhole, LoaderCircle, ScanSearch, Upload } from 'lucide-react'

export default function RecognitionResult({ file, previewUrl, result, loading, error, onRecognize, onUpload }) {
  const firstFace = result?.faces?.[0]
  return (
    <section className="result-panel panel-surface" aria-labelledby="result-title">
      <div className="section-kicker"><span className="kicker-line" /> DECISION PREVIEW</div>
      {previewUrl && <img className="result-preview" src={previewUrl} alt="Selected face recognition input" />}
      {!previewUrl && <div className="result-empty-icon"><ScanSearch size={24} /></div>}
      <h2 id="result-title">{loading ? 'Analyzing image' : firstFace ? (result.faces.length > 1 ? `${result.faces.length} faces analyzed` : (firstFace.recognized ? firstFace.name : 'Unknown person')) : 'Upload an image to begin'}</h2>
      <p>{error || (loading ? 'The local engine is processing the selected image.' : firstFace ? `${result.face_count} face${result.face_count === 1 ? '' : 's'} detected.` : 'Results will show identity, similarity, and processing data here.')}</p>
      {error && <div className="api-error" role="alert">{error}</div>}
      {result && <><div className="result-face-list">{result.faces.map((face, index) => <div className="result-face-row" key={`${face.name || 'unknown'}-${index}`}><div><strong>{face.name || 'Unknown person'}</strong><span className={face.recognized ? 'known-text' : 'unknown-text'}>Decision: {(face.status || (face.recognized ? 'Known' : 'Unknown')).toUpperCase()}</span></div><span>{`Similarity: ${face.similarity == null ? '—' : `${(face.similarity * 100).toFixed(2)}%`} · Threshold: ${((face.threshold ?? 0.5) * 100).toFixed(0)}%`}</span></div>)}</div><div className="result-facts"><div><span>Faces</span><strong>{result.face_count}</strong></div><div><span>Time</span><strong>{result.processing_time_ms} ms</strong></div><div><span>Threshold</span><strong>{((firstFace?.threshold ?? 0.5) * 100).toFixed(0)}%</strong></div><div><span>Decision</span><strong>{firstFace?.recognized ? 'KNOWN' : 'UNKNOWN'}</strong></div></div></>}
      {!result && <div className="result-facts"><div><span>Threshold</span><strong>0.50 configured</strong></div><div><span>Privacy</span><strong><Check size={14} /> Local-first</strong></div></div>}
      <div className="result-actions"><button className="button button-primary" type="button" onClick={onRecognize} disabled={!file || loading}>{loading ? <LoaderCircle className="spin" size={15} /> : <ScanSearch size={15} />} {loading ? 'Processing' : 'Recognize'}</button><button className="button button-secondary" type="button" onClick={onUpload}><Upload size={15} /> Choose image</button></div>
      <div className="result-note"><LockKeyhole size={14} /> Raw embeddings stay outside the interface.</div>
    </section>
  )
}
