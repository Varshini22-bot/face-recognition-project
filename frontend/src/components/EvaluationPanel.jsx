import { BarChart3, LoaderCircle } from 'lucide-react'

const metricLabels = [['accuracy', 'Accuracy'], ['precision', 'Precision'], ['recall', 'Recall'], ['f1', 'F1'], ['far', 'FAR'], ['frr', 'FRR']]

export default function EvaluationPanel({ data, loading, error }) {
  if (loading) return <section className="evaluation-panel panel-surface"><div className="evaluation-state"><LoaderCircle className="spin" size={21} /><strong>No evaluation data is being loaded.</strong></div></section>
  if (error) return <section className="evaluation-panel panel-surface"><div className="evaluation-state evaluation-error"><strong>Unable to load evaluation data.</strong><span>{error}</span></div></section>
  if (!data?.available) return <section className="evaluation-panel panel-surface"><div className="panel-heading"><div><span className="section-kicker"><span className="kicker-line" /> EVALUATION</span><h2>Evidence over assumptions.</h2></div><BarChart3 size={18} /></div><div className="evaluation-state"><strong>No evaluation report available yet.</strong><span>Run the local evaluation command to populate this dashboard with measured results.</span></div></section>

  const failures = (data.diagnostics || []).filter((item) => !item.accepted || item.detected_face_count !== 1)
  return <section className="evaluation-panel panel-surface">
    <div className="panel-heading"><div><span className="section-kicker"><span className="kicker-line" /> EVALUATION / MEASURED</span><h2>Evidence over assumptions.</h2></div><BarChart3 size={18} /></div>
    <div className="evaluation-meta"><span>{data.model} · {data.detector}</span><span>{data.images_evaluated} images / {data.faces_evaluated} faces</span></div>
    <div className="metric-grid">{metricLabels.map(([key, label]) => <div className="metric-cell" key={key}><span>{label}</span><strong>{(data[key] * 100).toFixed(1)}%</strong></div>)}</div>
    <div className="evaluation-facts"><div><span>Configured threshold</span><strong>{data.threshold.toFixed(2)}</strong></div><div><span>Best on current dataset</span><strong>{data.best_threshold.toFixed(2)}</strong></div><div><span>Known samples</span><strong>{data.known_samples}</strong></div><div><span>Unknown samples</span><strong>{data.unknown_samples}</strong></div></div>
    <p className="evaluation-warning">Baseline evaluation on a small local dataset. Results are indicative, not production accuracy.</p>
    {data.threshold_sweep?.length > 0 && <div className="sweep-table"><div className="sweep-heading"><span>Threshold sweep</span><span>F1 / FAR / FRR</span></div>{data.threshold_sweep.map((item) => <div className={`sweep-row ${item.threshold === data.threshold ? 'current' : ''}`} key={item.threshold}><span>{item.threshold.toFixed(2)}</span><span>{(item.f1 * 100).toFixed(1)}% / {(item.far * 100).toFixed(1)}% / {(item.frr * 100).toFixed(1)}%</span></div>)}</div>}
    {failures.length > 0 && <div className="failure-list"><span className="section-kicker"><span className="kicker-line" /> FAILURE CASES</span>{failures.map((item, index) => <div className="failure-row" key={`${item.image_filename}-${index}`}><strong>{item.image_filename}</strong><span>{item.detected_face_count === 0 ? 'No face' : item.detected_face_count > 1 ? `${item.detected_face_count} faces` : 'Rejected'} · {item.similarity == null ? '—' : item.similarity.toFixed(3)}</span></div>)}</div>}
  </section>
}
