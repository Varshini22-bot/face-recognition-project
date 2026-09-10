import { ArrowUpRight, BarChart3, Check, Database, FileUp, Gauge, Layers3, Play, ShieldCheck, Sparkles, UsersRound } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import Navbar from './components/Navbar.jsx'
import Pipeline from './components/Pipeline.jsx'
import RecognitionResult from './components/RecognitionResult.jsx'
import StatCard from './components/StatCard.jsx'
import PrivacyCard from './components/PrivacyCard.jsx'
import { recognitionService } from './services/recognitionService.js'
import { peopleService } from './services/peopleService.js'
import PeoplePanel from './components/PeoplePanel.jsx'
import RegisterPersonModal from './components/RegisterPersonModal.jsx'
import EvaluationPanel from './components/EvaluationPanel.jsx'
import { getEvaluation } from './services/evaluationService.js'

const navAnchors = { Dashboard: 'dashboard', Recognition: 'workspace', People: 'people', Evaluation: 'evaluation' }

function App() {
  const [activeNav, setActiveNav] = useState('Dashboard')
  const [activeStage, setActiveStage] = useState(0)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [recognitionResult, setRecognitionResult] = useState(null)
  const [recognitionError, setRecognitionError] = useState('')
  const [loading, setLoading] = useState(false)
  const [people, setPeople] = useState([])
  const [peopleLoading, setPeopleLoading] = useState(true)
  const [peopleError, setPeopleError] = useState('')
  const [registrationOpen, setRegistrationOpen] = useState(false)
  const [registrationLoading, setRegistrationLoading] = useState(false)
  const [registrationError, setRegistrationError] = useState('')
  const [evaluation, setEvaluation] = useState(null)
  const [evaluationLoading, setEvaluationLoading] = useState(true)
  const [evaluationError, setEvaluationError] = useState('')
  const fileInput = useRef(null)

  const refreshPeople = async () => {
    setPeopleLoading(true)
    try {
      const data = await peopleService.getPeople()
      setPeople(data.people)
      setPeopleError('')
    } catch (error) {
      setPeopleError(error.message)
    } finally {
      setPeopleLoading(false)
    }
  }

  useEffect(() => { refreshPeople() }, [])

  useEffect(() => {
    getEvaluation().then(setEvaluation).catch((error) => setEvaluationError(error.message)).finally(() => setEvaluationLoading(false))
  }, [])

  const navigate = (label) => {
    setActiveNav(label)
    document.getElementById(navAnchors[label])?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  const handleFile = (event) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setRecognitionResult(null)
      setRecognitionError('')
      document.getElementById('workspace')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  const recognizeSelectedImage = async () => {
    if (!selectedFile) return
    setLoading(true)
    setRecognitionError('')
    try {
      setRecognitionResult(await recognitionService.recognizeImage(selectedFile))
    } catch (error) {
      setRecognitionResult(null)
      setRecognitionError(error.message)
    } finally {
      setLoading(false)
    }
  }

  const registerPerson = async (name, file) => {
    setRegistrationLoading(true)
    setRegistrationError('')
    try {
      await peopleService.registerPerson(name, file)
      await refreshPeople()
      setRegistrationOpen(false)
    } catch (error) {
      setRegistrationError(error.message)
    } finally {
      setRegistrationLoading(false)
    }
  }

  const deletePerson = async (id) => {
    try {
      await peopleService.deletePerson(id)
      setPeople((current) => current.filter((person) => person.id !== id))
    } catch (error) {
      setPeopleError(error.message)
    }
  }

  return (
    <div className="app-shell" id="dashboard">
      <div className="ambient-grid" aria-hidden="true" />
      <Navbar active={activeNav} onNavigate={navigate} />
      <main>
        <section className="hero-section content-width" aria-labelledby="hero-title">
          <div className="hero-copy">
            <div className="eyebrow"><span className="eyebrow-pulse" /> COMPUTER VISION / LOCAL ENGINE</div>
            <h1 id="hero-title">Face recognition,<br /><em>made precise.</em></h1>
            <p className="hero-description">A transparent intelligence workspace for detecting faces, generating ArcFace embeddings, and making similarity-based identity decisions.</p>
            <div className="hero-actions">
              <button className="button button-primary" type="button" onClick={() => navigate('Recognition')}><Play size={16} fill="currentColor" /> Start recognition <ArrowUpRight size={16} /></button>
              <button className="button button-secondary" type="button" onClick={() => fileInput.current?.click()}><FileUp size={16} /> Upload image</button>
              <input ref={fileInput} className="visually-hidden" type="file" accept="image/*" onChange={handleFile} />
            </div>
            {selectedFile && <div className="file-ready"><Check size={14} /> {selectedFile.name} selected</div>}
          </div>
          <div className="hero-visual" aria-label="Recognition engine status illustration">
            <div className="visual-orbit orbit-one" /><div className="visual-orbit orbit-two" />
            <div className="scan-card">
              <div className="scan-card-top"><span>LIVE ENGINE</span><span className="live-label"><i /> STANDBY</span></div>
              <div className="face-frame"><div className="frame-corner top-left" /><div className="frame-corner top-right" /><div className="frame-corner bottom-left" /><div className="frame-corner bottom-right" /><ScanGlyph /></div>
              <div className="scan-card-bottom"><span>NO INPUT STREAM</span><span>LOCAL / 01</span></div>
            </div>
            <div className="hero-chip chip-top"><Sparkles size={14} /> 512-D EMBEDDING</div>
            <div className="hero-chip chip-bottom"><span className="signal-bars"><i /><i /><i /></span> 0.50 THRESHOLD</div>
          </div>
        </section>

        <section className="overview-row content-width" aria-label="System overview">
          <StatCard label="Registered people" value={peopleLoading ? '…' : people.length} detail={peopleError ? 'API unavailable' : 'Local identity roster'} icon={UsersRound} />
          <StatCard label="Recognition engine" value="ArcFace" detail="DeepFace · local model" icon={Sparkles} tone="accent" />
          <StatCard label="Matching threshold" value="0.50" detail="Configured baseline" icon={Gauge} />
          <StatCard label="Evaluation status" value="Ready" detail="Run evaluation to inspect" icon={BarChart3} />
        </section>

        <div className="content-width" id="people"><PeoplePanel people={people} loading={peopleLoading} error={peopleError} onRegister={() => { setRegistrationError(''); setRegistrationOpen(true) }} onDelete={deletePerson} /></div>

        <section className="workspace-section content-width" id="workspace">
          <div className="section-heading-row"><div><span className="section-kicker"><span className="kicker-line" /> INTELLIGENCE WORKSPACE</span><h2>See the decision forming.</h2></div><span className="section-index">01 / PIPELINE</span></div>
          <div className="workspace-grid">
            <div className="pipeline-panel panel-surface">
              <div className="panel-heading"><div><strong>Processing pipeline</strong><span>Each stage reflects the local recognition architecture.</span></div><span className="panel-live"><i /> IDLE</span></div>
              <Pipeline activeStage={activeStage} onStageChange={setActiveStage} />
              <div className="pipeline-detail"><span className="detail-number">0{activeStage + 1}</span><div><strong>{['Input is ready when you are.', 'YuNet will locate faces in the frame.', 'ArcFace will create a 512-dimensional representation.', 'Cosine similarity will rank registered candidates.', 'The configured threshold will decide known or unknown.'][activeStage]}</strong><span>Backend integration is intentionally not connected in this foundation.</span></div></div>
            </div>
            <RecognitionResult file={selectedFile} previewUrl={previewUrl} result={recognitionResult} loading={loading} error={recognitionError} onRecognize={recognizeSelectedImage} onUpload={() => fileInput.current?.click()} />
          </div>
        </section>

        <section className="lower-grid content-width" id="evaluation">
          <EvaluationPanel data={evaluation} loading={evaluationLoading} error={evaluationError} />
          <div className="pipeline-mini panel-surface"><div className="panel-heading"><div><span className="section-kicker"><span className="kicker-line" /> SYSTEM MODEL</span><h2>Engine profile</h2></div><Layers3 size={18} /></div><div className="model-list"><ModelRow label="Detector" value="YuNet" /><ModelRow label="Embedding" value="ArcFace" /><ModelRow label="Matcher" value="Cosine similarity" /><ModelRow label="Reference mode" value="Single / person" /></div><div className="model-note">Designed for explainable local experimentation.</div></div>
        </section>

        <section className="bottom-row content-width"><PrivacyCard /></section>
      </main>
      <footer className="footer content-width"><span>VISIONID / FACE RECOGNITION INTELLIGENCE</span><span>LOCAL ENGINE FOUNDATION <i /></span></footer>
      <RegisterPersonModal open={registrationOpen} loading={registrationLoading} error={registrationError} onClose={() => !registrationLoading && setRegistrationOpen(false)} onSubmit={registerPerson} />
    </div>
  )
}

function ModelRow({ label, value }) {
  return <div className="model-row"><span>{label}</span><strong>{value}</strong></div>
}

function ScanGlyph() {
  return <div className="scan-glyph"><span className="scan-eye eye-left" /><span className="scan-eye eye-right" /><span className="scan-nose" /><span className="scan-mouth" /></div>
}

export default App
