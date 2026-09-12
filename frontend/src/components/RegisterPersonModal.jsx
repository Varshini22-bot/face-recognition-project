import { Check, ImagePlus, LoaderCircle, X } from 'lucide-react'
import React, { useEffect, useRef, useState } from 'react'

export default function RegisterPersonModal({ open, loading, error, onClose, onSubmit }) {
  const [name, setName] = useState('')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const inputRef = useRef(null)

  useEffect(() => {
    if (!open) {
      setName('')
      setFile(null)
      setPreview('')
    }
  }, [open])

  if (!open) return null

  const selectFile = (event) => {
    const nextFile = event.target.files?.[0]
    if (!nextFile) return
    setFile(nextFile)
    setPreview(URL.createObjectURL(nextFile))
  }

  const submit = (event) => {
    event.preventDefault()
    if (name.trim() && file) onSubmit(name.trim(), file)
  }

  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <form className="register-modal" onSubmit={submit} aria-labelledby="register-title">
      <div className="modal-header"><div><span className="section-kicker"><span className="kicker-line" /> PEOPLE / ENROLLMENT</span><h2 id="register-title">Create a face profile</h2></div><button className="icon-button" type="button" onClick={onClose} aria-label="Close registration dialog"><X size={18} /></button></div>
      <p className="modal-intro">Add one reference image to the local recognition database. Registration requires exactly one visible face.</p>
      <label className="form-label" htmlFor="person-name">Name<input id="person-name" value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. Varshini" autoFocus /></label>
      <input ref={inputRef} className="visually-hidden" type="file" accept="image/*" onChange={selectFile} />
      <button className="image-picker" type="button" onClick={() => inputRef.current?.click()}><span className="picker-icon">{preview ? <img src={preview} alt="Selected reference preview" /> : <ImagePlus size={22} />}</span><span><strong>{file ? file.name : 'Choose reference image'}</strong><small>{file ? 'Ready to register' : 'JPG, PNG or WEBP · one face only'}</small></span></button>
      {error && <div className="api-error" role="alert">{error}</div>}
      <div className="modal-guidance"><Check size={14} /> The original image is stored locally by the registration workflow.</div>
      <div className="modal-actions"><button className="button button-secondary" type="button" onClick={onClose}>Cancel</button><button className="button button-primary" type="submit" disabled={!name.trim() || !file || loading}>{loading ? <LoaderCircle className="spin" size={15} /> : null}{loading ? 'Creating face profile...' : 'Register person'}</button></div>
    </form>
  </div>
}
