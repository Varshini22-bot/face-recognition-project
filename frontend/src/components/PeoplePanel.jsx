import React from 'react'
import { LoaderCircle, Trash2, UserPlus, UsersRound } from 'lucide-react'

export default function PeoplePanel({ people, loading, error, onRegister, onDelete }) {
  return <section className="people-panel panel-surface" aria-labelledby="people-title">
    <div className="panel-heading"><div><span className="section-kicker"><span className="kicker-line" /> REGISTERED PEOPLE</span><h2 id="people-title">Local identity roster</h2></div><button className="button button-primary compact-button" type="button" onClick={onRegister}><UserPlus size={15} /> Register person</button></div>
    {loading && <div className="people-state"><LoaderCircle className="spin" size={20} /><span>Loading registered people...</span></div>}
    {!loading && error && <div className="people-state people-error">{error}</div>}
    {!loading && !error && people.length === 0 && <div className="people-state"><UsersRound size={22} /><strong>No people registered yet</strong><span>Build the local roster one reference image at a time.</span><button className="button button-secondary" type="button" onClick={onRegister}>Register person</button></div>}
    {!loading && !error && people.length > 0 && <div className="people-list">{people.map((person) => <article className="person-row" key={person.id}><span className="person-avatar">{person.name.slice(0, 1).toUpperCase()}</span><div><strong>{person.name}</strong><small>Registered {new Date(person.created_at).toLocaleDateString()}</small></div><span className="registered-badge">ACTIVE</span><button className="icon-button delete-button" type="button" onClick={() => onDelete(person.id)} aria-label={`Delete ${person.name}`}><Trash2 size={15} /></button></article>)}</div>}
  </section>
}
