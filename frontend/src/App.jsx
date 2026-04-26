import React, {useState} from 'react'

export default function App(){
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [statusMsg, setStatusMsg] = useState('')
  const [ingesting, setIngesting] = useState(false)
  const [clearing, setClearing] = useState(false)

  const send = async () => {
    if(!input.trim()) return
    const userMsg = {from:'user', text: input}
    setMessages(m => [...m, userMsg])
    const payload = {message: input}
    setInput('')
    setLoading(true)
    try{
      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      })
      const data = await res.json()
      const botText = data.answer || (data.rag && data.rag.answer) || (data.web && data.web.answer) || 'No answer.'
      setMessages(m => [...m, {from:'bot', text: botText}])
    }catch(e){
      setMessages(m => [...m, {from:'bot', text: 'Error communicating with backend: '+String(e)}])
    }finally{ setLoading(false) }
  }

  const onKey = (e) => { if(e.key === 'Enter') send() }

  const ingest = async (force=false) => {
    setIngesting(true)
    setStatusMsg('Ingesting documents...')
    try{
      const url = `http://localhost:8000/ingest${force ? '?force=true' : ''}`
      const res = await fetch(url, {method:'POST'})
      const data = await res.json()
      setStatusMsg(JSON.stringify(data))
    }catch(e){
      setStatusMsg('Ingest failed: '+String(e))
    }finally{ setIngesting(false); setTimeout(()=>setStatusMsg(''), 5000) }
  }

  const clearDb = async () => {
    if(!window.confirm('Clear the document database? This cannot be undone.')) return
    setClearing(true)
    setStatusMsg('Clearing database...')
    try{
      const res = await fetch('http://localhost:8000/clear', {method:'POST'})
      const data = await res.json()
      setStatusMsg(JSON.stringify(data))
    }catch(e){
      setStatusMsg('Clear failed: '+String(e))
    }finally{ setClearing(false); setTimeout(()=>setStatusMsg(''), 5000) }
  }

  return (
    <div className="chat-root">
      <div className="topbar">
        <div className="title">Friendly Chat</div>
        <div className="subtitle">Ask questions or explore your documents</div>
        <div className="controls">
          <button onClick={()=>ingest(false)} disabled={ingesting || clearing}>{ingesting ? 'Ingesting...' : 'Ingest Docs'}</button>
          <button onClick={()=>ingest(true)} disabled={ingesting || clearing}>{ingesting ? 'Ingesting...' : 'Rebuild & Ingest (force)'}</button>
          <button onClick={clearDb} disabled={clearing || ingesting}>{clearing ? 'Clearing...' : 'Clear Database'}</button>
        </div>
      </div>

      <div className="chat-window">
        {messages.length===0 && (
          <div className="welcome">Hi! I'm here to help. Type a question below and press Enter or Send.</div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={"msg " + (m.from==='user' ? 'msg-user' : 'msg-bot')}>
            <div className="msg-from">{m.from==='user' ? 'You' : 'Assistant'}</div>
            <div className="msg-text">{m.text}</div>
          </div>
        ))}
      </div>

      <div className="status">{statusMsg}</div>

      <div className="chat-input">
        <input value={input} onChange={e=>setInput(e.target.value)} onKeyDown={onKey} placeholder="Ask anything..." />
        <button onClick={send} disabled={loading}>{loading ? '...' : 'Send'}</button>
      </div>
    </div>
  )
}
