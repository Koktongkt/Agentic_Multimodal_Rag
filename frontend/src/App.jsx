import React, { useState, useRef } from 'react'

export default function App(){
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [statusMsg, setStatusMsg] = useState('')
  const [ingesting, setIngesting] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const fileRef = useRef(null)
  const [history, setHistory] = useState([])

  const toBase64 = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onload = () => resolve(reader.result.split(',')[1])
    reader.onerror = reject
  })

  const send = async () => {
    if (!input.trim() && !image) return

    const userContent = input || '[Image]'

    const newHistory = [
      ...history,
      { role: 'user', content: userContent }
    ]

    setMessages(m => [...m, { from: 'user', text: userContent }])
    setLoading(true)

    try {
      let base64 = null

      if (image) {
        base64 = await toBase64(image)
        setImage(null)
        setPreview(null)
      }

      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          image_b64: base64,
          history: newHistory
        })
      })

      const data = await res.json()

      const botText =
        data.response?.answer ||
        data.answer ||
        data.rag?.answer ||
        'No answer.'

      const updatedHistory = [
        ...newHistory,
        { role: 'assistant', content: botText }
      ]

      setHistory(updatedHistory)
      setMessages(m => [...m, { from: 'bot', text: botText }])

    } finally {
      setLoading(false)
      setInput('')
    }
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

        {/* 📷 Vision Upload Section */}
        <div className="image-upload">
          <button onClick={() => fileRef.current.click()}>
            📷 Upload Image
          </button>

          {preview && (
            <img src={preview} className="image-preview" />
          )}
        </div>

        {/* hidden file input */}
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            const file = e.target.files[0]
            if (!file) return

            setImage(file)
            setPreview(URL.createObjectURL(file))
            e.target.value = null
          }}
        />

        {/* 💬 text input */}
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKey}
          placeholder="Ask anything..."
        />

        {/* 🚀 send button */}
        <button onClick={send} disabled={loading}>
          {loading ? '...' : 'Send'}
        </button>
      </div>
    </div>
  )
}
