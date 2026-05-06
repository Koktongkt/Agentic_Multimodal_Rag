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
  const combinedFileRef = useRef(null)
  const docFileRef = useRef(null)
  const [docFiles, setDocFiles] = useState(null)
  const [chatDoc, setChatDoc] = useState(null)
  const [history, setHistory] = useState([])
  const [showIngestModal, setShowIngestModal] = useState(false)

  const toBase64 = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onload = () => resolve(reader.result.split(',')[1])
    reader.onerror = reject
  })

  const send = async () => {
    if (!input.trim() && !image && !chatDoc) return

    const userContent = input || (image ? '[Image]' : (chatDoc ? `[Document: ${chatDoc.name}]` : ''))

    const newHistory = [
      ...history,
      { role: 'user', content: userContent }
    ]

    // Append user message then a loading placeholder for assistant
    setMessages(prev => [...prev, { from: 'user', text: userContent }])
    setMessages(prev => [...prev, { from: 'bot', text: '', loading: true }])
    setLoading(true)

    try {
      let base64 = null
      let docBase64 = null

      if (image) {
        base64 = await toBase64(image)
        setImage(null)
        setPreview(null)
      }

      if (chatDoc) {
        docBase64 = await toBase64(chatDoc)
        setChatDoc(null)
      }

      const payload = {
        message: input,
        image_b64: base64,
        file_b64: docBase64,
        filename: chatDoc ? chatDoc.name : undefined,
        history: newHistory
      }

      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      const data = await res.json()

      const botText =
        data.response?.answer ||
        data.answer ||
        data.rag?.answer ||
        'No answer.'

      // Replace the loading placeholder with real text (if present)
      setMessages(prev => {
        const idx = prev.findIndex(m => m.loading)
        if (idx >= 0) {
          const copy = [...prev]
          copy[idx] = { ...copy[idx], text: botText, loading: false }
          return copy
        }
        return [...prev, { from: 'bot', text: botText }]
      })

      const updatedHistory = [
        ...newHistory,
        { role: 'assistant', content: botText }
      ]

      setHistory(updatedHistory)

    } catch (e) {
      // replace placeholder with error message
      setMessages(prev => {
        const idx = prev.findIndex(m => m.loading)
        if (idx >= 0) {
          const copy = [...prev]
          copy[idx] = { ...copy[idx], text: 'Error: failed to get answer', loading: false }
          return copy
        }
        return [...prev, { from: 'bot', text: 'Error: failed to get answer' }]
      })
    } finally {
      setLoading(false)
      setInput('')
    }
  }

  const onKey = (e) => { if(e.key === 'Enter') send() }

  const handleStoreIngest = () => {
    if (docFiles && docFiles.length > 0) {
      ingest(false)
      return
    }
    setShowIngestModal(true)
  }

  const canSend = (input && input.trim().length > 0) || !!image || !!chatDoc

  const ingest = async (force=false, filesParam=null) => {
    setIngesting(true)
    setStatusMsg('Ingesting documents...')
    try{
      const url = `http://localhost:8000/ingest${force ? '?force=true' : ''}`
      let res
      const filesToUse = filesParam || docFiles
      if (filesToUse && filesToUse.length > 0) {
        const form = new FormData()
        for (let i=0;i<filesToUse.length;i++) {
          form.append('files', filesToUse[i])
        }
        res = await fetch(url, { method: 'POST', body: form })
        setDocFiles(null)
      } else {
        res = await fetch(url, {method:'POST'})
      }
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
        <div className="title-wrapper">
          <img src="/robot_icon.jpg" className="title-icon" />
          <div className="title">Agent Gemma</div>
        </div>
        <div className="subtitle">Ask questions, upload images or explore your documents!</div>
        <div className="controls">
          <button
            className="btn-primary"
            onClick={handleStoreIngest}
            disabled={ingesting || clearing}
          >
            {ingesting ? 'Ingesting...' : 'Store & Ingest Docs'}
          </button>

          <button
            className="btn-soft"
            onClick={() => ingest(true)}
            disabled={ingesting || clearing}
          >
            {ingesting ? 'Ingesting...' : 'Rebuild Database'}
          </button>

          <button
            className="btn-danger"
            onClick={clearDb}
            disabled={clearing || ingesting}
          >
            {clearing ? 'Clearing...' : 'Clear Database'}
          </button>

          {docFiles?.length > 0 && (
            <div className="file-badge">
              {docFiles.length} file(s)
            </div>
          )}
        </div>
      </div>

      <div className="chat-window">
        {messages.length===0 && (
          <div className="welcome">Hi! I'm here to help. Type a question below and press Enter or Send.</div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={"msg " + (m.from==='user' ? 'msg-user' : 'msg-bot')}>
            <div className="msg-from">{m.from==='user' ? 'You' : 'Assistant'}</div>
            <div className="msg-text">
              {m.loading ? (
                <span className="typing-dots">
                  <span></span><span></span><span></span>
                </span>
              ) : m.text}
            </div>
          </div>
        ))}
      </div>

      <div className="status">{statusMsg}</div>

      <div className="chat-input">

        {/* Upload file (image or document) */}
        <div className="image-upload">
          <button onClick={() => combinedFileRef.current.click()}>
            📁 
          </button>

          {preview && (
            <img src={preview} className="image-preview" />
          )}

          {chatDoc && (
            <div className="file-badge">{chatDoc.name}</div>
          )}
        </div>

        {/* hidden combined file input */}
        <input
          ref={combinedFileRef}
          type="file"
          accept="image/*,.pdf,.md,.txt,.docx,.xlsx"
          hidden
          onChange={(e) => {
            const file = e.target.files[0]
            if (!file) return

            if (file.type && file.type.startsWith('image/')) {
              setImage(file)
              setPreview(URL.createObjectURL(file))
            } else {
              setChatDoc(file)
            }
            e.target.value = null
          }}
        />

        {/* hidden docs file input (for ingesting multiple docs) */}
        <input
          ref={docFileRef}
          type="file"
          accept=".pdf,.md,.txt,.docx"
          hidden
          multiple
          onChange={(e) => {
            const files = Array.from(e.target.files || [])
            if (files.length === 0) return
            setDocFiles(files)
            ingest(false, files)
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
        <button className={`send-btn ${canSend ? 'active' : ''}`} onClick={send} disabled={!canSend || loading}>
          {loading ? '...' : (canSend ? '➤' : 'Send')}
        </button>
      </div>
      {showIngestModal && (
        <div className="modal-overlay">
          <div className="modal">
            <h3>Upload Documents?</h3>
            <p>
              Press Upload to select files first, or Skip to ingest current documents.
            </p>

            <div className="modal-actions">
              <button
                onClick={() => {
                  setShowIngestModal(false)
                  docFileRef.current.click()
                }}
              >
                Upload Files
              </button>

              <button
                onClick={() => {
                  setShowIngestModal(false)
                  ingest(false)
                }}
              >
                Skip
              </button>

              <button
                onClick={() => setShowIngestModal(false)}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}