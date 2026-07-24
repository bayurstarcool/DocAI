<script>
  import { refreshIcons } from "../lib/icons.js"
  import { onDestroy, onMount } from 'svelte'
  import { apiJson } from '../stores/auth.js'
  import { showToast } from '../stores/toast.js'

  let stats = { device: '...', restorer: '-', shadow: '-', enhancer: '-' }
  let selectedFile = null
  let selectedMode = 'restore'
  let resultBlob = null
  let resultUrl = ''
  let originalUrl = ''
  let processing = false
  let ocrResult = null
  let detectedText = ''
  let pipelineResult = null
  let trainedRuns = []
  let liveCheckpoints = []
  let selectedCheckpoint = 'checkpoints/document_restorer/best.pth'
  let checkpointTimer = null

  const checkpointModes = new Set(['restore', 'full_pipeline'])

  async function refreshCheckpoints() {
    try {
      const data = await apiJson('/api/training/runs')
      trainedRuns = (data.runs || []).filter(run => (run.checkpoints || []).length)
      liveCheckpoints = data.live_checkpoints || []
      const available = new Set([
        ...liveCheckpoints.map(item => item.path),
        ...trainedRuns.flatMap(run => (run.checkpoints || []).map(item => item.path))
      ])
      if (!available.has(selectedCheckpoint) && liveCheckpoints.length) selectedCheckpoint = liveCheckpoints[0].path
    } catch (e) {
      showToast(`Gagal refresh checkpoint: ${e.message}`, 'error')
    }
  }

  async function syncSelectedCheckpoint(token) {
    const response = await fetch('/api/model/reload', {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ checkpoint: selectedCheckpoint })
    })
    if (!response.ok) {
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || `Reload checkpoint gagal (${response.status})`)
    }
    return response.json()
  }

  const modes = [
    { id: 'restore', icon: 'wand-2', label: 'AI Restore' },
    { id: 'shadow_remove', icon: 'sun-dim', label: 'Shadow Remove' },
    { id: 'shadow_effective_bg', icon: 'sun', label: 'Shadow BG Est.' },
    { id: 'shadow_iterative', icon: 'refresh-cw', label: 'Shadow Iterative' },
    { id: 'shadow_so', icon: 'layers', label: 'Shadow SO' },
    { id: 'shadow_so_aggressive', icon: 'zap', label: 'Shadow SO Aggr.' },
    { id: 'full_pipeline', icon: 'workflow', label: 'Full Pipeline' },
    { id: 'enhance', icon: 'sparkles', label: 'AI Enhance' },
    { id: 'magic_enhance', icon: 'magic-wand', label: 'Magic Enhance' },
    { id: 'binarize', icon: 'contrast', label: 'Binarize' },
    { id: 'cleanup', icon: 'brush-cleaning', label: 'Full Cleanup' },
    { id: 'clahe', icon: 'sun', label: 'CLAHE' },
    { id: 'denoise', icon: 'shield-check', label: 'Denoise' },
    { id: 'sharpen', icon: 'focus', label: 'Sharpen' },
    { id: 'deskew', icon: 'rotate-cw', label: 'Deskew' },
  ]

  onMount(async () => {
    refreshIcons()
    try {
      const d = await apiJson('/api/health')
      stats.device = d.device || '?'
      stats.restorer = d.models?.document_restorer ? 'Loaded' : 'Not loaded'
      stats.shadow = d.models?.shadow_remover ? 'Loaded' : 'Not loaded'
      stats.enhancer = d.models?.doc_enhancer ? 'Loaded' : 'Not loaded'
    } catch(e) {}
    await refreshCheckpoints()
    checkpointTimer = setInterval(refreshCheckpoints, 5000)
  })

  onDestroy(() => {
    if (checkpointTimer) clearInterval(checkpointTimer)
    if (resultUrl?.startsWith('blob:')) URL.revokeObjectURL(resultUrl)
  })

  function handleDrop(e) {
    e.preventDefault()
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0])
  }

  function handleFile(file) {
    selectedFile = file
    resultUrl = ''
    resultBlob = null
    pipelineResult = null
    const r = new FileReader()
    r.onload = () => { originalUrl = r.result }
    r.readAsDataURL(file)
  }

  async function processImage() {
    if (!selectedFile) return
    processing = true
    ocrResult = null
    detectedText = ''

    if (selectedMode === 'detect_text') {
      const fd = new FormData()
      fd.append('file', selectedFile)
      const start = performance.now()
      try {
        const token = localStorage.getItem('docai_token')
        const res = await fetch('/api/ocr/detect', { method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: fd })
        if (!res.ok) {
          const err = await res.json().catch(() => ({}))
          throw new Error(err.detail || `OCR error ${res.status}`)
        }
        const data = await res.json()
        detectedText = data.text || '(Tidak ada teks terdeteksi)'
        ocrResult = data
        const r2 = new FileReader()
        r2.onload = () => { resultUrl = r2.result }
        r2.readAsDataURL(selectedFile)
        const ms = Math.round(performance.now() - start)
        showToast(`Teks terdeteksi: ${data.word_count || 0} kata dalam ${ms}ms`)
      } catch(e) { showToast(e.message, 'error') }
      finally { processing = false }
      return
    }

    if (selectedMode === 'full_pipeline') {
      const fd = new FormData()
      fd.append('file', selectedFile)
      const start = performance.now()
      try {
        const token = localStorage.getItem('docai_token')
        await syncSelectedCheckpoint(token)
        const res = await fetch('/api/pipeline/process-json', { method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: fd })
        if (!res.ok) {
          const err = await res.json().catch(() => ({}))
          throw new Error(err.detail || `Pipeline error ${res.status}`)
        }
        const data = await res.json()
        pipelineResult = data
        resultUrl = data.image
        resultBlob = null
        const done = data.steps?.filter(s => s.status === 'done').length || 0
        showToast(`Pipeline selesai: ${done} langkah, ${data.total_ms}ms`)
      } catch(e) { showToast(e.message, 'error') }
      finally { processing = false }
      return
    }

    // Normal image processing
    const fd = new FormData()
    fd.append('file', selectedFile)
    fd.append('mode', selectedMode)
    const start = performance.now()
    try {
      const token = localStorage.getItem('docai_token')
      if (checkpointModes.has(selectedMode)) await syncSelectedCheckpoint(token)
      const res = await fetch('/api/scan', { method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: fd })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `Server error ${res.status}`)
      }
      resultBlob = await res.blob()
      resultUrl = URL.createObjectURL(resultBlob)
      const ms = Math.round(performance.now() - start)
      showToast(`${selectedMode} completed in ${ms}ms`)
    } catch(e) { showToast(e.message, 'error') }
    finally { processing = false }
  }

  function downloadResult() {
    if (!resultBlob) return
    const a = document.createElement('a')
    a.href = URL.createObjectURL(resultBlob)
    a.download = `docai_${selectedMode}_${Date.now()}.png`
    a.click()
  }
</script>

<div class="page-header">
  <h1>Document Processing Dashboard</h1>
  <p>Upload gambar dokumen untuk diproses dengan AI.</p>
</div>

<div class="stats">
  <div class="stat-card"><div class="label">Device</div><div class="value accent">{stats.device}</div></div>
  <div class="stat-card"><div class="label">Restorer</div><div class="value success">{stats.restorer}</div></div>
  <div class="stat-card"><div class="label">Shadow Remover</div><div class="value success">{stats.shadow}</div></div>
  <div class="stat-card"><div class="label">Enhancer</div><div class="value" style="color:var(--text3)">{stats.enhancer}</div></div>
</div>

<div class="card">
  <h2><i data-lucide="upload-cloud"></i> Upload & Process</h2>
  <div class="upload-zone" ondrop={handleDrop} ondragover={(e) => e.preventDefault()}
    onclick={() => document.getElementById('fileInput').click()}>
    <i data-lucide="image-plus"></i>
    <p>Klik atau drag & drop gambar di sini</p>
    <div class="hint">PNG, JPG, WEBP, BMP — Maks 20MB</div>
    <input type="file" id="fileInput" accept="image/*" style="display:none"
      onchange={e => e.target.files[0] && handleFile(e.target.files[0])} />
  </div>

  <div style="margin-top:1.25rem">
    <div class="label">MODE PROCESSING</div>
    <div class="mode-grid">
      {#each modes as m}
        <button class="mode-btn" class:active={selectedMode === m.id} onclick={() => selectedMode = m.id}>
          <i data-lucide={m.icon}></i> {m.label}
        </button>
      {/each}
    </div>
  </div>

  {#if checkpointModes.has(selectedMode)}
    <div class="model-select" style="margin-top:1rem">
      <label for="homeCheckpoint">Trained checkpoint</label>
      <select id="homeCheckpoint" bind:value={selectedCheckpoint}>
        {#each liveCheckpoints as checkpoint}
          <option value={checkpoint.path}>{checkpoint.name}</option>
        {/each}
        {#each trainedRuns as run}
          <optgroup label={run.run_id}>
            {#each run.checkpoints || [] as checkpoint}
              <option value={checkpoint.path}>{checkpoint.immutable ? 'Immutable · ' : ''}{checkpoint.name}</option>
            {/each}
          </optgroup>
        {/each}
      </select>
      <p class="info-text">Auto-refresh 5 detik. Progress checkpoint belum divalidasi.</p>
    </div>
  {/if}

  <div class="process-bar">
    <button class="btn btn-primary" onclick={processImage} disabled={!selectedFile || processing}>
      {#if processing}<div class="spinner" style="width:16px;height:16px;border-width:2px"></div>{:else}<i data-lucide="play"></i>{/if}
      Process Image
    </button>
    {#if resultBlob}
      <button class="btn btn-secondary" onclick={downloadResult}><i data-lucide="download"></i> Download</button>
    {/if}
    {#if selectedFile}
      <span class="process-info">{selectedFile.name} ({Math.round(selectedFile.size / 1024)}KB)</span>
    {/if}
  </div>
</div>

{#if originalUrl}
  <div class="results-grid">
    <div class="result-box">
      <div class="header"><i data-lucide="image"></i> Original</div>
      <img src={originalUrl} alt="Original" />
    </div>
    {#if resultUrl}
      <div class="result-box">
        <div class="header"><i data-lucide="sparkles"></i> Result</div>
        <img src={resultUrl} alt="Result" />
      </div>
    {/if}
  </div>
{/if}

{#if pipelineResult?.steps?.length}
  <div class="card pipeline-card">
    <h2><i data-lucide="workflow"></i> Pipeline</h2>
    <div class="pipeline-timeline">
      {#each pipelineResult.steps as step, index}
        <div class="pipeline-step {step.status}">
          <div class="step-indicator">{index + 1}</div>
          <div class="step-info">
            <div class="step-label">{step.label}</div>
            <div class="step-detail">{step.info}</div>
          </div>
          <div class="step-time">{step.time_ms}ms</div>
        </div>
      {/each}
    </div>
    <div class="pipeline-summary">Total: {pipelineResult.total_ms}ms</div>
  </div>
{/if}

<div class="footer">DocAI v2.0 — Document Restoration AI Platform</div>

<style>
  .page-header { margin-bottom: 2rem; }
  .page-header h1 { font-size: 1.5rem; font-weight: 700; }
  .page-header p { color: var(--text2); margin-top: 0.25rem; font-size: 0.9rem; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
  .stat-card { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 1.25rem; }
  .stat-card .label { font-size: 0.75rem; font-weight: 700; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
  .stat-card .value { font-size: 1.5rem; font-weight: 700; }
  .stat-card .value.accent { color: var(--accent2); }
  .stat-card .value.success { color: var(--success); }
  .card { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.5rem; margin-bottom: 1.5rem; }
  .card :global(h2) { font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
  .upload-zone { border: 2px dashed var(--border); border-radius: var(--radius-sm); padding: 2.5rem; text-align: center; cursor: pointer; transition: all 0.2s; }
  .upload-zone:hover { border-color: var(--accent); background: rgba(6,182,212,0.05); }
  .upload-zone :global(i) { width: 48px; height: 48px; color: var(--text3); margin-bottom: 0.75rem; }
  .upload-zone p { font-weight: 500; color: var(--text2); }
  .hint { font-size: 0.75rem; color: var(--text3); margin-top: 0.5rem; }
  .label { font-size: 0.8rem; font-weight: 600; color: var(--text2); margin-bottom: 0.5rem; }
  .mode-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 0.5rem; }
  .mode-btn { display: flex; align-items: center; gap: 0.5rem; padding: 0.625rem 0.75rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); color: var(--text2); font-size: 0.8rem; font-weight: 500; transition: all 0.15s; }
  .mode-btn:hover { border-color: var(--accent); color: var(--text); }
  .mode-btn.active { border-color: var(--accent); background: rgba(6,182,212,0.1); color: var(--accent2); }
  .process-bar { display: flex; align-items: center; gap: 0.75rem; margin-top: 1.25rem; padding-top: 1.25rem; border-top: 1px solid var(--border); }
  .btn { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.625rem 1.25rem; border-radius: var(--radius-xs); font-size: 0.875rem; font-weight: 600; border: none; transition: all 0.15s; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-primary { background: var(--primary); color: white; }
  .btn-secondary { background: var(--surface); color: var(--text); border: 1px solid var(--border); }
  .process-info { color: var(--text2); font-size: 0.8rem; margin-left: auto; }
  .results-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1.5rem; }
  .result-box { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden; }
  .result-box .header { padding: 0.75rem 1rem; font-size: 0.8rem; font-weight: 600; color: var(--text2); border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 0.5rem; }
  .result-box img { width: 100%; display: block; }
  .footer { text-align: center; padding: 2rem 0; color: var(--text3); font-size: 0.8rem; }
  @media (max-width: 768px) { .results-grid { grid-template-columns: 1fr; } }
  /* Pipeline */
  .pipeline-card { margin-top: 1.5rem; }
  .pipeline-card :global(h2) { font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
  .pipeline-timeline { display: flex; flex-direction: column; gap: 0.5rem; }
  .pipeline-step { display: flex; align-items: center; gap: 0.75rem; padding: 0.625rem 0.875rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); }
  .pipeline-step.done { border-left: 3px solid var(--success); }
  .pipeline-step.skipped { border-left: 3px solid var(--text3); opacity: 0.6; }
  .step-indicator { width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center; font-size: 0.7rem; font-weight: 700; flex-shrink: 0; }
  .pipeline-step.done .step-indicator { background: rgba(16,185,129,0.15); color: var(--success); }
  .pipeline-step.skipped .step-indicator { background: rgba(113,113,122,0.15); color: var(--text3); }
  .step-info { flex: 1; }
  .step-label { font-size: 0.85rem; font-weight: 600; }
  .step-detail { font-size: 0.7rem; color: var(--text2); }
  .step-time { font-size: 0.7rem; color: var(--text3); flex-shrink: 0; }
  .pipeline-summary { margin-top: 0.75rem; text-align: right; font-size: 0.85rem; color: var(--text2); }

  /* OCR */
  .ocr-card { margin-top: 1.5rem; }
  .ocr-card :global(h2) { font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
  .ocr-meta { display: flex; gap: 1rem; font-size: 0.8rem; color: var(--text2); margin-bottom: 0.75rem; }
  .ocr-meta :global(i) { width: 14px; height: 14px; }
  .ocr-text { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); padding: 1rem; font-size: 0.85rem; line-height: 1.6; white-space: pre-wrap; color: var(--text); max-height: 300px; overflow-y: auto; }
  .ocr-details { margin-top: 0.75rem; }
  .ocr-details summary { font-size: 0.8rem; color: var(--text2); cursor: pointer; }
  .ocr-words { display: flex; flex-wrap: wrap; gap: 0.375rem; margin-top: 0.5rem; }
  .ocr-word { display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.2rem 0.5rem; background: var(--bg3); border: 1px solid var(--border); border-radius: 4px; font-size: 0.75rem; }
  .ocr-word.low-conf { opacity: 0.6; }
  .ocr-word small { color: var(--text3); font-size: 0.6rem; }

  .model-select { display:grid; gap:.45rem; min-width:0; }
  .model-select label { font-size:.72rem; font-weight:590; color:var(--text2); text-transform:uppercase; letter-spacing:.055em; }
  .model-select select { width:100%; padding:.72rem .8rem; border:1px solid var(--border); border-radius:var(--radius-xs); color:var(--text); }
  @media(max-width:768px){
    .mode-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.results-grid{gap:.8rem}.result-box img{max-height:72vh;object-fit:contain}.pipeline-step{align-items:flex-start}.step-time{display:none}
  }
</style>
