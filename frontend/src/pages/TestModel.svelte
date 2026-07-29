<script>
  import { refreshIcons } from "../lib/icons.js"
  import { onDestroy, onMount } from 'svelte'
  import { apiJson } from '../stores/auth.js'
  import { showToast } from '../stores/toast.js'

  let modelInfo = {}
  let rawResultBlob = null
  let resultCanvas
  let selectedFile = null
  let originalUrl = ''
  let selectedMode = 'restore'
  let pipelineMode = 'full'
  let shadowStrength = 1.0
  let adjBrightness = 1.0
  let adjContrast = 1.0
  let adjSaturation = 1.0
  let adjSharpness = 1.0
  let adjGamma = 1.0
  let adjWhiteBalance = false
  let adjClahe = 0.0
  function resetAdjust() { adjBrightness=1.0; adjContrast=1.0; adjSaturation=1.0; adjSharpness=1.0; adjGamma=1.0; adjWhiteBalance=false; adjClahe=0.0 }
  let resultUrl = ''
  let processing = false
  let docshadowWeights = []
  let trainedRuns = []
  let liveCheckpoints = []
  let checkpointTimer = null
  let selectedCheckpoint = 'checkpoints/document_restorer/best.pth'
  let loadedCheckpoint = ''

  const modes = ["restore","docres_base","docres_finetune","docres_onnx","magic_enhance","binarize","cleanup","clahe","denoise","sharpen","deskew"]
  const pipelineModes = [
    { value: 'full', label: 'Full Pipeline', desc: 'AI + white balance + whitening + CLAHE' },
    { value: 'ai_only', label: 'AI Only', desc: 'Model output langsung, warna terjaga' },
    { value: 'ai', label: 'AI + Shadow Fix', desc: 'AI + shadow correction, tanpa color processing' },
    { value: 'color', label: 'AI + CLAHE', desc: 'AI + contrast enhancement, tanpa white balance' },
  ]
  const pretrainedModes = ['SD7K', 'Jung', 'Kligler'].map(name => ({ name }))
  const modeLabels = { restore: "AI Restore", docres_base: "DocRes Base", docres_finetune: "DocRes Finetune", docres_onnx: "DocRes ONNX", magic_enhance: "Magic Enhance", binarize: "Binarize", cleanup: "Full Cleanup", clahe: "CLAHE", denoise: "Denoise", sharpen: "Sharpen", deskew: "Deskew" }
  $: availableDocshadowNames = new Set(docshadowWeights.map(w => w.name))
  $: docshadowModes = pretrainedModes.map(w => `docshadow:${w.name}`)
  $: allModeLabels = {
    ...modeLabels,
    ...Object.fromEntries(pretrainedModes.map(w => [`docshadow:${w.name}`, `Pretrained DocShadow ${w.name}`]))
  }
  $: selectedModelFamily = selectedMode.startsWith('docshadow:') ? 'docshadow' : 'docai'


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

  onMount(async () => {
    refreshIcons()
    try { modelInfo = await apiJson('/api/models/info') } catch(e) {}
    try {
      const d = await apiJson('/api/docshadow/weights')
      docshadowWeights = d.weights || []
    } catch(e) { docshadowWeights = [] }
    await refreshCheckpoints()
    checkpointTimer = setInterval(refreshCheckpoints, 5000)
  })

  onDestroy(() => {
    if (checkpointTimer) clearInterval(checkpointTimer)
    if (resultUrl) URL.revokeObjectURL(resultUrl)
  })

  function handleFile(file) {
    if (resultUrl) URL.revokeObjectURL(resultUrl)
    selectedFile = file
    resultUrl = ''
    originalUrl = ''

    const reader = new FileReader()
    reader.onload = () => {
      if (selectedFile === file) originalUrl = reader.result
    }
    reader.onerror = () => showToast('Gagal membaca preview original image', 'error')
    reader.readAsDataURL(file)
  }

  function handleDrop(event) {
    event.preventDefault()
    const file = event.dataTransfer?.files?.[0]
    if (file?.type?.startsWith('image/')) handleFile(file)
  }

  $: previewFilter = `brightness(${adjBrightness}) contrast(${adjContrast}) saturate(${adjSaturation})`

  function applyClientAdjustments() {
    // CSS filter updates preview reactively. Apply button bakes it into PNG.
  }

  function applyAdjustmentsToImage() {
    if (!rawResultBlob) return
    const sourceUrl = URL.createObjectURL(rawResultBlob)
    const img = new Image()
    img.onload = () => {
      const canvas = resultCanvas
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      const ctx = canvas.getContext('2d')
      ctx.filter = previewFilter
      ctx.drawImage(img, 0, 0)
      ctx.filter = 'none'
      canvas.toBlob(blob => {
        URL.revokeObjectURL(sourceUrl)
        if (!blob) return
        const oldUrl = resultUrl
        resultUrl = URL.createObjectURL(blob)
        rawResultBlob = blob
        setTimeout(() => { if (oldUrl) URL.revokeObjectURL(oldUrl) }, 250)
        showToast('Adjustment diterapkan')
      }, 'image/png')
    }
    img.onerror = () => URL.revokeObjectURL(sourceUrl)
    img.src = sourceUrl
  }

  function downloadResult() {
    if (!resultUrl) return
    const a = document.createElement('a')
    a.href = resultUrl
    a.download = 'result_' + (selectedFile?.name || 'output.png')
    a.click()
  }

  async function runTest() {
    if (!selectedFile) return
    processing = true
    const fd = new FormData()
    fd.append('file', selectedFile)
    fd.append('mode', selectedMode)
    // Adjustments applied client-side via OpenCV.js
      if (selectedMode === 'docres_finetune') { const vs = document.getElementById('finetune_variant'); fd.append('variant', vs ? vs.value : 'finetune_v3'); }
    if (selectedMode === 'restore') {
      fd.append('pipeline_mode', pipelineMode)
      fd.append('shadow_strength', shadowStrength)
    }
    const start = performance.now()
    try {
      const token = localStorage.getItem('docai_token')
      if (!selectedMode.startsWith('docshadow:') && selectedCheckpoint !== loadedCheckpoint) {
        const reload = await fetch('/api/model/reload', {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({ checkpoint: selectedCheckpoint })
        })
        if (!reload.ok) {
          const error = await reload.json().catch(() => ({}))
          throw new Error(error.detail || 'Gagal load checkpoint')
        }
        loadedCheckpoint = selectedCheckpoint
      }
      let endpoint = '/api/scan'
      if (selectedMode.startsWith('docshadow:')) {
        fd.set('mode', 'docshadow')
        fd.append('weight', selectedMode.split(':')[1])
        endpoint = '/api/docshadow/infer'
      }
      let res
      try {
        res = await fetch(endpoint, { method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: fd })
      } catch(fetchErr) {
        throw new Error('Gagal menghubungi server: ' + (fetchErr.message || 'network error'))
      }
      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        let errDetail = errText
        try { errDetail = JSON.parse(errText).detail || errText } catch(_) {}
        throw new Error(`Server error ${res.status}: ${errDetail || res.statusText}`)
      }
      rawResultBlob = await res.blob()
      if (resultUrl) URL.revokeObjectURL(resultUrl)
      resultUrl = URL.createObjectURL(rawResultBlob)
      const ms = Math.round(performance.now() - start)
      showToast(`${allModeLabels[selectedMode] || selectedMode} completed (${ms}ms)`)
    } catch(e) {
      let msg = e.message || 'Unknown error'
      if (msg === 'Failed to fetch') msg = 'Gagal menghubungi server. Periksa koneksi atau coba lagi.'
      showToast(msg, 'error')
    }
    finally { processing = false }
  }
</script>

<div class="page-header">
  <h1>Test Individual Model</h1>
  <p>Upload gambar dan test setiap mode processing.</p>
</div>

<div class="card">
  <h2><i data-lucide="cpu"></i> Model Info</h2>
  <div class="info-grid">
    {#each Object.entries(modelInfo) as [name, info]}
      <div class="info-item">
        <div class="k">{name}</div>
        <div class="v" style="color: {info.loaded !== false ? 'var(--success)' : 'var(--text3)'}">
          {info.total_params ? info.total_params.toLocaleString() + ' params' : (info.loaded !== false ? 'Loaded' : 'Not loaded')}
        </div>
      </div>
    {/each}
  </div>
</div>

<div class="card">
  <h2><i data-lucide="upload-cloud"></i> Upload & Run</h2>
  <div class="grid-2">
    <div>
      <div class="upload-zone" ondrop={handleDrop} ondragover={(e) => e.preventDefault()} onclick={() => document.getElementById('testFileInput').click()}>
        <i data-lucide="image-plus"></i>
        <p>Klik atau drag & drop gambar</p>
        <input type="file" id="testFileInput" accept="image/*" style="display:none"
          onchange={e => e.target.files[0] && handleFile(e.target.files[0])} />
      </div>
      <div class="model-select">
        <label>Model</label>
        <select bind:value={selectedMode}>
          <optgroup label="DocAI Custom / Local">
            {#each modes as m}
              <option value={m}>{modeLabels[m]}</option>
            {/each}
          </optgroup>
          <optgroup label="Pretrained DocShadow-SD7K">
            {#each docshadowModes as m}
              <option value={m} disabled={!availableDocshadowNames.has(m.split(':')[1])}>{allModeLabels[m]}{availableDocshadowNames.has(m.split(':')[1]) ? '' : ' (weight belum ada)'}</option>
            {/each}
          </optgroup>
        </select>
        <p class="info-text">
          {selectedModelFamily === 'docshadow'
            ? 'Pembanding pretrained dari DocShadow-SD7K releases.'
            : 'Model lokal dan mode enhancement DocAI.'}
        </p>
      </div>
      {#if selectedMode === 'docres_finetune'}
        <div class="model-select">
          <label>Finetune Variant</label>
          <select id="finetune_variant">
            <option value="finetune_v3" selected>v3 — Deshadow (iter 45k)</option>
            <option value="finetune_v2">v2 — Deshadow v2</option>
            <option value="finetune_v1">v1 — Original</option>
          </select>
          <p class="info-text">Pilih checkpoint fine-tuning.</p>
        </div>
      {/if}

      {#if selectedMode === 'restore'}
        <div class="model-select">
          <label>Trained checkpoint</label>
          <select bind:value={selectedCheckpoint}>
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
      {#if selectedMode === 'restore'}
        <div class="model-select" style="margin-top:1rem">
          <label>Pipeline Mode</label>
          <div class="pretrained-pills">
            {#each pipelineModes as pm}
              <button
                class="pill"
                class:active={pipelineMode === pm.value}
                onclick={() => pipelineMode = pm.value}
                title={pm.desc}
              >
                {pm.label}
              </button>
            {/each}
          </div>
          <p class="info-text">{pipelineModes.find(p => p.value === pipelineMode)?.desc}</p>
        </div>
        {#if pipelineMode !== 'ai_only'}
          <div class="model-select" style="margin-top:0.75rem">
            <label>Shadow Strength: {shadowStrength.toFixed(1)}</label>
            <input type="range" min="0" max="1" step="0.1" bind:value={shadowStrength}
              style="width:100%;accent-color:var(--primary)" />
            <p class="info-text">0.0 = tanpa shadow fix, 1.0 = full correction</p>
          </div>
        {/if}
      {/if}
      <div class="model-select" style="margin-top:1rem;border-top:1px solid var(--border,#333);padding-top:0.75rem">

      </div>
      <div style="margin-top:1rem;display:flex;align-items:center;gap:0.75rem">
        <button class="btn btn-primary" onclick={runTest} disabled={!selectedFile || processing}>
          {#if processing}<div class="spinner" style="width:16px;height:16px;border-width:2px"></div>{:else}<i data-lucide="play"></i>{/if}
          Run Test
        </button>
        {#if selectedFile}
          <span class="info-text">{selectedFile.name} ({Math.round(selectedFile.size / 1024)}KB)</span>
        {/if}
      </div>
    </div>
    <div>
      {#if originalUrl}
        <div class="img-box">
          <div class="label"><i data-lucide="image"></i> Original</div>
          <img src={originalUrl} alt="Original" />
        </div>
      {:else if selectedFile}
        <div class="img-box placeholder">
          <div class="label"><i data-lucide="image"></i> Original</div>
          <div class="empty-preview">Loading preview...</div>
        </div>
      {/if}
    </div>
  </div>
</div>

{#if resultUrl}
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center">
        <h2><i data-lucide="sparkles"></i> Result — {selectedMode}</h2>
        <button class="btn btn-primary" onclick={downloadResult} style="font-size:0.8rem;padding:0.4rem 0.8rem">
          <i data-lucide="download"></i> Download
        </button>
      </div>
    <div class="compare-grid">
      {#if originalUrl}
        <div class="img-box">
          <div class="label"><i data-lucide="image"></i> Original</div>
          <img src={originalUrl} alt="Original" />
        </div>
      {/if}
      <div class="img-box">
        <div class="label"><i data-lucide="sparkles"></i> Result</div>
        <img src={resultUrl} alt="Result" style:filter={previewFilter} />
        <canvas bind:this={resultCanvas} style="display:none"></canvas>
      </div>
    </div>
  </div>
{/if}
    <div class="adj-sliders">
      <div class="adj-row"><label>Brightness: {adjBrightness.toFixed(2)}</label>
        <input type="range" min="0.2" max="2.5" step="0.05" bind:value={adjBrightness} /></div>
      <div class="adj-row"><label>Contrast: {adjContrast.toFixed(2)}</label>
        <input type="range" min="0.2" max="2.5" step="0.05" bind:value={adjContrast} /></div>
      <div class="adj-row"><label>Saturation: {adjSaturation.toFixed(2)}</label>
        <input type="range" min="0" max="2.5" step="0.05" bind:value={adjSaturation} /></div>
      <div class="adj-check">
        <button class="btn" style="margin-left:auto;font-size:0.75rem;padding:0.35rem 0.65rem" onclick={resetAdjust}>Reset</button>
        <button class="btn btn-primary" style="font-size:0.75rem;padding:0.35rem 0.65rem" onclick={applyAdjustmentsToImage}>Apply</button>
      </div>
    </div>
  

<div class="footer">DocAI v2.0 — Model Testing</div>

<style>
  .page-header { margin-bottom: 2rem; }
  .page-header h1 { font-size: 1.5rem; font-weight: 700; }
  .page-header p { color: var(--text2); margin-top: 0.25rem; font-size: 0.9rem; }
  .card { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.5rem; margin-bottom: 1.5rem; }
  .card :global(h2) { font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  .compare-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; align-items: stretch; }
  .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; }
  .info-item { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); padding: 0.875rem; }
  .info-item .k { font-size: 0.7rem; font-weight: 700; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.375rem; }
  .info-item .v { font-size: 0.95rem; font-weight: 600; }
  .upload-zone { border: 2px dashed var(--border); border-radius: var(--radius-sm); padding: 2rem; text-align: center; cursor: pointer; transition: all 0.2s; }
  .upload-zone:hover { border-color: var(--accent); background: rgba(6,182,212,0.05); }
  .upload-zone :global(i) { width: 36px; height: 36px; color: var(--text3); margin-bottom: 0.5rem; }
  .upload-zone p { color: var(--text2); font-size: 0.85rem; }
  .model-select { margin-top: 1rem; display: grid; gap: 0.45rem; }
  .model-select label { font-size: 0.72rem; font-weight: 700; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; }
  .model-select select { width: 100%; background: var(--bg); border: 1px solid var(--border); color: var(--text); border-radius: var(--radius-xs); padding: 0.65rem 0.75rem; font-size: 0.85rem; }
  .model-select .info-text { margin-left: 0; }
  .mode-pills { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
  .pretrained-pills { margin-top: 0.5rem; }
  .pill { padding: 0.5rem 0.875rem; background: var(--bg); border: 1px solid var(--border); border-radius: 999px; color: var(--text2); font-size: 0.8rem; font-weight: 500; transition: all 0.15s; }
  .pill.pretrained { border-color: rgba(99,102,241,0.45); }
  .pill:hover { border-color: var(--accent); color: var(--text); }
  .pill.active { border-color: var(--accent); background: rgba(6,182,212,0.1); color: var(--accent2); }
  .pill:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; border-radius: var(--radius-xs); font-weight: 600; font-size: 0.8rem; border: none; transition: all 0.15s; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-primary { background: var(--primary); color: white; }
  .img-box { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden; }
  .img-box .label { padding: 0.625rem 0.75rem; border-bottom: 1px solid var(--border); font-size: 0.75rem; font-weight: 600; color: var(--text2); display: flex; align-items: center; gap: 0.5rem; }
  .img-box img { width: 100%; max-height: 70vh; object-fit: contain; display: block; background: var(--bg); }
  .empty-preview { min-height: 180px; display: grid; place-items: center; color: var(--text3); font-size: 0.85rem; }
  .info-text { font-size: 0.8rem; color: var(--text2); margin-left: auto; }
  .footer { text-align: center; padding: 2rem 0; color: var(--text3); font-size: 0.8rem; }
  @media (max-width: 768px) { .grid-2, .compare-grid { grid-template-columns: 1fr; } }
  .adj-sliders { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 0.4rem; margin-top: 0.75rem; padding: 0.75rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); }
  .adj-row { display: flex; flex-direction: column; gap: 0.15rem; }
  .adj-row label { font-size: 0.72rem; color: var(--text2); }
  .adj-row input[type="range"] { width: 100%; accent-color: var(--primary); }
  .adj-check { display: flex; align-items: center; gap: 0.4rem; font-size: 0.75rem; color: var(--text2); }
</style>
