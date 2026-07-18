<script>
  import { refreshIcons } from "../lib/icons.js"
  import { onMount } from 'svelte'
  import { apiJson } from '../stores/auth.js'
  import { showToast } from '../stores/toast.js'

  let modelInfo = {}
  let selectedFile = null
  let selectedMode = 'restore'
  let resultUrl = ''
  let processing = false
  let docshadowWeights = []

  const modes = ['restore','shadow_remove','enhance','magic_enhance','binarize','cleanup','clahe','denoise','sharpen','deskew']
  const pretrainedModes = ['SD7K', 'Jung', 'Kligler'].map(name => ({ name }))
  const modeLabels = { restore:'AI Restore', shadow_remove:'Shadow Remove', enhance:'AI Enhance', magic_enhance:'Magic Enhance', binarize:'Binarize', cleanup:'Full Cleanup', clahe:'CLAHE', denoise:'Denoise', sharpen:'Sharpen', deskew:'Deskew' }
  $: availableDocshadowNames = new Set(docshadowWeights.map(w => w.name))
  $: docshadowModes = pretrainedModes.map(w => `docshadow:${w.name}`)
  $: allModeLabels = {
    ...modeLabels,
    ...Object.fromEntries(pretrainedModes.map(w => [`docshadow:${w.name}`, `Pretrained DocShadow ${w.name}`]))
  }
  $: selectedModelFamily = selectedMode.startsWith('docshadow:') ? 'docshadow' : 'docai'

  onMount(async () => {
    refreshIcons()
    try { modelInfo = await apiJson('/api/models/info') } catch(e) {}
    try {
      const d = await apiJson('/api/docshadow/weights')
      docshadowWeights = d.weights || []
    } catch(e) { docshadowWeights = [] }
  })

  function handleFile(file) {
    selectedFile = file
    resultUrl = ''
  }

  async function runTest() {
    if (!selectedFile) return
    processing = true
    const fd = new FormData()
    fd.append('file', selectedFile)
    fd.append('mode', selectedMode)
    const start = performance.now()
    try {
      const token = localStorage.getItem('docai_token')
      let endpoint = '/api/scan'
      if (selectedMode.startsWith('docshadow:')) {
        fd.set('mode', 'docshadow')
        fd.append('weight', selectedMode.split(':')[1])
        endpoint = '/api/docshadow/infer'
      }
      const res = await fetch(endpoint, { method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: fd })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Test failed')
      }
      const blob = await res.blob()
      resultUrl = URL.createObjectURL(blob)
      const ms = Math.round(performance.now() - start)
      showToast(`${allModeLabels[selectedMode] || selectedMode} completed (${ms}ms)`)
    } catch(e) { showToast(e.message, 'error') }
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
      <div class="upload-zone" onclick={() => document.getElementById('testFileInput').click()}>
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
      {#if selectedFile}
        <div class="img-box">
          <div class="label"><i data-lucide="image"></i> Original</div>
          <img src={URL.createObjectURL(selectedFile)} alt="Original" />
        </div>
      {/if}
    </div>
  </div>
</div>

{#if resultUrl}
  <div class="card">
    <h2><i data-lucide="sparkles"></i> Result — {selectedMode}</h2>
    <div class="img-box">
      <img src={resultUrl} alt="Result" />
    </div>
  </div>
{/if}

<div class="footer">DocAI v2.0 — Model Testing</div>

<style>
  .page-header { margin-bottom: 2rem; }
  .page-header h1 { font-size: 1.5rem; font-weight: 700; }
  .page-header p { color: var(--text2); margin-top: 0.25rem; font-size: 0.9rem; }
  .card { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.5rem; margin-bottom: 1.5rem; }
  .card :global(h2) { font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
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
  .img-box img { width: 100%; display: block; }
  .info-text { font-size: 0.8rem; color: var(--text2); margin-left: auto; }
  .footer { text-align: center; padding: 2rem 0; color: var(--text3); font-size: 0.8rem; }
  @media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }
</style>
