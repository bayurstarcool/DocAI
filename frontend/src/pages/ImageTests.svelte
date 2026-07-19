<script>
  import { refreshIcons } from "../lib/icons.js"
  import { onMount } from 'svelte'
  import { apiJson } from '../stores/auth.js'
  import { showToast } from '../stores/toast.js'

  let tests = []
  let contactSheets = []
  let selectedTest = ''
  let selectedPath = 'input'
  let activeTab = 'contacts'
  let items = []
  let analysisPairs = []
  let lightboxOpen = false
  let lightboxImages = []
  let lightboxIdx = 0
  let analysisName = timestampName()
  let analysisInputFile = null
  let analysisOutputFile = null
  let uploadingPair = false
  let uploadingFile = false
  const maxUploadBytes = 20 * 1024 * 1024
  const uploadTimeoutMs = 60000

  onMount(() => {
    refreshIcons()
    loadContactSheets()
    loadTests()
  })

  function timestampName() {
    const date = new Date()
    const pad = (value) => String(value).padStart(2, '0')
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}_${pad(date.getHours())}-${pad(date.getMinutes())}-${pad(date.getSeconds())}`
  }

  async function loadContactSheets() {
    try {
      const d = await apiJson('/api/image-tests/contact-sheets')
      contactSheets = d.sheets || []
    } catch(e) { contactSheets = [] }
  }

  async function loadTests() {
    try {
      const d = await apiJson('/api/image-tests')
      tests = d.tests || []
      if (!selectedTest && tests.length) selectTest(tests[0].name)
      else if (selectedTest) loadFiles()
    } catch(e) {}
  }

  function selectTest(name) { selectedTest = name; selectedPath = 'input'; loadFiles(); loadAnalysisPairs() }

  async function loadFiles() {
    if (!selectedTest) return
    try {
      const d = await apiJson(`/api/image-tests/files?test=${encodeURIComponent(selectedTest)}&path=${encodeURIComponent(selectedPath)}`)
      items = d.items || []
    } catch(e) {}
  }

  async function loadAnalysisPairs() {
    if (!selectedTest) return
    try {
      const [inputData, outputData] = await Promise.all([
        apiJson(`/api/image-tests/files?test=${encodeURIComponent(selectedTest)}&path=input`),
        apiJson(`/api/image-tests/files?test=${encodeURIComponent(selectedTest)}&path=output`),
      ])
      const inputs = (inputData.items || []).filter(item => item.is_image)
      const outputs = (outputData.items || []).filter(item => item.is_image)
      analysisPairs = inputs.map((input, index) => {
        const sameName = outputs.find(output => output.name === input.name)
        const output = sameName || outputs[index] || null
        return { input, output }
      }).filter(pair => pair.input || pair.output)
    } catch(e) {
      analysisPairs = []
    }
  }

  function openFolder(path) { selectedPath = path; loadFiles() }

  async function deleteItem(item) {
    if (!selectedTest || !item) return
    const ok = confirm(`Hapus ${item.type === 'folder' ? 'folder' : 'file'} "${item.name}"?`)
    if (!ok) return
    try {
      await deletePath(selectedTest, item.path)
      showToast('Item dihapus')
      loadFiles()
      loadAnalysisPairs()
      loadTests()
    } catch(e) { showToast(e.message, 'error') }
  }

  async function deleteSelectedTest() {
    if (!selectedTest) return
    const ok = confirm(`Hapus seluruh test folder "${selectedTest}" beserta input/output?`)
    if (!ok) return
    try {
      await deletePath(selectedTest, '')
      showToast('Test folder dihapus')
      selectedTest = ''
      selectedPath = 'input'
      items = []
      analysisPairs = []
      loadTests()
    } catch(e) { showToast(e.message, 'error') }
  }

  async function deletePath(testName, itemPath) {
    const query = new URLSearchParams({ test: testName, path: itemPath })
    const res = await fetch(`/api/image-tests/item?${query.toString()}`, {
      method: 'DELETE',
      headers: authHeaders(),
      credentials: 'same-origin',
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || `Hapus gagal (${res.status})`)
    }
    return res.json()
  }

  function openLB(idx) {
    lightboxImages = items.filter(i => i.is_image).map(i => ({
      url: `/api/image-tests/preview?test=${encodeURIComponent(selectedTest)}&path=${encodeURIComponent(i.path)}`,
      name: i.name
    }))
    lightboxIdx = idx
    lightboxOpen = true
  }

  function closeLB() { lightboxOpen = false }
  function navLB(dir) { lightboxIdx = (lightboxIdx + dir + lightboxImages.length) % lightboxImages.length }

  function authHeaders() {
    const token = localStorage.getItem('docai_token')
    return token ? { Authorization: `Bearer ${token}` } : {}
  }

  async function fetchWithTimeout(url, options = {}, timeoutMs = uploadTimeoutMs) {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)
    try {
      return await fetch(url, { ...options, signal: controller.signal })
    } catch (error) {
      if (error.name === 'AbortError') {
        throw new Error(`Koneksi timeout setelah ${Math.round(timeoutMs / 1000)} detik. Cek server lalu coba lagi.`)
      }
      throw new Error(`Koneksi ke server gagal. Cek jaringan, login, atau restart server. Detail: ${error.message}`)
    } finally {
      clearTimeout(timer)
    }
  }

  async function uploadFile(e) {
    const file = e.target.files[0]
    if (!file || !selectedTest) return
    if (file.size > maxUploadBytes) {
      showToast(`File terlalu besar. Maks 20 MB: ${file.name} (${bytes(file.size)})`, 'error')
      e.target.value = ''
      return
    }
    const body = new FormData()
    body.append('test', selectedTest)
    body.append('destination', 'input')
    body.append('file', file)
    uploadingFile = true
    try {
      const res = await fetchWithTimeout('/api/image-tests/upload', { method: 'POST', headers: authHeaders(), credentials: 'same-origin', body })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Upload gagal')
      }
      showToast('Uploaded!')
      loadFiles()
      loadAnalysisPairs()
    } catch(e) { showToast(e.message, 'error') }
    finally { uploadingFile = false }
    e.target.value = ''
  }

  async function uploadAnalysisPair() {
    if (!analysisName.trim()) {
      showToast('Nama analisis wajib diisi', 'error')
      return
    }
    if (!analysisInputFile || !analysisOutputFile) {
      showToast('Pilih input dan output dulu', 'error')
      return
    }
    if (analysisInputFile.size > maxUploadBytes || analysisOutputFile.size > maxUploadBytes) {
      const tooLarge = [analysisInputFile, analysisOutputFile].filter(file => file.size > maxUploadBytes).map(file => `${file.name} (${bytes(file.size)})`).join(', ')
      showToast(`File terlalu besar. Maks 20 MB per file: ${tooLarge}`, 'error')
      return
    }
    uploadingPair = true
    try {
      try {
        await uploadAnalysisPairRequest(analysisName, analysisInputFile, analysisOutputFile)
      } catch (pairError) {
        await uploadAnalysisFile(analysisName, 'input', analysisInputFile)
        await uploadAnalysisFile(analysisName, 'output', analysisOutputFile)
      }
      await verifyAnalysisPair(analysisName)
      showToast('Pair analisis tersimpan')
      selectedTest = analysisName.trim()
      selectedPath = 'input'
      analysisInputFile = null
      analysisOutputFile = null
      analysisName = timestampName()
      await loadTests()
      await loadFiles()
      await loadAnalysisPairs()
    } catch(e) {
      showToast(e.message, 'error')
    } finally {
      uploadingPair = false
    }
  }

  async function uploadAnalysisPairRequest(testName, inputFile, outputFile) {
    const body = new FormData()
    body.append('test', testName.trim())
    body.append('input_file', inputFile)
    body.append('output_file', outputFile)
    let res
    try {
      res = await fetchWithTimeout('/api/image-tests/upload-pair', {
        method: 'POST',
        headers: authHeaders(),
        credentials: 'same-origin',
        body,
      })
    } catch (error) {
      throw error
    }
    if (res.status === 404) {
      await uploadAnalysisFile(testName, 'input', inputFile)
      await uploadAnalysisFile(testName, 'output', outputFile)
      return { success: true, fallback: true }
    }
    if (!res.ok) {
      throw new Error(await responseErrorMessage(res, 'Upload pair gagal'))
    }
    return res.json()
  }

  async function verifyAnalysisPair(testName) {
    const [inputData, outputData] = await Promise.all([
      apiJson(`/api/image-tests/files?test=${encodeURIComponent(testName.trim())}&path=input`),
      apiJson(`/api/image-tests/files?test=${encodeURIComponent(testName.trim())}&path=output`),
    ])
    const inputCount = (inputData.items || []).filter(item => item.is_image).length
    const outputCount = (outputData.items || []).filter(item => item.is_image).length
    if (!inputCount || !outputCount) {
      throw new Error(`Upload belum lengkap. Tersimpan: input=${inputCount}, output=${outputCount}`)
    }
  }

  async function uploadAnalysisFile(testName, destination, file) {
    const body = new FormData()
    body.append('test', testName.trim())
    body.append('destination', destination)
    body.append('file', file)
    let res
    try {
      res = await fetchWithTimeout('/api/image-tests/upload', {
        method: 'POST',
        headers: authHeaders(),
        credentials: 'same-origin',
        body,
      })
    } catch (error) {
      throw new Error(`Upload ${destination} gagal koneksi. File: ${file.name} (${bytes(file.size)}). Detail: ${error.message}`)
    }
    if (!res.ok) {
      throw new Error(await responseErrorMessage(res, `Upload ${destination} gagal`))
    }
    return res.json()
  }

  async function responseErrorMessage(res, prefix) {
    const text = await res.text().catch(() => '')
    let detail = text
    try {
      const json = JSON.parse(text)
      detail = Array.isArray(json.detail) ? json.detail.map(item => item.msg || JSON.stringify(item)).join('; ') : (json.detail || json.message || text)
    } catch(e) {}
    return `${prefix} (${res.status} ${res.statusText})${detail ? `: ${detail}` : ''}`
  }

  function bytes(n) { if (n == null) return ''; return n < 1024 ? n + ' B' : n < 1048576 ? (n/1024).toFixed(1) + ' KB' : (n/1048576).toFixed(1) + ' MB' }
  function previewUrl(path) { return `/api/image-tests/preview?test=${encodeURIComponent(selectedTest)}&path=${encodeURIComponent(path)}` }
</script>

<div class="page-header">
  <h1>Image Test Workspace</h1>
  <p>Browse input/output test folders, upload pair untuk analisis hasil model.</p>
</div>

<div class="tabs">
  <button class:active={activeTab === 'contacts'} onclick={() => activeTab = 'contacts'}><i data-lucide="scan-search"></i> Contact Sheets</button>
  <button class:active={activeTab === 'upload'} onclick={() => activeTab = 'upload'}><i data-lucide="upload-cloud"></i> Upload Pair</button>
  <button class:active={activeTab === 'workspace'} onclick={() => activeTab = 'workspace'}><i data-lucide="folder-open"></i> Workspace</button>
</div>

{#if activeTab === 'contacts'}
<section class="contact-sheet-card">
  <div class="contact-sheet-header">
    <div>
      <h2><i data-lucide="scan-search"></i> Contact Sheet Analysis</h2>
      <p>Daftar semua contact sheet analisis dari data/test.</p>
    </div>
  </div>
  {#if !contactSheets.length}
    <div class="contact-sheet-empty">Belum ada contact sheet.</div>
  {:else}
    <div class="contact-sheet-list">
      {#each contactSheets as sheet}
        <article class="contact-sheet-row">
          <a class="contact-sheet-preview" href={sheet.url} target="_blank" rel="noreferrer">
            <img class="contact-sheet-thumb" src={sheet.url} alt={sheet.name} loading="lazy" />
          </a>
          <div class="contact-sheet-meta">
            <div class="contact-sheet-name">{sheet.name}</div>
            <div class="contact-sheet-detail">{sheet.path}</div>
            <div class="contact-sheet-detail">{(sheet.size / 1024).toFixed(1)} KB</div>
          </div>
          <a class="btn btn-secondary contact-sheet-open" href={sheet.url} target="_blank" rel="noreferrer">
            <i data-lucide="maximize-2"></i> Buka
          </a>
        </article>
      {/each}
    </div>
  {/if}
</section>
{/if}

{#if activeTab === 'upload'}
<div class="analysis-card">
  <div>
    <h2><i data-lucide="flask-conical"></i> Upload Pair Analisis</h2>
    <p>Simpan gambar asli dan hasil model dalam satu folder untuk analisis sisa shadow, whiteness, contrast, dan color cast.</p>
  </div>
  <div class="analysis-grid">
    <div class="field">
      <label>Nama Analisis</label>
      <input type="text" bind:value={analysisName} placeholder="analysis_shadow_case_01" />
    </div>
    <div class="field">
      <label>Input / Original</label>
      <input type="file" accept="image/*" onchange={(e) => analysisInputFile = e.target.files?.[0] || null} />
      {#if analysisInputFile}<span>{analysisInputFile.name} ({bytes(analysisInputFile.size)})</span>{/if}
    </div>
    <div class="field">
      <label>Output / Result</label>
      <input type="file" accept="image/*" onchange={(e) => analysisOutputFile = e.target.files?.[0] || null} />
      {#if analysisOutputFile}<span>{analysisOutputFile.name} ({bytes(analysisOutputFile.size)})</span>{/if}
    </div>
    <button class="btn btn-primary" onclick={uploadAnalysisPair} disabled={uploadingPair}>
      {#if uploadingPair}<div class="spinner" style="width:14px;height:14px;border-width:2px"></div>{:else}<i data-lucide="upload-cloud"></i>{/if}
      Simpan Pair
    </button>
  </div>
</div>
{/if}

{#if activeTab === 'workspace'}
<div class="layout">
  <aside class="sidebar">
    <h3>Test Folders</h3>
    {#each tests as t}
      <button class="folder-btn" class:active={t.name === selectedTest} onclick={() => selectTest(t.name)}>
        <i data-lucide="folder"></i> {t.name} <span class="count">{t.image_count}</span>
      </button>
    {/each}
    {#if !tests.length}<p style="color:var(--text3);padding:1rem">No test folders</p>{/if}
  </aside>

  <div class="content">
    <div class="breadcrumb">
      <button onclick={() => openFolder('input')}>{selectedTest}</button>
      {#if selectedPath !== 'input'}
        {#each selectedPath.split('/').filter(Boolean) as part, i}
          <span>/</span>
          <button onclick={() => openFolder(selectedPath.split('/').slice(0, i + 2).join('/'))}>{part}</button>
        {/each}
      {/if}
    </div>

    <div class="toolbar">
      <button class="btn btn-primary btn-sm" onclick={() => document.getElementById('imgTestUpload').click()} disabled={uploadingFile}>
        {#if uploadingFile}
          <div class="spinner" style="width:12px;height:12px;border-width:2px"></div> Uploading…
        {:else}
          <i data-lucide="upload"></i> Upload
        {/if}
      </button>
      <button class="btn btn-secondary btn-sm" onclick={loadFiles}><i data-lucide="refresh-cw"></i> Refresh</button>
      <button class="btn btn-danger btn-sm" onclick={deleteSelectedTest} disabled={!selectedTest}><i data-lucide="trash-2"></i> Hapus</button>
      <span class="count-text">{items.length} items</span>
      <input type="file" id="imgTestUpload" accept="image/*" style="display:none" onchange={uploadFile} />
    </div>

    {#if analysisPairs.length}
      <div class="pair-section">
        <div class="pair-title"><i data-lucide="columns-2"></i> Input + Output Analisis</div>
        <div class="pair-grid">
          {#each analysisPairs as pair}
            <div class="pair-card">
              <div class="pair-image">
                <div class="pair-label">Input</div>
                {#if pair.input}
                  <img src={previewUrl(pair.input.path)} alt={pair.input.name} />
                {:else}
                  <div class="pair-empty">Tidak ada input</div>
                {/if}
              </div>
              <div class="pair-image">
                <div class="pair-label">Output</div>
                {#if pair.output}
                  <img src={previewUrl(pair.output.path)} alt={pair.output.name} />
                {:else}
                  <div class="pair-empty">Tidak ada output</div>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      </div>
    {/if}

    <div class="file-grid">
      {#each items as item, i}
        {#if item.type === 'folder'}
          <div class="file-card folder-card">
            <button class="card-main" onclick={() => openFolder(item.path)}>
              <i data-lucide="folder"></i>
              <span>{item.name}</span>
            </button>
            <button class="delete-btn" title="Hapus" onclick={() => deleteItem(item)}><i data-lucide="trash-2"></i></button>
          </div>
        {:else if item.is_image}
          {@const images = items.filter(x => x.is_image)}
          <div class="file-card">
            <button class="image-main" onclick={() => openLB(images.indexOf(item))}>
            <img loading="lazy" src={previewUrl(item.path)} alt={item.name} />
              <div class="info"><span class="fname">{item.name}</span><span>{bytes(item.size)}</span></div>
            </button>
            <button class="delete-btn" title="Hapus" onclick={() => deleteItem(item)}><i data-lucide="trash-2"></i></button>
          </div>
        {/if}
      {/each}
    </div>
    {#if !items.length}<div class="empty"><i data-lucide="folder-open"></i> Folder kosong</div>{/if}
  </div>
</div>
{/if}

{#if lightboxOpen}
  <div class="lightbox-overlay" onclick={closeLB}>
    <button class="lb-close" onclick={closeLB}><i data-lucide="x"></i></button>
    <button class="lb-nav prev" onclick={(e) => { e.stopPropagation(); navLB(-1) }}><i data-lucide="chevron-left"></i></button>
    <button class="lb-nav next" onclick={(e) => { e.stopPropagation(); navLB(1) }}><i data-lucide="chevron-right"></i></button>
    <img class="lb-img" src={lightboxImages[lightboxIdx]?.url} alt="" onclick={(e) => { e.stopPropagation(); closeLB() }} />
    <div class="lb-info">
      <span>{lightboxImages[lightboxIdx]?.name}</span>
      <span>{lightboxIdx + 1} / {lightboxImages.length}</span>
    </div>
  </div>
{/if}

<div class="footer">DocAI v2.0 — Image Tests</div>

<style>
  .contact-sheet-card { margin-bottom: 1.25rem; padding: 1rem; background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius-sm); }
  .contact-sheet-header { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 0.85rem; }
  .contact-sheet-header h2 { display: flex; align-items: center; gap: 0.5rem; margin: 0 0 0.25rem; font-size: 1rem; }
  .contact-sheet-header h2 :global(i) { width: 18px; height: 18px; color: var(--accent2); }
  .contact-sheet-header p { margin: 0; color: var(--text3); font-size: 0.8rem; }
  .contact-sheet-viewer { max-height: 720px; overflow: auto; background: #e5e7eb; border: 1px solid var(--border); border-radius: var(--radius-xs); }
  .contact-sheet-viewer img { display: block; width: 100%; height: auto; }
  .contact-sheet-path { margin-top: 0.55rem; color: var(--text3); font-family: monospace; font-size: 0.72rem; }
  .contact-sheet-empty { color: var(--text3); font-size: 0.85rem; padding: 0.75rem; border: 1px dashed var(--border); border-radius: var(--radius-xs); }
  .contact-sheet-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 360px), 1fr)); gap: 0.9rem; }
  .contact-sheet-row { min-width: 0; display: grid; grid-template-rows: auto 1fr auto; gap: 0.75rem; padding: 0.85rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); }
  .contact-sheet-preview { display: block; width: 100%; background: #e5e7eb; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .contact-sheet-thumb { width: 100%; height: clamp(180px, 28vw, 360px); object-fit: contain; display: block; }
  .contact-sheet-meta { min-width: 0; }
  .contact-sheet-name { color: var(--text); font-size: 0.9rem; font-weight: 700; margin-bottom: 0.25rem; }
  .contact-sheet-detail { color: var(--text3); font-family: monospace; font-size: 0.72rem; overflow-wrap: anywhere; }
  .contact-sheet-open { width: fit-content; justify-self: start; }
  .page-header { margin-bottom: 2rem; }
  .page-header h1 { font-size: 1.5rem; font-weight: 700; }
  .page-header p { color: var(--text2); margin-top: 0.25rem; font-size: 0.9rem; }
  .tabs { display: flex; gap: 0.5rem; margin-bottom: 1rem; overflow-x: auto; padding-bottom: 0.25rem; }
  .tabs button { flex: 0 0 auto; display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.6rem 0.9rem; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-xs); color: var(--text2); font-size: 0.8rem; font-weight: 700; cursor: pointer; }
  .tabs button.active { background: var(--primary); border-color: var(--primary); color: white; }
  .tabs button :global(i) { width: 16px; height: 16px; }
  .analysis-card { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem; margin-bottom: 1.5rem; display: grid; gap: 1rem; }
  .analysis-card h2 { font-size: 1.05rem; font-weight: 700; display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.35rem; }
  .analysis-card p { color: var(--text2); font-size: 0.85rem; }
  .analysis-grid { display: grid; grid-template-columns: 1.2fr 1fr 1fr auto; gap: 0.85rem; align-items: end; }
  .field { display: grid; gap: 0.4rem; }
  .field label { font-size: 0.72rem; font-weight: 700; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; }
  .field input[type="text"], .field input[type="file"] { width: 100%; background: var(--bg); border: 1px solid var(--border); color: var(--text); border-radius: var(--radius-xs); padding: 0.55rem 0.65rem; font-size: 0.8rem; }
  .field span { color: var(--text3); font-size: 0.72rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .layout { display: grid; grid-template-columns: 260px 1fr; gap: 1.5rem; }
  .sidebar { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem; position: sticky; top: 80px; height: fit-content; max-height: calc(100vh - 100px); overflow-y: auto; }
  .sidebar h3 { font-size: 0.75rem; font-weight: 700; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem; }
  .folder-btn { width: 100%; text-align: left; padding: 0.75rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); color: var(--text2); font-size: 0.85rem; display: flex; align-items: center; gap: 0.75rem; transition: all 0.15s; margin-bottom: 0.5rem; }
  .folder-btn:hover { border-color: var(--accent); color: var(--text); }
  .folder-btn.active { border-color: var(--accent); color: var(--accent2); background: rgba(6,182,212,0.05); }
  .folder-btn :global(i) { width: 18px; height: 18px; color: var(--warning); flex-shrink: 0; }
  .count { margin-left: auto; font-size: 0.7rem; background: var(--bg3); padding: 0.125rem 0.5rem; border-radius: 999px; }
  .content { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.5rem; }
  .breadcrumb { display: flex; align-items: center; gap: 0.25rem; margin-bottom: 1rem; flex-wrap: wrap; }
  .breadcrumb button { background: none; border: none; color: var(--accent2); cursor: pointer; font-size: 0.8rem; font-weight: 500; padding: 0.25rem 0.375rem; border-radius: 4px; }
  .breadcrumb button:hover { background: rgba(6,182,212,0.1); }
  .breadcrumb span { color: var(--text3); font-size: 0.8rem; }
  .toolbar { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }
  .count-text { font-size: 0.8rem; color: var(--text3); margin-left: auto; }
  .btn { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; border-radius: var(--radius-xs); font-size: 0.8rem; font-weight: 600; border: none; transition: all 0.15s; }
  .btn-sm { padding: 0.4rem 0.75rem; font-size: 0.75rem; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-primary { background: var(--primary); color: white; }
  .btn-secondary { background: var(--surface); color: var(--text); border: 1px solid var(--border); }
  .btn-danger { background: var(--error); color: white; }
  .pair-section { margin-bottom: 1.25rem; padding: 1rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); }
  .pair-title { display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; font-weight: 700; color: var(--text2); margin-bottom: 0.75rem; }
  .pair-title :global(i) { width: 16px; height: 16px; }
  .pair-grid { display: grid; gap: 1rem; }
  .pair-card { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-xs); padding: 0.75rem; }
  .pair-image { min-width: 0; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-xs); overflow: hidden; }
  .pair-label { padding: 0.45rem 0.65rem; font-size: 0.72rem; font-weight: 700; color: var(--text2); border-bottom: 1px solid var(--border); text-transform: uppercase; letter-spacing: 0.04em; }
  .pair-image img { width: 100%; max-height: 420px; object-fit: contain; display: block; background: var(--bg); }
  .pair-empty { height: 180px; display: grid; place-items: center; color: var(--text3); font-size: 0.8rem; }
  .file-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 0.75rem; }
  .file-card { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); overflow: hidden; cursor: pointer; transition: all 0.15s; position: relative; }
  .file-card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .image-main, .card-main { width: 100%; background: none; border: none; color: inherit; text-align: left; padding: 0; cursor: pointer; }
  .file-card img { width: 100%; aspect-ratio: 3/4; object-fit: cover; display: block; background: var(--surface); }
  .file-card .info { padding: 0.5rem 0.75rem; font-size: 0.7rem; color: var(--text2); display: flex; justify-content: space-between; }
  .fname { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 70%; }
  .folder-card { display: flex; align-items: center; gap: 0.75rem; padding: 1rem; }
  .folder-card .card-main { display: flex; align-items: center; gap: 0.75rem; }
  .folder-card :global(i) { width: 28px; height: 28px; color: var(--warning); }
  .delete-btn { position: absolute; top: 0.4rem; right: 0.4rem; width: 30px; height: 30px; display: grid; place-items: center; background: rgba(239,68,68,0.92); border: none; border-radius: var(--radius-xs); color: white; opacity: 0; transition: opacity 0.15s; }
  .file-card:hover .delete-btn { opacity: 1; }
  .delete-btn :global(i) { width: 15px; height: 15px; color: white; }
  .empty { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 4rem 0; color: var(--text3); gap: 0.75rem; }
  .empty :global(i) { width: 48px; height: 48px; opacity: 0.3; }
  .lightbox-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.92); z-index: 600; display: flex; align-items: center; justify-content: center; flex-direction: column; }
  .lb-close { position: absolute; top: 1rem; right: 1rem; background: rgba(255,255,255,0.1); border: none; color: white; padding: 0.5rem; border-radius: var(--radius-xs); z-index: 10; }
  .lb-nav { position: absolute; top: 50%; transform: translateY(-50%); background: rgba(255,255,255,0.1); border: none; color: white; padding: 1rem; border-radius: var(--radius-xs); }
  .lb-nav.prev { left: 1rem; }
  .lb-nav.next { right: 1rem; }
  .lb-img { max-width: 90vw; max-height: 85vh; object-fit: contain; border-radius: 4px; }
  .lb-info { position: absolute; bottom: 1rem; color: rgba(255,255,255,0.7); font-size: 0.8rem; display: flex; gap: 1rem; }
  .footer { text-align: center; padding: 2rem 0; color: var(--text3); font-size: 0.8rem; }
  @media (max-width: 1100px) { .analysis-grid { grid-template-columns: 1fr 1fr; } }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } .content { padding: 1rem; } }
  @media (max-width: 768px) { .page-header { margin-bottom: 1rem; } .page-header h1 { font-size: 1.25rem; } .tabs { position: sticky; top: 0; z-index: 5; background: var(--bg); margin-left: -0.25rem; margin-right: -0.25rem; padding: 0.25rem; } .tabs button { flex: 1 0 auto; justify-content: center; } .contact-sheet-card { padding: 0.75rem; } .contact-sheet-header { align-items: flex-start; flex-direction: column; } .contact-sheet-list { grid-template-columns: 1fr; } .contact-sheet-thumb { height: min(70vh, 420px); } .contact-sheet-open { width: 100%; justify-content: center; } .file-grid { grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); } .analysis-grid, .pair-card { grid-template-columns: 1fr; } .toolbar { gap: 0.35rem; } .toolbar .btn { width: 100%; justify-content: center; } .count-text { width: 100%; margin-left: 0; text-align: center; } }
</style>
