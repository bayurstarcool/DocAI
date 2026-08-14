<script>
  import { onMount, onDestroy } from 'svelte'
  import { refreshIcons } from '../lib/icons.js'
  import { showToast } from '../stores/toast.js'

  const API = ''
  let authOk = false
  let inputs = []
  let uploads = []
  let selectedInput = ''      // "source:name"
  let count = 5
  let candidates = []
  let picked = new Set()
  let loading = true
  let generating = false
  let job = null
  let pollTimer = null
  let selectedCount = 0
  let rejectedCount = 0
  let customCount = 0
  let customItems = []
  let customPicked = new Set()
  let showCustom = false
  let preview = null
  let uploading = false

  onMount(async () => {
    try { authOk = JSON.parse(atob(localStorage.getItem('docai_token'))).user === 'admin' } catch(e) { authOk = false }
    if (authOk) { await loadInputs(); await loadCandidates(); await loadCustom(); await resumeJob() }
    refreshIcons()
  })
  onDestroy(() => { if (pollTimer) clearInterval(pollTimer); window.removeEventListener('keydown', onKey) })

  function onKey(e) { if (e.key === 'Escape') preview = null }
  onMount(() => window.addEventListener('keydown', onKey))

  async function apiFetch(url, opts = {}) {
    const res = await fetch(API + url, { ...opts, headers: { Authorization: 'Bearer ' + (localStorage.getItem('docai_token') || ''), ...(opts.headers || {}) } })
    if (!res.ok) throw new Error(await res.text())
    return await res.json()
  }

  async function loadInputs() {
    loading = true
    try {
      const d = await apiFetch('/api/synthetic-dewarp/clean-inputs')
      inputs = d.items || []
      uploads = d.uploads || []
      if (!selectedInput) {
        const first = uploads[0] || inputs.find(x => x.clean) || inputs[0]
        if (first) selectedInput = first.source + ':' + first.name
      }
    } catch(e) { showToast('Gagal load input: ' + e.message, 'error') }
    finally { loading = false }
  }

  async function loadCandidates() {
    const d = await apiFetch('/api/synthetic-dewarp/candidates')
    candidates = d.candidates || []
    selectedCount = d.selected_count || 0
    rejectedCount = d.rejected_count || 0
    customCount = d.custom_count || 0
  }

  function parseSel() {
    const idx = selectedInput.indexOf(':')
    return { source: selectedInput.slice(0, idx), name: selectedInput.slice(idx + 1) }
  }

  $: currentPreviewUrl = (() => {
    if (!selectedInput) return ''
    const { source, name } = parseSel()
    return source === 'upload' ? `/api/synthetic-dewarp/upload-preview/${name}` : `/api/synthetic-dewarp/clean-preview/${name}`
  })()

  async function onUpload(e) {
    const file = e.target.files?.[0]
    if (!file) return
    if (file.size > 5 * 1024 * 1024) {
      showToast(`Ukuran gambar maksimal 5MB (file ${(file.size/1024/1024).toFixed(1)}MB)`, 'error')
      e.target.value = ''
      return
    }
    uploading = true
    try {
      const fd = new FormData(); fd.append('file', file)
      const d = await apiFetch('/api/synthetic-dewarp/upload', { method: 'POST', body: fd })
      showToast('Upload ok: ' + d.name, 'success')
      await loadInputs()
      selectedInput = 'upload:' + d.name
    } catch(err) { showToast('Upload gagal: ' + err.message, 'error') }
    finally { uploading = false; e.target.value = '' }
  }

  async function generate() {
    if (!selectedInput) return
    const { source, name } = parseSel()
    generating = true
    try {
      const d = await apiFetch('/api/synthetic-dewarp/generate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: name, source, count })
      })
      job = { job_id: d.job_id, status: 'starting', done: 0, count, items: [] }
      localStorage.setItem('synth_dewarp_job', d.job_id)
      startPolling()
    } catch(e) { showToast('Generate gagal: ' + e.message, 'error'); generating = false }
  }

  function startPolling() {
    if (pollTimer) clearInterval(pollTimer)
    generating = true
    pollTimer = setInterval(async () => {
      if (!job) return
      try {
        const d = await apiFetch('/api/synthetic-dewarp/job/' + job.job_id)
        const prevDone = job.done
        job = d
        if (d.done > prevDone) await loadCandidates()   // reveal ready samples immediately
        if (['done', 'error', 'cancelled'].includes(d.status)) {
          clearInterval(pollTimer); pollTimer = null; generating = false
          localStorage.removeItem('synth_dewarp_job')
          await loadCandidates()
          if (d.status === 'done') showToast(`Selesai — ${d.done} sample siap`, 'success')
          if (d.status === 'error') showToast('Job error: ' + (d.error || ''), 'error')
        }
      } catch(e) { /* keep polling */ }
    }, 2000)
  }

  async function resumeJob() {
    const jid = localStorage.getItem('synth_dewarp_job')
    if (!jid) return
    try {
      const d = await apiFetch('/api/synthetic-dewarp/job/' + jid)
      if (['done', 'error', 'cancelled'].includes(d.status)) {
        localStorage.removeItem('synth_dewarp_job')
        return
      }
      job = d
      count = d.count
      showToast('Melanjutkan job generate yang masih berjalan...', 'info')
      startPolling()
    } catch(e) {
      localStorage.removeItem('synth_dewarp_job')   // job gone (server restarted)
    }
  }

  async function cancelJob() {
    if (!job) return
    try { await apiFetch('/api/synthetic-dewarp/job/' + job.job_id + '/cancel', { method: 'POST' }) } catch(e) {}
    localStorage.removeItem('synth_dewarp_job')
  }

  function toggle(name) { const s = new Set(picked); s.has(name) ? s.delete(name) : s.add(name); picked = s }

  async function mark(action) {
    const names = [...picked]
    if (!names.length) return showToast('Pilih sample dulu', 'error')
    try {
      await apiFetch('/api/synthetic-dewarp/mark', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ names, action }) })
      showToast(`${names.length} ${action === 'select' ? 'disimpan' : 'direject'}`, 'success')
      picked = new Set(); await loadCandidates()
    } catch(e) { showToast('Gagal: ' + e.message, 'error') }
  }

  async function commitCustom() {
    try {
      const d = await apiFetch('/api/synthetic-dewarp/commit-custom', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) })
      showToast(`${d.committed.length} sample masuk custom dewarp (total ${d.custom_count})`, 'success')
      await loadCandidates()
      if (showCustom) await loadCustom()
    } catch(e) { showToast('Commit gagal: ' + e.message, 'error') }
  }

  async function loadCustom() {
    const d = await apiFetch('/api/synthetic-dewarp/custom')
    customItems = d.items || []
    customCount = d.count || 0
    customPicked = new Set()
  }

  async function toggleCustomView() {
    showCustom = !showCustom
    if (showCustom) await loadCustom()
  }

  function toggleCustom(name) { const s = new Set(customPicked); s.has(name) ? s.delete(name) : s.add(name); customPicked = s }

  async function removeCustom(toSelected) {
    const names = [...customPicked]
    if (!names.length) return showToast('Pilih sample dulu', 'error')
    const msg = toSelected ? 'Kembalikan ke Selected?' : 'Hapus permanen dari custom dataset?'
    if (!confirm(`${names.length} sample. ${msg}`)) return
    try {
      const d = await apiFetch('/api/synthetic-dewarp/custom/remove', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ names, to_selected: toSelected }) })
      showToast(`${d.removed.length} sample ${toSelected ? 'dikembalikan' : 'dihapus'}`, 'success')
      await loadCustom(); await loadCandidates()
    } catch(e) { showToast('Gagal: ' + e.message, 'error') }
  }

  $: progressPct = job && job.count ? Math.round((job.done / job.count) * 100) : 0
</script>

{#if !authOk}
  <section class="card denied"><h1>Admin only</h1><p>Halaman Synthetic Dewarp hanya untuk admin.</p></section>
{:else}
  <section class="page-head">
    <div>
      <p class="eyebrow">Control Points Dataset</p>
      <h1>Synthetic Dewarp</h1>
      <p class="page-desc">Generate realtime, preview, pilih sample bagus, lalu masukkan ke dataset custom dewarp.</p>
    </div>
    <button class="btn" onclick={loadCandidates}><i data-lucide="refresh-cw"></i> Refresh</button>
  </section>

  <section class="grid2">
    <div class="card">
      <h2>1. Clean input</h2>
      {#if loading}<p>Loading...</p>{:else}
        <select bind:value={selectedInput}>
          {#if uploads.length}
            <optgroup label="Upload">
              {#each uploads as u}<option value={'upload:' + u.name}>⬆ {u.name}</option>{/each}
            </optgroup>
          {/if}
          <optgroup label="Inv3D target">
            {#each inputs as item}<option value={'inv3d:' + item.name}>{item.clean ? '✓' : '•'} {item.name}</option>{/each}
          </optgroup>
        </select>
        <label class="upload-btn">
          <input type="file" accept="image/*" onchange={onUpload} style="display:none" />
          <i data-lucide="upload"></i> {uploading ? 'Uploading...' : 'Upload gambar custom'}
        </label>
        {#if currentPreviewUrl}<div class="clean-preview"><img src={currentPreviewUrl} alt="clean" /></div>{/if}
      {/if}
    </div>

    <div class="card">
      <h2>2. Generate realtime</h2>
      <label>Jumlah variants</label>
      <input type="number" min="1" max="20" bind:value={count} />
      {#if !generating}
        <button class="btn primary" disabled={!selectedInput} onclick={generate}><i data-lucide="wand-sparkles"></i> Generate {count}</button>
      {:else}
        <button class="btn danger" onclick={cancelJob}><i data-lucide="square"></i> Cancel</button>
      {/if}

      {#if job}
        <div class="progress-wrap">
          <div class="progress-bar"><div class="progress-fill" style="width:{progressPct}%"></div></div>
          <div class="progress-meta">
            <span>{job.status}</span>
            <span>{job.done} / {job.count} siap</span>
          </div>
        </div>
      {/if}

      <div class="stats">
        <span>Candidate <b>{candidates.length}</b></span>
        <span>Selected <b>{selectedCount}</b></span>
        <span>Rejected <b>{rejectedCount}</b></span>
        <span>Custom <b>{customCount}</b></span>
      </div>
      <p class="hint">Sample yang sudah siap langsung muncul di bawah, tanpa tunggu semua selesai.</p>
    </div>
  </section>

  <section class="card">
    <div class="section-head">
      <div><h2>3. Review candidates</h2><p class="hint">Pilih sample bagus: crop benar, dokumen realistis, control points ikut bentuk.</p></div>
      <div class="actions">
        <button class="btn ok" onclick={() => mark('select')} disabled={picked.size === 0}>Save selected ({picked.size})</button>
        <button class="btn danger" onclick={() => mark('reject')} disabled={picked.size === 0}>Reject</button>
        <button class="btn accent" onclick={commitCustom} disabled={selectedCount === 0}><i data-lucide="database"></i> Masukkan ke Custom Dewarp ({selectedCount})</button>
      </div>
    </div>

    {#if candidates.length === 0}
      <div class="empty">{generating ? 'Menunggu sample pertama...' : 'Belum ada candidate. Generate dulu.'}</div>
    {:else}
      <div class="cards">
        {#each candidates as c (c.name)}
          <div class="sample" class:selected={picked.has(c.name)}>
            <label class="check"><input type="checkbox" checked={picked.has(c.name)} onchange={() => toggle(c.name)} /> pilih</label>
            <button class="imgbtn" onclick={() => preview = c}><img src={c.points_url} alt={c.name} loading="lazy" /></button>
            <div class="meta"><b title={c.stem}>{c.stem}</b><span>{c.size_mb} MB</span></div>
          </div>
        {/each}
      </div>
    {/if}
  </section>

  <section class="card">
    <div class="section-head">
      <div><h2>4. Custom Dewarp Dataset</h2><p class="hint">Sample yang sudah di-commit ke training dataset. Audit, preview, atau kembalikan ke selected.</p></div>
      <div class="actions">
        <button class="btn" onclick={toggleCustomView}><i data-lucide={showCustom ? 'chevron-up' : 'chevron-down'}></i> {showCustom ? 'Tutup' : `Lihat (${customCount})`}</button>
      </div>
    </div>

    {#if showCustom}
      {#if customItems.length === 0}
        <div class="empty">Belum ada sample di custom dewarp dataset.</div>
      {:else}
        <div class="actions" style="margin-bottom:1rem">
          <button class="btn" onclick={() => customPicked = new Set(customItems.map(c => c.name))} disabled={customPicked.size === customItems.length}>Pilih semua ({customItems.length})</button>
          <button class="btn danger" onclick={() => removeCustom(false)} disabled={customPicked.size === 0}><i data-lucide="trash-2"></i> Hapus ({customPicked.size})</button>
          <button class="btn accent" onclick={() => removeCustom(true)} disabled={customPicked.size === 0}><i data-lucide="arrow-left"></i> Kembali ke Selected ({customPicked.size})</button>
        </div>
        <div class="cards">
          {#each customItems as c (c.name)}
            <div class="sample" class:selected={customPicked.has(c.name)}>
              <label class="check"><input type="checkbox" checked={customPicked.has(c.name)} onchange={() => toggleCustom(c.name)} /> pilih</label>
              <button class="imgbtn" onclick={() => preview = c}><img src={c.points_url} alt={c.name} loading="lazy" /></button>
              <div class="meta"><b title={c.stem}>{c.stem}</b><span>{c.size_mb} MB</span></div>
            </div>
          {/each}
        </div>
      {/if}
    {/if}
  </section>
{/if}

{#if preview}
  <div class="modal" onclick={() => preview = null}>
    <div class="modalbox" onclick={(e) => e.stopPropagation()}>
      <div class="modalhead">
        <h3>{preview.stem}</h3>
        <button class="btn close-btn" onclick={() => preview = null} aria-label="Close"><i data-lucide="x"></i><span>Close</span></button>
      </div>
      <div class="previewgrid">
        <div><b>Output</b><img src={preview.image_url} alt="output" /></div>
        <div><b>Control points</b><img src={preview.points_url} alt="points" /></div>
      </div>
      <p class="hint" style="text-align:center;margin-top:.75rem">Klik luar, tombol Close, atau tekan Esc untuk menutup.</p>
    </div>
  </div>
{/if}

<style>
  .page-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:1.5rem}.eyebrow{color:#38bdf8;text-transform:uppercase;font-size:.75rem;font-weight:700;letter-spacing:.12em;margin:0 0 .4rem}h1{font-size:2rem;margin:0;color:#f8fafc}h2{margin:0 0 1rem;color:#f8fafc}.page-desc,.hint{color:#94a3b8}.hint{font-size:.85rem}.card{background:rgba(15,23,42,.75);border:1px solid rgba(148,163,184,.18);border-radius:18px;padding:1.25rem;margin-bottom:1rem}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.btn{display:inline-flex;align-items:center;gap:.4rem;border:1px solid rgba(148,163,184,.25);background:#1e293b;color:#e2e8f0;border-radius:10px;padding:.6rem .9rem;cursor:pointer;font-weight:700}.btn:disabled{opacity:.5;cursor:not-allowed}.primary{background:#0284c7}.ok{background:#16a34a}.danger{background:#dc2626}.accent{background:#7c3aed}select,input[type=number]{width:100%;padding:.75rem;border-radius:10px;border:1px solid rgba(148,163,184,.25);background:#020617;color:#e2e8f0;margin-bottom:.75rem}.upload-btn{display:inline-flex;align-items:center;gap:.4rem;cursor:pointer;color:#cbd5e1;border:1px dashed rgba(148,163,184,.4);border-radius:10px;padding:.55rem .8rem;margin-bottom:.75rem}.clean-preview{height:300px;background:#020617;border-radius:12px;display:flex;align-items:center;justify-content:center;overflow:hidden}.clean-preview img{max-width:100%;max-height:100%;object-fit:contain}.progress-wrap{margin:.75rem 0}.progress-bar{height:10px;background:#020617;border-radius:999px;overflow:hidden;border:1px solid rgba(148,163,184,.2)}.progress-fill{height:100%;background:linear-gradient(90deg,#0ea5e9,#22c55e);transition:width .4s}.progress-meta{display:flex;justify-content:space-between;color:#94a3b8;font-size:.8rem;margin-top:.35rem}.stats{display:flex;gap:.6rem;flex-wrap:wrap;margin-top:1rem}.stats span{background:#020617;border:1px solid rgba(148,163,184,.18);border-radius:999px;padding:.35rem .65rem}.section-head{display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;flex-wrap:wrap}.actions{display:flex;gap:.5rem;flex-wrap:wrap}.empty{text-align:center;color:#94a3b8;padding:2rem}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:1rem}.sample{border:1px solid rgba(148,163,184,.18);border-radius:14px;background:#020617;padding:.6rem}.sample.selected{outline:2px solid #22c55e}.check{display:flex;gap:.35rem;color:#cbd5e1;font-size:.85rem;margin-bottom:.5rem}.check input{width:auto;margin:0}.imgbtn{width:100%;height:220px;border:0;background:#0f172a;border-radius:10px;cursor:pointer;padding:0;overflow:hidden}.imgbtn img{width:100%;height:100%;object-fit:contain}.meta{display:flex;justify-content:space-between;gap:.5rem;margin-top:.5rem;color:#94a3b8;font-size:.78rem}.meta b{color:#e2e8f0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.modal{position:fixed;inset:0;background:rgba(0,0,0,.85);display:flex;align-items:center;justify-content:center;z-index:1000;padding:1rem}.modalbox{background:#0f172a;border:1px solid rgba(148,163,184,.25);border-radius:16px;width:min(1200px,96vw);max-height:94vh;overflow:auto;padding:1rem;position:relative}.modalhead{display:flex;justify-content:space-between;align-items:center;gap:.75rem;position:sticky;top:-1rem;background:#0f172a;padding:.75rem 0;margin:-1rem 0 .5rem;z-index:2}.modalhead h3{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:1rem}.close-btn{flex-shrink:0;min-height:44px;min-width:44px;justify-content:center}.previewgrid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.previewgrid div{background:#020617;border-radius:12px;padding:.75rem}.previewgrid b{display:block;margin-bottom:.5rem;color:#e2e8f0}.previewgrid img{width:100%;max-height:78vh;object-fit:contain}@media(max-width:900px){.grid2,.previewgrid{grid-template-columns:1fr}.section-head,.page-head{flex-direction:column}.clean-preview{height:240px}}
</style>
