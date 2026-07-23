<script>
  import { onMount } from 'svelte'
  import { navigate, currentRoute } from '../lib/router.svelte.js'

  const API = ""
  let slug = ""
  let dataset = null
  let loading = true
  let error = null

  // Upload
  let shadowFile = null
  let cleanFile = null
  let shadowPreview = null
  let cleanPreview = null
  let auditResult = null
  let adding = false
  let added = false
  let validating = false
  let validationResult = null
  let previewPair = null

  $: {
    const parts = ($currentRoute || "").split("/")
    slug = parts[2] || ""
  }

  $: if (slug) loadDataset()

  onMount(() => { if (slug) loadDataset() })

  async function runValidate() {
    validating = true
    validationResult = null
    try {
      const res = await fetch(API + '/api/datasets/custom/' + slug + '/validate', {
        headers: { Authorization: 'Bearer ' + (localStorage.getItem('token') || '') }
      })
      if (!res.ok) throw new Error(await res.text())
      validationResult = await res.json()
    } catch (e) {
      error = e.message
    } finally {
      validating = false
    }
  }

function openPreview(pair) { previewPair = pair }
  function closePreview() { previewPair = null }

  async function loadDataset() {
    loading = true
    error = null
    try {
      const res = await fetch(`${API}/api/datasets/custom/${slug}`, {
        headers: { Authorization: `Bearer ${localStorage.getItem("token") || ""}` }
      })
      if (!res.ok) throw new Error("Dataset not found")
      dataset = await res.json()
    } catch (e) {
      error = e.message
    } finally {
      loading = false
    }
  }

  function handleShadow(e) {
    const file = e.target.files[0]
    if (!file) return
    shadowFile = file
    shadowPreview = URL.createObjectURL(file)
    auditResult = null
    added = false
  }

  function handleClean(e) {
    const file = e.target.files[0]
    if (!file) return
    cleanFile = file
    cleanPreview = URL.createObjectURL(file)
    auditResult = null
    added = false
  }

  async function runAudit() {
    if (!shadowFile || !cleanFile) return
    error = null
    auditResult = null
    const fd = new FormData()
    fd.append("shadow", shadowFile)
    fd.append("clean", cleanFile)
    try {
      const res = await fetch(`${API}/api/datasets/custom/${slug}/audit`, {
        method: "POST",
        headers: { Authorization: `Bearer ${localStorage.getItem("token") || ""}` },
        body: fd
      })
      if (!res.ok) throw new Error(await res.text())
      auditResult = await res.json()
    } catch (e) {
      error = e.message
    }
  }

  async function runAdd() {
    if (!shadowFile || !cleanFile) return
    adding = true
    error = null
    const fd = new FormData()
    fd.append("shadow", shadowFile)
    fd.append("clean", cleanFile)
    try {
      const res = await fetch(`${API}/api/datasets/custom/${slug}/add`, {
        method: "POST",
        headers: { Authorization: `Bearer ${localStorage.getItem("token") || ""}` },
        body: fd
      })
      if (!res.ok) throw new Error(await res.text())
      added = true
      shadowFile = null
      cleanFile = null
      shadowPreview = null
      cleanPreview = null
      auditResult = null
      await loadDataset()
    } catch (e) {
      error = e.message
    } finally {
      adding = false
    }
  }

  async function deletePair(filename) {
    if (!confirm(`Hapus pair ${filename}?`)) return
    try {
      await fetch(`${API}/api/datasets/custom/${slug}/${filename}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${localStorage.getItem("token") || ""}` }
      })
      await loadDataset()
    } catch (e) {
      error = e.message
    }
  }

  function gradeColor(g) {
    return g === "A" ? "#22c55e" : g === "B" ? "#eab308" : g === "C" ? "#f97316" : "#ef4444"
  }
</script>

<div class="page">
  <a class="back" href="/dataset-manager" onclick={(e) => { e.preventDefault(); navigate("/dataset-manager") }}>
    <i data-lucide="arrow-left"></i> Kembali
  </a>

  {#if error}
    <div class="alert error">{error}</div>
  {/if}

  {#if loading}
    <p class="loading">Loading...</p>
  {:else if !dataset}
    <p>Dataset tidak ditemukan.</p>
  {:else}
    <div class="header-row">
      <div>
        <h1>{dataset.name}</h1>
        <p class="slug">/{dataset.slug}</p>
        {#if dataset.description}
          <p class="desc">{dataset.description}</p>
        {/if}
      </div>
      <div class="stats">
        <span class="stat">{dataset.paired_count} pair</span>
        <span class="stat">{dataset.input_count} input</span>
        <span class="stat">{dataset.target_count} target</span>
      </div>
    </div>

    <!-- Add Pair -->
    <div class="section">
      <h2>Tambah Pair</h2>
      <div class="upload-row">
        <div class="upload-box">
          <label>Shadow (Input)</label>
          {#if shadowPreview}
            <img src={shadowPreview} alt="shadow" class="preview" />
          {:else}
            <div class="placeholder">Drop atau klik</div>
          {/if}
          <input type="file" accept="image/*" onchange={handleShadow} />
        </div>
        <div class="upload-box">
          <label>Clean (Target)</label>
          {#if cleanPreview}
            <img src={cleanPreview} alt="clean" class="preview" />
          {:else}
            <div class="placeholder">Drop atau klik</div>
          {/if}
          <input type="file" accept="image/*" onchange={handleClean} />
        </div>
      </div>

      <div class="action-row">
        <button class="btn" disabled={!shadowFile || !cleanFile || !!auditResult} onclick={runAudit}>
          {auditResult ? "Sudah Di-Audit" : "Audit"}
        </button>
        <button class="btn success" disabled={!auditResult || adding} onclick={runAdd}>
          {added ? "Tersimpan!" : adding ? "Menyimpan..." : "Audit + Simpan"}
        </button>
      </div>

      <div class="action-row" style="margin-top:1rem">
        <button class="btn" disabled={validating || dataset.paired_count === 0} onclick={runValidate}>
          {validating ? "Memvalidasi..." : "Validasi Semua Pair"}
        </button>
      </div>

      {#if validationResult}
        <div class="audit-card" style="margin-top:1rem">
          <h3 style="margin-bottom:0.75rem">Hasil Validasi ({validationResult.total_pairs} pair)</h3>
          <div class="audit-grid">
            <div class="audit-item">
              <span class="label">Average Score</span>
              <span class="value" style="font-weight:700">{validationResult.average_score}/100</span>
            </div>
            <div class="audit-item">
              <span class="label">Min / Max</span>
              <span class="value">{validationResult.min_score} / {validationResult.max_score}</span>
            </div>
            <div class="audit-item">
              <span class="label">Grade</span>
              <span class="value">
                {#each Object.entries(validationResult.grade_distribution || {}) as [g, c]}
                  {#if c > 0}
                    <span class="grade-badge" style="background:{g === 'A' ? '#22c55e' : g === 'B' ? '#eab308' : g === 'C' ? '#f97316' : '#ef4444'}">{g}: {c}</span>
                  {/if}
                {/each}
              </span>
            </div>
            <div class="audit-item">
              <span class="label">Siap Training</span>
              <span class="value">{validationResult.recommended ? "Ya" : "Perlu perbaikan"}</span>
            </div>
          </div>
          {#if validationResult.recommendations?.length}
            <div class="reasons" style="margin-top:0.75rem">
              {#each validationResult.recommendations as r}
                <span class="reason-tag">{r}</span>
              {/each}
            </div>
          {/if}
          {#if validationResult.pairs?.length}
            <details style="margin-top:0.75rem">
              <summary style="cursor:pointer;font-size:0.85rem;color:var(--text2)">Detail per pair ({validationResult.pairs.length})</summary>
              <div class="pairs-table" style="margin-top:0.5rem">
                {#each validationResult.pairs as p}
                  <div class="pair-row" class:unpaired={p.score < 50}>
                    <span class="pair-name">{p.name}</span>
                    <span class="grade-badge small" style="background:{p.grade === 'A' ? '#22c55e' : p.grade === 'B' ? '#eab308' : p.grade === 'C' ? '#f97316' : '#ef4444'}">{p.grade}</span>
                    <span class="pair-badge">{p.score}pts</span>
                    {#if p.issues?.length}
                      <span class="pair-badge" title={p.issues.join(', ')}>{p.issues.length} issue</span>
                    {/if}
                  </div>
                {/each}
              </div>
            </details>
          {/if}
        </div>
      {/if}

      {#if auditResult}
        <div class="audit-card">
          <div class="audit-header">
            <span class="grade" style="background:{gradeColor(auditResult.grade)}">{auditResult.grade}</span>
            <span class="score">Score: {auditResult.score}/100</span>
          </div>
          <div class="audit-grid">
            <div class="audit-item">
              <span class="label">Resolusi</span>
              <span class="value">
                {auditResult.dimensions.shadow.w}x{auditResult.dimensions.shadow.h} / {auditResult.dimensions.clean.w}x{auditResult.dimensions.clean.h}
                {auditResult.dimensions.match ? "Sama" : "Akan di-resize"}
              </span>
            </div>
            <div class="audit-item">
              <span class="label">Shadow</span>
              <span class="value">{auditResult.luminance.shadow_pct}% (shadow) vs {auditResult.luminance.clean_dark_pct}% (clean)</span>
            </div>
            <div class="audit-item">
              <span class="label">Alignment</span>
              <span class="value">
                {auditResult.alignment.success ? `${auditResult.alignment.inliers} inliers` : "Gagal"}
                {#if auditResult.alignment.success}
                  (tx={auditResult.alignment.tx.toFixed(1)}px)
                {/if}
              </span>
            </div>
            <div class="audit-item">
              <span class="label">Warna</span>
              <span class="value">R:{auditResult.color_diff.R.diff} G:{auditResult.color_diff.G.diff} B:{auditResult.color_diff.B.diff}</span>
            </div>
          </div>
          {#if auditResult.reasons.length > 0}
            <div class="reasons">
              {#each auditResult.reasons as r}
                <span class="reason-tag">{r}</span>
              {/each}
            </div>
          {/if}
        </div>
      {/if}
    </div>

    <!-- Pairs List -->
    <div class="section">
      <h2>Daftar Pair ({dataset.pairs?.length || 0})</h2>
      {#if !dataset.pairs || dataset.pairs.length === 0}
        <p class="empty-text">Belum ada pair. Upload di atas untuk menambah.</p>
      {:else}
        <div class="pairs-grid">
          {#each dataset.pairs as pair}
            <div class="pair-card" onclick={() => openPreview(pair)}>
              <button class="btn-icon danger delete-btn" onclick={(e) => { e.stopPropagation(); deletePair(pair.name + '.png') }} title="Hapus">
                <i data-lucide="trash-2"></i>
              </button>
              <div class="pair-thumbs">
                <div class="thumb-wrap">
                  <span class="thumb-label">Input</span>
                  <img src={API + '/api/datasets/explorer/image?dataset=' + encodeURIComponent('datasets/paired/custom/' + slug) + '&path=' + encodeURIComponent(pair.input)} alt="input" class="thumb" loading="lazy" />
                </div>
                {#if pair.target}
                  <div class="thumb-wrap">
                    <span class="thumb-label">Target</span>
                    <img src={API + '/api/datasets/explorer/image?dataset=' + encodeURIComponent('datasets/paired/custom/' + slug) + '&path=' + encodeURIComponent(pair.target)} alt="target" class="thumb" loading="lazy" />
                  </div>
                {:else}
                  <div class="thumb-wrap empty">No target</div>
                {/if}
              </div>
              <span class="pair-name">{pair.name}</span>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}

  {#if previewPair}
    <div class="lightbox" onclick={closePreview} tabindex="-1">
      <div class="lb-content" onclick={(e) => e.stopPropagation()}>
        <button class="lb-close" onclick={closePreview}>&times;</button>
        <h3>{previewPair.name}</h3>
        <div class="lb-pair">
          <div class="lb-side">
            <span class="thumb-label">Input</span>
            <img src={API + '/api/datasets/explorer/image?dataset=' + encodeURIComponent('datasets/paired/custom/' + slug) + '&path=' + encodeURIComponent(previewPair.input)} alt="input" class="lb-img" />
          </div>
          {#if previewPair.target}
            <div class="lb-side">
              <span class="thumb-label">Target</span>
              <img src={API + '/api/datasets/explorer/image?dataset=' + encodeURIComponent('datasets/paired/custom/' + slug) + '&path=' + encodeURIComponent(previewPair.target)} alt="target" class="lb-img" />
            </div>
          {/if}
        </div>
        <div class="lb-actions">
          <button class="btn danger" onclick={() => { deletePair(previewPair.name + '.png'); closePreview() }}>
            <i data-lucide="trash-2"></i> Hapus
          </button>
        </div>
      </div>
    </div>
  {/if}

</div>>

<style>
  .page { max-width: 1000px; margin: 0 auto; padding: 2rem; }
  .back { display: inline-flex; align-items: center; gap: 0.4rem; color: var(--text2); font-size: 0.85rem; margin-bottom: 1rem; text-decoration: none; }
  .back:hover { color: var(--text); }
  .header-row { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 2rem; }
  h1 { font-size: 1.4rem; }
  .slug { font-family: monospace; color: var(--accent); font-size: 0.85rem; }
  .desc { color: var(--text2); font-size: 0.85rem; margin-top: 0.3rem; }
  .stats { display: flex; gap: 0.75rem; }
  .stat { font-size: 0.8rem; padding: 0.3rem 0.75rem; background: var(--bg2); border: 1px solid var(--border); border-radius: 999px; }
  .section { margin-bottom: 2rem; }
  .section h2 { font-size: 1rem; margin-bottom: 0.75rem; }
  .upload-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1rem; }
  .upload-box { position: relative; border: 2px dashed var(--border); border-radius: var(--radius); padding: 1rem; text-align: center; }
  .upload-box label { display: block; font-size: 0.8rem; font-weight: 700; color: var(--text2); text-transform: uppercase; margin-bottom: 0.5rem; }
  .upload-box input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .preview { max-width: 100%; max-height: 200px; object-fit: contain; }
  .placeholder { color: var(--text2); font-size: 0.85rem; padding: 2rem; }
  .action-row { display: flex; gap: 0.75rem; margin-bottom: 1rem; }
  .btn { padding: 0.6rem 1.25rem; border: none; border-radius: var(--radius-xs); font-weight: 600; cursor: pointer; font-size: 0.85rem; background: var(--bg2); color: var(--text); border: 1px solid var(--border); }
  .btn:disabled { opacity: 0.5; }
  .btn.success { background: #22c55e; color: white; border: none; }
  .audit-card { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem; }
  .audit-header { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem; }
  .grade { display: inline-flex; align-items: center; justify-content: center; width: 2.2rem; height: 2.2rem; border-radius: 50%; color: white; font-weight: 800; font-size: 1.1rem; }
  .score { font-weight: 600; }
  .audit-grid { display: grid; gap: 0.4rem; }
  .audit-item { display: flex; gap: 0.75rem; font-size: 0.82rem; padding: 0.35rem 0; border-bottom: 1px solid var(--border); }
  .audit-item .label { font-weight: 600; min-width: 90px; color: var(--text2); }
  .reasons { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.5rem; }
  .grade-badge { display: inline-flex; padding: 0.15rem 0.5rem; border-radius: 999px; color: white; font-size: 0.72rem; font-weight: 700; margin-right: 0.3rem; }
  .grade-badge.small { padding: 0.1rem 0.4rem; font-size: 0.68rem; }
  .reason-tag { background: rgba(239,68,68,0.1); color: #ef4444; padding: 0.2rem 0.5rem; border-radius: 999px; font-size: 0.72rem; }
  .pairs-table { display: flex; flex-direction: column; gap: 0.3rem; }
  .pair-row { display: flex; align-items: center; gap: 1rem; padding: 0.5rem 0.75rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); }
  .pair-row.unpaired { opacity: 0.6; }
  .pair-name { font-family: monospace; font-size: 0.85rem; flex: 1; }
  .pair-badge { font-size: 0.7rem; padding: 0.15rem 0.5rem; border-radius: 999px; background: rgba(255,255,255,0.05); color: var(--text2); }
  .pair-badge.ok { background: rgba(34,197,94,0.15); color: #22c55e; }
  .btn-icon { background: none; border: none; color: var(--text2); cursor: pointer; padding: 0.25rem; }
  .btn-icon.danger:hover { color: #ef4444; }
  .alert { padding: 0.75rem 1rem; border-radius: var(--radius-xs); margin-bottom: 1rem; font-size: 0.85rem; }
  .alert.error { background: rgba(239,68,68,0.1); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
  .loading { color: var(--text2); }
  .empty-text { color: var(--text2); font-size: 0.85rem; }

  .pairs-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 0.75rem; margin-top: 0.75rem; }
  .pair-card { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 0.75rem; cursor: pointer; transition: border-color 0.15s; position: relative; }
  .pair-card:hover { border-color: var(--accent); }
  .pair-thumbs { display: flex; gap: 0.5rem; margin-bottom: 0.5rem; }
  .thumb-wrap { flex: 1; position: relative; background: var(--bg2); border-radius: var(--radius-xs); overflow: hidden; aspect-ratio: 4/3; }
  .thumb-wrap.empty { display: flex; align-items: center; justify-content: center; color: var(--text3); font-size: 0.7rem; }
  .thumb-label { position: absolute; top: 4px; left: 4px; font-size: 0.55rem; font-weight: 700; text-transform: uppercase; color: white; background: rgba(0,0,0,0.6); padding: 0.1rem 0.3rem; border-radius: 3px; z-index: 1; }
  .thumb { width: 100%; height: 100%; object-fit: contain; }
  .delete-btn { position: absolute; top: 6px; right: 6px; z-index: 2; background: rgba(0,0,0,0.5) !important; border-radius: 50% !important; width: 28px !important; height: 28px !important; min-height: 28px !important; }
  .delete-btn:hover { background: rgba(239,68,68,0.85) !important; }
  .lightbox { position: fixed; inset: 0; background: rgba(0,0,0,0.8); z-index: 1000; display: flex; align-items: center; justify-content: center; padding: 2rem; }
  .lb-content { background: var(--bg); border-radius: var(--radius); padding: 1.5rem; max-width: 900px; width: 100%; max-height: 90vh; overflow-y: auto; position: relative; }
  .lb-close { position: absolute; top: 0.75rem; right: 0.75rem; background: none; border: none; color: var(--text2); font-size: 1.5rem; cursor: pointer; }
  .lb-close:hover { color: var(--text); }
  .lb-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
  .lb-side { text-align: center; }
  .lb-img { max-width: 100%; max-height: 60vh; object-fit: contain; border-radius: var(--radius-xs); background: var(--bg2); }
  .lb-actions { margin-top: 1rem; text-align: center; }
  .btn.danger { background: #ef4444; color: white; border: none; padding: 0.5rem 1.25rem; border-radius: var(--radius-xs); font-weight: 600; cursor: pointer; }

</style>
