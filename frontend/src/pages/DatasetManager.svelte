<script>
  import { onMount } from 'svelte'
  import { navigate, currentRoute } from '../lib/router.svelte.js'

  const API = ""

  let datasets = []
  let loading = true
  let showCreate = false
  let newSlug = ""
  let newName = ""
  let newDesc = ""
  let creating = false
  let page = 1
  let pageSize = 9
  let error = null

  onMount(loadDatasets)

  async function loadDatasets() {
    loading = true
    error = null
    try {
      const res = await fetch(`${API}/api/datasets/custom`, {
        headers: { Authorization: `Bearer ${localStorage.getItem("token") || ""}` }
      })
      const data = await res.json()
      datasets = (data.datasets || []).sort((a,b) => (b.updated_at || 0) - (a.updated_at || 0))
      page = 1
    } catch (e) {
      error = e.message
    } finally {
      loading = false
    }
  }

  $: totalPages = Math.max(1, Math.ceil(datasets.length / pageSize))
  $: if (page > totalPages) page = totalPages
  $: pagedDatasets = datasets.slice((page - 1) * pageSize, page * pageSize)

  async function createDataset() {
    if (!newSlug.trim()) return
    creating = true
    error = null
    const fd = new FormData()
    fd.append("slug", newSlug.trim())
    fd.append("name", newName.trim() || newSlug.trim())
    fd.append("description", newDesc.trim())
    try {
      const res = await fetch(`${API}/api/datasets/custom`, {
        method: "POST",
        headers: { Authorization: `Bearer ${localStorage.getItem("token") || ""}` },
        body: fd
      })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      showCreate = false
      newSlug = ""
      newName = ""
      newDesc = ""
      navigate(`/dataset-manager/${data.slug}`)
    } catch (e) {
      error = e.message
    } finally {
      creating = false
    }
  }

  async function deleteDataset(slug) {
    if (!confirm(`Hapus dataset "${slug}"?`)) return
    try {
      await fetch(`${API}/api/datasets/custom/${slug}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${localStorage.getItem("token") || ""}` }
      })
      await loadDatasets()
    } catch (e) {
      error = e.message
    }
  }
</script>

<div class="page">
  <div class="header-row">
    <div>
      <h1>Dataset Manager</h1>
      <p class="subtitle">Buat & kelola dataset custom untuk training</p>
    </div>
    <button class="btn primary" onclick={() => showCreate = !showCreate}>
      {showCreate ? "Batal" : "+ Buat Dataset"}
    </button>
  </div>

  {#if error}
    <div class="alert error">{error}</div>
  {/if}

  {#if showCreate}
    <div class="create-card">
      <h3>Dataset Baru</h3>
      <div class="form-grid">
        <div class="form-field">
          <label>Slug (ID unik)</label>
          <input type="text" bind:value={newSlug} placeholder="my-shadow-dataset" />
        </div>
        <div class="form-field">
          <label>Nama</label>
          <input type="text" bind:value={newName} placeholder="My Shadow Dataset" />
        </div>
        <div class="form-field full">
          <label>Deskripsi</label>
          <input type="text" bind:value={newDesc} placeholder="Dataset untuk shadow hand..." />
        </div>
      </div>
      <button class="btn success" disabled={!newSlug.trim() || creating} onclick={createDataset}>
        {creating ? "Membuat..." : "Buat Dataset"}
      </button>
    </div>
  {/if}

  {#if loading}
    <p class="loading">Loading...</p>
  {:else if datasets.length === 0}
    <div class="empty">
      <p>Belum ada dataset.</p>
      <p class="hint">Klik "+ Buat Dataset" untuk mulai.</p>
    </div>
  {:else}
    <div class="grid">
      {#each pagedDatasets as ds}
        <div class="ds-card" onclick={() => navigate(`/dataset-manager/${ds.slug}`)}>
          <div class="ds-top">
            <span class="ds-status" class:ready={ds.status === 'ready'}>{ds.status}</span>
            <button class="btn-icon danger" onclick={(e) => { e.stopPropagation(); deleteDataset(ds.slug) }} title="Hapus">
              <i data-lucide="trash-2"></i>
            </button>
          </div>
          <h3 class="ds-name">{ds.name}</h3>
          <p class="ds-slug">/{ds.slug}</p>
          {#if ds.description}
            <p class="ds-desc">{ds.description}</p>
          {/if}
          <div class="ds-stats">
            <span>{ds.paired_count} pair</span>
            <span>{ds.input_count} input</span>
            <span>{ds.target_count} target</span>
          </div>
          <p class="ds-date">{ds.created_at}</p>
        </div>
      {/each}
    </div>
    <div class="pagination">
      <button disabled={page <= 1} onclick={() => page--}>Sebelumnya</button>
      <span>Halaman {page} / {totalPages} · {datasets.length} dataset</span>
      <button disabled={page >= totalPages} onclick={() => page++}>Berikutnya</button>
    </div>
  {/if}
</div>

<style>
  .page { max-width: 1000px; margin: 0 auto; padding: 2rem; }
  .header-row { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem; }
  h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
  .subtitle { color: var(--text2); font-size: 0.9rem; }
  .btn { padding: 0.6rem 1.25rem; border: none; border-radius: var(--radius-xs); font-weight: 600; cursor: pointer; font-size: 0.85rem; }
  .btn:disabled { opacity: 0.5; }
  .btn.primary { background: var(--accent); color: white; }
  .btn.success { background: #22c55e; color: white; margin-top: 1rem; }
  .create-card { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem; margin-bottom: 1.5rem; }
  .create-card h3 { font-size: 1rem; margin-bottom: 1rem; }
  .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .form-field { display: flex; flex-direction: column; gap: 0.3rem; }
  .form-field.full { grid-column: 1 / -1; }
  .form-field label { font-size: 0.75rem; font-weight: 700; color: var(--text2); text-transform: uppercase; }
  .form-field input { padding: 0.6rem 0.75rem; background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius-xs); color: var(--text); font-size: 0.85rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }
  .ds-card { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem; cursor: pointer; transition: border-color 0.15s; }
  .ds-card:hover { border-color: var(--accent); }
  .ds-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
  .ds-status { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; padding: 0.2rem 0.6rem; border-radius: 999px; background: rgba(255,255,255,0.05); color: var(--text2); }
  .ds-status.ready { background: rgba(34,197,94,0.15); color: #22c55e; }
  .ds-name { font-size: 1.05rem; margin-bottom: 0.15rem; }
  .ds-slug { font-size: 0.8rem; color: var(--accent); font-family: monospace; margin-bottom: 0.5rem; }
  .ds-desc { font-size: 0.82rem; color: var(--text2); margin-bottom: 0.5rem; }
  .ds-stats { display: flex; gap: 1rem; font-size: 0.78rem; color: var(--text2); margin-bottom: 0.3rem; }
  .ds-date { font-size: 0.72rem; color: var(--text3); }
  .btn-icon { background: none; border: none; color: var(--text2); cursor: pointer; padding: 0.3rem; border-radius: 4px; }
  .btn-icon.danger:hover { color: #ef4444; }
  .alert { padding: 0.75rem 1rem; border-radius: var(--radius-xs); margin-bottom: 1rem; font-size: 0.85rem; }
  .alert.error { background: rgba(239,68,68,0.1); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
  .pagination { display:flex; justify-content:center; align-items:center; gap:.75rem; margin-top:1.25rem; color:var(--text2); font-size:.8rem; }
  .pagination button { padding:.4rem .75rem; border:1px solid var(--border); background:var(--bg2); color:var(--text); border-radius:6px; cursor:pointer; }
  .pagination button:disabled { opacity:.4; cursor:not-allowed; }
  .loading { color: var(--text2); }
  .empty { text-align: center; padding: 3rem; color: var(--text2); }
  .empty .hint { font-size: 0.85rem; margin-top: 0.5rem; }
</style>
