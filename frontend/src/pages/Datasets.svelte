<script>
  import { refreshIcons } from "../lib/icons.js"
  import { navigate, currentRoute } from '../lib/router.svelte.js'
  import { onMount } from 'svelte'
  import { apiJson } from '../stores/auth.js'
  import { showToast } from '../stores/toast.js'

  let datasets = []
  let loading = true
  let selectedCategory = 'all'
  let searchQuery = ''
  let selectedSlug = null
  let singleDataset = null
  let singleImages = []
  let singleLoading = false

  // Lightbox
  let lbOpen = false
  let lbIdx = 0

  const API = ''

  $: {
    const parts = ($currentRoute || '').split('/')
    const newSlug = (parts.length === 3 && parts[1] === 'datasets') ? parts[2] : null
    if (newSlug !== selectedSlug) {
      selectedSlug = newSlug
      if (selectedSlug) loadDetail()
      else { singleDataset = null; singleImages = [] }
    }
  }

  onMount(() => {
    if (!selectedSlug) loadDatasets()
    refreshIcons()
  })

  async function loadDatasets() {
    loading = true
    try {
      const d = await apiJson('/api/datasets')
      datasets = d.datasets || []
    } catch(e) { showToast('Gagal memuat dataset', 'error') }
    finally { loading = false }
  }

  async function loadDetail() {
    if (!selectedSlug) return
    singleLoading = true
    try {
      // Load dataset info
      const d = await apiJson('/api/datasets/validate?path=datasets/paired/' + selectedSlug)
      singleDataset = d

      // Load images via explorer
      const fm = await apiJson('/api/datasets/explorer?dataset=datasets/paired/' + selectedSlug + '&path=train/input&limit=200')
      const entries = fm.entries || []
      singleImages = entries.filter(e => e.kind === 'image')
    } catch(e) { showToast('Gagal memuat detail', 'error') }
    finally { singleLoading = false }
  }

  function getFiltered() {
    let items = datasets
    if (selectedCategory !== 'all') items = items.filter(d => d.kind === selectedCategory)
    if (searchQuery) {
      const q = searchQuery.toLowerCase()
      items = items.filter(d => d.name.toLowerCase().includes(q) || (d.path || '').toLowerCase().includes(q))
    }
    return items
  }

  function getTotalImages(ds) {
    return ds.kind === 'paired' ? (ds.pair_count || ds.train_pairs || 0) : (ds.image_count || 0)
  }

  function goBack() {
    navigate('/datasets')
    selectedSlug = null
    singleDataset = null
    singleImages = []
  }

  function openLB(idx) {
    lbIdx = idx
    lbOpen = true
  }
  function closeLB() { lbOpen = false }
  function navLB(dir) {
    lbIdx = (lbIdx + dir + singleImages.length) % singleImages.length
  }
  function onKeydown(e) {
    if (e.key === 'Escape') closeLB()
    if (e.key === 'ArrowLeft') navLB(-1)
    if (e.key === 'ArrowRight') navLB(1)
  }

  function imgURL(entry) {
    return `${API}/api/datasets/explorer/image?dataset=${encodeURIComponent('datasets/paired/' + selectedSlug)}&path=${encodeURIComponent(entry.path)}&size=200`
  }
</script>

<div class="page">
  {#if !selectedSlug}
    <!-- LIST VIEW -->
    <div class="page-header">
      <div>
        <h1>Dataset Browser</h1>
        <p class="subtitle">Jelajahi dataset yang tersedia.</p>
      </div>
      <button class="btn-refresh" onclick={loadDatasets}>
        <i data-lucide="refresh-cw"></i> Refresh
      </button>
    </div>

    <div class="filters">
      <div class="category-tabs">
        {#each [{ id: 'all', label: 'All' }, { id: 'paired', label: 'Paired' }, { id: 'clean', label: 'Clean' }] as cat}
          <button class="cat-tab" class:active={selectedCategory === cat.id} onclick={() => selectedCategory = cat.id}>
            {cat.label} ({selectedCategory === cat.id ? getFiltered().length : datasets.filter(d => cat.id === 'all' || d.kind === cat.id).length})
          </button>
        {/each}
      </div>
      <div class="search-box">
        <i data-lucide="search"></i>
        <input type="text" placeholder="Cari dataset..." bind:value={searchQuery} />
      </div>
    </div>

    {#if loading}
      <p class="loading-text">Loading...</p>
    {:else if getFiltered().length === 0}
      <p class="empty-text">Tidak ada dataset.</p>
    {:else}
      <div class="ds-grid">
        {#each getFiltered() as ds}
          <div class="ds-card" class:paired={ds.kind === 'paired'} onclick={() => navigate('/datasets/' + (ds.name || ds.path))}>
            <div class="ds-card-header">
              <span class="ds-badge" class:paired={ds.kind === 'paired'}>{ds.kind}</span>
            </div>
            <h3 class="ds-name">{ds.name}</h3>
            <div class="ds-stats">
              <span>{getTotalImages(ds)} {ds.kind === 'paired' ? 'pairs' : 'images'}</span>
              {#if ds.kind === 'paired' && ds.has_train_test}
                <span>Train + Test</span>
              {/if}
            </div>
            <div class="ds-progress">
              <div class="progress-fill" style="width: {ds.ready ? 100 : 0}%"></div>
            </div>
            <span class="ds-status" class:ready={ds.ready}>Ready: {ds.ready ? 'Yes' : 'No'}</span>
            <span class="ds-path">{ds.source || ds.path}</span>
          </div>
        {/each}
      </div>
    {/if}

  {:else}
    <!-- DETAIL VIEW -->
    <a class="back-link" href="/datasets" onclick={(e) => { e.preventDefault(); goBack() }}>
      <i data-lucide="arrow-left"></i> Kembali ke Datasets
    </a>

    {#if singleLoading}
      <p class="loading-text">Loading...</p>
    {:else if !singleDataset}
      <p class="empty-text">Dataset tidak ditemukan.</p>
    {:else}
      <div class="detail-header">
        <div>
          <h1>{singleDataset.name}</h1>
          <p class="detail-path">{singleDataset.path}</p>
        </div>
        <div class="detail-stats">
          <span class="stat-badge">{singleDataset.kind}</span>
          <span class="stat-badge">{getTotalImages(singleDataset)} images</span>
          {#if singleDataset.has_train_test}
            <span class="stat-badge">Train + Test</span>
          {/if}
          <span class="stat-badge" class:ready={singleDataset.ready}>{singleDataset.ready ? 'Ready' : 'Not Ready'}</span>
        </div>
      </div>

      {#if singleImages.length === 0}
        <p class="empty-text">Tidak ada gambar ditemukan.</p>
      {:else}
        <div class="thumb-grid">
          {#each singleImages as img, i}
            <div class="thumb-card" onclick={() => openLB(i)}>
              <img src={imgURL(img)} alt={img.name} loading="lazy" class="thumb-img" />
              <span class="thumb-name">{img.name}</span>
            </div>
          {/each}
        </div>
      {/if}
    {/if}
  {/if}
</div>

<!-- LIGHTBOX -->
{#if lbOpen && singleImages.length > 0}
  <div class="lb-overlay" onclick={closeLB} onkeydown={onKeydown} tabindex="-1" role="dialog">
    <button class="lb-close" onclick={closeLB}>×</button>
    {#if singleImages.length > 1}
      <button class="lb-prev" onclick={() => navLB(-1)}>‹</button>
      <button class="lb-next" onclick={() => navLB(1)}>›</button>
    {/if}
    <div class="lb-content" onclick={(e) => e.stopPropagation()}>
      <img src={imgURL(singleImages[lbIdx])} alt={singleImages[lbIdx]?.name} class="lb-img" />
      <div class="lb-info">
        <span>{singleImages[lbIdx]?.name}</span>
        <span class="lb-counter">{lbIdx + 1} / {singleImages.length}</span>
      </div>
    </div>
  </div>
{/if}

<style>
  .page { max-width: 1100px; margin: 0 auto; padding: 2rem; }
  .page-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem; }
  h1 { font-size: 1.4rem; margin-bottom: 0.2rem; }
  .subtitle { color: var(--text2); font-size: 0.9rem; }
  .btn-refresh { display: flex; align-items: center; gap: 0.4rem; padding: 0.5rem 1rem; background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius-xs); color: var(--text2); cursor: pointer; font-size: 0.85rem; }
  .filters { display: flex; gap: 1rem; align-items: center; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .category-tabs { display: flex; gap: 0.25rem; }
  .cat-tab { padding: 0.4rem 0.8rem; border: 1px solid var(--border); border-radius: 6px; background: transparent; color: var(--text2); font-size: 0.8rem; font-weight: 600; cursor: pointer; transition: all 0.15s; }
  .cat-tab:hover { border-color: var(--accent); color: var(--text); }
  .cat-tab.active { background: rgba(113,112,255,0.14); color: var(--accent); border-color: rgba(130,143,255,0.3); }
  .search-box { display: flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.75rem; background: var(--bg2); border: 1px solid var(--border); border-radius: 6px; }
  .search-box input { border: none; background: transparent; color: var(--text); font-size: 0.85rem; outline: none; min-width: 150px; }
  .ds-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 0.75rem; }
  .ds-card { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem; cursor: pointer; transition: border-color 0.15s; }
  .ds-card:hover { border-color: var(--accent); }
  .ds-card-header { display: flex; justify-content: space-between; margin-bottom: 0.4rem; }
  .ds-badge { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; padding: 0.15rem 0.5rem; border-radius: 999px; background: rgba(255,255,255,0.05); color: var(--text2); }
  .ds-badge.paired { background: rgba(99,102,241,0.15); color: #818cf8; }
  .ds-name { font-size: 1rem; margin-bottom: 0.3rem; }
  .ds-stats { display: flex; gap: 0.75rem; font-size: 0.78rem; color: var(--text2); margin-bottom: 0.4rem; }
  .ds-progress { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; margin-bottom: 0.3rem; }
  .progress-fill { height: 100%; background: var(--accent); border-radius: 2px; transition: width 0.3s; }
  .ds-status { font-size: 0.72rem; color: var(--text3); }
  .ds-status.ready { color: #22c55e; }
  .ds-path { display: block; font-size: 0.7rem; color: var(--text3); margin-top: 0.2rem; }
  .loading-text { color: var(--text2); font-size: 0.9rem; }
  .empty-text { color: var(--text2); text-align: center; padding: 2rem; }

  /* Detail view */
  .back-link { display: inline-flex; align-items: center; gap: 0.4rem; color: var(--text2); font-size: 0.85rem; margin-bottom: 1rem; text-decoration: none; }
  .back-link:hover { color: var(--text); }
  .detail-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem; }
  .detail-path { font-family: monospace; font-size: 0.8rem; color: var(--accent); }
  .detail-stats { display: flex; gap: 0.5rem; flex-wrap: wrap; }
  .stat-badge { font-size: 0.72rem; padding: 0.2rem 0.6rem; border-radius: 999px; background: var(--bg2); border: 1px solid var(--border); color: var(--text2); }
  .stat-badge.ready { background: rgba(34,197,94,0.12); color: #22c55e; border-color: rgba(34,197,94,0.3); }
  .thumb-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 0.75rem; }
  .thumb-card { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); overflow: hidden; cursor: pointer; transition: border-color 0.15s; }
  .thumb-card:hover { border-color: var(--accent); }
  .thumb-img { width: 100%; aspect-ratio: 4/3; object-fit: contain; background: var(--bg2); display: block; }
  .thumb-name { display: block; padding: 0.35rem 0.5rem; font-size: 0.7rem; font-family: monospace; color: var(--text2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  /* Lightbox */
  .lb-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 1000; display: flex; align-items: center; justify-content: center; }
  .lb-content { position: relative; max-width: 90vw; max-height: 90vh; }
  .lb-img { max-width: 100%; max-height: 80vh; object-fit: contain; }
  .lb-close { position: fixed; top: 1rem; right: 1.5rem; background: none; border: none; color: white; font-size: 2rem; cursor: pointer; z-index: 1001; }
  .lb-prev, .lb-next { position: fixed; top: 50%; transform: translateY(-50%); background: rgba(0,0,0,0.5); border: none; color: white; font-size: 2.5rem; cursor: pointer; padding: 0.5rem 1rem; z-index: 1001; }
  .lb-prev { left: 1rem; }
  .lb-next { right: 1rem; }
  .lb-prev:hover, .lb-next:hover { background: rgba(0,0,0,0.8); }
  .lb-info { display: flex; justify-content: space-between; align-items: center; color: var(--text2); font-size: 0.85rem; margin-top: 0.5rem; }
  .lb-counter { color: var(--text3); }
</style>
