<script>
  import { refreshIcons } from "../lib/icons.js"
  import { onMount } from 'svelte'
  import { apiJson } from '../stores/auth.js'
  import { showToast } from '../stores/toast.js'

  let datasets = []
  let loading = true
  let selectedCategory = 'all'
  let searchQuery = ''

  // File manager state
  let fmOpen = false
  let fmDataset = ''
  let fmPath = ''
  let fmEntries = []
  let fmLoading = false
  let fmView = 'grid'
  let fmImages = []
  let fmTotal = 0

  // Lightbox
  let lightboxOpen = false
  let lightboxIdx = 0
  let lightboxLoading = true

  const categories = [
    { id: 'all', label: 'All Datasets', icon: 'database' },
    { id: 'paired', label: 'Paired', icon: 'layers' },
    { id: 'clean', label: 'Clean', icon: 'image' },
  ]

  onMount(async () => {
    await loadDatasets()
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

  function getFiltered() {
    let items = datasets
    if (selectedCategory !== 'all') {
      items = items.filter(d => d.kind === selectedCategory)
    }
    if (searchQuery) {
      const q = searchQuery.toLowerCase()
      items = items.filter(d => d.name.toLowerCase().includes(q) || (d.path || '').toLowerCase().includes(q))
    }
    return items
  }

  function getTotalImages(ds) {
    if (ds.kind === 'paired') return ds.pair_count || ds.train_pairs || 0
    return ds.image_count || 0
  }

  function getBadgeClass(ds) {
    if (ds.kind === 'paired') return 'paired'
    if (ds.kind === 'clean') return 'clean'
    return 'unknown'
  }

  function openFM(ds) {
    fmDataset = ds.path || ds.name
    fmPath = ''
    fmOpen = true
    browsePath('')
  }

  function closeFM() {
    fmOpen = false
    fmEntries = []
    fmImages = []
  }

  async function browsePath(path) {
    fmLoading = true
    fmEntries = []
    try {
      const d = await apiJson(`/api/datasets/explorer?dataset=${encodeURIComponent(fmDataset)}&path=${encodeURIComponent(path)}&limit=120`)
      fmPath = d.path || ''
      fmEntries = d.entries || []
      fmTotal = d.total || 0
      fmImages = fmEntries.filter(e => e.kind === 'image')
    } catch(e) { showToast('Gagal memuat folder', 'error') }
    finally { fmLoading = false }
  }

  function openLB(idx) {
    lightboxIdx = idx
    lightboxOpen = true
    lightboxLoading = true
  }

  function closeLB() { lightboxOpen = false }

  function navLB(dir) {
    lightboxIdx = (lightboxIdx + dir + fmImages.length) % fmImages.length
    lightboxLoading = true
  }

  function onLBKeydown(e) {
    if (e.key === 'Escape') closeLB()
    if (e.key === 'ArrowLeft') navLB(-1)
    if (e.key === 'ArrowRight') navLB(1)
  }

  function onFMKeydown(e) {
    if (e.key === 'Backspace' && fmPath !== '') {
      e.preventDefault()
      goBack()
    }
  }

  function imgURL(img) {
    return `/api/datasets/explorer/image?dataset=${encodeURIComponent(fmDataset)}&path=${encodeURIComponent(img.path)}`
  }

  function fmtSize(n) {
    if (!n) return ''
    return n > 1048576 ? (n/1048576).toFixed(1)+' MB' : (n/1024).toFixed(0)+' KB'
  }

  function goBack() {
    const parts = fmPath ? fmPath.split('/').filter(Boolean) : []
    if (parts.length === 0) return
    browsePath(parts.slice(0, -1).join('/'))
  }

  function parentDir() {
    const parts = fmPath ? fmPath.split('/').filter(Boolean) : []
    return parts.length > 0 ? parts[parts.length - 2] || '...' : ''
  }

  function bcParts() {
    return fmPath ? fmPath.split('/').filter(Boolean) : []
  }

  function skeleton(count) {
    return Array.from({ length: count }, (_, i) => i)
  }
</script>

<div class="page-header">
  <div>
    <h1>Dataset Browser</h1>
    <p>Jelajahi, kelola, dan gunakan dataset untuk training model.</p>
  </div>
  <button class="btn-refresh" onclick={loadDatasets}>
    <i data-lucide="refresh-cw"></i> Refresh
  </button>
</div>

<!-- Stats -->
<div class="stats">
  <div class="stat-card stat-total">
    <div class="stat-icon"><i data-lucide="database"></i></div>
    <div class="stat-body">
      <div class="stat-value">{datasets.length}</div>
      <div class="stat-label">Total Datasets</div>
    </div>
  </div>
  <div class="stat-card stat-paired">
    <div class="stat-icon"><i data-lucide="layers"></i></div>
    <div class="stat-body">
      <div class="stat-value">{datasets.filter(d => d.kind === 'paired').length}</div>
      <div class="stat-label">Paired Datasets</div>
    </div>
  </div>
  <div class="stat-card stat-clean">
    <div class="stat-icon"><i data-lucide="image"></i></div>
    <div class="stat-body">
      <div class="stat-value">{datasets.reduce((s, d) => s + getTotalImages(d), 0).toLocaleString()}</div>
      <div class="stat-label">Total Images</div>
    </div>
  </div>
  <div class="stat-card stat-ready">
    <div class="stat-icon"><i data-lucide="check-circle"></i></div>
    <div class="stat-body">
      <div class="stat-value">{datasets.filter(d => d.ready || getTotalImages(d) > 0).length}</div>
      <div class="stat-label">Ready to Use</div>
    </div>
  </div>
</div>

<!-- Filters -->
<div class="filters">
  <div class="category-tabs">
    {#each categories as cat}
      <button
        class="cat-tab"
        class:active={selectedCategory === cat.id}
        onclick={() => selectedCategory = cat.id}
      >
        <i data-lucide={cat.icon}></i>
        {cat.label}
        <span class="cat-count">
          {cat.id === 'all' ? datasets.length : datasets.filter(d => d.kind === cat.id).length}
        </span>
      </button>
    {/each}
  </div>
  <div class="search-box">
    <i data-lucide="search"></i>
    <input
      type="text"
      placeholder="Cari dataset..."
      bind:value={searchQuery}
      oninput={refreshIcons}
    />
    {#if searchQuery}
      <button class="search-clear" onclick={() => searchQuery = ''}>
        <i data-lucide="x"></i>
      </button>
    {/if}
  </div>
</div>

<!-- Dataset Grid -->
{#if loading}
  <div class="dataset-grid">
    {#each skeleton(4) as i}
      <div class="skeleton-card">
        <div class="skeleton-icon"></div>
        <div class="skeleton-line w-60"></div>
        <div class="skeleton-line w-40"></div>
        <div class="skeleton-line w-80"></div>
      </div>
    {/each}
  </div>
{:else if getFiltered().length === 0}
  <div class="empty">
    <i data-lucide="folder-open"></i>
    <p class="empty-title">Tidak ada dataset ditemukan</p>
    <p class="empty-desc">Coba ubah filter atau tambahkan dataset baru.</p>
  </div>
{:else}
  <div class="dataset-grid">
    {#each getFiltered() as ds}
      <div class="dataset-card" class:paired={ds.kind === 'paired'} onclick={() => openFM(ds)}>
        <div class="card-glow"></div>
        <div class="card-header">
          <div class="card-icon-wrap">
            <i data-lucide={ds.kind === 'paired' ? 'layers' : 'image'}></i>
          </div>
          <div class="card-badge {getBadgeClass(ds)}">
            {ds.kind}
          </div>
        </div>
        <div class="card-body">
          <h3 class="card-title">{ds.name}</h3>
          <div class="card-meta">
            <span class="meta-item">
              <i data-lucide="file"></i>
              {ds.kind === 'paired'
                ? (ds.pair_count || ds.train_pairs || 0) + ' pairs'
                : (ds.image_count || 0) + ' images'}
            </span>
            {#if ds.kind === 'paired'}
              <span class="meta-item">
                <i data-lucide="layers"></i>
                {ds.has_train_test ? 'Train + Test' : 'Train only'}
              </span>
            {/if}
          </div>
          {#if ds.kind === 'paired'}
            <div class="progress-bar">
              <div class="progress-fill" style="width: {ds.ready ? 100 : 0}%"></div>
            </div>
            <div class="progress-label">
              <span>Ready: {ds.ready ? 'Yes' : 'No'}</span>
              {#if ds.test_count != null}
                <span>{ds.train_pairs || 0} train · {ds.test_count} test</span>
              {/if}
            </div>
          {:else}
            <div class="progress-bar muted">
              <div class="progress-fill" style="width: {ds.image_count > 0 ? 100 : 0}%"></div>
            </div>
          {/if}
        </div>
        <div class="card-footer">
          <span class="card-path">{ds.source || ds.path}</span>
          <i data-lucide="chevron-right" class="card-arrow"></i>
        </div>
      </div>
    {/each}
  </div>
{/if}

<!-- File Manager Panel -->
{#if fmOpen}
  <div class="fm-overlay" onclick={closeFM}></div>
  <div class="fm-panel" id="fmPanel" onkeydown={onFMKeydown}>
    <div class="fm-toolbar">
      <div class="fm-title">
        <i data-lucide="folder-open"></i>
        <span>{fmDataset}</span>
      </div>
      <div class="breadcrumb">
        <button class="bc-back" onclick={goBack} disabled={fmPath === ''} title="Kembali ke folder sebelumnya">
          <i data-lucide="arrow-left"></i>
        </button>
        <button class="bc-root" onclick={() => browsePath('')}>
          <i data-lucide="home"></i>
        </button>
        {#each bcParts() as part, i}
          <span class="bc-sep"><i data-lucide="chevron-right"></i></span>
          <button
            class="bc-part"
            class:active={i === bcParts().length - 1}
            onclick={() => browsePath(bcParts().slice(0, i + 1).join('/'))}
          >{part}</button>
        {/each}
      </div>
      <div class="fm-actions">
        <div class="view-toggle">
          <button class:active={fmView === 'grid'} onclick={() => fmView = 'grid'} title="Grid view">
            <i data-lucide="grid-3x3"></i>
          </button>
          <button class:active={fmView === 'list'} onclick={() => fmView = 'list'} title="List view">
            <i data-lucide="list"></i>
          </button>
        </div>
        <span class="fm-count">{fmEntries.length} / {fmTotal} items</span>
        <button class="fm-close" onclick={closeFM} title="Close">
          <i data-lucide="x"></i>
        </button>
      </div>
    </div>

    <div class="fm-body">
      {#if fmLoading}
        <div class="fm-loading">
          <div class="spinner"></div>
          <span>Loading...</span>
        </div>
      {:else if fmEntries.length === 0}
        <div class="fm-empty">
          <i data-lucide="folder-open"></i>
          <p>Folder kosong</p>
        </div>
      {:else if fmView === 'grid'}
        <div class="fm-grid">
          {#each fmEntries as entry}
            {#if entry.kind === 'folder'}
              <div class="fm-item folder" onclick={() => browsePath(entry.path)}>
                <div class="folder-icon-wrap">
                  <i data-lucide="folder"></i>
                </div>
                <span class="fm-item-name">{entry.name}</span>
              </div>
            {:else}
              <div class="fm-item image" onclick={() => openLB(fmImages.indexOf(entry))}>
                <div class="img-wrap">
                  <img class="thumb" src={imgURL(entry)} loading="lazy" alt={entry.name} />
                </div>
                <div class="img-info">
                  <span class="img-name">{entry.name}</span>
                  <span class="img-size">{fmtSize(entry.size)}</span>
                </div>
              </div>
            {/if}
          {/each}
        </div>
      {:else}
        <div class="fm-list">
          {#each fmEntries as entry}
            {#if entry.kind === 'folder'}
              <div class="fm-list-item folder" onclick={() => browsePath(entry.path)}>
                <div class="list-icon"><i data-lucide="folder"></i></div>
                <span class="list-name">{entry.name}</span>
                <span class="list-meta">Folder</span>
                <i data-lucide="chevron-right" class="list-arrow"></i>
              </div>
            {:else}
              <div class="fm-list-item image" onclick={() => openLB(fmImages.indexOf(entry))}>
                <img class="list-thumb" src={imgURL(entry)} alt={entry.name} />
                <span class="list-name">{entry.name}</span>
                <span class="list-meta">{fmtSize(entry.size)}</span>
                <i data-lucide="eye" class="list-arrow"></i>
              </div>
            {/if}
          {/each}
        </div>
      {/if}
    </div>
  </div>
{/if}

<!-- Lightbox -->
{#if lightboxOpen}
  <div class="lightbox-overlay" onclick={closeLB} onkeydown={onLBKeydown} tabindex="0" role="dialog">
    <button class="lb-close" onclick={closeLB}><i data-lucide="x"></i></button>
    {#if fmImages.length > 1}
      <button class="lb-nav prev" onclick={(e) => { e.stopPropagation(); navLB(-1) }}><i data-lucide="chevron-left"></i></button>
      <button class="lb-nav next" onclick={(e) => { e.stopPropagation(); navLB(1) }}><i data-lucide="chevron-right"></i></button>
    {/if}
    <div class="lb-img-wrap">
      {#if lightboxLoading}
        <div class="lb-spinner"><div class="spinner"></div></div>
      {/if}
      <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
      <img
        class="lb-img"
        src={imgURL(fmImages[lightboxIdx])}
        alt={fmImages[lightboxIdx]?.name}
        onload={() => lightboxLoading = false}
        onerror={() => lightboxLoading = false}
        class:loading={lightboxLoading}
        onclick={(e) => e.stopPropagation()}
      />
    </div>
    <div class="lb-info">
      <span class="lb-name">{fmImages[lightboxIdx]?.name}</span>
      <span class="lb-counter">{lightboxIdx + 1} / {fmImages.length}</span>
    </div>
  </div>
{/if}

<div class="footer">DocAI Dataset Browser v2.0</div>

<style>
  .page-header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 1.5rem; }
  .page-header h1 { font-size: 1.5rem; font-weight: 700; }
  .page-header p { color: var(--text2); margin-top: 0.25rem; font-size: 0.9rem; }
  .btn-refresh { display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-xs); color: var(--text2); font-size: 0.8rem; transition: all 0.15s; }
  .btn-refresh:hover { border-color: var(--accent); color: var(--text); }

  /* Stats */
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
  .stat-card { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 1.25rem; display: flex; align-items: center; gap: 1rem; transition: all 0.2s; }
  .stat-card:hover { transform: translateY(-2px); border-color: var(--accent); }
  .stat-icon { width: 44px; height: 44px; border-radius: var(--radius-xs); display: grid; place-items: center; flex-shrink: 0; }
  .stat-total .stat-icon { background: rgba(99,102,241,0.15); color: var(--primary); }
  .stat-paired .stat-icon { background: rgba(16,185,129,0.15); color: var(--success); }
  .stat-clean .stat-icon { background: rgba(6,182,212,0.15); color: var(--accent); }
  .stat-ready .stat-icon { background: rgba(245,158,11,0.15); color: var(--warning); }
  .stat-value { font-size: 1.75rem; font-weight: 700; line-height: 1; }
  .stat-label { font-size: 0.75rem; color: var(--text2); margin-top: 0.25rem; }

  /* Filters */
  .filters { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .category-tabs { display: flex; gap: 0.25rem; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-xs); padding: 0.25rem; }
  .cat-tab { display: flex; align-items: center; gap: 0.375rem; padding: 0.45rem 0.75rem; border: none; background: transparent; color: var(--text2); font-size: 0.8rem; font-weight: 500; border-radius: 6px; cursor: pointer; transition: all 0.15s; }
  .cat-tab:hover { color: var(--text); }
  .cat-tab.active { background: var(--primary); color: white; }
  .cat-tab :global(i) { width: 14px; height: 14px; }
  .cat-count { font-size: 0.65rem; padding: 0.1rem 0.4rem; border-radius: 999px; background: rgba(255,255,255,0.1); }
  .cat-tab.active .cat-count { background: rgba(255,255,255,0.2); }
  .search-box { display: flex; align-items: center; gap: 0.5rem; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-xs); padding: 0.45rem 0.75rem; flex: 1; max-width: 320px; transition: border-color 0.15s; }
  .search-box:focus-within { border-color: var(--accent); }
  .search-box :global(i) { width: 16px; height: 16px; color: var(--text3); flex-shrink: 0; }
  .search-box input { flex: 1; background: none; border: none; color: var(--text); font-size: 0.85rem; outline: none; }
  .search-box input::placeholder { color: var(--text3); }
  .search-clear { background: none; border: none; color: var(--text3); padding: 0; display: flex; cursor: pointer; }
  .search-clear:hover { color: var(--text); }

  /* Dataset Grid */
  .dataset-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
  .dataset-card { position: relative; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; cursor: pointer; transition: all 0.25s; }
  .dataset-card:hover { transform: translateY(-4px); border-color: var(--accent); box-shadow: 0 8px 30px rgba(6,182,212,0.08); }
  .dataset-card.paired:hover { border-color: var(--success); box-shadow: 0 8px 30px rgba(16,185,129,0.08); }
  .card-glow { position: absolute; inset: 0; opacity: 0; transition: opacity 0.3s; background: radial-gradient(600px circle at var(--mouse-x,50%) var(--mouse-y,50%), rgba(6,182,212,0.06), transparent 60%); pointer-events: none; }
  .dataset-card:hover .card-glow { opacity: 1; }
  .card-header { display: flex; align-items: center; justify-content: space-between; padding: 1rem 1.25rem 0; }
  .card-icon-wrap { width: 40px; height: 40px; border-radius: var(--radius-xs); background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(6,182,212,0.15)); display: grid; place-items: center; color: var(--accent); }
  .dataset-card.paired .card-icon-wrap { background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.15)); color: var(--success); }
  .card-badge { padding: 0.2rem 0.625rem; border-radius: 999px; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }
  .card-badge.paired { background: rgba(16,185,129,0.15); color: var(--success); }
  .card-badge.clean { background: rgba(99,102,241,0.15); color: var(--primary); }
  .card-badge.unknown { background: rgba(113,113,122,0.15); color: var(--text3); }
  .card-body { padding: 0.75rem 1.25rem; }
  .card-title { font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .card-meta { display: flex; gap: 1rem; font-size: 0.75rem; color: var(--text2); margin-bottom: 0.75rem; flex-wrap: wrap; }
  .meta-item { display: flex; align-items: center; gap: 0.3rem; }
  .meta-item :global(i) { width: 12px; height: 12px; }
  .progress-bar { height: 4px; background: var(--border); border-radius: 999px; overflow: hidden; margin-bottom: 0.375rem; }
  .progress-bar.muted .progress-fill { background: var(--text3); }
  .progress-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--primary), var(--accent)); transition: width 0.5s ease; }
  .dataset-card.paired .progress-fill { background: linear-gradient(90deg, var(--success), #34d399); }
  .progress-label { display: flex; justify-content: space-between; font-size: 0.65rem; color: var(--text3); }
  .card-footer { display: flex; align-items: center; justify-content: space-between; padding: 0.625rem 1.25rem; background: var(--bg2); border-top: 1px solid var(--border); }
  .card-path { font-size: 0.65rem; color: var(--text3); font-family: monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .card-arrow { width: 14px; height: 14px; color: var(--text3); flex-shrink: 0; transition: transform 0.2s; }
  .dataset-card:hover .card-arrow { transform: translateX(3px); color: var(--accent); }

  /* Skeleton */
  .skeleton-card { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem; }
  .skeleton-icon { width: 40px; height: 40px; border-radius: var(--radius-xs); background: var(--surface); margin-bottom: 1rem; animation: pulse 1.5s infinite; }
  .skeleton-line { height: 12px; background: var(--surface); border-radius: 4px; margin-bottom: 0.625rem; animation: pulse 1.5s infinite; }
  .skeleton-line.w-60 { width: 60%; }
  .skeleton-line.w-40 { width: 40%; }
  .skeleton-line.w-80 { width: 80%; }
  @keyframes pulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 0.8; } }

  /* Empty */
  .empty { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 4rem 0; color: var(--text3); gap: 0.75rem; }
  .empty :global(i) { width: 64px; height: 64px; opacity: 0.2; }
  .empty-title { font-size: 1.1rem; font-weight: 600; color: var(--text2); }
  .empty-desc { font-size: 0.85rem; }

  /* File Manager */
  .fm-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.6); z-index: 300; animation: fadeIn 0.2s; }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
  .fm-panel { position: fixed; inset: 5vh 5vw; z-index: 301; background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius); display: flex; flex-direction: column; overflow: hidden; animation: slideUp 0.25s ease; }
  @keyframes slideUp { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
  @media (max-width: 768px) { .fm-panel { inset: 0; border-radius: 0; } }
  .fm-toolbar { display: flex; flex-direction: column; gap: 0.5rem; padding: 0.75rem 1rem; background: var(--bg3); border-bottom: 1px solid var(--border); }
  .fm-title { display: flex; align-items: center; gap: 0.5rem; font-weight: 600; font-size: 0.9rem; }
  .fm-title :global(i) { width: 16px; height: 16px; color: var(--warning); }
  .breadcrumb { display: flex; align-items: center; gap: 0.125rem; overflow-x: auto; white-space: nowrap; padding: 0.25rem 0; }
  .bc-back { display: flex; align-items: center; background: none; border: none; color: var(--text2); cursor: pointer; padding: 0.25rem 0.4rem; border-radius: 4px; transition: all 0.15s; }
  .bc-back:hover { background: var(--bg); color: var(--accent2); }
  .bc-back:disabled { opacity: 0.3; cursor: default; }
  .bc-back:disabled:hover { background: none; color: var(--text2); }
  .bc-back :global(i) { width: 16px; height: 16px; }
  .bc-root { display: flex; align-items: center; background: none; border: none; color: var(--accent2); cursor: pointer; padding: 0.25rem; border-radius: 4px; }
  .bc-root:hover { background: rgba(6,182,212,0.1); }
  .bc-root :global(i) { width: 14px; height: 14px; }
  .bc-sep { display: flex; align-items: center; color: var(--text3); }
  .bc-sep :global(i) { width: 12px; height: 12px; }
  .bc-part { background: none; border: none; color: var(--text2); cursor: pointer; font-size: 0.8rem; padding: 0.2rem 0.4rem; border-radius: 4px; white-space: nowrap; }
  .bc-part:hover { background: var(--bg); color: var(--text); }
  .bc-part.active { color: var(--accent2); font-weight: 600; }
  .fm-actions { display: flex; align-items: center; gap: 0.75rem; }
  .view-toggle { display: flex; border: 1px solid var(--border); border-radius: var(--radius-xs); overflow: hidden; }
  .view-toggle button { background: none; border: none; color: var(--text3); padding: 0.375rem 0.5rem; display: flex; cursor: pointer; transition: all 0.15s; }
  .view-toggle button:hover { color: var(--text); }
  .view-toggle button.active { background: var(--primary); color: white; }
  .fm-count { font-size: 0.7rem; color: var(--text3); white-space: nowrap; }
  .fm-close { background: none; border: 1px solid var(--border); color: var(--text2); padding: 0.375rem; border-radius: var(--radius-xs); display: flex; cursor: pointer; transition: all 0.15s; margin-left: auto; }
  .fm-close:hover { border-color: var(--error); color: var(--error); }

  .fm-body { flex: 1; overflow-y: auto; padding: 1rem; }
  .fm-loading { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 1rem; padding: 3rem; color: var(--text3); }
  .fm-empty { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 0.75rem; padding: 3rem; color: var(--text3); }
  .fm-empty :global(i) { width: 48px; height: 48px; opacity: 0.3; }

  .fm-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 0.75rem; }
  .fm-item { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-xs); overflow: hidden; cursor: pointer; transition: all 0.2s; }
  .fm-item:hover { transform: translateY(-2px); border-color: var(--accent); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
  .fm-item.folder { display: flex; align-items: center; gap: 0.75rem; padding: 1rem; }
  .fm-item.folder:hover { border-color: var(--warning); }
  .folder-icon-wrap { width: 36px; height: 36px; border-radius: var(--radius-xs); background: rgba(245,158,11,0.1); display: grid; place-items: center; flex-shrink: 0; }
  .folder-icon-wrap :global(i) { width: 20px; height: 20px; color: var(--warning); }
  .fm-item-name { font-size: 0.8rem; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .img-wrap { width: 100%; aspect-ratio: 4/3; overflow: hidden; background: var(--surface); }
  .thumb { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform 0.3s; }
  .fm-item:hover .thumb { transform: scale(1.05); }
  .img-info { padding: 0.5rem 0.75rem; display: flex; justify-content: space-between; align-items: center; }
  .img-name { font-size: 0.7rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 65%; color: var(--text); }
  .img-size { font-size: 0.65rem; color: var(--text3); flex-shrink: 0; }

  /* List view */
  .fm-list { display: flex; flex-direction: column; gap: 2px; }
  .fm-list-item { display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0.75rem; background: var(--bg3); border: 1px solid transparent; border-radius: var(--radius-xs); cursor: pointer; transition: all 0.15s; }
  .fm-list-item:hover { border-color: var(--border); background: var(--bg); }
  .list-icon { width: 24px; height: 24px; display: grid; place-items: center; color: var(--warning); flex-shrink: 0; }
  .list-thumb { width: 36px; height: 36px; object-fit: cover; border-radius: 4px; background: var(--surface); flex-shrink: 0; }
  .list-name { flex: 1; font-size: 0.8rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .list-meta { font-size: 0.7rem; color: var(--text3); flex-shrink: 0; }
  .list-arrow { width: 14px; height: 14px; color: var(--text3); flex-shrink: 0; }
  .fm-list-item:hover .list-arrow { color: var(--accent); }

  /* Lightbox */
  .lightbox-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.95); z-index: 500; display: flex; align-items: center; justify-content: center; flex-direction: column; animation: fadeIn 0.2s; }
  .lb-close { position: absolute; top: 1rem; right: 1rem; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.1); color: white; padding: 0.5rem; border-radius: 50%; z-index: 10; cursor: pointer; transition: all 0.15s; display: flex; }
  .lb-close:hover { background: rgba(255,255,255,0.15); }
  .lb-nav { position: absolute; top: 50%; transform: translateY(-50%); background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.1); color: white; padding: 0.75rem; border-radius: 50%; cursor: pointer; transition: all 0.15s; display: flex; z-index: 10; }
  .lb-nav:hover { background: rgba(255,255,255,0.15); }
  .lb-nav.prev { left: 1.5rem; }
  .lb-nav.next { right: 1.5rem; }
  .lb-img-wrap { position: relative; max-width: 90vw; max-height: 80vh; display: flex; align-items: center; justify-content: center; }
  .lb-img { max-width: 90vw; max-height: 80vh; object-fit: contain; border-radius: 4px; transition: opacity 0.3s; }
  .lb-img.loading { opacity: 0; }
  .lb-spinner { position: absolute; }
  .lb-info { position: absolute; bottom: -2.5rem; color: rgba(255,255,255,0.6); font-size: 0.8rem; display: flex; gap: 1.5rem; }
  .lb-name { max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  .footer { text-align: center; padding: 1.5rem 0; color: var(--text3); font-size: 0.75rem; }

  @media (max-width: 768px) {
    .stats { grid-template-columns: repeat(2, 1fr); }
    .dataset-grid { grid-template-columns: 1fr; }
    .fm-grid { grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); }
    .cat-tab span:not(:first-child) { display: none; }
  }
</style>
