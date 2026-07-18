<script>
  import { refreshIcons } from "../lib/icons.js"
  import { onMount } from 'svelte'
  import { apiJson } from '../stores/auth.js'
  import { showToast } from '../stores/toast.js'

  let tests = []
  let selectedTest = ''
  let selectedPath = 'input'
  let items = []
  let lightboxOpen = false
  let lightboxImages = []
  let lightboxIdx = 0

  onMount(loadTests)

  async function loadTests() {
    try {
      const d = await apiJson('/api/image-tests')
      tests = d.tests || []
      if (!selectedTest && tests.length) selectTest(tests[0].name)
      else if (selectedTest) loadFiles()
    } catch(e) {}
  }

  function selectTest(name) { selectedTest = name; selectedPath = 'input'; loadFiles() }

  async function loadFiles() {
    if (!selectedTest) return
    try {
      const d = await apiJson(`/api/image-tests/files?test=${encodeURIComponent(selectedTest)}&path=${encodeURIComponent(selectedPath)}`)
      items = d.items || []
    } catch(e) {}
  }

  function openFolder(path) { selectedPath = path; loadFiles() }

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

  async function uploadFile(e) {
    const file = e.target.files[0]
    if (!file || !selectedTest) return
    const body = new FormData()
    body.append('test', selectedTest)
    body.append('destination', 'input')
    body.append('file', file)
    try {
      const res = await fetch('/api/image-tests/upload', { method: 'POST', headers: { Authorization: `Bearer ${localStorage.getItem('docai_token')}` }, body })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Upload gagal')
      }
      showToast('Uploaded!')
      loadFiles()
    } catch(e) { showToast(e.message, 'error') }
    e.target.value = ''
  }

  function bytes(n) { if (n == null) return ''; return n < 1024 ? n + ' B' : n < 1048576 ? (n/1024).toFixed(1) + ' KB' : (n/1048576).toFixed(1) + ' MB' }
</script>

<div class="page-header">
  <h1>Image Test Workspace</h1>
  <p>Browse input/output test folders, preview and upload images.</p>
</div>

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
      <button class="btn btn-primary" onclick={() => document.getElementById('imgTestUpload').click()}>
        <i data-lucide="upload"></i> Upload
      </button>
      <button class="btn btn-secondary" onclick={loadFiles}><i data-lucide="refresh-cw"></i> Refresh</button>
      <span class="count-text">{items.length} items</span>
      <input type="file" id="imgTestUpload" accept="image/*" style="display:none" onchange={uploadFile} />
    </div>

    <div class="file-grid">
      {#each items as item, i}
        {#if item.type === 'folder'}
          <div class="file-card folder-card" onclick={() => openFolder(item.path)}>
            <i data-lucide="folder"></i>
            <span>{item.name}</span>
          </div>
        {:else if item.is_image}
          {@const images = items.filter(x => x.is_image)}
          <div class="file-card" onclick={() => openLB(images.indexOf(item))}>
            <img loading="lazy" src="/api/image-tests/preview?test={encodeURIComponent(selectedTest)}&path={encodeURIComponent(item.path)}" alt={item.name} />
            <div class="info"><span class="fname">{item.name}</span><span>{bytes(item.size)}</span></div>
          </div>
        {/if}
      {/each}
    </div>
    {#if !items.length}<div class="empty"><i data-lucide="folder-open"></i> Folder kosong</div>{/if}
  </div>
</div>

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
  .page-header { margin-bottom: 2rem; }
  .page-header h1 { font-size: 1.5rem; font-weight: 700; }
  .page-header p { color: var(--text2); margin-top: 0.25rem; font-size: 0.9rem; }
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
  .toolbar { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; }
  .count-text { font-size: 0.8rem; color: var(--text3); margin-left: auto; }
  .btn { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; border-radius: var(--radius-xs); font-size: 0.8rem; font-weight: 600; border: none; transition: all 0.15s; }
  .btn-primary { background: var(--primary); color: white; }
  .btn-secondary { background: var(--surface); color: var(--text); border: 1px solid var(--border); }
  .file-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 0.75rem; }
  .file-card { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); overflow: hidden; cursor: pointer; transition: all 0.15s; }
  .file-card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .file-card img { width: 100%; aspect-ratio: 3/4; object-fit: cover; display: block; background: var(--surface); }
  .file-card .info { padding: 0.5rem 0.75rem; font-size: 0.7rem; color: var(--text2); display: flex; justify-content: space-between; }
  .fname { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 70%; }
  .folder-card { display: flex; align-items: center; gap: 0.75rem; padding: 1rem; }
  .folder-card :global(i) { width: 28px; height: 28px; color: var(--warning); }
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
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
  @media (max-width: 768px) { .file-grid { grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); } }
</style>
