<script>
  import { refreshIcons } from "../lib/icons.js"
  import { onMount, onDestroy } from 'svelte'
  import { apiJson } from '../stores/auth.js'
  import { showToast } from '../stores/toast.js'

  let datasets = []
  let selectedDatasets = []
  let status = { running: false, epochs: 0, has_history: false, best_exists: false, history: { train_loss: [], val_loss: [], val_psnr: [], val_ssim: [] } }
  let log = ''
  let logOffset = 0
  let pollTimer = null
  let training = false

  const config = { epochs: 100, batchSize: 8, size: 512, lr: 0.0002, baseChannels: 32 }

  onMount(async () => {
    refreshIcons()
    await loadDatasets()
    await refreshStatus()
  })

  onDestroy(() => { if (pollTimer) clearInterval(pollTimer) })

  async function loadDatasets() {
    try {
      const d = await apiJson('/api/datasets')
      datasets = d.datasets || []
      // Auto-select all paired datasets
      selectedDatasets = datasets.filter(ds => ds.kind === 'paired' && ds.ready).map(ds => ds.path)
    } catch(e) {}
  }

  async function refreshStatus() {
    try {
      status = await apiJson('/api/training/status')
      training = status.running
    } catch(e) {}
  }

  function toggleDataset(path) {
    if (selectedDatasets.includes(path)) {
      selectedDatasets = selectedDatasets.filter(p => p !== path)
    } else {
      selectedDatasets = [...selectedDatasets, path]
    }
  }

  async function startTraining() {
    if (selectedDatasets.length === 0) {
      showToast('Pilih minimal satu dataset', 'error')
      return
    }

    try {
      const formData = new FormData()
      // Send all selected datasets as comma-separated
      formData.append('paired_data', selectedDatasets.join(','))
      formData.append('epochs', String(config.epochs))
      formData.append('batch_size', String(config.batchSize))
      formData.append('size', String(config.size))
      formData.append('lr', String(config.lr))
      formData.append('base_channels', String(config.baseChannels))
      formData.append('output', 'checkpoints/document_restorer')

      const token = localStorage.getItem('docai_token')
      const res = await fetch('/api/training/start', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Gagal start training')
      }
      showToast('Training started!')
      training = true
      log = ''
      logOffset = 0
      pollTimer = setInterval(pollLog, 2000)
      pollLog()
    } catch(e) { showToast(e.message, 'error') }
  }

  async function stopTraining() {
    try {
      const token = localStorage.getItem('docai_token')
      await fetch('/api/training/stop', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` }
      })
      showToast('Training stopped')
      training = false
      if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
      refreshStatus()
    } catch(e) { showToast(e.message, 'error') }
  }

  async function reloadAndTest() {
    try {
      showToast('Loading model from best.pth...')
      const token = localStorage.getItem('docai_token')
      const res = await fetch('/api/model/reload', { method: 'POST', headers: { Authorization: `Bearer ${token}` } })
      const d = await res.json()
      if (d.success) {
        showToast('Model loaded! Beralih ke Dashboard untuk testing...')
        window.history.pushState({}, '', '/dashboard')
        setTimeout(() => window.history.pushState({}, '', '/test'), 200)
      } else {
        showToast('Gagal load model: ' + (d.detail || 'unknown'), 'error')
      }
    } catch(e) { showToast(e.message, 'error') }
  }

  async function pollLog() {
    try {
      const r = await apiJson(`/api/training/log?offset=${logOffset}`)
      if (r.lines && r.lines.length) {
        log += r.lines.join('\n') + '\n'
        logOffset = r.total
        // Auto-scroll
        setTimeout(() => {
          const el = document.querySelector('.log-box')
          if (el) el.scrollTop = el.scrollHeight
        }, 50)
      }
      if (!r.running) {
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
        training = false
        refreshStatus()
        showToast('✅ Training selesai! Model tersimpan di checkpoints/document_restorer')
      }
    } catch(e) {}
  }

  function getTotalPairs() {
    return selectedDatasets.reduce((sum, path) => {
      const ds = datasets.find(d => d.path === path)
      return sum + (ds ? (ds.pair_count || ds.train_pairs || 0) : 0)
    }, 0)
  }

  function getTotalTest() {
    return selectedDatasets.reduce((sum, path) => {
      const ds = datasets.find(d => d.path === path)
      return sum + (ds ? (ds.test_count || 0) : 0)
    }, 0)
  }

  // Chart helpers
  function chartData(values, color = '#6366f1') {
    if (!values || values.length < 2) return ''
    const max = Math.max(...values)
    const min = Math.min(...values)
    const range = max - min || 1
    const w = 400
    const h = 120
    const pts = values.map((v, i) => {
      const x = (i / (values.length - 1)) * w
      const y = h - ((v - min) / range) * (h - 10) - 5
      return `${x},${y}`
    }).join(' ')
    return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"/>`
  }
</script>

<div class="page-header">
  <div>
    <h1>Training</h1>
    <p>Latih model document restoration dengan dataset yang tersedia.</p>
  </div>
  <div class="header-badges">
    {#if status.best_exists}
      <span class="badge success"><i data-lucide="check-circle"></i> Model Ready</span>
    {/if}
    {#if training}
      <span class="badge warning"><i data-lucide="activity"></i> Training Running</span>
    {/if}
  </div>
</div>

<div class="grid-2">
  <!-- Left: Config -->
  <div>
    <div class="card">
      <h2><i data-lucide="settings"></i> Configuration</h2>

      <div class="form-group">
        <label>Dataset <span class="hint">(pilih satu atau lebih)</span></label>
        <div class="dataset-list">
          {#each datasets as ds}
            {@const paired = ds.kind === 'paired'}
            {@const count = paired ? (ds.pair_count || ds.train_pairs || 0) : (ds.image_count || 0)}
            {@const selected = selectedDatasets.includes(ds.path)}
            <div
              class="dataset-option"
              class:selected
              class:disabled={!ds.ready && paired}
              onclick={() => ds.ready !== false && toggleDataset(ds.path)}
            >
              <div class="ds-check">
                {#if selected}
                  <i data-lucide="check-circle"></i>
                {:else}
                  <i data-lucide="circle"></i>
                {/if}
              </div>
              <div class="ds-info">
                <div class="ds-name">{ds.name}</div>
                <div class="ds-meta">
                  {paired ? `${count} train · ${ds.test_count || 0} test` : `${count} images`}
                  {#if !ds.ready && paired}
                    <span class="ds-warn">⚠️ not ready</span>
                  {/if}
                </div>
              </div>
              <div class="ds-badge {ds.kind}">{ds.kind}</div>
            </div>
          {:else}
            <div class="empty-datasets">Tidak ada dataset tersedia</div>
          {/each}
        </div>
        {#if selectedDatasets.length > 0}
          <div class="ds-summary">
            <strong>{selectedDatasets.length}</strong> dataset dipilih —
            <strong>{getTotalPairs().toLocaleString()}</strong> train pairs —
            <strong>{getTotalTest().toLocaleString()}</strong> test pairs
          </div>
        {/if}
      </div>

      <div class="form-row">
        <div class="form-group">
          <label>Epochs</label>
          <input type="number" bind:value={config.epochs} min="1" />
        </div>
        <div class="form-group">
          <label>Batch Size</label>
          <input type="number" bind:value={config.batchSize} min="1" max="64" />
        </div>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label>Image Size</label>
          <input type="number" bind:value={config.size} min="64" step="64" />
        </div>
        <div class="form-group">
          <label>Learning Rate</label>
          <input type="number" bind:value={config.lr} step="0.0001" min="0" />
        </div>
      </div>

      <div class="form-group">
        <label>Base Channels</label>
        <input type="number" bind:value={config.baseChannels} min="8" step="8" />
      </div>

      <div class="form-actions">
        {#if training}
          <button class="btn btn-danger" onclick={stopTraining}>
            <i data-lucide="square"></i> Stop Training
          </button>
        {:else}
          <button class="btn btn-primary" onclick={startTraining} disabled={selectedDatasets.length === 0}>
            <i data-lucide="play"></i> Start Training
          </button>
        {/if}
      </div>
    </div>

    <!-- Status -->
    <div class="card">
      <h2><i data-lucide="activity"></i> Model Status</h2>
      <div class="status-grid">
        <div class="status-item wide">
          <span class="k">Status</span>
          <span class="v" style="color:{training ? 'var(--warning)' : (status.best_exists ? 'var(--success)' : 'var(--text3)')}">
            {training ? 'Running' : (status.best_exists ? '✓ Model Ready' : '— No Model')}
          </span>
        </div>
        {#if training && status.current_epoch != null}
          <div class="status-item wide">
            <span class="k">Progress</span>
            <div class="progress-wrap">
              <div class="progress-bar">
                <div class="progress-fill" style="width: {((status.current_epoch / status.total_epochs) * 100) || 0}%"></div>
              </div>
              <span class="progress-text">{status.current_epoch}/{status.total_epochs} epochs ({Math.round((status.current_epoch / status.total_epochs) * 100)}%)</span>
            </div>
          </div>
        {:else}
          <div class="status-item">
            <span class="k">Epochs</span>
            <span class="v">{status.epochs || 0}</span>
          </div>
        {/if}
        {#if status.eta}
          <div class="status-item">
            <span class="k">Estimasi Sisa</span>
            <span class="v eta">{status.eta}</span>
          </div>
        {/if}
        <div class="status-item">
          <span class="k">Best Val Loss</span>
          <span class="v">{status.best_loss != null && status.best_loss !== Infinity ? Number(status.best_loss).toFixed(6) : '—'}</span>
        </div>
        {#if status.last_val_loss}
          <div class="status-item">
            <span class="k">Last Val Loss</span>
            <span class="v">{Number(status.last_val_loss).toFixed(6)}</span>
          </div>
        {/if}
        {#if status.last_val_psnr}
          <div class="status-item">
            <span class="k">Last PSNR</span>
            <span class="v">{Number(status.last_val_psnr).toFixed(2)} dB</span>
          </div>
        {/if}
        {#if status.last_val_ssim}
          <div class="status-item">
            <span class="k">Last SSIM</span>
            <span class="v">{Number(status.last_val_ssim).toFixed(4)}</span>
          </div>
        {/if}
        <div class="status-item">
          <span class="k">Checkpoint</span>
          <span class="v">{status.best_exists ? 'best.pth ✓' : '—'}</span>
        </div>
      </div>

      {#if status.model_available_for_test}
        <div style="margin-top:0.75rem">
          <button class="btn btn-outline" onclick={reloadAndTest}>
            <i data-lucide="test-tube"></i> Test Model (dari best.pth)
          </button>
        </div>
      {/if}
    </div>

    {#if status.latest_preview_url}
      <div class="card">
        <h2><i data-lucide="image"></i> Latest Validation Preview</h2>
        <p class="info-text" style="margin-bottom:0.75rem">Baris: input, hasil model, target bersih, mask prediksi.</p>
        <div class="preview-box">
          <img src={status.latest_preview_url} alt="Latest validation preview" />
        </div>
      </div>
    {/if}

    <!-- Loss Chart -->
    {#if status.history?.train_loss?.length > 1}
      <div class="card">
        <h2><i data-lucide="trending-down"></i> Training History</h2>
        <div class="chart-wrap">
          <svg viewBox="0 0 400 130" preserveAspectRatio="none" class="chart">
            {@html chartData(status.history.train_loss, '#6366f1')}
            {@html chartData(status.history.val_loss, '#06b6d4')}
          </svg>
          <div class="chart-legend">
            <span><span class="dot primary"></span> Train Loss</span>
            <span><span class="dot accent"></span> Val Loss</span>
          </div>
        </div>
      </div>
    {/if}
  </div>

  <!-- Right: Log -->
  <div>
    <div class="card log-card">
      <div class="log-header">
        <h2><i data-lucide="scroll-text"></i> Training Log</h2>
        {#if log}
          <button class="btn-small" onclick={() => { log = ''; logOffset = 0 }}>
            <i data-lucide="trash-2"></i> Clear
          </button>
        {/if}
      </div>
      <pre class="log-box">{log || 'Belum ada log. Mulai training untuk melihat output.'}</pre>
    </div>
  </div>
</div>

<div class="footer">DocAI v2.0 — Document Restoration Training</div>

<style>
  .page-header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 2rem; }
  .page-header h1 { font-size: 1.5rem; font-weight: 700; }
  .page-header p { color: var(--text2); margin-top: 0.25rem; font-size: 0.9rem; }
  .header-badges { display: flex; gap: 0.5rem; }
  .badge { display: flex; align-items: center; gap: 0.35rem; padding: 0.3rem 0.75rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
  .badge.success { background: rgba(16,185,129,0.15); color: var(--success); }
  .badge.warning { background: rgba(245,158,11,0.15); color: var(--warning); }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  .card { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.5rem; margin-bottom: 1.5rem; }
  .card :global(h2) { font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
  .form-group { margin-bottom: 1rem; }
  .form-group label { display: flex; align-items: center; gap: 0.5rem; font-size: 0.75rem; font-weight: 700; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.375rem; }
  .form-group .hint { font-weight: 400; text-transform: none; letter-spacing: 0; color: var(--text3); font-size: 0.7rem; }
  .form-group input { width: 100%; padding: 0.625rem 0.75rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); color: var(--text); font-family: inherit; font-size: 0.85rem; }
  .form-group input:focus { outline: none; border-color: var(--accent); }
  .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }

  /* Dataset selection */
  .dataset-list { display: flex; flex-direction: column; gap: 0.375rem; max-height: 280px; overflow-y: auto; padding: 0.25rem 0; }
  .dataset-option { display: flex; align-items: center; gap: 0.625rem; padding: 0.625rem 0.75rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); cursor: pointer; transition: all 0.15s; }
  .dataset-option:hover { border-color: var(--accent); }
  .dataset-option.selected { border-color: var(--accent); background: rgba(6,182,212,0.05); }
  .dataset-option.disabled { opacity: 0.4; cursor: not-allowed; }
  .ds-check { color: var(--text3); flex-shrink: 0; }
  .dataset-option.selected .ds-check { color: var(--accent); }
  .ds-check :global(i) { width: 18px; height: 18px; }
  .ds-info { flex: 1; min-width: 0; }
  .ds-name { font-size: 0.85rem; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .ds-meta { font-size: 0.7rem; color: var(--text3); display: flex; gap: 0.5rem; }
  .ds-warn { color: var(--warning); }
  .ds-badge { font-size: 0.6rem; font-weight: 700; text-transform: uppercase; padding: 0.15rem 0.5rem; border-radius: 999px; flex-shrink: 0; }
  .ds-badge.paired { background: rgba(16,185,129,0.15); color: var(--success); }
  .ds-badge.clean { background: rgba(99,102,241,0.15); color: var(--primary); }
  .ds-badge.unknown { background: rgba(113,113,122,0.15); color: var(--text3); }
  .empty-datasets { padding: 2rem; text-align: center; color: var(--text3); font-size: 0.85rem; }
  .ds-summary { margin-top: 0.5rem; font-size: 0.75rem; color: var(--text2); text-align: right; }

  .form-actions { margin-top: 1.25rem; padding-top: 1.25rem; border-top: 1px solid var(--border); }
  .btn { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.625rem 1.25rem; border-radius: var(--radius-xs); font-size: 0.875rem; font-weight: 600; border: none; transition: all 0.15s; cursor: pointer; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary { background: var(--primary); color: white; }
  .btn-primary:not(:disabled):hover { background: #5558e6; }
  .btn-danger { background: var(--error); color: white; }
  .btn-danger:hover { background: #dc2626; }
  .btn-small { display: flex; align-items: center; gap: 0.3rem; padding: 0.3rem 0.6rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); color: var(--text3); font-size: 0.7rem; cursor: pointer; }
  .btn-small:hover { color: var(--text); }

  /* Status */
  .status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
  .status-item { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); padding: 0.75rem; }
  .status-item .k { display: block; font-size: 0.7rem; font-weight: 700; color: var(--text2); text-transform: uppercase; margin-bottom: 0.25rem; }
  .status-item .v { font-size: 1.1rem; font-weight: 700; }

  /* Chart */
  .chart-wrap { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); padding: 1rem; }
  .chart { width: 100%; height: 120px; }
  .chart-legend { display: flex; gap: 1.5rem; justify-content: center; margin-top: 0.75rem; font-size: 0.7rem; color: var(--text2); }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.35rem; }
  .dot.primary { background: #6366f1; }
  .dot.accent { background: #06b6d4; }
  .preview-box { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); padding: 0.75rem; overflow: auto; }
  .preview-box img { width: 100%; height: auto; display: block; border-radius: var(--radius-xs); }

  /* Log */
  .log-card { display: flex; flex-direction: column; height: calc(100vh - 200px); min-height: 400px; }
  .log-header { display: flex; align-items: center; justify-content: space-between; }
  .log-box { flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); padding: 1rem; font-size: 0.75rem; color: var(--text2); overflow-y: auto; white-space: pre-wrap; word-break: break-all; font-family: 'SF Mono', 'Fira Code', monospace; margin: 0; }

  .status-item.wide { grid-column: 1 / -1; }
  .progress-wrap { display: flex; flex-direction: column; gap: 0.25rem; }
  .progress-bar { height: 6px; background: var(--bg); border-radius: 999px; overflow: hidden; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, var(--primary), var(--accent)); border-radius: 999px; transition: width 0.5s ease; }
  .progress-text { font-size: 0.7rem; color: var(--text2); }
  .eta { font-size: 0.9rem; color: var(--accent) !important; font-weight: 600; }
  .btn-outline { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.625rem 1.25rem; border-radius: var(--radius-xs); font-size: 0.85rem; font-weight: 600; border: 1px solid var(--accent); color: var(--accent); background: transparent; cursor: pointer; transition: all 0.15s; }
  .btn-outline:hover { background: rgba(6,182,212,0.1); }

  .footer { text-align: center; padding: 2rem 0; color: var(--text3); font-size: 0.8rem; }

  @media (max-width: 768px) {
    .grid-2 { grid-template-columns: 1fr; }
    .form-row { grid-template-columns: 1fr; }
  }
</style>
