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
  let systemTimer = null
  let statusTimer = null
  let training = false
  let systemStatus = null
  let evaluation = { exists: false, summary: null, preview_url: null, metrics_url: null }
  let runs = []
  let selectedCheckpoint = 'checkpoints/document_restorer/best.pth'
  let resumeCheckpoint = 'checkpoints/document_restorer/best.pth'
  let resumeMode = 'weights'

  const serverPresets = [
    { id: 'balanced', name: 'Balanced L4', note: 'Baseline kuat: semua paired dataset, mask supervised, size 768.', epochs: 80, batchSize: 4, size: 768, lr: 0.0001, workers: 12, perceptualWeight: 0.05, ssimWeight: 0.1, shadowLossWeight: 1.8, illuminationWeight: 0.55, maskLossWeight: 0.35, gradientWeight: 0.15, colorWeight: 0.08, identityWeight: 0.12, earlyStopPatience: 10, minDelta: 0.0001, gradClipNorm: 1.0, resumeBest: false },
    { id: 'quality', name: 'Fine-tune Quality', note: 'Final quality: size 1024, LR kecil, resume best.pth weights-only.', epochs: 160, batchSize: 2, size: 1024, lr: 0.00004, workers: 8, perceptualWeight: 0.04, ssimWeight: 0.15, shadowLossWeight: 2.0, illuminationWeight: 0.65, maskLossWeight: 0.4, gradientWeight: 0.2, colorWeight: 0.08, identityWeight: 0.12, earlyStopPatience: 12, minDelta: 0.0001, gradClipNorm: 0.8, resumeBest: true },
    { id: 'speed', name: 'Speed 512', note: 'Cepat untuk sanity check awal.', epochs: 30, batchSize: 8, size: 512, lr: 0.0002, workers: 12, perceptualWeight: 0.04, ssimWeight: 0.1, shadowLossWeight: 1.6, illuminationWeight: 0.45, maskLossWeight: 0.3, gradientWeight: 0.1, colorWeight: 0.06, identityWeight: 0.1, earlyStopPatience: 6, minDelta: 0.0001, gradClipNorm: 1.0, resumeBest: false }
  ]

  let selectedPreset = 'balanced'
  const config = { epochs: 80, batchSize: 4, size: 768, lr: 0.0001, baseChannels: 32, workers: 12, perceptualWeight: 0.05, ssimWeight: 0.1, shadowLossWeight: 1.8, illuminationWeight: 0.55, maskLossWeight: 0.35, gradientWeight: 0.15, colorWeight: 0.08, identityWeight: 0.12, earlyStopPatience: 10, minDelta: 0.0001, gradClipNorm: 1.0, resumeBest: false, device: 'cuda', maxTrainSamples: 0, maxValSamples: 0, maxEvalSamples: 0 }
  $: estimatedVram = estimateVram(config.size, config.batchSize)
  $: vramTotal = systemStatus?.gpu?.items?.[0]?.vram_total_gb || 23
  $: vramLevel = estimatedVram > vramTotal * 0.92 ? 'danger' : estimatedVram > vramTotal * 0.75 ? 'warning' : 'safe'
  $: validationDatasets = selectedDatasets.map(path => validationPathFor(path)).filter(Boolean)
  $: displayProgress = Number(status.progress_percent ?? ((status.current_epoch / status.total_epochs) * 100) ?? 0)

  onMount(async () => {
    refreshIcons()
    await loadDatasets()
    await refreshStatus()
    await refreshSystemStatus()
    await refreshEvaluation()
    await refreshRuns()
    systemTimer = setInterval(refreshSystemStatus, 2000)
    statusTimer = setInterval(refreshStatus, 2000)
    if (training) {
      log = ''
      logOffset = 0
      pollTimer = setInterval(pollLog, 2000)
      pollLog()
    }
  })

  onDestroy(() => {
    if (pollTimer) clearInterval(pollTimer)
    if (systemTimer) clearInterval(systemTimer)
    if (statusTimer) clearInterval(statusTimer)
  })

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

  async function refreshSystemStatus() {
    try {
      systemStatus = await apiJson('/api/system/status')
    } catch(e) {}
  }

  async function refreshEvaluation() {
    try {
      evaluation = await apiJson('/api/training/evaluation/status')
    } catch(e) {}
  }

  async function refreshRuns() {
    try {
      const data = await apiJson('/api/training/runs')
      runs = data.runs || []
    } catch(e) {}
  }

  function pct(value) {
    return Math.max(0, Math.min(100, Number(value) || 0))
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
      const selected = selectedDatasets.map(path => datasets.find(ds => ds.path === path)).filter(Boolean)
      const paired = selected.filter(ds => ds.kind === 'paired').map(ds => ds.path)
      const clean = selected.filter(ds => ds.kind === 'clean').map(ds => ds.path)
      const identity = selected.filter(ds => ds.kind === 'identity').map(ds => ds.path)
      formData.append('paired_data', paired.join(','))
      formData.append('clean_data', clean.join(','))
      formData.append('identity_data', identity.join(','))
      formData.append('epochs', String(config.epochs))
      formData.append('batch_size', String(config.batchSize))
      formData.append('size', String(config.size))
      formData.append('lr', String(config.lr))
      formData.append('base_channels', String(config.baseChannels))
      formData.append('workers', String(config.workers))
      formData.append('device', config.device)
      formData.append('pipeline', 'true')
      formData.append('perceptual_weight', String(config.perceptualWeight))
      formData.append('ssim_weight', String(config.ssimWeight))
      formData.append('shadow_loss_weight', String(config.shadowLossWeight))
      formData.append('illumination_weight', String(config.illuminationWeight))
      formData.append('mask_loss_weight', String(config.maskLossWeight))
      formData.append('gradient_weight', String(config.gradientWeight))
      formData.append('color_weight', String(config.colorWeight))
      formData.append('identity_weight', String(config.identityWeight))
      if (Number(config.maxTrainSamples) > 0) formData.append('max_train_samples', String(config.maxTrainSamples))
      if (Number(config.maxValSamples) > 0) formData.append('max_val_samples', String(config.maxValSamples))
      if (validationDatasets.length) formData.append('validation_paired_data', validationDatasets.join(','))
      formData.append('early_stop_patience', String(config.earlyStopPatience))
      formData.append('min_delta', String(config.minDelta))
      formData.append('grad_clip_norm', String(config.gradClipNorm))
      if (config.resumeBest && resumeCheckpoint) {
        formData.append('resume', resumeCheckpoint)
        if (resumeMode === 'weights') formData.append('resume_weights_only', 'true')
      }
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

  async function runEvaluation() {
    if (!status.best_exists) {
      showToast('best.pth belum ada', 'error')
      return
    }
    const pairedData = validationDatasets[0] || 'datasets/ShadowDocument7K/test'
    try {
      const formData = new FormData()
      formData.append('paired_data', pairedData)
      formData.append('checkpoint', selectedCheckpoint)
      formData.append('output', 'evaluation/document_restorer')
      formData.append('size', String(config.size))
      formData.append('batch_size', '1')
      formData.append('workers', String(Math.min(Number(config.workers) || 4, 8)))
      formData.append('device', config.device)
      if (Number(config.maxEvalSamples) > 0) formData.append('max_samples', String(config.maxEvalSamples))

      const token = localStorage.getItem('docai_token')
      const res = await fetch('/api/training/evaluate', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Gagal mulai evaluasi')
      }
      showToast('Evaluasi dimulai')
      training = true
      log = ''
      logOffset = 0
      pollTimer = setInterval(pollLog, 2000)
      pollLog()
    } catch(e) { showToast(e.message, 'error') }
  }

  async function exportMobile() {
    try {
      const formData = new FormData()
      formData.append('checkpoint', selectedCheckpoint)
      const token = localStorage.getItem('docai_token')
      const response = await fetch('/api/models/export-mobile', {
        method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: formData
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(data.detail || 'Export mobile gagal')
      showToast(`ONNX ready: ${data.model}`)
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
      await refreshStatus()
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
        refreshEvaluation()
        refreshRuns()
        showToast('✅ Proses selesai')
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

  function validationPathFor(path) {
    const ds = datasets.find(d => d.path === path)
    if (!ds || ds.kind !== 'paired' || !ds.test_count) return ''
    if (path.endsWith('/train')) return path.replace(/\/train$/, '/test')
    return `${path}/test`
  }

  function applyPreset(id) {
    selectedPreset = id
    const preset = serverPresets.find(p => p.id === id)
    if (!preset) return
    config.epochs = preset.epochs
    config.batchSize = preset.batchSize
    config.size = preset.size
    config.lr = preset.lr
    config.workers = preset.workers
    config.perceptualWeight = preset.perceptualWeight
    config.ssimWeight = preset.ssimWeight
    config.shadowLossWeight = preset.shadowLossWeight
    config.illuminationWeight = preset.illuminationWeight
    config.maskLossWeight = preset.maskLossWeight
    config.gradientWeight = preset.gradientWeight
    config.colorWeight = preset.colorWeight
    config.identityWeight = preset.identityWeight
    config.earlyStopPatience = preset.earlyStopPatience
    config.minDelta = preset.minDelta
    config.gradClipNorm = preset.gradClipNorm
    config.resumeBest = preset.resumeBest
  }

  function estimateVram(size, batchSize) {
    const base = 0.21
    const perSampleAt512 = 1.33
    return base + perSampleAt512 * batchSize * Math.pow((Number(size) || 512) / 512, 2)
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

      <div class="preset-grid">
        {#each serverPresets as preset}
          <button class="preset-card" class:active={selectedPreset === preset.id} onclick={() => applyPreset(preset.id)} type="button">
            <span>{preset.name}</span>
            <small>{preset.note}</small>
          </button>
        {/each}
      </div>

      <div class="server-advice {vramLevel}">
        <div><strong>Estimasi VRAM</strong> {estimatedVram.toFixed(1)} / {vramTotal.toFixed(0)} GB</div>
        <span>{vramLevel === 'safe' ? 'Aman untuk NVIDIA L4' : vramLevel === 'warning' ? 'Mepet, tutup proses GPU lain' : 'Risiko OOM, turunkan batch/size'}</span>
      </div>

      <div class="form-group">
        <label>Dataset <span class="hint">(pilih satu atau lebih)</span></label>
        <div class="guide-callout">
          <strong>Panduan pemilihan dataset</strong>
          <span class="guide-item"><strong>paired</strong>: paling penting untuk akurasi shadow removal. Pilih semua yang ready.</span>
          <span class="guide-item"><strong>clean</strong>: hanya gambar bersih. Model membuat degradasi sintetis otomatis.</span>
          <span class="guide-item"><strong>identity</strong>: input=target bersih. Jaga warna asli area non-shadow. Cocok untuk 20–100 gambar bersihmu sendiri.</span>
          <span class="guide-item">Kalau belum ada identity, tidak masalah; fokus dulu ke semua paired.</span>
        </div>
        <div class="dataset-list">
          {#each datasets as ds}
            {@const paired = ds.kind === 'paired'}
            {@const count = paired ? (ds.pair_count || ds.train_pairs || 0) : (ds.image_count || 0)}
            {@const selected = selectedDatasets.includes(ds.path)}
            {@const selectable = ['paired', 'clean', 'identity'].includes(ds.kind) && ds.ready !== false}
            <div
              class="dataset-option"
              class:selected
              class:disabled={!selectable}
              onclick={() => selectable && toggleDataset(ds.path)}
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
                  {#if !selectable}
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

      <div class="form-row">
        <div class="form-group">
          <label>Workers</label>
          <input type="number" bind:value={config.workers} min="0" max="16" />
        </div>
        <div class="form-group">
          <label>Device</label>
          <select bind:value={config.device}>
            <option value="cuda">CUDA</option>
            <option value="auto">Auto</option>
            <option value="cpu">CPU</option>
          </select>
        </div>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label>Perceptual Weight</label>
          <input type="number" bind:value={config.perceptualWeight} step="0.01" min="0" />
        </div>
        <div class="form-group">
          <label>SSIM Weight</label>
          <input type="number" bind:value={config.ssimWeight} step="0.01" min="0" />
        </div>
      </div>

      <div class="form-row">
        <div class="form-group">
          <label>Shadow Loss Weight</label>
          <input type="number" bind:value={config.shadowLossWeight} step="0.05" min="0" />
        </div>
        <div class="form-group">
          <label>Illumination Weight</label>
          <input type="number" bind:value={config.illuminationWeight} step="0.05" min="0" />
        </div>
      </div>

      <div class="form-group">
        <label>Mask Loss Weight</label>
        <input type="number" bind:value={config.maskLossWeight} step="0.05" min="0" />
      </div>

      <div class="form-row">
        <div class="form-group">
          <label>Gradient Weight</label>
          <input type="number" bind:value={config.gradientWeight} step="0.05" min="0" />
        </div>
        <div class="form-group">
          <label>Color Weight</label>
          <input type="number" bind:value={config.colorWeight} step="0.01" min="0" />
        </div>
      </div>

      <div class="form-group">
        <label>Non-shadow Identity Weight</label>
        <input type="number" bind:value={config.identityWeight} step="0.05" min="0" />
      </div>

      <label class="check-row">
        <input type="checkbox" bind:checked={config.resumeBest} disabled={!status.best_exists} />
        <span>Fine-tune dari `best.pth` dengan optimizer/LR baru</span>
      </label>
      {#if config.resumeBest}
        <div class="field">
          <label>Resume source</label>
          <select bind:value={resumeCheckpoint}>
            <option value="checkpoints/document_restorer/best.pth">Live best.pth</option>
            {#each runs as run}
              {#each run.checkpoints || [] as checkpoint}
                <option value={checkpoint.path}>{run.run_id} · {checkpoint.name}</option>
              {/each}
            {/each}
          </select>
        </div>
        <div class="field">
          <label>Resume mode</label>
          <select bind:value={resumeMode}>
            <option value="weights">Fine-tune: weights only</option>
            <option value="full">Continue: optimizer + scheduler + epoch</option>
          </select>
          <p class="info-text">Fine-tune memakai LR/config baru. Continue melanjutkan state training lama.</p>
        </div>
      {/if}

      <div class="form-row">
        <div class="form-group">
          <label>Max Train Samples <span class="hint">0 = semua</span></label>
          <input type="number" bind:value={config.maxTrainSamples} min="0" />
        </div>
        <div class="form-group">
          <label>Max Val Samples <span class="hint">0 = semua</span></label>
          <input type="number" bind:value={config.maxValSamples} min="0" />
        </div>
      </div>

      <div class="form-group">
        <label>Max Eval Samples <span class="hint">0 = semua</span></label>
        <input type="number" bind:value={config.maxEvalSamples} min="0" />
      </div>

      {#if validationDatasets.length}
        <div class="validation-note">
          Validation otomatis: {validationDatasets.join(', ')}
        </div>
      {/if}

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
                <div class="progress-fill" style="width: {displayProgress || 0}%"></div>
              </div>
              <span class="progress-text">
                Epoch {status.current_epoch}/{status.total_epochs}
                {#if status.current_batch != null && status.total_batches}
                  · Batch {status.current_batch}/{status.total_batches}
                {/if}
                · {(displayProgress || 0).toFixed(1)}%
              </span>
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
        {#if status.started_at_wib}
          <div class="status-item">
            <span class="k">Jenis Run</span>
            <span class="v time-text">{status.process_kind || 'training'}</span>
          </div>
          <div class="status-item">
            <span class="k">Mulai Run</span>
            <span class="v time-text">{status.started_at_wib}</span>
          </div>
        {/if}
        {#if status.estimated_finish_wib}
          <div class="status-item">
            <span class="k">Estimasi Selesai</span>
            <span class="v time-text eta">{status.estimated_finish_wib}</span>
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
        <div class="model-actions">
          <button class="btn btn-outline" onclick={reloadAndTest}>
            <i data-lucide="test-tube"></i> Test Model (dari best.pth)
          </button>
          <button class="btn btn-outline" onclick={runEvaluation} disabled={training}>
            <i data-lucide="bar-chart-3"></i> Evaluate best.pth
          </button>
          <button class="btn btn-outline" onclick={exportMobile} disabled={training}>
            <i data-lucide="smartphone"></i> Export Flutter ONNX
          </button>
        </div>
      {/if}
      <div class="field" style="margin-top:1rem">
        <label>Checkpoint untuk evaluasi</label>
        <select bind:value={selectedCheckpoint}>
          <option value="checkpoints/document_restorer/best.pth">Live best.pth</option>
          <option value="checkpoints/document_restorer/best_psnr.pth">Live best_psnr.pth</option>
          <option value="checkpoints/document_restorer/best_ssim.pth">Live best_ssim.pth</option>
          {#each runs as run}
            {#each run.checkpoints || [] as checkpoint}
              <option value={checkpoint.path}>{run.run_id} · {checkpoint.name}</option>
            {/each}
          {/each}
        </select>
        <p class="info-text">Setiap training tersimpan di folder run berbeda. Live best diperbarui atomik dan aman dicoba saat training.</p>
      </div>
    </div>

    {#if evaluation.exists}
      <div class="card">
        <h2><i data-lucide="bar-chart-3"></i> Evaluation Result</h2>
        <div class="eval-grid">
          <div class="status-item">
            <span class="k">Samples</span>
            <span class="v">{evaluation.summary?.samples || 0}</span>
          </div>
          <div class="status-item">
            <span class="k">PSNR</span>
            <span class="v">{evaluation.summary?.psnr ? Number(evaluation.summary.psnr).toFixed(2) + ' dB' : '—'}</span>
          </div>
          <div class="status-item">
            <span class="k">SSIM</span>
            <span class="v">{evaluation.summary?.ssim ? Number(evaluation.summary.ssim).toFixed(4) : '—'}</span>
          </div>
          <div class="status-item">
            <span class="k">L1</span>
            <span class="v">{evaluation.summary?.l1 ? Number(evaluation.summary.l1).toFixed(5) : '—'}</span>
          </div>
        </div>
        <div class="model-actions">
          {#if evaluation.metrics_url}
            <a class="btn btn-outline" href={evaluation.metrics_url} target="_blank" rel="noreferrer"><i data-lucide="download"></i> Metrics CSV</a>
          {/if}
          <button class="btn btn-outline" onclick={refreshEvaluation}><i data-lucide="refresh-cw"></i> Refresh</button>
        </div>
        {#if evaluation.preview_url}
          <div class="preview-box eval-preview">
            <img src={evaluation.preview_url} alt="Evaluation preview" />
          </div>
        {/if}
      </div>
    {/if}

    {#if systemStatus}
      <div class="card">
        <h2><i data-lucide="gauge"></i> System Monitor</h2>
        <div class="monitor-grid">
          <div class="monitor-item">
            <div class="monitor-head"><span>CPU</span><strong>{pct(systemStatus.cpu?.percent).toFixed(0)}%</strong></div>
            <div class="meter"><div class="meter-fill cpu" style="width:{pct(systemStatus.cpu?.percent)}%"></div></div>
            <div class="monitor-meta">{systemStatus.cpu?.count || 0} cores{systemStatus.cpu?.load_average ? ` · load ${systemStatus.cpu.load_average[0].toFixed(2)}` : ''}</div>
          </div>
          <div class="monitor-item">
            <div class="monitor-head"><span>RAM</span><strong>{pct(systemStatus.ram?.percent).toFixed(0)}%</strong></div>
            <div class="meter"><div class="meter-fill ram" style="width:{pct(systemStatus.ram?.percent)}%"></div></div>
            <div class="monitor-meta">{systemStatus.ram?.used_gb} / {systemStatus.ram?.total_gb} GB · free {systemStatus.ram?.available_gb} GB</div>
          </div>
          <div class="monitor-item">
            <div class="monitor-head"><span>Disk</span><strong>{pct(systemStatus.disk?.percent).toFixed(0)}%</strong></div>
            <div class="meter"><div class="meter-fill disk" style="width:{pct(systemStatus.disk?.percent)}%"></div></div>
            <div class="monitor-meta">{systemStatus.disk?.used_gb} / {systemStatus.disk?.total_gb} GB · free {systemStatus.disk?.free_gb} GB</div>
          </div>
          {#if systemStatus.gpu?.available && systemStatus.gpu?.items?.length}
            {#each systemStatus.gpu.items as gpu}
              <div class="monitor-item wide">
                <div class="monitor-head"><span>GPU {gpu.index}: {gpu.name}</span><strong>Util {gpu.utilization_percent ?? '—'}%</strong></div>
                <div class="gpu-bars">
                  <div>
                    <div class="mini-label"><span>GPU Util</span><span>{gpu.utilization_percent ?? '—'}%</span></div>
                    <div class="meter"><div class="meter-fill gpu" style="width:{pct(gpu.utilization_percent)}%"></div></div>
                  </div>
                  <div>
                    <div class="mini-label"><span>VRAM</span><span>{pct(gpu.vram_percent).toFixed(0)}%</span></div>
                    <div class="meter"><div class="meter-fill ram" style="width:{pct(gpu.vram_percent)}%"></div></div>
                  </div>
                </div>
                <div class="monitor-meta">
                  VRAM {gpu.vram_used_gb} / {gpu.vram_total_gb} GB · free {gpu.vram_free_gb} GB
                  {#if gpu.temperature_c != null} · {gpu.temperature_c}°C{/if}
                  {#if gpu.power_draw_w != null} · {gpu.power_draw_w} / {gpu.power_limit_w} W{/if}
                  {#if gpu.stats_source} · {gpu.stats_source}{/if}
                </div>
              </div>
            {/each}
          {:else}
            <div class="monitor-item wide muted">
              <div class="monitor-head"><span>GPU</span><strong>Not available</strong></div>
              <div class="monitor-meta">Training akan pakai CPU jika CUDA tidak tersedia.</div>
            </div>
          {/if}
        </div>
      </div>
    {/if}

    {#if status.latest_preview_url}
      <div class="card">
        <h2><i data-lucide="image"></i> Latest Validation Preview</h2>
        <p class="info-text" style="margin-bottom:0.75rem">Baris: input, hasil model, target bersih, target mask, predicted mask.</p>
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
  .form-group input, .form-group select { width: 100%; padding: 0.625rem 0.75rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); color: var(--text); font-family: inherit; font-size: 0.85rem; }
  .form-group input:focus, .form-group select:focus { outline: none; border-color: var(--accent); }
  .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }

  .preset-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.625rem; margin-bottom: 1rem; }
  .preset-card { text-align: left; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); padding: 0.75rem; color: var(--text); cursor: pointer; transition: all 0.15s; }
  .preset-card:hover, .preset-card.active { border-color: var(--accent); background: rgba(6,182,212,0.06); }
  .preset-card span { display: block; font-size: 0.82rem; font-weight: 700; margin-bottom: 0.25rem; }
  .preset-card small { display: block; color: var(--text3); font-size: 0.68rem; line-height: 1.35; }
  .server-advice { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; border: 1px solid var(--border); border-radius: var(--radius-xs); padding: 0.75rem; margin-bottom: 1rem; font-size: 0.78rem; }
  .server-advice.safe { border-color: rgba(16,185,129,0.45); background: rgba(16,185,129,0.08); }
  .server-advice.warning { border-color: rgba(245,158,11,0.5); background: rgba(245,158,11,0.08); }
  .server-advice.danger { border-color: rgba(239,68,68,0.55); background: rgba(239,68,68,0.08); }
  .server-advice span { color: var(--text2); }
  .check-row { display: flex; align-items: center; gap: 0.5rem; padding: 0.625rem 0; color: var(--text2); font-size: 0.82rem; }
  .check-row input { width: auto; }
  .validation-note { color: var(--text2); font-size: 0.76rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); padding: 0.625rem 0.75rem; margin-bottom: 1rem; word-break: break-all; }

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
  .ds-badge.identity { background: rgba(245,158,11,0.15); color: var(--warning); }
  .ds-badge.unknown { background: rgba(113,113,122,0.15); color: var(--text3); }
  .empty-datasets { padding: 2rem; text-align: center; color: var(--text3); font-size: 0.85rem; }
  .ds-summary { margin-top: 0.5rem; font-size: 0.75rem; color: var(--text2); text-align: right; }
  .guide-callout { background: var(--bg); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: var(--radius-xs); padding: 0.75rem; margin-top: 0.5rem; display: flex; flex-direction: column; gap: 0.35rem; font-size: 0.78rem; color: var(--text2); }
  .guide-callout strong { color: var(--text); font-size: 0.82rem; }
  .guide-item { display: block; }

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
  .eval-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 0.75rem; }
  .status-item { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); padding: 0.75rem; }
  .status-item .k { display: block; font-size: 0.7rem; font-weight: 700; color: var(--text2); text-transform: uppercase; margin-bottom: 0.25rem; }
  .status-item .v { font-size: 1.1rem; font-weight: 700; }

  /* Monitor */
  .monitor-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
  .monitor-item { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); padding: 0.85rem; }
  .monitor-item.wide { grid-column: 1 / -1; }
  .monitor-item.muted { opacity: 0.7; }
  .monitor-head { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; margin-bottom: 0.6rem; }
  .monitor-head span { font-size: 0.75rem; font-weight: 700; color: var(--text2); text-transform: uppercase; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .monitor-head strong { font-size: 1rem; color: var(--text); }
  .monitor-meta { margin-top: 0.5rem; font-size: 0.7rem; color: var(--text3); }
  .meter { height: 7px; background: var(--bg3); border-radius: 999px; overflow: hidden; }
  .meter-fill { height: 100%; border-radius: 999px; transition: width 0.5s ease; }
  .meter-fill.cpu { background: linear-gradient(90deg, #06b6d4, #3b82f6); }
  .meter-fill.ram { background: linear-gradient(90deg, #8b5cf6, #6366f1); }
  .meter-fill.disk { background: linear-gradient(90deg, #f59e0b, #f97316); }
  .meter-fill.gpu { background: linear-gradient(90deg, #10b981, #22c55e); }
  .gpu-bars { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 0.45rem; }
  .mini-label { display: flex; justify-content: space-between; color: var(--text2); font-size: 0.68rem; margin-bottom: 0.25rem; }

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
  .time-text { font-size: 0.86rem !important; line-height: 1.35; }
  .btn-outline { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.625rem 1.25rem; border-radius: var(--radius-xs); font-size: 0.85rem; font-weight: 600; border: 1px solid var(--accent); color: var(--accent); background: transparent; cursor: pointer; transition: all 0.15s; }
  .btn-outline:hover { background: rgba(6,182,212,0.1); }
  .model-actions { margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.625rem; }
  .model-actions a { text-decoration: none; }
  .eval-preview { margin-top: 0.75rem; }

  .footer { text-align: center; padding: 2rem 0; color: var(--text3); font-size: 0.8rem; }

  @media (max-width: 768px) {
    .grid-2 { grid-template-columns: 1fr; }
    .form-row, .eval-grid { grid-template-columns: 1fr; }
    .preset-grid { grid-template-columns: 1fr; }
  }
</style>
