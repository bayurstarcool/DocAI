<script>
  import { onMount, onDestroy } from 'svelte'
  import { refreshIcons } from '../lib/icons.js'

  let status = { running: false, status: "no data" }
  let sys = { cpu_percent: 0, cpu_count: 0, ram_used_gb: 0, ram_total_gb: 0, ram_percent: 0, gpu: null }
  let history = { points: [] }
  let timer = null

  async function poll() {
    try {
      const h = { Authorization: "Bearer " + (localStorage.getItem("docai_token") || "") }
      const [sr, syr, hr] = await Promise.all([
        fetch("/api/training/docres/status", { headers: h }).then(r => r.ok ? r.json() : {}),
        fetch("/api/training/docres/system", { headers: h }).then(r => r.ok ? r.json() : {}),
        fetch("/api/training/docres/history", { headers: h }).then(r => r.ok ? r.json() : {}),
      ])
      status = sr || status
      sys = syr || sys
      history = hr || history
    } catch(e) {}
  }

  onMount(() => {
    refreshIcons()
    poll()
    timer = setInterval(poll, 3000)
  })

  onDestroy(() => { if (timer) clearInterval(timer) })

  $: progress = status.progress || 0
  $: etaH = status.eta_seconds ? Math.floor(status.eta_seconds / 3600) : 0
  $: etaM = status.eta_seconds ? Math.floor((status.eta_seconds % 3600) / 60) : 0
  $: lossPoints = (history.points || []).map((p, i) => ({ x: i, y: p.loss, iter: p.iter }))
  $: maxLoss = Math.max(0.1, ...lossPoints.map(p => p.y))
</script>

<div class="page">
  <h1><i data-lucide="brain"></i> DocRes Fine-tuning</h1>
  <p class="subtitle">Monitor fine-tuning model DocRes Restormer untuk deshadowing.</p>

  <!-- Status -->
  <div class="card">
    <h2><i data-lucide="activity"></i> Status Training</h2>
    {#if status.running}
      <div class="status-grid">
        <div class="status-item"><span>STATUS</span><strong class="running">{status.status || 'Running'}</strong></div>
        <div class="status-item"><span>ITERASI</span><strong>{status.iter || 0} / {(status.total_iter || 0).toLocaleString()}</strong></div>
        <div class="status-item"><span>PROGRESS</span><strong>{progress}%</strong></div>
        <div class="status-item"><span>LOSS (L1)</span><strong>{status.loss ?? '-'}</strong></div>
        <div class="status-item"><span>BEST SCORE</span><strong>{status.best_score ?? 'none yet'}</strong></div>
        <div class="status-item"><span>SPEED</span><strong>{status.speed_it_per_sec ?? '-'} it/s</strong></div>
        <div class="status-item"><span>ETA</span><strong>{etaH}h {etaM}m</strong></div>
        <div class="status-item"><span>LR</span><strong>{status.lr ?? '-'}</strong></div>
      </div>
      <div class="progress-bar"><div class="progress-fill" style="width: {progress}%"></div></div>
    {:else if status.status === "done"}
      <div class="done-banner"><span>Training selesai!</span><strong>Best: {status.best_score}</strong></div>
    {:else}
      <p class="idle-text">Tidak ada training aktif.</p>
    {/if}
  </div>

  <!-- System Monitor -->
  <div class="card">
    <h2><i data-lucide="monitor"></i> System Monitor</h2>
    <div class="sys-grid">
      <div class="sys-item">
        <div class="sys-label">CPU</div>
        <div class="sys-bar"><div class="sys-fill cpu" style="width:{sys.cpu_percent || 0}%"></div></div>
        <div class="sys-val">{sys.cpu_percent || 0}% ({sys.cpu_count || 0} cores)</div>
      </div>
      <div class="sys-item">
        <div class="sys-label">RAM</div>
        <div class="sys-bar"><div class="sys-fill ram" style="width:{sys.ram_percent || 0}%"></div></div>
        <div class="sys-val">{sys.ram_used_gb || 0}/{sys.ram_total_gb || 0} GB ({sys.ram_percent || 0}%)</div>
      </div>
      {#if sys.gpu}
        <div class="sys-item">
          <div class="sys-label">GPU ({sys.gpu.name})</div>
          <div class="sys-bar"><div class="sys-fill gpu" style="width:{(sys.gpu.vram_used / sys.gpu.vram_total * 100) || 0}%"></div></div>
          <div class="sys-val">{sys.gpu.vram_used}/{sys.gpu.vram_total} MB ({sys.gpu.utilization}% util, {sys.gpu.temperature}C)</div>
        </div>
      {/if}
    </div>
  </div>

  <!-- Loss Chart -->
  <div class="card">
    <h2><i data-lucide="line-chart"></i> Training Loss</h2>
    {#if lossPoints.length > 1}
      <div class="chart-container">
        <svg viewBox="0 0 800 200" class="chart-svg">
          <!-- Grid -->
          {#each [0, 0.25, 0.5, 0.75, 1] as frac}
            <line x1="50" y1={10 + frac * 180} x2="790" y2={10 + frac * 180} stroke="rgba(255,255,255,0.06)" stroke-width="1"/>
            <text x="45" y={14 + frac * 180} text-anchor="end" fill="rgba(255,255,255,0.3)" font-size="10">{(maxLoss * (1 - frac)).toFixed(3)}</text>
          {/each}
          <!-- Loss line -->
          <polyline
            points={lossPoints.map((p, i) => {
              const x = 50 + (i / Math.max(1, lossPoints.length - 1)) * 740
              const y = 190 - (p.y / maxLoss) * 180
              return `${x},${y}`
            }).join(' ')}
            fill="none"
            stroke="#7070ff"
            stroke-width="1.5"
            stroke-linejoin="round"
          />
          <!-- Current point -->
          {#if lossPoints.length > 0}
            {@const last = lossPoints[lossPoints.length - 1]}
            <circle cx={50 + ((lossPoints.length - 1) / Math.max(1, lossPoints.length - 1)) * 740} cy={190 - (last.y / maxLoss) * 180} r="4" fill="#7070ff"/>
          {/if}
        </svg>
        <div class="chart-legend">
          <span>Iter {lossPoints[0]?.iter || 0}</span>
          <span>Loss: {lossPoints[lossPoints.length-1]?.y?.toFixed(4) || '-'}</span>
          <span>Iter {lossPoints[lossPoints.length-1]?.iter || 0}</span>
        </div>
      </div>
    {:else}
      <p class="idle-text">Menunggu data loss...</p>
    {/if}
  </div>

  <!-- Config -->
  <div class="card">
    <h2><i data-lucide="settings"></i> Config</h2>
    <div class="cfg-grid">
      <div><span>Loss</span><strong>L1 only</strong></div>
      <div><span>Optimizer</span><strong>AdamW (wd=5e-4)</strong></div>
      <div><span>LR</span><strong>1e-4 to 1e-6 cosine</strong></div>
      <div><span>Sampling</span><strong>WeightedRandomSampler</strong></div>
      <div><span>Augment</span><strong>rotate +/-180, scale 0.7-1.5</strong></div>
      <div><span>Im size</span><strong>512x512</strong></div>
      <div><span>Train scope</span><strong>Decoder only (16%)</strong></div>
      <div><span>Iters</span><strong>50,000</strong></div>
    </div>
  </div>
</div>

<style>
  .page { display: flex; flex-direction: column; gap: 1rem; }
  h1 { display: flex; align-items: center; gap: .5rem; font-size: 1.4rem; }
  .subtitle { color: var(--text3); margin-top: -.5rem; font-size: .85rem; }
  .card { background: var(--card, rgba(255,255,255,.03)); border: 1px solid var(--border, #222); border-radius: 10px; padding: 1rem; }
  .card h2 { display: flex; align-items: center; gap: .4rem; font-size: .95rem; margin-bottom: .6rem; }

  /* Status */
  .status-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: .5rem; }
  .status-item { padding: .55rem; background: rgba(0,0,0,.15); border-radius: 6px; display: flex; flex-direction: column; gap: .15rem; }
  .status-item span { font-size: .6rem; color: var(--text3); text-transform: uppercase; letter-spacing: .05em; }
  .status-item strong { font-size: .85rem; }
  .running { color: var(--success, #22c55e); }
  .progress-bar { margin-top: .6rem; background: rgba(0,0,0,.2); border-radius: 6px; height: 7px; overflow: hidden; }
  .progress-fill { height: 100%; background: var(--primary, #7070ff); transition: width .5s ease; border-radius: 6px; }
  .done-banner { display: flex; align-items: center; gap: 1rem; padding: .7rem 1rem; background: rgba(34,197,94,.08); border: 1px solid rgba(34,197,94,.2); border-radius: 8px; color: var(--success); font-size: .9rem; }
  .idle-text { color: var(--text3); font-size: .85rem; }

  /* System Monitor */
  .sys-grid { display: flex; flex-direction: column; gap: .5rem; }
  .sys-item { display: grid; grid-template-columns: 90px 1fr auto; gap: .5rem; align-items: center; }
  .sys-label { font-size: .75rem; color: var(--text2); font-weight: 600; }
  .sys-bar { background: rgba(0,0,0,.2); border-radius: 4px; height: 8px; overflow: hidden; }
  .sys-fill { height: 100%; border-radius: 4px; transition: width .5s; }
  .sys-fill.cpu { background: #22c55e; }
  .sys-fill.ram { background: #3b82f6; }
  .sys-fill.gpu { background: #f59e0b; }
  .sys-val { font-size: .7rem; color: var(--text3); white-space: nowrap; }

  /* Chart */
  .chart-container { margin-top: .5rem; }
  .chart-svg { width: 100%; height: auto; }
  .chart-legend { display: flex; justify-content: space-between; font-size: .7rem; color: var(--text3); margin-top: .3rem; }

  /* Config */
  .cfg-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: .45rem; }
  .cfg-grid > div { padding: .5rem; background: rgba(0,0,0,.1); border: 1px solid var(--border, #222); border-radius: 6px; display: flex; flex-direction: column; gap: .1rem; }
  .cfg-grid span { font-size: .6rem; color: var(--text3); text-transform: uppercase; }
  .cfg-grid strong { font-size: .78rem; }

  @media(max-width:600px) {
    .status-grid { grid-template-columns: repeat(2, 1fr); }
    .cfg-grid { grid-template-columns: repeat(2, 1fr); }
    .sys-item { grid-template-columns: 70px 1fr; }
    .sys-val { grid-column: 1/-1; font-size: .65rem; }
  }
</style>
