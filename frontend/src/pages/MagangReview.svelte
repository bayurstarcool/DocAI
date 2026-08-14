<script>
  import { onMount } from "svelte"
  import { refreshIcons } from "../lib/icons.js"
  import { showToast } from "../stores/toast.js"

  const API = ""
  let users = []
  let loading = true
  let totalPairs = 0
  let selectedUser = null
  let moving = false
  let validating = false
  let previewPair = null
  let previewSide = "both"   // both | input | target
  let previewFull = false    // full-resolution toggle in modal

  // bulk selection: Set of "username::pairName"
  let selected = new Set()
  let bulkMoving = false
  let bulkProgress = ""

  onMount(() => {
    refreshIcons()
    loadReview()
  })

  async function loadReview() {
    loading = true
    try {
      const res = await fetch(API + "/api/magang/review", {
        headers: { Authorization: "Bearer " + (localStorage.getItem("docai_token") || "") }
      })
      if (!res.ok) throw new Error(await res.text())
      const d = await res.json()
      users = d.users || []
      totalPairs = d.total_pairs || 0
      selected = new Set()
    } catch (e) {
      showToast("Gagal load: " + e.message, "error")
    } finally {
      loading = false
    }
  }

  async function validatePair(username, pairName) {
    validating = true
    try {
      const res = await fetch(API + "/api/magang/validate-pair", {
        method: "POST",
        headers: {
          Authorization: "Bearer " + (localStorage.getItem("docai_token") || ""),
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ username, pair_name: pairName })
      })
      if (!res.ok) throw new Error(await res.text())
      const d = await res.json()
      users = users.map(u => u.username !== username ? u : {
        ...u,
        pairs: u.pairs.map(p => p.name !== pairName ? p : { ...p, validation: d, valid: d.valid })
      })
      if (previewPair?.user === username && previewPair?.name === pairName) {
        previewPair = { ...previewPair, validation: d, valid: d.valid }
      }
      if (d.valid) showToast(`${pairName} valid — skor ${Math.round(d.audit?.score || 0)} / ${d.audit?.grade || '-'}`, "success")
      else showToast(`${pairName} invalid: ${(d.issues || []).join(', ')}`, "error")
      return d.valid
    } catch (e) {
      showToast("Gagal validasi: " + e.message, "error")
      return false
    } finally {
      validating = false
    }
  }

  async function movePair(username, pairName) {
    const ok = await validatePair(username, pairName)
    if (!ok) return
    if (!confirm(`Pair valid. Pindahkan ${pairName} dari ${username} ke dataset custom?`)) return
    moving = true
    try {
      const res = await fetch(API + "/api/magang/move", {
        method: "POST",
        headers: {
          Authorization: "Bearer " + (localStorage.getItem("docai_token") || ""),
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ username, pair_name: pairName })
      })
      if (!res.ok) throw new Error(await res.text())
      showToast(`${pairName} dipindahkan ke custom!`, "success")
      await loadReview()
    } catch (e) {
      showToast("Gagal pindah: " + e.message, "error")
    } finally {
      moving = false
    }
  }

  // ---- bulk selection helpers ----
  function keyOf(username, pairName) { return username + "::" + pairName }

  function toggleSelect(username, pairName) {
    const k = keyOf(username, pairName)
    const next = new Set(selected)
    if (next.has(k)) next.delete(k); else next.add(k)
    selected = next
  }

  function isSelected(username, pairName) {
    return selected.has(keyOf(username, pairName))
  }

  function selectedCountForUser(user) {
    return user.pairs.filter(p => selected.has(keyOf(user.username, p.name))).length
  }

  function selectAllPaired(user) {
    const next = new Set(selected)
    for (const p of user.pairs) if (p.paired) next.add(keyOf(user.username, p.name))
    selected = next
  }

  function clearUserSelection(user) {
    const next = new Set(selected)
    for (const p of user.pairs) next.delete(keyOf(user.username, p.name))
    selected = next
  }

  async function bulkMove(user) {
    const names = user.pairs
      .filter(p => p.paired && selected.has(keyOf(user.username, p.name)))
      .map(p => p.name)
    if (names.length === 0) { showToast("Belum ada pair dipilih", "error"); return }
    if (!confirm(`Pindahkan ${names.length} pair dari ${user.username} ke custom? Pair invalid akan dilewati.`)) return
    bulkMoving = true
    bulkProgress = `Memproses ${names.length} pair...`
    try {
      const res = await fetch(API + "/api/magang/move-bulk", {
        method: "POST",
        headers: {
          Authorization: "Bearer " + (localStorage.getItem("docai_token") || ""),
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ username: user.username, pair_names: names })
      })
      if (!res.ok) throw new Error(await res.text())
      const d = await res.json()
      const skipped = (d.results || []).filter(r => !r.moved)
      showToast(`${d.moved}/${d.requested} pair dipindahkan.` + (skipped.length ? ` ${skipped.length} dilewati (invalid).` : ""), skipped.length ? "warn" : "success")
      if (skipped.length) {
        const detail = skipped.map(r => r.pair_name + (r.error ? `: ${r.error}` : "")).join(", ")
        bulkProgress = "Dilewati: " + detail
      } else {
        bulkProgress = ""
      }
      await loadReview()
      // reopen the same user after reload
      selectedUser = users.find(u => u.username === user.username) || null
    } catch (e) {
      showToast("Gagal bulk move: " + e.message, "error")
    } finally {
      bulkMoving = false
    }
  }

  // full-res when size omitted; thumbnails use size param
  function imgURL(subpath, size) {
    if (!subpath) return ""
    let u = API + "/api/datasets/explorer/image?dataset=" + encodeURIComponent("datasets/magang") + "&path=" + encodeURIComponent(subpath)
    if (size) u += "&size=" + size
    return u
  }

  function openPreview(user, pair, side) {
    previewSide = side || "both"
    previewFull = false
    previewPair = { user: user.username, ...pair }
  }

  function toggleUser(user) {
    selectedUser = selectedUser?.username === user.username ? null : user
  }
</script>

<div class="page">
  <div class="header-row">
    <div>
      <h1><i data-lucide="clipboard-check"></i> Review Dataset Magang</h1>
      <p class="subtitle">Validasi, preview full-res, dan pindahkan dataset dari intern ke custom.</p>
    </div>
    <button class="btn refresh-btn" onclick={loadReview}>
      <i data-lucide="refresh-cw"></i> Refresh
    </button>
  </div>

  {#if loading}
    <p class="loading">Loading...</p>
  {:else if users.length === 0}
    <div class="empty">
      <i data-lucide="inbox"></i>
      <p>Belum ada dataset dari intern.</p>
    </div>
  {:else}
    <div class="summary">
      <span class="sum-card"><strong>{users.length}</strong> intern</span>
      <span class="sum-card"><strong>{totalPairs}</strong> total pair</span>
      {#if selected.size}<span class="sum-card sel"><strong>{selected.size}</strong> dipilih</span>{/if}
    </div>

    <div class="user-list">
      {#each users as user}
        <div class="user-card">
          <div class="user-header" onclick={() => toggleUser(user)}>
            <i data-lucide="user"></i>
            <h3>{user.username}</h3>
            <div class="user-stats">
              <span>{user.paired_count} pair</span>
              <span>{user.input_count} input</span>
              <span>{user.target_count} target</span>
              {#if selectedCountForUser(user)}<span class="sel">{selectedCountForUser(user)} dipilih</span>{/if}
            </div>
            <span class="expand-icon">{selectedUser?.username === user.username ? "▲" : "▼"}</span>
          </div>

          {#if selectedUser?.username === user.username}
            <div class="pairs-section">
              {#if user.pairs.length === 0}
                <p class="empty-text">Belum ada pair.</p>
              {:else}
                <div class="bulk-bar">
                  <button class="btn-small" onclick={(e) => { e.stopPropagation(); selectAllPaired(user) }}>Pilih semua paired</button>
                  <button class="btn-small" onclick={(e) => { e.stopPropagation(); clearUserSelection(user) }}>Bersihkan pilihan</button>
                  <button class="btn-move bulk" disabled={bulkMoving || selectedCountForUser(user) === 0} onclick={(e) => { e.stopPropagation(); bulkMove(user) }}>
                    <i data-lucide="arrow-right-to-line"></i> Pindah {selectedCountForUser(user)} pair
                  </button>
                  {#if bulkMoving}<span class="bulk-progress">{bulkProgress || "Memproses..."}</span>{/if}
                </div>
                {#if bulkProgress && !bulkMoving}<div class="bulk-note">{bulkProgress}</div>{/if}
                <div class="pairs-grid">
                  {#each user.pairs as pair}
                    <div class="pair-card" class:unpaired={!pair.paired} class:checked={isSelected(user.username, pair.name)}>
                      {#if pair.paired}
                        <label class="pair-check" onclick={(e) => e.stopPropagation()}>
                          <input type="checkbox" checked={isSelected(user.username, pair.name)} onchange={() => toggleSelect(user.username, pair.name)} />
                          <span>Pilih</span>
                        </label>
                      {/if}
                      <div class="pair-thumbs">
                        {#if pair.input}
                          <button class="thumb-wrap" title="Preview input full-res" onclick={(e) => { e.stopPropagation(); openPreview(user, pair, 'input') }}>
                            <span class="tl">Input</span>
                            <img src={imgURL(pair.input, 200)} alt="input" class="thumb" loading="lazy" />
                          </button>
                        {:else}
                          <div class="thumb-wrap empty">N/A</div>
                        {/if}
                        {#if pair.target}
                          <button class="thumb-wrap" title="Preview target full-res" onclick={(e) => { e.stopPropagation(); openPreview(user, pair, 'target') }}>
                            <span class="tl">Target</span>
                            <img src={imgURL(pair.target, 200)} alt="target" class="thumb" loading="lazy" />
                          </button>
                        {:else}
                          <div class="thumb-wrap empty">N/A</div>
                        {/if}
                      </div>
                      <div class="pair-footer">
                        <span class="pair-name">{pair.name}</span>
                        <span class:badge-ok={pair.valid} class:badge-warn={!pair.valid}>
                          {#if pair.validation?.audit}
                            Skor {Math.round(pair.validation.audit.score)} / {pair.validation.audit.grade}
                          {:else}
                            {pair.valid ? "Valid" : (pair.validation?.issues?.[0] || "Invalid")}
                          {/if}
                        </span>
                      </div>
                      <div class="pair-actions">
                        <button class="btn-small" onclick={(e) => { e.stopPropagation(); openPreview(user, pair, 'both') }}>Preview</button>
                        {#if pair.paired}
                          <button class="btn-small" disabled={validating} onclick={(e) => { e.stopPropagation(); validatePair(user.username, pair.name) }}>Validasi ulang</button>
                          <button class="btn-move" disabled={moving || !pair.valid} onclick={(e) => { e.stopPropagation(); movePair(user.username, pair.name) }}>
                            <i data-lucide="arrow-right"></i> Pindah
                          </button>
                        {:else}
                          <span class="badge-warn">Unpaired</span>
                        {/if}
                      </div>
                    </div>
                  {/each}
                </div>
              {/if}
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {/if}
</div>

{#if previewPair}
  <div class="preview-modal" onclick={() => previewPair = null}>
    <div class="preview-box" onclick={(e) => e.stopPropagation()}>
      <div class="preview-head">
        <h3>{previewPair.user} / {previewPair.name}</h3>
        <div class="preview-head-actions">
          <div class="seg">
            <button class:active={previewSide==='both'} onclick={() => previewSide='both'}>Both</button>
            <button class:active={previewSide==='input'} onclick={() => previewSide='input'}>Input</button>
            <button class:active={previewSide==='target'} onclick={() => previewSide='target'}>Target</button>
          </div>
          <button class="btn" class:active={previewFull} onclick={() => previewFull = !previewFull}>
            <i data-lucide="maximize"></i> {previewFull ? "Fit" : "Full res"}
          </button>
        </div>
      </div>
      <div class="preview-status" class:ok={previewPair.valid}>
        {previewPair.valid ? "VALID" : "INVALID"}
        {#if previewPair.validation?.audit}
          — Skor {Math.round(previewPair.validation.audit.score)} / Grade {previewPair.validation.audit.grade}
        {/if}
        {#if previewPair.validation?.issues?.length}— {previewPair.validation.issues.join(', ')}{/if}
      </div>
      {#if previewPair.validation?.audit?.reasons?.length}
        <div class="audit-reasons">
          <b>Alasan:</b> {previewPair.validation.audit.reasons.join(', ')}
        </div>
      {/if}
      <div class="preview-images" class:single={previewSide!=='both'} class:full={previewFull}>
        {#if previewSide !== 'target'}
          <div><b>Input</b>{#if previewPair.input}<img src={imgURL(previewPair.input)} alt="input" class:fullres={previewFull} />{/if}</div>
        {/if}
        {#if previewSide !== 'input'}
          <div><b>Target</b>{#if previewPair.target}<img src={imgURL(previewPair.target)} alt="target" class:fullres={previewFull} />{/if}</div>
        {/if}
      </div>
      <div class="preview-footer">
        <button class="btn" onclick={() => previewPair = null}><i data-lucide="x"></i> Close</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .page { max-width: 1000px; margin: 0 auto; padding: 2rem; }
  .header-row { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem; }
  h1 { font-size: 1.4rem; display: flex; align-items: center; gap: 0.5rem; }
  h1 :global(i) { width: 28px; height: 28px; color: var(--accent); }
  .subtitle { color: var(--text2); font-size: 0.9rem; margin-top: 0.25rem; }
  .btn { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.5rem 1rem; border: 1px solid var(--border); border-radius: var(--radius-xs); font-weight: 600; cursor: pointer; font-size: 0.85rem; background: var(--bg2); color: var(--text); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn.active { border-color: var(--accent); color: var(--accent2); background: rgba(6,182,212,0.1); }
  .btn :global(i) { width: 16px; height: 16px; }
  .refresh-btn { background: var(--bg2); }

  .loading { color: var(--text2); }
  .empty { text-align: center; padding: 3rem; color: var(--text2); }
  .empty :global(i) { width: 48px; height: 48px; color: var(--text3); margin-bottom: 0.5rem; }

  .summary { display: flex; gap: 0.75rem; margin-bottom: 1.5rem; }
  .sum-card { padding: 0.5rem 1rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); font-size: 0.9rem; }
  .sum-card.sel { border-color: var(--accent); color: var(--accent2); }
  .sel { color: var(--accent2); }

  .user-list { display: flex; flex-direction: column; gap: 0.5rem; }
  .user-card { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
  .user-header { display: flex; align-items: center; gap: 0.75rem; padding: 1rem; cursor: pointer; transition: background 0.15s; }
  .user-header:hover { background: rgba(255,255,255,0.02); }
  .user-header :global(i) { width: 20px; height: 20px; color: var(--accent); }
  .user-header h3 { font-size: 1rem; flex-shrink: 0; }
  .user-stats { display: flex; gap: 0.5rem; color: var(--text2); font-size: 0.8rem; flex-wrap: wrap; }
  .user-stats span { padding: 0.15rem 0.5rem; background: var(--bg2); border-radius: 999px; }
  .user-stats span.sel { background: rgba(6,182,212,0.12); color: var(--accent2); }
  .expand-icon { margin-left: auto; color: var(--text3); font-size: 0.8rem; }

  .pairs-section { padding: 0 1rem 1rem; border-top: 1px solid var(--border); padding-top: 0.75rem; }
  .bulk-bar { display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; margin-bottom:.75rem; padding:.5rem; background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius-xs); }
  .bulk-progress { font-size:.75rem; color:var(--text2); }
  .bulk-note { font-size:.75rem; color:#f59e0b; margin-bottom:.5rem; }
  .pairs-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.75rem; }
  .pair-card { padding: 0.5rem; border: 1px solid var(--border); border-radius: var(--radius-xs); background: var(--bg2); position: relative; }
  .pair-card.unpaired { opacity: 0.6; }
  .pair-card.checked { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent) inset; }
  .pair-check { display:flex; align-items:center; gap:.3rem; font-size:.7rem; color:var(--text2); margin-bottom:.4rem; cursor:pointer; user-select:none; }
  .pair-check input { width:15px; height:15px; accent-color: var(--accent); cursor:pointer; }
  .pair-thumbs { display: flex; gap: 0.35rem; margin-bottom: 0.5rem; }
  .thumb-wrap { flex: 1; position: relative; aspect-ratio: 4/3; background: var(--bg); border-radius: 4px; overflow: hidden; padding:0; border:none; cursor: zoom-in; }
  .thumb-wrap.empty { display: flex; align-items: center; justify-content: center; color: var(--text3); font-size: 0.65rem; cursor: default; }
  .tl { position: absolute; top: 2px; left: 2px; font-size: 0.5rem; font-weight: 700; color: white; background: rgba(0,0,0,0.6); padding: 0.05rem 0.2rem; border-radius: 2px; z-index: 1; }
  .thumb { width: 100%; height: 100%; object-fit: contain; }
  .pair-footer { display: flex; justify-content: space-between; align-items: center; }
  .pair-name { font-family: monospace; font-size: 0.75rem; }
  .btn-move { padding: 0.25rem 0.6rem; background: #22c55e; color: white; border: none; border-radius: 4px; font-size: 0.7rem; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 0.2rem; }
  .btn-move.bulk { padding: 0.4rem 0.8rem; font-size:.75rem; }
  .btn-move:disabled { opacity: 0.5; cursor:not-allowed; }
  .btn-move :global(i) { width: 12px; height: 12px; }
  .badge-warn { font-size: 0.65rem; padding: 0.1rem 0.4rem; background: rgba(239,68,68,0.1); color: #ef4444; border-radius: 999px; }
  .pair-actions { display:flex; gap:.35rem; flex-wrap:wrap; align-items:center; margin-top:.5rem; }
  .btn-small { padding:0.25rem .5rem; background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:4px; font-size:.68rem; cursor:pointer; }
  .badge-ok { font-size:0.65rem; padding:0.1rem 0.4rem; background:rgba(34,197,94,0.12); color:#22c55e; border-radius:999px; }
  .preview-modal { position:fixed; inset:0; background:rgba(0,0,0,.75); z-index:50; display:flex; align-items:center; justify-content:center; padding:1rem; }
  .preview-box { width:min(1100px,96vw); max-height:92vh; overflow:auto; background:var(--bg); border:1px solid var(--border); border-radius:var(--radius); padding:1rem; }
  .preview-head { display:flex; justify-content:space-between; align-items:center; gap:1rem; margin-bottom:.75rem; flex-wrap:wrap; }
  .preview-head-actions { display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; }
  .seg { display:inline-flex; border:1px solid var(--border); border-radius:var(--radius-xs); overflow:hidden; }
  .seg button { padding:.4rem .7rem; background:var(--bg2); color:var(--text2); border:none; cursor:pointer; font-size:.78rem; }
  .seg button.active { background:rgba(6,182,212,0.15); color:var(--accent2); }
  .preview-status { padding:.5rem .75rem; border-radius:8px; background:rgba(239,68,68,.12); color:#ef4444; margin-bottom:.75rem; font-weight:700; }
  .preview-status.ok { background:rgba(34,197,94,.12); color:#22c55e; }
  .audit-reasons { padding:.5rem .75rem; margin-bottom:.75rem; border:1px solid var(--border); border-radius:8px; color:var(--text2); background:var(--bg2); font-size:.85rem; }
  .preview-images { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
  .preview-images.single { grid-template-columns:1fr; }
  .preview-images div { background:var(--bg2); border:1px solid var(--border); border-radius:10px; padding:.75rem; }
  .preview-images b { display:block; margin-bottom:.5rem; }
  .preview-images img { width:100%; max-height:72vh; object-fit:contain; background:#111; border-radius:8px; }
  /* full-res: show at natural size, scroll within panel */
  .preview-images.full div { overflow:auto; max-height:80vh; }
  .preview-images img.fullres { width:auto; max-width:none; max-height:none; object-fit:none; }
  .preview-footer { display:flex; justify-content:flex-end; margin-top:1rem; padding-top:.75rem; border-top:1px solid var(--border); }
  .preview-footer .btn { padding:.55rem 1.4rem; }

  .empty-text { color: var(--text3); font-size: 0.8rem; padding: 1rem 0; }
</style>
