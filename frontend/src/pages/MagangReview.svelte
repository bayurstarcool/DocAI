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
    } catch (e) {
      showToast("Gagal load: " + e.message, "error")
    } finally {
      loading = false
    }
  }

  async function movePair(username, pairName) {
    if (!confirm(`Pindahkan ${pairName} dari ${username} ke dataset custom?`)) return
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

  function imgURL(subpath) {
    if (!subpath) return ""
    return API + "/api/datasets/explorer/image?dataset=" + encodeURIComponent("datasets/magang") + "&path=" + encodeURIComponent(subpath) + "&size=200"
  }

  function toggleUser(user) {
    selectedUser = selectedUser?.username === user.username ? null : user
  }
</script>

<div class="page">
  <div class="header-row">
    <div>
      <h1><i data-lucide="clipboard-check"></i> Review Dataset Magang</h1>
      <p class="subtitle">Validasi dan pindahkan dataset dari intern ke custom.</p>
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
    </div>

    <div class="user-list">
      {#each users as user}
        <div class="user-card" onclick={() => toggleUser(user)}>
          <div class="user-header">
            <i data-lucide="user"></i>
            <h3>{user.username}</h3>
            <div class="user-stats">
              <span>{user.paired_count} pair</span>
              <span>{user.input_count} input</span>
              <span>{user.target_count} target</span>
            </div>
            <span class="expand-icon">{selectedUser?.username === user.username ? "▲" : "▼"}</span>
          </div>

          {#if selectedUser?.username === user.username}
            <div class="pairs-section">
              {#if user.pairs.length === 0}
                <p class="empty-text">Belum ada pair.</p>
              {:else}
                <div class="pairs-grid">
                  {#each user.pairs as pair}
                    <div class="pair-card" class:unpaired={!pair.paired}>
                      <div class="pair-thumbs">
                        {#if pair.input}
                          <div class="thumb-wrap">
                            <span class="tl">Input</span>
                            <img src={imgURL(pair.input)} alt="input" class="thumb" loading="lazy" />
                          </div>
                        {:else}
                          <div class="thumb-wrap empty">N/A</div>
                        {/if}
                        {#if pair.target}
                          <div class="thumb-wrap">
                            <span class="tl">Target</span>
                            <img src={imgURL(pair.target)} alt="target" class="thumb" loading="lazy" />
                          </div>
                        {:else}
                          <div class="thumb-wrap empty">N/A</div>
                        {/if}
                      </div>
                      <div class="pair-footer">
                        <span class="pair-name">{pair.name}</span>
                        {#if pair.paired}
                          <button class="btn-move" disabled={moving} onclick={(e) => { e.stopPropagation(); movePair(user.username, pair.name) }}>
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

<style>
  .page { max-width: 1000px; margin: 0 auto; padding: 2rem; }
  .header-row { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem; }
  h1 { font-size: 1.4rem; display: flex; align-items: center; gap: 0.5rem; }
  h1 :global(i) { width: 28px; height: 28px; color: var(--accent); }
  .subtitle { color: var(--text2); font-size: 0.9rem; margin-top: 0.25rem; }
  .btn { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.5rem 1rem; border: 1px solid var(--border); border-radius: var(--radius-xs); font-weight: 600; cursor: pointer; font-size: 0.85rem; background: var(--bg2); color: var(--text); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn :global(i) { width: 16px; height: 16px; }
  .refresh-btn { background: var(--bg2); }

  .loading { color: var(--text2); }
  .empty { text-align: center; padding: 3rem; color: var(--text2); }
  .empty :global(i) { width: 48px; height: 48px; color: var(--text3); margin-bottom: 0.5rem; }

  .summary { display: flex; gap: 0.75rem; margin-bottom: 1.5rem; }
  .sum-card { padding: 0.5rem 1rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); font-size: 0.9rem; }

  .user-list { display: flex; flex-direction: column; gap: 0.5rem; }
  .user-card { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
  .user-header { display: flex; align-items: center; gap: 0.75rem; padding: 1rem; cursor: pointer; transition: background 0.15s; }
  .user-header:hover { background: rgba(255,255,255,0.02); }
  .user-header :global(i) { width: 20px; height: 20px; color: var(--accent); }
  .user-header h3 { font-size: 1rem; flex-shrink: 0; }
  .user-stats { display: flex; gap: 0.5rem; color: var(--text2); font-size: 0.8rem; }
  .user-stats span { padding: 0.15rem 0.5rem; background: var(--bg2); border-radius: 999px; }
  .expand-icon { margin-left: auto; color: var(--text3); font-size: 0.8rem; }

  .pairs-section { padding: 0 1rem 1rem; border-top: 1px solid var(--border); padding-top: 0.75rem; }
  .pairs-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.75rem; }
  .pair-card { padding: 0.5rem; border: 1px solid var(--border); border-radius: var(--radius-xs); background: var(--bg2); }
  .pair-card.unpaired { opacity: 0.6; }
  .pair-thumbs { display: flex; gap: 0.35rem; margin-bottom: 0.5rem; }
  .thumb-wrap { flex: 1; position: relative; aspect-ratio: 4/3; background: var(--bg); border-radius: 4px; overflow: hidden; }
  .thumb-wrap.empty { display: flex; align-items: center; justify-content: center; color: var(--text3); font-size: 0.65rem; }
  .tl { position: absolute; top: 2px; left: 2px; font-size: 0.5rem; font-weight: 700; color: white; background: rgba(0,0,0,0.6); padding: 0.05rem 0.2rem; border-radius: 2px; z-index: 1; }
  .thumb { width: 100%; height: 100%; object-fit: contain; }
  .pair-footer { display: flex; justify-content: space-between; align-items: center; }
  .pair-name { font-family: monospace; font-size: 0.75rem; }
  .btn-move { padding: 0.25rem 0.6rem; background: #22c55e; color: white; border: none; border-radius: 4px; font-size: 0.7rem; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 0.2rem; }
  .btn-move:disabled { opacity: 0.5; }
  .btn-move :global(i) { width: 12px; height: 12px; }
  .badge-warn { font-size: 0.65rem; padding: 0.1rem 0.4rem; background: rgba(239,68,68,0.1); color: #ef4444; border-radius: 999px; }
  .empty-text { color: var(--text3); font-size: 0.8rem; padding: 1rem 0; }
</style>
