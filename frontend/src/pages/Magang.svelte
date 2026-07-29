<script>
  import { refreshIcons } from "../lib/icons.js"
  import { onMount } from "svelte"

  const API = ""

  // Auth
  let pageMode = "login"
  let username = ""
  let password = ""
  let password2 = ""
  let authError = ""
  let authLoading = false
  let loggedIn = false
  let loggedUser = ""
  let isAdmin = false

  // Upload
  let shadowFile = null
  let cleanFile = null
  let shadowPreview = null
  let cleanPreview = null
  let auditResult = null
  let adding = false
  let added = false
  let error = ""

  // Stats
  let datasetInfo = null
  let loadingStats = false

  // Previews
  let previewPair = null
  let pairsList = []

  onMount(() => {
    refreshIcons()
    checkLogin()
  })

  function checkLogin() {
    const token = localStorage.getItem("docai_token")
    if (!token) return
    fetch(API + "/api/auth/check", {
      headers: { Authorization: "Bearer " + token }
    }).then(r => r.json()).then(async d => {
      if (d.authenticated) {
        try {
          const payload = JSON.parse(atob(token))
          loggedUser = payload.user || "Magang"
        } catch (e) {
          loggedUser = "Magang"
        }
        // Check if admin
        const checkRes = await fetch(API + "/api/auth/check", {
          headers: { Authorization: "Bearer " + token }
        })
        const checkData = await checkRes.json()
        isAdmin = checkData.authenticated && loggedUser === "admin"
        if (!isAdmin) {
          loggedIn = true
          loadMagangInfo()
        }
      }
    }).catch(() => {})
  }

  async function doLogin() {
    if (!username || !password) { authError = "Isi username & password"; return }
    authLoading = true
    authError = ""
    try {
      const res = await fetch(API + "/api/auth/login/intern", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: "username=" + encodeURIComponent(username) + "&password=" + encodeURIComponent(password)
      })
      const d = await res.json()
      if (d.success) {
        localStorage.setItem("docai_token", d.token)
        loggedIn = true
        loggedUser = d.user
        isAdmin = false
        loadMagangInfo()
      } else {
        authError = "Username atau password salah"
      }
    } catch (e) {
      authError = "Gagal login: " + e.message
    } finally {
      authLoading = false
    }
  }

  async function doRegister() {
    if (!username || !password) { authError = "Isi username & password"; return }
    if (password.length < 4) { authError = "Password minimal 4 karakter"; return }
    if (password !== password2) { authError = "Password tidak cocok"; return }
    authLoading = true
    authError = ""
    try {
      const res = await fetch(API + "/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: "username=" + encodeURIComponent(username) + "&password=" + encodeURIComponent(password)
      })
      const d = await res.json()
      if (d.success) {
        localStorage.setItem("docai_token", d.token)
        loggedIn = true
        loggedUser = d.user
        isAdmin = false
        loadMagangInfo()
      } else {
        authError = d.error || "Gagal registrasi"
      }
    } catch (e) {
      authError = "Gagal registrasi: " + e.message
    } finally {
      authLoading = false
    }
  }

  function doLogout() {
    localStorage.removeItem("docai_token")
    loggedIn = false
    loggedUser = ""
    isAdmin = false
    username = ""
    password = ""
    password2 = ""
    shadowFile = null
    cleanFile = null
    shadowPreview = null
    cleanPreview = null
    auditResult = null
  }

  async function loadMagangInfo() {
    loadingStats = true
    try {
      const res = await fetch(API + "/api/magang/list", {
        headers: { Authorization: "Bearer " + (localStorage.getItem("docai_token") || "") }
      })
      if (res.ok) {
        const d = await res.json()
        if (d.datasets && d.datasets.length > 0) {
          datasetInfo = d.datasets[0]
        } else {
          datasetInfo = { paired_count: 0, input_count: 0, target_count: 0, username: loggedUser }
        }
        // Also get pairs via review API (admin only) or via list
        loadPairs()
      }
    } catch (e) {}
    finally { loadingStats = false }
  }

  async function loadPairs() {
    try {
      const userFolder = "datasets/magang/" + loggedUser
      // Coba load input folder, kalo 404 artinya belum ada file
      let inputFiles = []
      const res = await fetch(API + "/api/datasets/explorer?dataset=" + encodeURIComponent(userFolder) + "&path=input&limit=200", {
        headers: { Authorization: "Bearer " + (localStorage.getItem("docai_token") || "") }
      })
      if (res.ok) {
        const d = await res.json()
        inputFiles = d.entries?.filter(e => e.kind === "image") || []
      }
      
      let targetMap = {}
      const res2 = await fetch(API + "/api/datasets/explorer?dataset=" + encodeURIComponent(userFolder) + "&path=target&limit=200", {
        headers: { Authorization: "Bearer " + (localStorage.getItem("docai_token") || "") }
      })
      if (res2.ok) {
        const d2 = await res2.json()
        const targetFiles = d2.entries?.filter(e => e.kind === "image") || []
        targetFiles.forEach(f => {
          const stem = f.name.replace(/\.[^.]+$/, "")
          targetMap[stem] = f.path
        })
      }
      
      const pairs = []
      inputFiles.forEach(f => {
        const stem = f.name.replace(/\.[^.]+$/, "")
        pairs.push({
          name: stem,
          input: f.path,
          target: targetMap[stem] || null,
          paired: !!targetMap[stem]
        })
      })
      pairsList = pairs.sort((a, b) => b.name.localeCompare(a.name))
      
      const inputCount = inputFiles.length
      const targetCount = Object.keys(targetMap).length
      datasetInfo = {
        ...datasetInfo,
        input_count: inputCount,
        target_count: targetCount,
        paired_count: inputFiles.filter(f => {
          const stem = f.name.replace(/\.[^.]+$/, "")
          return !!targetMap[stem]
        }).length
      }
    } catch (e) {
      console.error("loadPairs err:", e)
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

  let validating = false

  async function runAudit() {
    if (!shadowFile || !cleanFile) return
    validating = true
    error = ""
    auditResult = null
    const fd = new FormData()
    fd.append("shadow", shadowFile)
    fd.append("clean", cleanFile)
    try {
      const res = await fetch(API + "/api/magang/audit", {
        method: "POST",
        headers: { Authorization: "Bearer " + (localStorage.getItem("docai_token") || "") },
        body: fd
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({detail: "Gagal validasi"}))
        throw new Error(err.detail || "Gagal validasi")
      }
      auditResult = await res.json()
    } catch (e) {
      error = e.message
    } finally {
      validating = false
    }
  }

  async function runAdd() {
    if (!auditResult?.passed) return
    adding = true
    error = ""
    const fd = new FormData()
    fd.append("shadow", shadowFile)
    fd.append("clean", cleanFile)
    try {
      const res = await fetch(API + "/api/magang/upload", {
        method: "POST",
        headers: { Authorization: "Bearer " + (localStorage.getItem("docai_token") || "") },
        body: fd
      })
      if (!res.ok) {
        let errMsg = "Gagal simpan"
        try { const d = await res.json(); errMsg = d.detail || errMsg } catch(_) { errMsg = await res.text().catch(() => errMsg) }
        throw new Error(errMsg)
      }
      added = true
      shadowFile = null
      cleanFile = null
      shadowPreview = null
      cleanPreview = null
      auditResult = null
      await loadMagangInfo()
      setTimeout(() => { added = false }, 3000)
    } catch (e) {
      error = e.message
    } finally {
      adding = false
    }
  }

  function openPreview(pair) { previewPair = pair }
  function closePreview() { previewPair = null }

  function imgURL(path) {
    if (!path) return ""
    return API + "/api/datasets/explorer/image?dataset=" + encodeURIComponent("datasets/magang/" + loggedUser) + "&path=" + encodeURIComponent(path) + "&size=200"
  }

  function gradeColor(g) {
    return g === "A" ? "#22c55e" : g === "B" ? "#eab308" : g === "C" ? "#f97316" : "#ef4444"
  }
</script>

<div class="page">
  {#if !loggedIn}
    <!-- AUTH -->
    <div class="auth-card">
      <div class="auth-header">
        <h1><i data-lucide="graduation-cap"></i> Portal Dataset Magang</h1>
        <p class="subtitle">Upload dataset shadow untuk training DocAI</p>
        <p class="subtitle" style="font-size:0.75rem;color:var(--text3);margin-top:0.5rem">
          Pastikan foto shadow dan clean sudah sesuai. Sistem akan validasi otomatis.
        </p>
      </div>

      <div class="tab-bar">
        <button class="tab" class:active={pageMode === "login"} onclick={() => { pageMode = "login"; authError = "" }}>
          Masuk
        </button>
        <button class="tab" class:active={pageMode === "register"} onclick={() => { pageMode = "register"; authError = "" }}>
          Daftar Baru
        </button>
      </div>

      {#if authError}
        <div class="alert error">{authError}</div>
      {/if}

      <div class="form">
        <div class="field">
          <label>Username</label>
          <input type="text" bind:value={username} placeholder="Username" />
        </div>
        <div class="field">
          <label>Password</label>
          <input type="password" bind:value={password} placeholder="Password" />
        </div>
        {#if pageMode === "register"}
          <div class="field">
            <label>Konfirmasi Password</label>
            <input type="password" bind:value={password2} placeholder="Ulangi password" />
          </div>
        {/if}
        <button class="btn primary" onclick={pageMode === "login" ? doLogin : doRegister} disabled={authLoading}>
          {authLoading ? "Memproses..." : pageMode === "login" ? "Masuk" : "Daftar"}
        </button>
      </div>

      {#if pageMode === "register"}
        <p class="info-text">Setelah daftar, kamu langsung bisa upload dataset. Admin akan review hasil upload.</p>
      {:else}
        <p class="info-text">Punya akun? Masuk. Belum punya? Klik "Daftar Baru".</p>
      {/if}
    </div>

  {:else}
    <!-- MAIN UPLOAD PAGE -->
    <div class="header-row">
      <div>
        <h1><i data-lucide="graduation-cap"></i> Upload Dataset</h1>
        <p class="subtitle">Halo, {loggedUser}! Upload pasangan shadow + clean di sini.</p>
      </div>
      <button class="btn logout-btn" onclick={doLogout}>
        <i data-lucide="log-out"></i> Keluar
      </button>
    </div>

    {#if error}
      <div class="alert error">{error}</div>
    {/if}

    <!-- Stats -->
    {#if datasetInfo}
      <div class="stats-row">
        <span class="stat"><strong>{datasetInfo.paired_count}</strong> pair</span>
        <span class="stat"><strong>{datasetInfo.input_count}</strong> input</span>
        <span class="stat"><strong>{datasetInfo.target_count}</strong> target</span>
        <span class="stat" style="color:var(--text3)">Folder: datasets/magang/{loggedUser}/</span>
      </div>
    {:else if loadingStats}
      <div class="stats-row"><span class="stat">Memuat statistik...</span></div>
    {/if}

    <!-- Upload Form -->
    <div class="upload-section">
      <h2>Tambah Pair Baru</h2>
      <p class="section-hint">Upload foto dokumen dengan shadow (input) dan foto bersih tanpa shadow (target).</p>

      <div class="upload-row">
        <div class="upload-box">
          <label>Shadow (Input) — foto HP dengan shadow</label>
          {#if shadowPreview}
            <img src={shadowPreview} alt="shadow" class="preview" />
          {:else}
            <div class="placeholder">Klik untuk pilih foto shadow</div>
          {/if}
          <input type="file" accept="image/*" onchange={handleShadow} />
        </div>
        <div class="upload-box">
          <label>Clean (Target) — foto bersih tanpa bayangan</label>
          {#if cleanPreview}
            <img src={cleanPreview} alt="clean" class="preview" />
          {:else}
            <div class="placeholder">Klik untuk pilih foto clean</div>
          {/if}
          <input type="file" accept="image/*" onchange={handleClean} />
        </div>
      </div>

      <div class="action-row">
        <button class="btn" disabled={!shadowFile || !cleanFile || adding} onclick={runAudit}>
          <i data-lucide="search-check"></i>
          {validating ? "Memvalidasi..." : "Validasi Pair"}
        </button>
        <button class="btn success" disabled={!auditResult?.passed || adding} onclick={runAdd}>
          {added ? "✓ Tersimpan!" : adding ? "Menyimpan..." : "Simpan Pair"}
        </button>
      </div>

      {#if error}
        <div class="alert error">{error}</div>
      {/if}

      <!-- Audit Result -->
      {#if auditResult}
        <div class="audit-card">
          <div class="audit-header">
            <span>Grade:</span>
            <span class="grade" style="background:{gradeColor(auditResult.grade)}">{auditResult.grade}</span>
            <span class="score">Score: {auditResult.score}/100</span>
            <span class="score" style="color:{auditResult.passed ? '#22c55e' : '#ef4444'};font-size:0.8rem">
              {auditResult.passed ? "✓ LULUS" : "✗ TIDAK LULUS (min 90)"}
            </span>
          </div>
          <div class="audit-grid">
            <div class="audit-item">
              <span class="label">Resolusi</span>
              <span class="value">{auditResult.dimensions.shadow.w}x{auditResult.dimensions.shadow.h} / {auditResult.dimensions.clean.w}x{auditResult.dimensions.clean.h}
                {auditResult.dimensions.match ? " ✓ Sama" : " ⚠ Akan di-resize"}
              </span>
            </div>
            <div class="audit-item">
              <span class="label">Shadow</span>
              <span class="value">{auditResult.luminance.shadow_pct}% area gelap di shadow vs {auditResult.luminance.clean_dark_pct}% di clean</span>
            </div>
          </div>
          {#if auditResult.reasons.length > 0}
            <div class="reasons">
              {#each auditResult.reasons as r}
                <span class="reason-tag">{r}</span>
              {/each}
            </div>
          {:else}
            <div class="reasons">
              <span class="reason-tag ok">✓ Pair ini bagus!</span>
            </div>
          {/if}
        </div>
      {/if}
    </div>

    <!-- Pairs List -->
    {#if pairsList.length > 0}
      <div class="section">
        <h2>Pair Saya ({pairsList.length})</h2>
        <div class="pairs-grid">
          {#each pairsList as pair}
            <div class="pair-card" onclick={() => openPreview(pair)}>
              <div class="pair-thumbs">
                {#if pair.input}
                  <div class="thumb-wrap">
                    <span class="thumb-label">Input</span>
                    <img src={imgURL(pair.input)} alt="input" class="thumb" loading="lazy" />
                  </div>
                {:else}
                  <div class="thumb-wrap empty">No input</div>
                {/if}
                {#if pair.target}
                  <div class="thumb-wrap">
                    <span class="thumb-label">Target</span>
                    <img src={imgURL(pair.target)} alt="target" class="thumb" loading="lazy" />
                  </div>
                {:else}
                  <div class="thumb-wrap empty">No target</div>
                {/if}
              </div>
              <span class="pair-name">{pair.name}</span>
              <span class="pair-status" class:paired={pair.paired}>
                {pair.paired ? "✓ Paired" : "⚠ Unpaired"}
              </span>
            </div>
          {/each}
        </div>
      </div>
    {:else if !loadingStats}
      <div class="empty-state">
        <i data-lucide="image-plus"></i>
        <p>Belum ada pair. Upload di atas untuk mulai.</p>
      </div>
    {/if}

    <!-- Tips -->
    <div class="tips">
      <h2>Tips Upload</h2>
      <ul>
        <li>Pastikan shadow dan clean adalah <strong>dokumen yang sama</strong></li>
        <li>Shadow = foto HP dengan bayangan tangan/benda</li>
        <li>Clean = foto scan atau foto bersih tanpa bayangan</li>
        <li>Pair dengan score &ge; 50 baru bisa disimpan</li>
        <li>Admin akan review dan memindahkan ke dataset utama</li>
      </ul>
    </div>
  {/if}

  <!-- Preview Lightbox -->
  {#if previewPair}
    <div class="lightbox" onclick={closePreview} tabindex="-1">
      <div class="lb-content" onclick={(e) => e.stopPropagation()}>
        <button class="lb-close" onclick={closePreview}>&times;</button>
        <h3>{previewPair.name}</h3>
        <div class="lb-pair">
          {#if previewPair.input}
            <div class="lb-side">
              <span class="thumb-label">Input</span>
              <img src={imgURL(previewPair.input)} alt="input" class="lb-img" />
            </div>
          {/if}
          {#if previewPair.target}
            <div class="lb-side">
              <span class="thumb-label">Target</span>
              <img src={imgURL(previewPair.target)} alt="target" class="lb-img" />
            </div>
          {/if}
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .page { max-width: 800px; margin: 0 auto; padding: 2rem; }
  h1 { font-size: 1.4rem; display: flex; align-items: center; gap: 0.5rem; }
  h1 :global(i) { width: 28px; height: 28px; color: var(--accent); }
  .subtitle { color: var(--text2); font-size: 0.9rem; margin-top: 0.25rem; }

  /* Auth */
  .auth-card { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 2rem; max-width: 420px; margin: 2rem auto; }
  .auth-header { text-align: center; margin-bottom: 1.5rem; }
  .tab-bar { display: flex; gap: 0; margin-bottom: 1rem; border: 1px solid var(--border); border-radius: var(--radius-xs); overflow: hidden; }
  .tab { flex: 1; padding: 0.6rem; border: none; background: transparent; color: var(--text2); font-weight: 600; font-size: 0.85rem; cursor: pointer; }
  .tab.active { background: var(--accent); color: white; }
  .form { display: flex; flex-direction: column; gap: 0.75rem; }
  .field { display: flex; flex-direction: column; gap: 0.3rem; }
  .field label { font-size: 0.72rem; font-weight: 700; color: var(--text2); text-transform: uppercase; }
  .field input { padding: 0.65rem 0.75rem; background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius-xs); color: var(--text); font-size: 0.9rem; }
  .info-text { text-align: center; color: var(--text3); font-size: 0.8rem; margin-top: 1rem; }

  /* Main */
  .header-row { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem; }
  .logout-btn { background: transparent; border: 1px solid var(--border); color: var(--text2); padding: 0.4rem 0.75rem; }
  .logout-btn :global(i) { width: 16px; height: 16px; }
  .stats-row { display: flex; gap: 0.75rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .stat { padding: 0.35rem 0.75rem; background: var(--bg2); border: 1px solid var(--border); border-radius: 999px; font-size: 0.8rem; color: var(--text2); }

  /* Upload */
  .upload-section { margin-bottom: 2rem; }
  .section { margin-bottom: 2rem; }
  .section h2, .upload-section h2 { font-size: 1rem; margin-bottom: 0.5rem; }
  .section-hint { color: var(--text3); font-size: 0.8rem; margin-bottom: 1rem; }
  .upload-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1rem; }
  .upload-box { position: relative; border: 2px dashed var(--border); border-radius: var(--radius); padding: 1rem; text-align: center; min-height: 160px; display: flex; flex-direction: column; align-items: center; justify-content: center; }
  .upload-box label { display: block; font-size: 0.75rem; font-weight: 700; color: var(--text2); margin-bottom: 0.5rem; }
  .upload-box input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .preview { max-width: 100%; max-height: 200px; object-fit: contain; }
  .placeholder { color: var(--text3); font-size: 0.85rem; padding: 1rem; }
  .action-row { display: flex; gap: 0.75rem; margin-bottom: 1rem; }

  .btn { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.6rem 1.25rem; border: none; border-radius: var(--radius-xs); font-weight: 600; cursor: pointer; font-size: 0.85rem; background: var(--bg2); color: var(--text); border: 1px solid var(--border); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn :global(i) { width: 16px; height: 16px; }
  .btn.primary { background: var(--accent); color: white; border: none; width: 100%; justify-content: center; padding: 0.75rem; font-size: 1rem; }
  .btn.success { background: #22c55e; color: white; border: none; }

  .alert { padding: 0.75rem 1rem; border-radius: var(--radius-xs); margin-bottom: 1rem; font-size: 0.85rem; }
  .alert.error { background: rgba(239,68,68,0.1); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

  /* Audit */
  .audit-card { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem; }
  .audit-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem; font-size: 0.85rem; color: var(--text2); }
  .grade { display: inline-flex; align-items: center; justify-content: center; width: 2.2rem; height: 2.2rem; border-radius: 50%; color: white; font-weight: 800; font-size: 1.1rem; }
  .score { font-weight: 600; font-size: 0.9rem; }
  .audit-grid { display: flex; flex-direction: column; gap: 0.3rem; }
  .audit-item { display: flex; gap: 0.75rem; font-size: 0.82rem; padding: 0.35rem 0; border-bottom: 1px solid var(--border); }
  .audit-item .label { font-weight: 600; min-width: 80px; color: var(--text2); }
  .reasons { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.75rem; }
  .reason-title { font-size: 0.82rem; color: var(--text2); font-weight: 600; width: 100%; }
  .reason-tag { background: rgba(239,68,68,0.1); color: #ef4444; padding: 0.2rem 0.5rem; border-radius: 999px; font-size: 0.72rem; }
  .reason-tag.ok { background: rgba(34,197,94,0.1); color: #22c55e; }

  /* Pairs */
  .pairs-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 0.75rem; }
  .pair-card { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 0.75rem; cursor: pointer; transition: border-color 0.15s; }
  .pair-card:hover { border-color: var(--accent); }
  .pair-thumbs { display: flex; gap: 0.5rem; margin-bottom: 0.5rem; }
  .thumb-wrap { flex: 1; position: relative; background: var(--bg2); border-radius: var(--radius-xs); overflow: hidden; aspect-ratio: 4/3; }
  .thumb-wrap.empty { display: flex; align-items: center; justify-content: center; color: var(--text3); font-size: 0.7rem; }
  .thumb-label { position: absolute; top: 4px; left: 4px; font-size: 0.55rem; font-weight: 700; text-transform: uppercase; color: white; background: rgba(0,0,0,0.6); padding: 0.1rem 0.3rem; border-radius: 3px; z-index: 1; }
  .thumb { width: 100%; height: 100%; object-fit: contain; }
  .pair-name { font-family: monospace; font-size: 0.85rem; display: block; }
  .pair-status { font-size: 0.7rem; color: var(--text3); }
  .pair-status.paired { color: #22c55e; }

  /* Empty */
  .empty-state { text-align: center; padding: 3rem; color: var(--text2); }
  .empty-state :global(i) { width: 48px; height: 48px; color: var(--text3); margin-bottom: 0.5rem; }

  /* Tips */
  .tips { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem; }
  .tips h2 { font-size: 1rem; margin-bottom: 0.5rem; }
  .tips ul { padding-left: 1.25rem; font-size: 0.85rem; color: var(--text2); display: flex; flex-direction: column; gap: 0.4rem; }
  .tips li { line-height: 1.5; }

  /* Lightbox */
  .lightbox { position: fixed; inset: 0; background: rgba(0,0,0,0.8); z-index: 1000; display: flex; align-items: center; justify-content: center; padding: 2rem; }
  .lb-content { background: var(--bg); border-radius: var(--radius); padding: 1.5rem; max-width: 900px; width: 100%; max-height: 90vh; overflow-y: auto; position: relative; }
  .lb-close { position: absolute; top: 0.75rem; right: 0.75rem; background: none; border: none; color: var(--text2); font-size: 1.5rem; cursor: pointer; }
  .lb-close:hover { color: var(--text); }
  .lb-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
  .lb-side { text-align: center; }
  .lb-img { max-width: 100%; max-height: 60vh; object-fit: contain; border-radius: var(--radius-xs); background: var(--bg2); }
</style>
