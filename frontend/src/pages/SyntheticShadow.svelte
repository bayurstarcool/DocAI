<script>
  import { onMount, onDestroy } from 'svelte'
  import { refreshIcons } from '../lib/icons.js'
  import { showToast } from '../stores/toast.js'

  const API = window.location.origin
  let files = []
  let processedImages = []
  let processing = false
  let position = 'bottom'
  let objType = 'none'
  let intensity = 0.7
  let widthPct = 0.4
  let marginX = 0.0
  let marginY = 0.0
  let saving = false
  let selected = new Set()
  let selectAll = true
  let fileInput

  const objTypes = [
    { id: 'none', label: 'Gradient' },
    { id: 'hand', label: 'Tangan' },
    { id: 'phone', label: 'HP' },
    { id: 'finger', label: 'Jari' },
  ]

  const positions = [
    { id: 'bottom', label: 'Bawah' },
    { id: 'top', label: 'Atas' },
    { id: 'left', label: 'Kiri' },
    { id: 'right', label: 'Kanan' },
    { id: 'all', label: 'Semua Sisi' },
    { id: 'bottom_left', label: 'Bawah Kiri' },
    { id: 'bottom_right', label: 'Bawah Kanan' },
  ]

  let filePreviews = []

  function onFilesChange(e) {
    const newFiles = Array.from(e.target.files || [])
    if (newFiles.length === 0) return
    files = [...files, ...newFiles]
    // Generate preview thumbnails immediately
    for (const f of newFiles) {
      const url = URL.createObjectURL(f)
      filePreviews = [...filePreviews, url]
    }
    selected = new Set(files.map((_, i) => i))
    selectAll = true
    e.target.value = ''
  }

  function removeFile(idx) {
    files = files.filter((_, i) => i !== idx)
    if (filePreviews[idx]) URL.revokeObjectURL(filePreviews[idx])
    filePreviews = filePreviews.filter((_, i) => i !== idx)
    processedImages = processedImages.filter((_, i) => i !== idx)
    selected.delete(idx)
    const newSelected = new Set()
    for (const s of selected) {
      if (s > idx) newSelected.add(s - 1)
      else if (s < idx) newSelected.add(s)
    }
    selected = newSelected
  }



  function toggleSelect(idx) {
    if (selected.has(idx)) selected.delete(idx)
    else selected.add(idx)
    selected = selected // trigger reactivity
  }

  function toggleAll() {
    if (selectAll) {
      selected = new Set()
      selectAll = false
    } else {
      selected = new Set(files.map((_, i) => i))
      selectAll = true
    }
  }

  async function resizeImage(file, maxDim) {
    return new Promise((resolve) => {
      const img = new Image()
      const url = URL.createObjectURL(file)
      img.onload = () => {
        let w = img.width, h = img.height
        if (Math.max(w, h) > maxDim) {
          const scale = maxDim / Math.max(w, h)
          w = Math.round(w * scale)
          h = Math.round(h * scale)
        }
        const canvas = document.createElement('canvas')
        canvas.width = w
        canvas.height = h
        canvas.getContext('2d').drawImage(img, 0, 0, w, h)
        canvas.toBlob((blob) => {
          URL.revokeObjectURL(url)
          resolve(new File([blob], file.name, { type: 'image/jpeg' }))
        }, 'image/jpeg', 0.9)
      }
      img.onerror = () => { URL.revokeObjectURL(url); resolve(file) }
      img.src = url
    })
  }

  async function generateAll() {
    if (files.length === 0) return
    processing = true
    processedImages = []

    for (let i = 0; i < files.length; i++) {
      try {
        // Resize large images client-side (max 1024px)
        const resized = await resizeImage(files[i], 1024)
        const fd = new FormData()
        fd.append('file', resized)
        fd.append('position', position)
        fd.append('intensity', String(intensity))
        fd.append('width_pct', String(widthPct))
        fd.append('obj_type', objType)
        fd.append('margin_x', String(marginX))
        fd.append('margin_y', String(marginY))

        const ctrl = new AbortController()
        const t = setTimeout(() => ctrl.abort(), 120000)
        const res = await fetch(API + '/api/synthetic-shadow/generate', {
          method: 'POST',
          body: fd,
          signal: ctrl.signal,
        })
        clearTimeout(t)

        if (!res.ok) {
          let errText = ''
          try { errText = (await res.json()).detail || '' } catch(e) { errText = res.statusText }
          const errLabel = {400: 'Format gambar salah', 413: 'Gambar terlalu besar', 500: 'Server error', 502: 'Server down', 504: 'Server timeout'}
          const label = errLabel[res.status] || 'Server error'
          throw new Error(`${label} (${res.status}): ${errText}`)
        }

        const data = await res.json()
        processedImages = [...processedImages, {
          original: URL.createObjectURL(files[i]),
          shadow: data.image,
          name: files[i].name,
          idx: i,
        }]
      } catch (e) {
        let msg = e.message || 'Unknown error'
        if (e.name === 'AbortError') msg = 'Timeout (>120s) - coba gambar lebih kecil'
        else if (e.message && e.message.includes('Failed to fetch')) msg = 'Network error - cek koneksi / coba gambar lebih kecil'
        else if (e.message && e.message.includes('NetworkError')) msg = 'Network error - cek koneksi / coba gambar lebih kecil'
        else if (e.message && e.message.includes('ResizeError')) msg = 'Gagal resize gambar - coba format lain'
        showToast(`Gagal generate ${files[i].name}: ${msg}`, 'error')
        processedImages = [...processedImages, {
          original: URL.createObjectURL(files[i]),
          shadow: null,
          name: files[i].name,
          idx: i,
          error: true,
        }]
      }
    }
    processing = false
    if (processedImages.length > 0) {
      const ok = processedImages.filter(i => !i.error).length
      const fail = processedImages.filter(i => i.error).length
      showToast(`Generate selesai: ${ok} berhasil, ${fail} gagal`, ok > 0 ? 'success' : 'error')
    }
  }

  async function saveSelected() {
    const toSave = processedImages.filter(img => selected.has(img.idx) && !img.error)
    if (toSave.length === 0) return showToast('Pilih gambar yang ingin disimpan', 'error')

    saving = true
    try {
      const fd = new FormData()
      // Convert base64 shadow images back to blobs
      for (const img of toSave) {
        const resp = await fetch(img.shadow)
        const blob = await resp.blob()
        fd.append('files', blob, img.name)
      }
      fd.append('position', position)
      fd.append('intensity', String(intensity))
      fd.append('width_pct', String(widthPct))
      fd.append('obj_type', objType)
      fd.append('margin_x', String(marginX))
      fd.append('margin_y', String(marginY))
      fd.append('output_dir', 'datasets/paired/custom')

      const res = await fetch(API + '/api/synthetic-shadow/save', {
        method: 'POST',
        body: fd,
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'HTTP ' + res.status + ' ' + res.statusText }))
        throw new Error(err.detail || 'Failed')
      }

      const data = await res.json()
      showToast(`Tersimpan ${data.saved} pair ke dataset`, 'success')
    } catch (e) {
      let msg = e.message || 'Unknown error'
      if (e.name === 'AbortError') msg = 'Timeout - coba gambar lebih kecil'
      else if (e.message && e.message.includes('Failed to fetch')) msg = 'Network error - cek koneksi internet'
      showToast(`Gagal simpan: ${msg}`, 'error')
    }
    saving = false
  }

  onMount(() => {
    refreshIcons()
  })
</script>

<div class="page">
  <h1 class="page-title">
    <i data-lucide="sun-dim"></i> Synthetic Shadow Generator
  </h1>
  <p class="page-desc">Upload gambar bersih → generate shadow synthetic untuk training data</p>

  <!-- Settings -->
  <div class="settings-card">
    <div class="settings-row">
      <div class="setting">
        <label>Posisi Shadow</label>
        <div class="position-tabs">
          {#each positions as pos}
            <button
              class="pos-tab"
              class:active={position === pos.id}
              onclick={() => position = pos.id}
            >{pos.label}</button>
          {/each}
        </div>
      </div>
      <div class="setting">
        <label>Jenis Shadow</label>
        <div class="position-tabs">
          {#each objTypes as ot}
            <button
              class="pos-tab"
              class:active={objType === ot.id}
              onclick={() => objType = ot.id}
            >{ot.label}</button>
          {/each}
        </div>
      </div>
    </div>
    <div class="settings-row">
      <div class="setting">
        <label>Intensitas: {intensity.toFixed(2)}</label>
        <input type="range" min="0.1" max="1.0" step="0.05" bind:value={intensity} class="slider" />
      </div>
      <div class="setting">
        <label>Lebar Shadow: {(widthPct * 100).toFixed(0)}%</label>
        <input type="range" min="0.1" max="0.8" step="0.05" bind:value={widthPct} class="slider" />
      </div>
    </div>
    <div class="settings-row">
      <div class="setting">
        <label>Margin Horizontal: {(marginX * 100).toFixed(0)}% (0=edge, 100=center)</label>
        <input type="range" min="0.0" max="0.8" step="0.05" bind:value={marginX} class="slider" />
      </div>
      <div class="setting">
        <label>Margin Vertical: {(marginY * 100).toFixed(0)}% (0=edge, 100=center)</label>
        <input type="range" min="0.0" max="0.8" step="0.05" bind:value={marginY} class="slider" />
      </div>
    </div>
  </div>

  <!-- Upload -->
  <div class="upload-zone" ondrop={(e) => { e.preventDefault(); onFilesChange({target: {files: e.dataTransfer.files}}) }} ondragover={(e) => e.preventDefault()}>
    <input type="file" accept="image/*" multiple bind:this={fileInput} onchange={onFilesChange} style="display:none" />
    <button class="btn-upload" onclick={() => fileInput.click()}>
      <i data-lucide="upload"></i> Upload Gambar Bersih
    </button>
    <p class="upload-hint">Drag & drop atau klik untuk upload. Format: PNG, JPG, WEBP</p>
  </div>

  <!-- Generate Button -->
  {#if files.length > 0}
    <div class="action-bar">
      <span class="file-count">{files.length} gambar</span>
      <button class="btn-generate" onclick={generateAll} disabled={processing}>
        {#if processing}
          <i data-lucide="loader-2" class="spin"></i> Generating...
        {:else}
          <i data-lucide="wand-2"></i> Generate Shadow
        {/if}
      </button>
    </div>
  {/if}

  <!-- Results Grid -->
  {#if processedImages.length > 0}
    <div class="results-header">
      <div class="results-actions">
        <label class="checkbox-label">
          <input type="checkbox" checked={selectAll} onchange={toggleAll} />
          Pilih Semua ({processedImages.filter(i => !i.error).length})
        </label>
        <button class="btn-save" onclick={saveSelected} disabled={saving}>
          {#if saving}
            <i data-lucide="loader-2" class="spin"></i> Saving...
          {:else}
            <i data-lucide="save"></i> Simpan Terpilih
          {/if}
        </button>
      </div>
    </div>

    <div class="results-grid">
      {#each processedImages as img, idx}
        <div class="result-card" class:error={img.error} class:selected={selected.has(img.idx)}>
          <div class="result-header">
            <label class="checkbox-label">
              <input type="checkbox" checked={selected.has(img.idx)} onchange={() => toggleSelect(img.idx)} disabled={img.error} />
              {img.name}
            </label>
            <button class="btn-remove" onclick={() => removeFile(img.idx)} title="Hapus">
              <i data-lucide="x"></i>
            </button>
          </div>
          <div class="result-images">
            <div class="img-panel">
              <span class="img-label">Original</span>
              <img src={img.original} alt="Original" />
            </div>
            <div class="img-panel">
              <span class="img-label">Shadow</span>
              {#if img.error}
                <div class="img-error">Gagal generate</div>
              {:else}
                <img src={img.shadow} alt="Shadow" />
              {/if}
            </div>
          </div>
        </div>
      {/each}
    </div>
  {/if}

  {#if files.length > 0 && processedImages.length === 0 && !processing}
    <div class="preview-files">
      <h3>Preview gambar:</h3>
      <div class="preview-grid">
        {#each files as f, idx}
          <div class="preview-card">
            {#if filePreviews[idx]}
              <img src={filePreviews[idx]} alt={f.name} class="preview-thumb" />
            {/if}
            <div class="preview-info">
              <span class="preview-name">{f.name}</span>
              <span class="preview-size">{(f.size / 1024).toFixed(0)} KB</span>
              <button class="btn-remove-sm" onclick={() => removeFile(idx)}><i data-lucide="x"></i></button>
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .page { max-width: 1200px; margin: 0 auto; }
  .page-title { font-size: 1.6rem; font-weight: 700; display: flex; align-items: center; gap: .6rem; margin-bottom: .3rem; }
  .page-desc { color: var(--text2); margin-bottom: 1.5rem; font-size: .9rem; }

  .settings-card {
    background: rgba(255,255,255,.03); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.2rem; margin-bottom: 1.5rem;
  }
  .settings-row { display: flex; gap: 1.5rem; margin-bottom: 1rem; flex-wrap: wrap; }
  .settings-row:last-child { margin-bottom: 0; }
  .setting { flex: 1; min-width: 200px; }
  .setting label { display: block; font-size: .82rem; font-weight: 600; color: var(--text2); margin-bottom: .5rem; }
  .position-tabs { display: flex; gap: .35rem; flex-wrap: wrap; }
  .pos-tab {
    padding: .35rem .7rem; border-radius: 6px; font-size: .78rem; font-weight: 500;
    background: rgba(255,255,255,.04); border: 1px solid var(--border);
    color: var(--text2); cursor: pointer; transition: all .15s;
  }
  .pos-tab:hover { background: rgba(255,255,255,.08); color: var(--text); }
  .pos-tab.active { background: rgba(113,112,255,.15); border-color: rgba(130,143,255,.3); color: #e8e9ff; }
  .slider {
    width: 100%; height: 6px; -webkit-appearance: none; appearance: none;
    background: rgba(255,255,255,.1); border-radius: 999px; outline: none;
  }
  .slider::-webkit-slider-thumb {
    -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%;
    background: #7170ff; cursor: pointer; border: 2px solid #fff;
  }

  .upload-zone {
    border: 2px dashed var(--border); border-radius: 12px; padding: 2.5rem 1.5rem;
    text-align: center; margin-bottom: 1.5rem; transition: border-color .2s;
  }
  .upload-zone:hover { border-color: rgba(113,112,255,.4); }
  .btn-upload {
    display: inline-flex; align-items: center; gap: .5rem; padding: .7rem 1.5rem;
    background: linear-gradient(135deg, #7170ff, #5359b7); color: white;
    border: none; border-radius: 8px; font-weight: 600; font-size: .9rem; cursor: pointer;
  }
  .btn-upload:hover { opacity: .9; }
  .upload-hint { color: var(--text2); font-size: .8rem; margin-top: .5rem; }

  .action-bar {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.5rem; gap: 1rem; flex-wrap: wrap;
  }
  .file-count { color: var(--text2); font-size: .9rem; font-weight: 500; }
  .btn-generate {
    display: inline-flex; align-items: center; gap: .4rem; padding: .6rem 1.3rem;
    background: linear-gradient(135deg, #22c55e, #16a34a); color: white;
    border: none; border-radius: 8px; font-weight: 600; font-size: .88rem; cursor: pointer;
  }
  .btn-generate:hover { opacity: .9; }
  .btn-generate:disabled { opacity: .5; cursor: not-allowed; }

  .results-header { margin-bottom: 1rem; }
  .results-actions { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
  .checkbox-label { display: flex; align-items: center; gap: .4rem; font-size: .85rem; color: var(--text2); cursor: pointer; }
  .checkbox-label input { accent-color: #7170ff; }
  .btn-save {
    display: inline-flex; align-items: center; gap: .4rem; padding: .5rem 1.2rem;
    background: linear-gradient(135deg, #6366f1, #4f46e5); color: white;
    border: none; border-radius: 8px; font-weight: 600; font-size: .85rem; cursor: pointer;
  }
  .btn-save:hover { opacity: .9; }
  .btn-save:disabled { opacity: .5; cursor: not-allowed; }

  .results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1rem; }
  .result-card {
    background: rgba(255,255,255,.03); border: 1px solid var(--border);
    border-radius: 10px; padding: .8rem; transition: all .2s;
  }
  .result-card.selected { border-color: rgba(113,112,255,.3); background: rgba(113,112,255,.05); }
  .result-card.error { border-color: rgba(239,68,68,.3); }
  .result-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: .6rem; }
  .btn-remove {
    background: none; border: none; color: var(--text2); cursor: pointer; padding: .2rem;
    border-radius: 4px; display: flex; align-items: center;
  }
  .btn-remove:hover { color: #f87171; background: rgba(239,68,68,.1); }
  .result-images { display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; }
  .img-panel { position: relative; border-radius: 6px; overflow: hidden; background: rgba(0,0,0,.2); }
  .img-panel img { width: 100%; height: auto; display: block; }
  .img-label {
    position: absolute; top: 4px; left: 4px; background: rgba(0,0,0,.7);
    color: #fff; font-size: .65rem; padding: 2px 6px; border-radius: 4px; font-weight: 600;
  }
  .img-error { padding: 2rem; text-align: center; color: #f87171; font-size: .8rem; }

  .preview-files { margin-top: 1.5rem; }
  .preview-files h3 { font-size: .9rem; color: var(--text2); margin-bottom: .5rem; }
  .file-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: .4rem .6rem; background: rgba(255,255,255,.03); border-radius: 6px;
    margin-bottom: .3rem; font-size: .82rem; color: var(--text2);
  }
  .btn-remove-sm { background: none; border: none; color: var(--text2); cursor: pointer; display: flex; padding: .1rem; }
  .btn-remove-sm:hover { color: #f87171; }

  @keyframes spin { to { transform: rotate(360deg); } }
  .spin { animation: spin 1s linear infinite; }
</style>
