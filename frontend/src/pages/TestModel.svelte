<script>
  import { refreshIcons } from "../lib/icons.js"
  import { onDestroy, onMount, tick } from 'svelte'
  import { apiJson } from '../stores/auth.js'
  import { showToast } from '../stores/toast.js'

  let modelInfo = {}
  let rawResultBlob = null
  let resultCanvas
  let selectedFile = null
  let originalUrl = ''
  let selectedMode = 'restore'
  let pipelineMode = 'full'
  let shadowStrength = 1.0
  let adjBrightness = 1.0
  let adjContrast = 1.0
  let adjSaturation = 1.0
  let adjSharpness = 1.0
  let adjGamma = 1.0
  let adjWhiteBalance = false
  let adjClahe = 0.0
  function resetAdjust() { adjBrightness=1.0; adjContrast=1.0; adjSaturation=1.0; adjSharpness=1.0; adjGamma=1.0; adjWhiteBalance=false; adjClahe=0.0 }
  let resultUrl = ''
  let processing = false
  let docshadowWeights = []
  let trainedRuns = []
  let liveCheckpoints = []
  let checkpointTimer = null
  let selectedCheckpoint = 'checkpoints/document_restorer/best.pth'
  let loadedCheckpoint = ''

  let cropEnabled = false
  let cropPreset = 'smart'
  let cropCanvas
  let cropImage = null
  let cropImageLoaded = false
  let cropScale = 1
  let cropDisplay = { width: 0, height: 0 }
  let cropPoints = []
  let draggingPoint = -1
  let croppedFile = null
  let croppedUrl = ''
  let edgeCanvas
  let edgeStatus = 'Belum deteksi'
  let edgeScore = 0
  let edgeMethod = 'auto'
  let warpAnimating = false
  let warpProgress = 0

  const modes = ["restore","docres_base","docres_finetune","docres_onnx","docres_dewarping","docres_appearance","docres_appearance_mixed","docres_appearance_mixed_v5","docres_appearance_mixed_v5_lift12","docres_appearance_mixed_v5_tiled_lift12","docres_appearance_mixed_v5_tiled_lift12_base_detail","docres_deblurring","docres_end2end","dewarpnet","opencv_dewarp","controlpoints_dewarp","geotr_doc3d_dewarp","geotr_doc3d_docres","sam_geotr_docres","sam_perspective_appearance","textline_refine_dewarp","auto_dewarp","magic_enhance","binarize","cleanup","clahe","denoise","sharpen","deskew"]
  const pipelineModes = [
    { value: 'full', label: 'Full Pipeline', desc: 'AI + white balance + whitening + CLAHE' },
    { value: 'ai_only', label: 'AI Only', desc: 'Model output langsung, warna terjaga' },
    { value: 'ai', label: 'AI + Shadow Fix', desc: 'AI + shadow correction, tanpa color processing' },
    { value: 'color', label: 'AI + CLAHE', desc: 'AI + contrast enhancement, tanpa white balance' },
  ]
  const pretrainedModes = ['SD7K', 'Jung', 'Kligler'].map(name => ({ name }))
  const modeLabels = { restore: "AI Restore", docres_base: "DocRes Base", docres_finetune: "DocRes Finetune", docres_onnx: "DocRes ONNX", docres_dewarping: "DocRes Dewarping", docres_appearance: "DocRes Appearance", docres_appearance_mixed: "Appearance Mixed Custom (1,419 pairs)", docres_appearance_mixed_v5: "Appearance Mixed → V5 Blend75 + Lift15", docres_appearance_mixed_v5_lift12: "Appearance Mixed → V5 Blend75 + LIFT12", docres_appearance_mixed_v5_tiled_lift12: "Appearance Mixed → V5 Tiled 576 + Blend75 + LIFT12", docres_appearance_mixed_v5_tiled_lift12_base_detail: "Appearance Mixed → V5 Tiled + LIFT12 + Base Appearance Detail", docres_deblurring: "DocRes Deblurring", docres_end2end: "DocRes End2End (Dewarp+Deshadow+Appear)", dewarpnet: "DewarpNet (ICCV)", opencv_dewarp: "OpenCV Dewarp", controlpoints_dewarp: "Control Points Dewarp", geotr_doc3d_dewarp: "GeoTr Doc3D Dewarp", geotr_doc3d_docres: "GeoTr Doc3D + Deshadow + Appearance", sam_geotr_docres: "SAM Edge + GeoTr + Deshadow + Appearance", sam_perspective_appearance: "SAM Perspective + Appearance", textline_refine_dewarp: "Textline Refine Dewarp (Exp)", auto_dewarp: "Auto Dewarp (Smart)", magic_enhance: "Magic Enhance", binarize: "Binarize", cleanup: "Full Cleanup", clahe: "CLAHE", denoise: "Denoise", sharpen: "Sharpen", deskew: "Deskew" }
  $: availableDocshadowNames = new Set(docshadowWeights.map(w => w.name))
  $: docshadowModes = pretrainedModes.map(w => `docshadow:${w.name}`)
  $: allModeLabels = {
    ...modeLabels,
    ...Object.fromEntries(pretrainedModes.map(w => [`docshadow:${w.name}`, `Pretrained DocShadow ${w.name}`]))
  }
  $: selectedModelFamily = selectedMode.startsWith('docshadow:') ? 'docshadow' : 'docai'


  async function refreshCheckpoints() {
    try {
      const data = await apiJson('/api/training/runs')
      trainedRuns = (data.runs || []).filter(run => (run.checkpoints || []).length)
      liveCheckpoints = data.live_checkpoints || []
      const available = new Set([
        ...liveCheckpoints.map(item => item.path),
        ...trainedRuns.flatMap(run => (run.checkpoints || []).map(item => item.path))
      ])
      if (!available.has(selectedCheckpoint) && liveCheckpoints.length) selectedCheckpoint = liveCheckpoints[0].path
    } catch (e) {
      showToast(`Gagal refresh checkpoint: ${e.message}`, 'error')
    }
  }

  onMount(async () => {
    refreshIcons()
    try { modelInfo = await apiJson('/api/models/info') } catch(e) {}
    try {
      const d = await apiJson('/api/docshadow/weights')
      docshadowWeights = d.weights || []
    } catch(e) { docshadowWeights = [] }
    await refreshCheckpoints()
    checkpointTimer = setInterval(refreshCheckpoints, 5000)
  })

  onDestroy(() => {
    if (checkpointTimer) clearInterval(checkpointTimer)
    if (resultUrl) URL.revokeObjectURL(resultUrl)
    if (croppedUrl) URL.revokeObjectURL(croppedUrl)
  })

  function handleFile(file) {
    console.log('[TestModel] handleFile', file?.name)
    if (resultUrl) URL.revokeObjectURL(resultUrl)
    if (croppedUrl) URL.revokeObjectURL(croppedUrl)
    selectedFile = file
    cropEnabled = true
    croppedFile = null
    croppedUrl = ''
    resultUrl = ''
    originalUrl = ''
    cropImageLoaded = false
    cropPoints = []
    edgeStatus = 'Memuat gambar'
    edgeScore = 0
    warpProgress = 0

    const reader = new FileReader()
    reader.onload = () => {
      if (selectedFile !== file) return
      originalUrl = reader.result
      loadCropImage(reader.result)
    }
    reader.onerror = () => showToast('Gagal membaca preview original image', 'error')
    reader.readAsDataURL(file)
  }

  function loadCropImage(src) {
    const img = new Image()
    img.onload = () => {
      cropImage = img
      cropImageLoaded = true
      let tries = 0
      function waitForCanvas() {
        if (!cropCanvas) {
          const el = document.querySelector('.crop-stage canvas')
          if (el) cropCanvas = el
        }
        if (cropCanvas || tries++ > 15) { resetCrop(); setTimeout(autoDetectEdge, 100); return }
        requestAnimationFrame(waitForCanvas)
      }
      waitForCanvas()
    }
    img.onerror = () => showToast('Gagal memuat gambar untuk pangkas', 'error')
    img.src = src
  }

  function resetCrop() {
    if (!cropImageLoaded || !cropCanvas) return
    const maxW = Math.min(720, cropCanvas.parentElement?.clientWidth || 720)
    cropScale = Math.min(maxW / cropImage.naturalWidth, 560 / cropImage.naturalHeight, 1)
    cropDisplay = { width: Math.round(cropImage.naturalWidth * cropScale), height: Math.round(cropImage.naturalHeight * cropScale) }
    cropCanvas.width = cropDisplay.width
    cropCanvas.height = cropDisplay.height
    const padX = cropImage.naturalWidth * 0.015
    const padY = cropImage.naturalHeight * 0.015
    cropPoints = [
      { x: padX, y: padY },
      { x: cropImage.naturalWidth - padX, y: padY },
      { x: cropImage.naturalWidth - padX, y: cropImage.naturalHeight - padY },
      { x: padX, y: cropImage.naturalHeight - padY },
    ]
    edgeStatus = cropPreset==='smart' ? 'Smart crop aktif. Geser titik jika perlu.' : cropPreset==='full' ? 'Full image. Bisa pakai Auto Edge bila perlu.' : 'Potong tepi aktif. Geser titik jika perlu.'
    edgeScore = 0
    drawCrop()
  }

  function applyCropPreset() {
    if (!cropImageLoaded || cropPreset === 'manual') return
    const w = cropImage.naturalWidth, h = cropImage.naturalHeight
    if (cropPreset === 'full') {
      cropPoints = [{ x: 0, y: 0 }, { x: w, y: 0 }, { x: w, y: h }, { x: 0, y: h }]
      edgeStatus = 'Full image. Bisa pakai Auto Edge bila perlu.'
    } else {
      const mx = w * 0.028, my = h * 0.018
      const x0 = mx, x1 = w - mx
      const y0 = my, y1 = h - my
      if (cropPreset === 'smart') {
        cropPoints = [{ x: x0, y: y0 }, { x: x1, y: y0 }, { x: x1, y: y1 }, { x: x0, y: y1 }]
        edgeStatus = 'Smart crop aktif. Geser titik jika perlu.'
      } else {
        const tx = w * 0.02, ty = h * 0.065
        cropPoints = [{ x: x0, y: ty }, { x: x1, y: ty }, { x: x1, y: y1 }, { x: x0, y: y1 }]
        edgeStatus = 'Potong tepi aktif. Geser titik jika perlu.'
      }
    }
    edgeScore = 0
    drawCrop()
  }

  function imageDataGray(ctx, w, h) {
    const data = ctx.getImageData(0, 0, w, h).data
    const gray = new Uint8ClampedArray(w * h)
    for (let i = 0, j = 0; i < data.length; i += 4, j++) gray[j] = Math.round(0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2])
    return gray
  }





  



  function autoDetectEdge() {
    if (!cropImageLoaded || !selectedFile) return
    edgeStatus = 'Mendeteksi tepi...'
    drawCrop()
    const token = localStorage.getItem('docai_token') || ''
    const fd = new FormData()
    fd.append('file', selectedFile)
    fd.append('method', edgeMethod)
    fetch('/api/detect_edges', { method: 'POST', headers: { 'Authorization': 'Bearer ' + token }, body: fd })
    .then(r => { if (!r.ok) throw new Error(r.status); return r.json() })
    .then(data => {
      if (!data.corners || data.corners.length !== 4) { edgeStatus = 'Deteksi gagal'; drawCrop(); return }
      // corners in ORIGINAL image coords, order tl,tr,br,bl -> set draggable points
      cropPoints = data.corners.map(c => ({ x: c[0], y: c[1] }))
      cropPreset = 'manual'
      const ml = (data.method || edgeMethod).toUpperCase()
      if (data.is_full) {
        edgeScore = 0
        edgeStatus = ml + ': tepi tak terdeteksi (mungkin sudah ter-crop). Geser 4 titik bila perlu.'
      } else {
        edgeScore = Math.round(data.confidence * 100)
        edgeStatus = ml + ' auto edge: ' + edgeScore + '% — geser titik bila perlu, lalu Terapkan.'
      }
      drawCrop()
    })
    .catch(() => { edgeStatus = 'Deteksi tepi gagal'; drawCrop() })
  }

  function drawOverlayOn(ctx) {
    if (!cropPoints.length) return
    const pts = cropPoints.map(p => ({ x: p.x * cropScale, y: p.y * cropScale }))
    ctx.save(); ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 3; ctx.beginPath()
    pts.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)); ctx.closePath(); ctx.stroke(); ctx.restore()
  }

  function drawCrop() {
    if (!cropCanvas || !cropImageLoaded) return
    const ctx = cropCanvas.getContext('2d')
    ctx.clearRect(0, 0, cropCanvas.width, cropCanvas.height)
    ctx.drawImage(cropImage, 0, 0, cropDisplay.width, cropDisplay.height)
    const pts = cropPoints.map(p => ({ x: p.x * cropScale, y: p.y * cropScale }))
    ctx.save()
    ctx.fillStyle = 'rgba(0, 180, 255, .14)'
    ctx.strokeStyle = '#38bdf8'
    ctx.lineWidth = 3
    ctx.beginPath()
    pts.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))
    ctx.closePath(); ctx.fill(); ctx.stroke()
    pts.forEach((p, i) => {
      ctx.beginPath(); ctx.arc(p.x, p.y, 11, 0, Math.PI * 2)
      ctx.fillStyle = draggingPoint === i ? '#0ea5e9' : 'rgba(255,255,255,.75)'
      ctx.fill(); ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 4; ctx.stroke()
    })
    ctx.restore()
  }

  function cropPointer(event) {
    const r = cropCanvas.getBoundingClientRect()
    return { x: (event.clientX - r.left) / cropScale, y: (event.clientY - r.top) / cropScale }
  }

  function startCropDrag(event) {
    if (!cropImageLoaded) return
    const p = cropPointer(event)
    let best = -1, bestD = Infinity
    cropPoints.forEach((pt, i) => {
      const d = Math.hypot(pt.x - p.x, pt.y - p.y)
      if (d < bestD) { bestD = d; best = i }
    })
    if (bestD < 36 / cropScale) draggingPoint = best
    moveCropDrag(event)
  }

  function moveCropDrag(event) {
    if (draggingPoint < 0) return
    event.preventDefault()
    const p = cropPointer(event)
    cropPoints[draggingPoint] = {
      x: Math.max(0, Math.min(cropImage.naturalWidth, p.x)),
      y: Math.max(0, Math.min(cropImage.naturalHeight, p.y)),
    }
    drawCrop()
  }

  function endCropDrag() { draggingPoint = -1; drawCrop() }

  function orderCropPoints(points) {
    const c = points.reduce((a, p) => ({ x: a.x + p.x / points.length, y: a.y + p.y / points.length }), { x: 0, y: 0 })
    const sorted = [...points].sort((a, b) => Math.atan2(a.y - c.y, a.x - c.x) - Math.atan2(b.y - c.y, b.x - c.x))
    const tlIndex = sorted.reduce((best, p, i) => (p.x + p.y < sorted[best].x + sorted[best].y ? i : best), 0)
    return [...sorted.slice(tlIndex), ...sorted.slice(0, tlIndex)]
  }

  function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y) }

  function solvePerspective(src, dst) {
    const A = [], b = []
    for (let i = 0; i < 4; i++) {
      const { x, y } = dst[i]
      const u = src[i].x, v = src[i].y
      A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u)
      A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v)
    }
    for (let i = 0; i < 8; i++) {
      let max = i
      for (let r = i + 1; r < 8; r++) if (Math.abs(A[r][i]) > Math.abs(A[max][i])) max = r
      ;[A[i], A[max]] = [A[max], A[i]]; [b[i], b[max]] = [b[max], b[i]]
      const div = A[i][i] || 1e-12
      for (let c = i; c < 8; c++) A[i][c] /= div
      b[i] /= div
      for (let r = 0; r < 8; r++) if (r !== i) {
        const f = A[r][i]
        for (let c = i; c < 8; c++) A[r][c] -= f * A[i][c]
        b[r] -= f * b[i]
      }
    }
    return [...b, 1]
  }

  function applyCrop() {
    if (!cropImageLoaded || cropPoints.length !== 4) return
    const src = orderCropPoints(cropPoints)
    const width = Math.max(64, Math.round(Math.max(dist(src[0], src[1]), dist(src[3], src[2]))))
    const height = Math.max(64, Math.round(Math.max(dist(src[0], src[3]), dist(src[1], src[2]))))
    const out = document.createElement('canvas')
    out.width = width; out.height = height
    const ctx = out.getContext('2d')
    const dst = [{x:0,y:0},{x:width-1,y:0},{x:width-1,y:height-1},{x:0,y:height-1}]
    const m = solvePerspective(src, dst)
    const imgData = ctx.createImageData(width, height)
    const s = document.createElement('canvas')
    s.width = cropImage.naturalWidth; s.height = cropImage.naturalHeight
    const sctx = s.getContext('2d')
    sctx.drawImage(cropImage, 0, 0)
    const srcData = sctx.getImageData(0, 0, s.width, s.height).data
    const outData = imgData.data
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const den = m[6] * x + m[7] * y + 1
      const sx = (m[0] * x + m[1] * y + m[2]) / den
      const sy = (m[3] * x + m[4] * y + m[5]) / den
      const ix = Math.max(0, Math.min(s.width - 1, Math.round(sx)))
      const iy = Math.max(0, Math.min(s.height - 1, Math.round(sy)))
      const si = (iy * s.width + ix) * 4, oi = (y * width + x) * 4
      outData[oi] = srcData[si]; outData[oi+1] = srcData[si+1]; outData[oi+2] = srcData[si+2]; outData[oi+3] = 255
    }
    ctx.putImageData(imgData, 0, 0)
    out.toBlob(blob => {
      if (!blob) return
      if (croppedUrl) URL.revokeObjectURL(croppedUrl)
      croppedFile = new File([blob], 'cropped_' + (selectedFile?.name || 'input.png').replace(/\.[^.]+$/, '.png'), { type: 'image/png' })
      croppedUrl = URL.createObjectURL(blob)
      showToast('Pemangkasan diterapkan')
      animateWarpPreview()
    }, 'image/png')
  }


  function animateWarpPreview() {
    if (!cropCanvas || !cropImageLoaded || !cropPoints.length) return
    warpAnimating = true
    warpProgress = 0
    const src = orderCropPoints(cropPoints).map(p => ({ x: p.x * cropScale, y: p.y * cropScale }))
    const minX = Math.min(...src.map(p=>p.x)), maxX = Math.max(...src.map(p=>p.x))
    const minY = Math.min(...src.map(p=>p.y)), maxY = Math.max(...src.map(p=>p.y))
    const dst = [{x:minX,y:minY},{x:maxX,y:minY},{x:maxX,y:maxY},{x:minX,y:maxY}]
    const start = performance.now()
    const step = (t) => {
      warpProgress = Math.min(1, (t - start) / 900)
      drawCrop()
      const ctx = cropCanvas.getContext('2d')
      ctx.save(); ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2
      for (let g = 0; g <= 8; g++) {
        const a = g / 8
        const top = interp(src[0], src[1], a), bot = interp(src[3], src[2], a)
        const l = interp(src[0], src[3], a), r = interp(src[1], src[2], a)
        const top2 = interp(dst[0], dst[1], a), bot2 = interp(dst[3], dst[2], a)
        const l2 = interp(dst[0], dst[3], a), r2 = interp(dst[1], dst[2], a)
        line(ctx, interp(top, top2, warpProgress), interp(bot, bot2, warpProgress))
        line(ctx, interp(l, l2, warpProgress), interp(r, r2, warpProgress))
      }
      ctx.restore()
      if (warpProgress < 1) requestAnimationFrame(step); else warpAnimating = false
    }
    requestAnimationFrame(step)
  }

  function interp(a,b,t){ return { x:a.x+(b.x-a.x)*t, y:a.y+(b.y-a.y)*t } }
  function line(ctx,a,b){ ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke() }

  function skipCrop() {
    croppedFile = null
    if (croppedUrl) URL.revokeObjectURL(croppedUrl)
    croppedUrl = ''
    cropEnabled = false
    showToast('Pangkas dilewati')
  }

  function handleDrop(event) {
    event.preventDefault()
    const file = event.dataTransfer?.files?.[0]
    if (file?.type?.startsWith('image/')) handleFile(file)
  }

  $: previewFilter = `brightness(${adjBrightness}) contrast(${adjContrast}) saturate(${adjSaturation})`
  $: if (cropCanvas && cropImageLoaded && cropPoints.length && !warpAnimating) drawCrop()

  function applyClientAdjustments() {
    // CSS filter updates preview reactively. Apply button bakes it into PNG.
  }

  function applyAdjustmentsToImage() {
    if (!rawResultBlob) return
    const sourceUrl = URL.createObjectURL(rawResultBlob)
    const img = new Image()
    img.onload = () => {
      const canvas = resultCanvas
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      const ctx = canvas.getContext('2d')
      ctx.filter = previewFilter
      ctx.drawImage(img, 0, 0)
      ctx.filter = 'none'
      canvas.toBlob(blob => {
        URL.revokeObjectURL(sourceUrl)
        if (!blob) return
        const oldUrl = resultUrl
        resultUrl = URL.createObjectURL(blob)
        rawResultBlob = blob
        setTimeout(() => { if (oldUrl) URL.revokeObjectURL(oldUrl) }, 250)
        showToast('Adjustment diterapkan')
      }, 'image/png')
    }
    img.onerror = () => URL.revokeObjectURL(sourceUrl)
    img.src = sourceUrl
  }

  function downloadResult() {
    if (!resultUrl) return
    const a = document.createElement('a')
    a.href = resultUrl
    a.download = 'result_' + (selectedFile?.name || 'output.png')
    a.click()
  }

  async function runTest() {
    if (!selectedFile) return
    processing = true
    const fd = new FormData()
    const inputFile = croppedFile || selectedFile
    fd.append('file', inputFile)
    fd.append('mode', selectedMode)
    // Adjustments applied client-side via OpenCV.js
      if (selectedMode === 'docres_finetune') { const vs = document.getElementById('finetune_variant'); fd.append('variant', vs ? vs.value : 'finetune_v3'); }
    if (selectedMode === 'restore') {
      fd.append('pipeline_mode', pipelineMode)
      fd.append('shadow_strength', shadowStrength)
    }
    const start = performance.now()
    try {
      const token = localStorage.getItem('docai_token')
      if (!selectedMode.startsWith('docshadow:') && selectedCheckpoint !== loadedCheckpoint) {
        const reload = await fetch('/api/model/reload', {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({ checkpoint: selectedCheckpoint })
        })
        if (!reload.ok) {
          const error = await reload.json().catch(() => ({}))
          throw new Error(error.detail || 'Gagal load checkpoint')
        }
        loadedCheckpoint = selectedCheckpoint
      }
      let endpoint = '/api/scan'
      if (selectedMode.startsWith('docshadow:')) {
        fd.set('mode', 'docshadow')
        fd.append('weight', selectedMode.split(':')[1])
        endpoint = '/api/docshadow/infer'
      }
      let res
      try {
        res = await fetch(endpoint, { method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: fd })
      } catch(fetchErr) {
        throw new Error('Gagal menghubungi server: ' + (fetchErr.message || 'network error'))
      }
      if (!res.ok) {
        const errText = await res.text().catch(() => '')
        let errDetail = errText
        try { errDetail = JSON.parse(errText).detail || errText } catch(_) {}
        throw new Error(`Server error ${res.status}: ${errDetail || res.statusText}`)
      }
      rawResultBlob = await res.blob()
      if (resultUrl) URL.revokeObjectURL(resultUrl)
      resultUrl = URL.createObjectURL(rawResultBlob)
      const ms = Math.round(performance.now() - start)
      showToast(`${allModeLabels[selectedMode] || selectedMode} completed (${ms}ms)`)
    } catch(e) {
      let msg = e.message || 'Unknown error'
      if (msg === 'Failed to fetch') msg = 'Gagal menghubungi server. Periksa koneksi atau coba lagi.'
      showToast(msg, 'error')
    }
    finally { processing = false }
  }

  let compareMode = 'deshadowing'
  let compareResult = null
  let compareView = 'finetuned'
  let comparing = false
  const isDocresMode = (m) => ['docres_base','docres_finetune','docres_appearance','docres_appearance_mixed','docres_appearance_mixed_v5','docres_appearance_mixed_v5_lift12','docres_appearance_mixed_v5_tiled_lift12','docres_appearance_mixed_v5_tiled_lift12_base_detail','docres_deblurring'].includes(m)

  async function runCompare() {
    if (!selectedFile) return
    comparing = true
    compareResult = null
    const fd = new FormData()
    fd.append('file', croppedFile || selectedFile)
    fd.append('task', compareMode)
    fd.append('selected_mode', selectedMode)
    const start = performance.now()
    try {
      const token = localStorage.getItem('docai_token')
      const res = await fetch('/api/docres/compare', { method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: fd })
      if (!res.ok) throw new Error('Compare error')
      compareResult = await res.json()
      compareResult._ms = Math.round(performance.now() - start)
      showToast(`Compare selesai: ${compareResult._ms}ms`)
    } catch(e) { showToast(e.message, 'error') }
    finally { comparing = false }
  }
</script>

<div class="page-header">
  <h1>Test Individual Model</h1>
  <p>Upload gambar dan test setiap mode processing.</p>
</div>

<div class="card">
  <h2><i data-lucide="cpu"></i> Model Info</h2>
  <div class="info-grid">
    {#each Object.entries(modelInfo) as [name, info]}
      <div class="info-item">
        <div class="k">{name}</div>
        <div class="v" style="color: {info.loaded !== false ? 'var(--success)' : 'var(--text3)'}">
          {info.total_params ? info.total_params.toLocaleString() + ' params' : (info.loaded !== false ? 'Loaded' : 'Not loaded')}
        </div>
      </div>
    {/each}
  </div>
</div>

<div class="card">
  <h2><i data-lucide="upload-cloud"></i> Upload & Run</h2>
  <div class="grid-2">
    <div>
      <div class="upload-zone" ondrop={handleDrop} ondragover={(e) => e.preventDefault()} onclick={() => document.getElementById('testFileInput').click()}>
        <i data-lucide="image-plus"></i>
        <p>Klik atau drag & drop gambar</p>
        <input type="file" id="testFileInput" accept="image/*" style="display:none"
          onchange={e => e.target.files[0] && handleFile(e.target.files[0])} />
      </div>
      {#if selectedFile && cropEnabled}
        <div class="crop-pills">
          <button class="pill" class:active={cropPreset==='smart'} onclick={() => { cropPreset='smart'; applyCropPreset() }}>Smart Crop</button>
          <button class="pill" class:active={cropPreset==='edges'} onclick={() => { cropPreset='edges'; applyCropPreset() }}>Potong Tepi</button>
          <button class="pill" class:active={cropPreset==='full'} onclick={() => { cropPreset='full'; applyCropPreset() }}>Full Image</button>
          <button class="pill" class:active={cropPreset==='manual'} onclick={() => { cropPreset='manual' }}>Manual</button>
        </div>
        <div class="crop-panel" style="border:2px solid #22c55e; background:#020617">

          <div class="crop-head">
            <div>
              <label>Pangkas / Edge sebelum test</label>
              <p class="info-text">{edgeStatus}. Geser 4 titik bila perlu, lalu Terapkan Pangkas.</p>
            </div>
            <button class="btn" onclick={skipCrop}>Tanpa Pangkas</button>
          </div>
          <div class="crop-stage">
            <canvas bind:this={cropCanvas}
              onpointerdown={startCropDrag}
              onpointermove={moveCropDrag}
              onpointerup={endCropDrag}
              onpointerleave={endCropDrag}></canvas>
          </div>
          <div class="crop-method">
            <span class="method-label">Metode deteksi:</span>
            <button class="pill sm" class:active={edgeMethod==='auto'} onclick={() => { edgeMethod='auto'; autoDetectEdge() }}>Auto</button>
            <button class="pill sm" class:active={edgeMethod==='opencv'} onclick={() => { edgeMethod='opencv'; autoDetectEdge() }}>OpenCV (cepat)</button>
            <button class="pill sm" class:active={edgeMethod==='grabcut'} onclick={() => { edgeMethod='grabcut'; autoDetectEdge() }}>GrabCut (miring)</button>
          </div>
          <div class="crop-actions">
            <button class="btn" onclick={autoDetectEdge}>Auto Edge</button>
            <button class="btn" onclick={resetCrop}>Reset Titik</button>
            <button class="btn btn-primary" onclick={applyCrop}>Terapkan Pangkas</button>
          </div>
          <div class="edge-preview">
            <div class="label"><i data-lucide="scan-line"></i> Edge preview {#if edgeScore}({edgeScore}%){/if}</div>
            <canvas bind:this={edgeCanvas}></canvas>
          </div>
          {#if warpAnimating}<div class="warpbar"><span style={`width:${Math.round(warpProgress*100)}%`}></span></div>{/if}
        </div>
      {/if}
      {#if croppedUrl}
        <div class="img-box crop-result">
          <div class="label"><i data-lucide="crop"></i> Hasil Pangkas</div>
          <img src={croppedUrl} alt="Hasil Pangkas" />
        </div>
      {/if}
      <div class="model-select">
        <label>Model</label>
        <select bind:value={selectedMode}>
          <optgroup label="DocAI Custom / Local">
            {#each modes as m}
              <option value={m}>{modeLabels[m]}</option>
            {/each}
          </optgroup>
          <optgroup label="Pretrained DocShadow-SD7K">
            {#each docshadowModes as m}
              <option value={m} disabled={!availableDocshadowNames.has(m.split(':')[1])}>{allModeLabels[m]}{availableDocshadowNames.has(m.split(':')[1]) ? '' : ' (weight belum ada)'}</option>
            {/each}
          </optgroup>
        </select>
        <p class="info-text">
          {selectedModelFamily === 'docshadow'
            ? 'Pembanding pretrained dari DocShadow-SD7K releases.'
            : 'Model lokal dan mode enhancement DocAI.'}
        </p>
      </div>
      {#if selectedMode === 'docres_finetune'}
        <div class="model-select">
          <label>Finetune Variant</label>
          <select id="finetune_variant">
            <option value="finetune_576">576 — Deshadow 576 (training best)</option>
            <option value="finetune_v5" selected>V5 Deshadow — Smooth Detail (epoch 1)</option>
            <option value="finetune_v3">v3 — Deshadow (iter 45k)</option>
            <option value="finetune_v2">v2 — Deshadow v2</option>
            <option value="finetune_v1">v1 — Original</option>
          </select>
          <p class="info-text">Pilih checkpoint fine-tuning.</p>
        </div>
      {/if}

      {#if selectedMode === 'restore'}
        <div class="model-select">
          <label>Trained checkpoint</label>
          <select bind:value={selectedCheckpoint}>
            {#each liveCheckpoints as checkpoint}
              <option value={checkpoint.path}>{checkpoint.name}</option>
            {/each}
            {#each trainedRuns as run}
              <optgroup label={run.run_id}>
                {#each run.checkpoints || [] as checkpoint}
                  <option value={checkpoint.path}>{checkpoint.immutable ? 'Immutable · ' : ''}{checkpoint.name}</option>
                {/each}
              </optgroup>
            {/each}
          </select>
          <p class="info-text">Auto-refresh 5 detik. Progress checkpoint belum divalidasi.</p>
        </div>
      {/if}
      {#if selectedMode === 'restore'}
        <div class="model-select" style="margin-top:1rem">
          <label>Pipeline Mode</label>
          <div class="pretrained-pills">
            {#each pipelineModes as pm}
              <button
                class="pill"
                class:active={pipelineMode === pm.value}
                onclick={() => pipelineMode = pm.value}
                title={pm.desc}
              >
                {pm.label}
              </button>
            {/each}
          </div>
          <p class="info-text">{pipelineModes.find(p => p.value === pipelineMode)?.desc}</p>
        </div>
        {#if pipelineMode !== 'ai_only'}
          <div class="model-select" style="margin-top:0.75rem">
            <label>Shadow Strength: {shadowStrength.toFixed(1)}</label>
            <input type="range" min="0" max="1" step="0.1" bind:value={shadowStrength}
              style="width:100%;accent-color:var(--primary)" />
            <p class="info-text">0.0 = tanpa shadow fix, 1.0 = full correction</p>
          </div>
        {/if}
      {/if}
      <div class="model-select" style="margin-top:1rem;border-top:1px solid var(--border,#333);padding-top:0.75rem">

      </div>
      <div style="margin-top:1rem;display:flex;align-items:center;gap:0.75rem">
        <button class="btn btn-primary" onclick={runTest} disabled={!selectedFile || processing || (cropEnabled && !croppedFile)}>
          {#if processing}<div class="spinner" style="width:16px;height:16px;border-width:2px"></div>{:else}<i data-lucide="play"></i>{/if}
          {cropEnabled && !croppedFile ? 'Pangkas dulu' : 'Run Test'}
        </button>
        {#if selectedFile}
          <span class="info-text">{selectedFile.name} ({Math.round(selectedFile.size / 1024)}KB)</span>
        {/if}
      </div>
      {#if isDocresMode(selectedMode)}
        <div style="margin-top:0.75rem;display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap">
          <select bind:value={compareMode} style="padding:0.4rem;border-radius:6px;background:#1a1a2e;color:white;border:1px solid #333">
            <option value="deshadowing">Deshadowing</option>
            <option value="appearance">Appearance</option>
            <option value="deblurring">Deblurring</option>
          </select>
          <button class="btn btn-secondary" onclick={runCompare} disabled={!selectedFile || comparing}>
            {#if comparing}<div class="spinner" style="width:14px;height:14px;border-width:2px"></div>{:else}<i data-lucide="columns"></i>{/if}
            Compare Base vs Selected Model
          </button>
        </div>
      {/if}
      {#if compareResult}
        <div class="card" style="margin-top:1rem">
          <div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.75rem">
            <h3 style="margin:0">Compare: {compareMode}</h3>
            <span class="info-text">Total: {compareResult._ms}ms</span>
          </div>
          <div style="display:flex;gap:0.5rem;margin-bottom:0.75rem">
            <button class="btn" class:btn-primary={compareView==='base'} class:btn-secondary={compareView!=='base'} onclick={()=>compareView='base'} style="font-size:0.8rem;padding:0.35rem 0.8rem">Base ({compareResult.base?.ms || 0}ms)</button>
            <button class="btn" class:btn-primary={compareView==='finetuned'} class:btn-secondary={compareView!=='finetuned'} onclick={()=>compareView='finetuned'} style="font-size:0.8rem;padding:0.35rem 0.8rem">{compareResult.selected?.label || 'Selected'} ({compareResult.selected?.ms || 0}ms)</button>
            <button class="btn" class:btn-primary={compareView==='both'} class:btn-secondary={compareView!=='both'} onclick={()=>compareView='both'} style="font-size:0.8rem;padding:0.35rem 0.8rem">Side-by-side</button>
          </div>
          {#if compareView === 'both'}
            <div class="compare-grid">
              <div class="img-box">
                <div class="img-label">Base</div>
                {#if compareResult.base?.image}<img src={compareResult.base.image} alt="base" style:filter={previewFilter} />{/if}
              </div>
              <div class="img-box">
                <div class="img-label">{compareResult.selected?.label || 'Selected'}</div>
                {#if compareResult.finetuned?.image}<img src={compareResult.finetuned.image} alt="finetuned" style:filter={previewFilter} />{/if}
              </div>
            </div>
          {:else}
            <div class="result-box">
              {#if compareView === 'base' && compareResult.base?.image}
                <img src={compareResult.base.image} alt="base" style:filter={previewFilter} />
              {:else if compareResult.finetuned?.image}
                <img src={compareResult.finetuned.image} alt="finetuned" style:filter={previewFilter} />
              {/if}
            </div>
          {/if}
          <div class="adj-row" style="margin-top:0.75rem"><label>Brightness: {adjBrightness.toFixed(2)}</label>
            <input type="range" min="0.2" max="2.5" step="0.05" bind:value={adjBrightness} /></div>
          <div class="adj-row"><label>Contrast: {adjContrast.toFixed(2)}</label>
            <input type="range" min="0.2" max="2.5" step="0.05" bind:value={adjContrast} /></div>
          <div class="adj-row"><label>Saturation: {adjSaturation.toFixed(2)}</label>
            <input type="range" min="0" max="3" step="0.05" bind:value={adjSaturation} /></div>
        </div>
      {/if}
    </div>
    <div>
      {#if originalUrl}
        <div class="img-box">
          <div class="label"><i data-lucide="image"></i> {croppedUrl ? 'Original (belum dipangkas)' : 'Original'}</div>
          <img src={originalUrl} alt="Original" />
        </div>
      {:else if selectedFile}
        <div class="img-box placeholder">
          <div class="label"><i data-lucide="image"></i> Original</div>
          <div class="empty-preview">Loading preview...</div>
        </div>
      {/if}
    </div>
  </div>
</div>

{#if resultUrl}
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center">
        <h2><i data-lucide="sparkles"></i> Result — {selectedMode}</h2>
        <button class="btn btn-primary" onclick={downloadResult} style="font-size:0.8rem;padding:0.4rem 0.8rem">
          <i data-lucide="download"></i> Download
        </button>
      </div>
    <div class="compare-grid">
      {#if originalUrl}
        <div class="img-box">
          <div class="label"><i data-lucide="image"></i> Original</div>
          <img src={originalUrl} alt="Original" />
        </div>
      {/if}
      <div class="img-box">
        <div class="label"><i data-lucide="sparkles"></i> Result</div>
        <img src={resultUrl} alt="Result" style:filter={previewFilter} />
        <canvas bind:this={resultCanvas} style="display:none"></canvas>
      </div>
    </div>
  </div>
{/if}
    <div class="adj-sliders">
      <div class="adj-row"><label>Brightness: {adjBrightness.toFixed(2)}</label>
        <input type="range" min="0.2" max="2.5" step="0.05" bind:value={adjBrightness} /></div>
      <div class="adj-row"><label>Contrast: {adjContrast.toFixed(2)}</label>
        <input type="range" min="0.2" max="2.5" step="0.05" bind:value={adjContrast} /></div>
      <div class="adj-row"><label>Saturation: {adjSaturation.toFixed(2)}</label>
        <input type="range" min="0" max="2.5" step="0.05" bind:value={adjSaturation} /></div>
      <div class="adj-check">
        <button class="btn" style="margin-left:auto;font-size:0.75rem;padding:0.35rem 0.65rem" onclick={resetAdjust}>Reset</button>
        <button class="btn btn-primary" style="font-size:0.75rem;padding:0.35rem 0.65rem" onclick={applyAdjustmentsToImage}>Apply</button>
      </div>
    </div>
  

<div class="footer">DocAI v2.0 — Model Testing</div>

<style>
  .page-header { margin-bottom: 2rem; }
  .page-header h1 { font-size: 1.5rem; font-weight: 700; }
  .page-header p { color: var(--text2); margin-top: 0.25rem; font-size: 0.9rem; }
  .card { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.5rem; margin-bottom: 1.5rem; }
  .card :global(h2) { font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  .compare-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; align-items: stretch; }
  .img-label { font-size: 0.85rem; font-weight: 600; color: var(--text2); margin-bottom: 0.5rem; text-align: center; }
  .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; }
  .info-item { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-xs); padding: 0.875rem; }
  .info-item .k { font-size: 0.7rem; font-weight: 700; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.375rem; }
  .info-item .v { font-size: 0.95rem; font-weight: 600; }
  .upload-zone { border: 2px dashed var(--border); border-radius: var(--radius-sm); padding: 2rem; text-align: center; cursor: pointer; transition: all 0.2s; }
  .upload-zone:hover { border-color: var(--accent); background: rgba(6,182,212,0.05); }
  .upload-zone :global(i) { width: 36px; height: 36px; color: var(--text3); margin-bottom: 0.5rem; }
  .upload-zone p { color: var(--text2); font-size: 0.85rem; }
  .model-select { margin-top: 1rem; display: grid; gap: 0.45rem; }
  .crop-panel { margin-top: 1rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 0.9rem; }
  .crop-head { display: flex; justify-content: space-between; gap: 0.75rem; align-items: flex-start; margin-bottom: 0.75rem; }
  .crop-head label { font-size: 0.72rem; font-weight: 700; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; }
  .inline-check { display:flex; align-items:center; gap:.5rem; font-size:.82rem; color:var(--text); text-transform:none !important; letter-spacing:0 !important; }
  .inline-check input { width:18px; height:18px; accent-color:var(--primary); }
  .crop-pills { display:flex; flex-wrap:wrap; gap:.4rem; margin-bottom:.6rem; }
  .crop-head .info-text { margin: 0.25rem 0 0; }
  .crop-stage { width: 100%; overflow: auto; background: #020617; border-radius: var(--radius-xs); display: grid; place-items: center; touch-action: none; }
  .crop-stage canvas { max-width: 100%; display: block; cursor: crosshair; touch-action: none; }
  .crop-actions { display: flex; gap: 0.5rem; margin-top: 0.75rem; flex-wrap: wrap; }
  .crop-method { display:flex; align-items:center; flex-wrap:wrap; gap:.4rem; margin-top:.6rem; }
  .method-label { font-size:.78rem; color:var(--text2); margin-right:.2rem; }
  .pill.sm { padding:.35rem .7rem; font-size:.75rem; }
  .crop-result { margin-top: 1rem; }
  .edge-preview { margin-top:.75rem; border:1px solid var(--border); border-radius:var(--radius-xs); overflow:hidden; background:#020617; }
  .edge-preview .label { padding:.45rem .65rem; border-bottom:1px solid var(--border); font-size:.72rem; color:var(--text2); display:flex; gap:.4rem; align-items:center; }
  .edge-preview canvas { width:100%; max-height:260px; object-fit:contain; display:block; }
  .warpbar { height:6px; background:#111827; border-radius:999px; overflow:hidden; margin-top:.6rem; }
  .warpbar span { display:block; height:100%; background:linear-gradient(90deg,#38bdf8,#f59e0b); transition:width .05s; }
  .model-select label { font-size: 0.72rem; font-weight: 700; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; }
  .model-select select { width: 100%; background: var(--bg); border: 1px solid var(--border); color: var(--text); border-radius: var(--radius-xs); padding: 0.65rem 0.75rem; font-size: 0.85rem; }
  .model-select .info-text { margin-left: 0; }
  .mode-pills { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
  .pretrained-pills { margin-top: 0.5rem; }
  .pill { padding: 0.5rem 0.875rem; background: var(--bg); border: 1px solid var(--border); border-radius: 999px; color: var(--text2); font-size: 0.8rem; font-weight: 500; transition: all 0.15s; }
  .pill.pretrained { border-color: rgba(99,102,241,0.45); }
  .pill:hover { border-color: var(--accent); color: var(--text); }
  .pill.active { border-color: var(--accent); background: rgba(6,182,212,0.1); color: var(--accent2); }
  .pill:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; border-radius: var(--radius-xs); font-weight: 600; font-size: 0.8rem; border: none; transition: all 0.15s; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-primary { background: var(--primary); color: white; }
  .img-box { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden; }
  .img-box .label { padding: 0.625rem 0.75rem; border-bottom: 1px solid var(--border); font-size: 0.75rem; font-weight: 600; color: var(--text2); display: flex; align-items: center; gap: 0.5rem; }
  .img-box img { width: 100%; max-height: 70vh; object-fit: contain; display: block; background: var(--bg); }
  .empty-preview { min-height: 180px; display: grid; place-items: center; color: var(--text3); font-size: 0.85rem; }
  .info-text { font-size: 0.8rem; color: var(--text2); margin-left: auto; }
  .footer { text-align: center; padding: 2rem 0; color: var(--text3); font-size: 0.8rem; }
  @media (max-width: 768px) { .grid-2, .compare-grid { grid-template-columns: 1fr; } }
  .adj-sliders { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 0.4rem; margin-top: 0.75rem; padding: 0.75rem; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); }
  .adj-row { display: flex; flex-direction: column; gap: 0.15rem; }
  .adj-row label { font-size: 0.72rem; color: var(--text2); }
  .adj-row input[type="range"] { width: 100%; accent-color: var(--primary); }
  .adj-check { display: flex; align-items: center; gap: 0.4rem; font-size: 0.75rem; color: var(--text2); }
</style>
