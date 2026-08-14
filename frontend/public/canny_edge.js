/**
 * Client-side document crop detection
 * Find paper border by looking for the largest dark-to-light transition
 */
(function() {
  'use strict'

  window.__cannyEdgeDetect = function(cropImage) {
    if (!cropImage) return null
    const maxDim = 600
    const scale = Math.min(maxDim / cropImage.naturalWidth, maxDim / cropImage.naturalHeight, 1)
    const w = Math.max(32, Math.round(cropImage.naturalWidth * scale))
    const h = Math.max(32, Math.round(cropImage.naturalHeight * scale))
    const tmp = document.createElement('canvas')
    tmp.width = w; tmp.height = h
    const ctx = tmp.getContext('2d')
    ctx.drawImage(cropImage, 0, 0, w, h)
    const imgData = ctx.getImageData(0, 0, w, h)
    const data = imgData.data

    // Grayscale
    const gray = new Uint8Array(w * h)
    for (let i = 0, j = 0; i < data.length; i += 4, j++)
      gray[j] = Math.round(0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2])

    // Find paper: scan from edges inward to find where brightness jumps
    // Left edge: scan right until brightness > threshold
    // Right edge: scan left until brightness > threshold
    // Top edge: scan down until brightness > threshold
    // Bottom edge: scan up until brightness > threshold
    
    const sorted = Array.from(gray).sort((a,b) => a-b)
    const p25 = sorted[Math.floor(sorted.length * 0.25)]
    const p75 = sorted[Math.floor(sorted.length * 0.75)]
    const threshold = Math.floor((p25 + p75) / 2)

    // Scan from each edge to find paper boundary
    function scanFromLeft() {
      for (let x = 0; x < w; x++) {
        let bright = 0
        for (let y = Math.floor(h*0.1); y < Math.floor(h*0.9); y += 2) {
          if (gray[y * w + x] > threshold) bright++
        }
        if (bright > h * 0.3) return x
      }
      return 0
    }
    function scanFromRight() {
      for (let x = w - 1; x >= 0; x--) {
        let bright = 0
        for (let y = Math.floor(h*0.1); y < Math.floor(h*0.9); y += 2) {
          if (gray[y * w + x] > threshold) bright++
        }
        if (bright > h * 0.3) return x
      }
      return w - 1
    }
    function scanFromTop() {
      for (let y = 0; y < h; y++) {
        let bright = 0
        for (let x = Math.floor(w*0.1); x < Math.floor(w*0.9); x += 2) {
          if (gray[y * w + x] > threshold) bright++
        }
        if (bright > w * 0.3) return y
      }
      return 0
    }
    function scanFromBottom() {
      for (let y = h - 1; y >= 0; y--) {
        let bright = 0
        for (let x = Math.floor(w*0.1); x < Math.floor(w*0.9); x += 2) {
          if (gray[y * w + x] > threshold) bright++
        }
        if (bright > w * 0.3) return y
      }
      return h - 1
    }

    const left = scanFromLeft()
    const right = scanFromRight()
    const top = scanFromTop()
    const bottom = scanFromBottom()

    if (right <= left || bottom <= top) return null

    // Add small padding
    const pad = 2
    const corners = [
      { x: Math.max(0, left - pad) / scale, y: Math.max(0, top - pad) / scale },
      { x: Math.min(w - 1, right + pad) / scale, y: Math.max(0, top - pad) / scale },
      { x: Math.min(w - 1, right + pad) / scale, y: Math.min(h - 1, bottom + pad) / scale },
      { x: Math.max(0, left - pad) / scale, y: Math.min(h - 1, bottom + pad) / scale }
    ]

    return { edgePoints: corners.slice(0, 4), w, h, scale, edgeMask: gray }
  }

  window.__findDocCorners = function(edgePoints, imgW, imgH, scale) {
    if (!edgePoints || edgePoints.length < 4) return null
    const result = edgePoints.map(p => ({ x: p.x, y: p.y }))

    const area = Math.abs(
      (result[0].x*result[1].y - result[1].x*result[0].y) +
      (result[1].x*result[2].y - result[2].x*result[1].y) +
      (result[2].x*result[3].y - result[3].x*result[2].y) +
      (result[3].x*result[0].y - result[0].x*result[3].y)
    ) / 2
    const imgArea = imgW * imgH
    const ratio = area / imgArea

    let score = 0
    if (ratio > 0.15 && ratio < 0.95) score = 90
    else if (ratio > 0.05) score = 70
    else score = 40

    result.score = Math.min(95, score)
    return result
  }

  window.__drawEdgePrev = function(mask, ew, eh, scale, img, disp) {
    const ec = document.querySelector('.edge-preview canvas')
    if (!ec) return
    ec.width = disp.width; ec.height = disp.height
    const ctx = ec.getContext('2d')
    ctx.drawImage(img, 0, 0, disp.width, disp.height)
  }
})()
