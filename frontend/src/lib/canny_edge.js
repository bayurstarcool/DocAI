/**
 * Client-side Canny edge detection for document crop
 * Pure JS, no dependencies
 */
export function cannyEdgeDetect(cropImage) {
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
  const gray = new Float32Array(w * h)
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    gray[j] = 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2]
  }
  // Gaussian blur 5x5
  const kernel = [1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1]
  const kSum = 273
  const blurred = new Float32Array(w * h)
  for (let y = 2; y < h-2; y++) for (let x = 2; x < w-2; x++) {
    let sum = 0, ki = 0
    for (let ky = -2; ky <= 2; ky++) for (let kx = -2; kx <= 2; kx++) {
      sum += gray[(y+ky)*w + (x+kx)] * kernel[ki++]
    }
    blurred[y*w+x] = sum / kSum
  }
  // Sobel
  const gx = new Float32Array(w * h)
  const gy = new Float32Array(w * h)
  const mag = new Float32Array(w * h)
  for (let y = 1; y < h-1; y++) for (let x = 1; x < w-1; x++) {
    const i = y*w+x
    gx[i] = -blurred[(y-1)*w+x-1] - 2*blurred[y*w+x-1] - blurred[(y+1)*w+x-1]
           + blurred[(y-1)*w+x+1] + 2*blurred[y*w+x+1] + blurred[(y+1)*w+x+1]
    gy[i] = -blurred[(y-1)*w+x-1] - 2*blurred[(y-1)*w+x] - blurred[(y-1)*w+x+1]
           + blurred[(y+1)*w+x-1] + 2*blurred[(y+1)*w+x] + blurred[(y+1)*w+x+1]
    mag[i] = Math.sqrt(gx[i]*gx[i] + gy[i]*gy[i])
  }
  // Non-maximum suppression
  const nms = new Float32Array(w * h)
  for (let y = 1; y < h-1; y++) for (let x = 1; x < w-1; x++) {
    const i = y*w+x
    const angle = Math.atan2(gy[i], gx[i]) * 180 / Math.PI
    const a = ((angle % 180) + 180) % 180
    let q = 0, r = 0
    if ((a < 22.5) || (a >= 157.5)) { q = mag[y*w+x+1]; r = mag[y*w+x-1] }
    else if (a < 67.5) { q = mag[(y+1)*w+x-1]; r = mag[(y-1)*w+x+1] }
    else if (a < 112.5) { q = mag[(y+1)*w+x]; r = mag[(y-1)*w+x] }
    else { q = mag[(y+1)*w+x+1]; r = mag[(y-1)*w+x-1] }
    nms[i] = (mag[i] >= q && mag[i] >= r) ? mag[i] : 0
  }
  // Double threshold + hysteresis
  const highT = 80, lowT = 30
  const strong = 255, weak = 128
  const edge = new Uint8ClampedArray(w * h)
  for (let i = 0; i < nms.length; i++) {
    edge[i] = nms[i] >= highT ? strong : (nms[i] >= lowT ? weak : 0)
  }
  for (let iter = 0; iter < 3; iter++) {
    for (let y = 1; y < h-1; y++) for (let x = 1; x < w-1; x++) {
      const i = y*w+x
      if (edge[i] === weak) {
        for (let ky = -1; ky <= 1; ky++) for (let kx = -1; kx <= 1; kx++) {
          if (edge[(y+ky)*w+(x+kx)] === strong) { edge[i] = strong; break }
        }
      }
    }
  }
  const edgePoints = []
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (edge[y*w+x] === strong) edgePoints.push({ x, y })
  }
  return { edgePoints, w, h, scale, edgeMask: edge }
}

export function findDocumentCorners(edgePoints, imgW, imgH, scale) {
  if (!edgePoints || edgePoints.length < 40) return null
  const cx = imgW / 2, cy = imgH / 2
  const quads = [[], [], [], []]
  for (const p of edgePoints) {
    if (p.x < cx && p.y < cy) quads[0].push(p)
    else if (p.x >= cx && p.y < cy) quads[1].push(p)
    else if (p.x >= cx && p.y >= cy) quads[2].push(p)
    else quads[3].push(p)
  }
  const corners = [
    { x: 0, y: 0 }, { x: imgW, y: 0 },
    { x: imgW, y: imgH }, { x: 0, y: imgH },
  ]
  const result = []
  for (let i = 0; i < 4; i++) {
    if (quads[i].length < 5) return null
    let best = quads[i][0], bestD = Infinity
    for (const p of quads[i]) {
      const d = Math.hypot(p.x - corners[i].x, p.y - corners[i].y)
      if (d < bestD) { bestD = d; best = p }
    }
    result.push({ x: best.x / scale, y: best.y / scale })
  }
  const area = Math.abs(
    (result[0].x*result[1].y - result[1].x*result[0].y) +
    (result[1].x*result[2].y - result[2].x*result[1].y) +
    (result[2].x*result[3].y - result[3].x*result[2].y) +
    (result[3].x*result[0].y - result[0].x*result[3].y)
  ) / 2
  const imgArea = imgW * imgH
  if (area < imgArea * 0.05 || area > imgArea * 0.99) return null
  let score = 0
  for (let i = 0; i < 4; i++) {
    const nearEdges = edgePoints.filter(p =>
      Math.hypot(p.x/scale - result[i].x, p.y/scale - result[i].y) < 40
    ).length
    score += Math.min(25, nearEdges * 2)
  }
  result.score = Math.min(95, score)
  return result
}
