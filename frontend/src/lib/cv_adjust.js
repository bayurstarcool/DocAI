let ready = false

export function isCvReady() {
  return ready
}

export function markCvReady() {
  ready = true
}

export function adjustCanvas(srcCanvas, dstCanvas, p) {
  if (!ready || !srcCanvas || !dstCanvas || typeof cv === 'undefined') return false

  let mat = null
  try {
    mat = cv.imread(srcCanvas)

    if (p.gamma !== 1) {
      const lut = new cv.Mat(1, 256, cv.CV_8UC1)
      for (let i = 0; i < 256; i++) {
        lut.ucharPtr(0, i)[0] = Math.min(255, Math.max(0,
          Math.round(255 * Math.pow(i / 255, 1 / p.gamma))))
      }
      const channels = new cv.MatVector()
      cv.split(mat, channels)
      for (let i = 0; i < channels.size(); i++) {
        const channel = channels.get(i)
        cv.LUT(channel, lut, channel)
        channels.set(i, channel)
        channel.delete()
      }
      cv.merge(channels, mat)
      channels.delete()
      lut.delete()
    }

    if (p.b !== 1 || p.c !== 1) {
      mat.convertTo(mat, -1, p.c, (p.b - 1) * 128)
    }

    if (p.s !== 1) {
      const hsv = new cv.Mat()
      const channels = new cv.MatVector()
      cv.cvtColor(mat, hsv, cv.COLOR_RGBA2RGB)
      cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV)
      cv.split(hsv, channels)
      const sat = channels.get(1)
      sat.convertTo(sat, -1, p.s, 0)
      channels.set(1, sat)
      cv.merge(channels, hsv)
      cv.cvtColor(hsv, hsv, cv.COLOR_HSV2RGB)
      cv.cvtColor(hsv, mat, cv.COLOR_RGB2RGBA)
      sat.delete()
      channels.delete()
      hsv.delete()
    }

    if (p.sh !== 1) {
      const blurred = new cv.Mat()
      const kernel = cv.Mat.ones(3, 3, cv.CV_32F)
      kernel.convertTo(kernel, cv.CV_32F, 1 / 9)
      cv.filter2D(mat, blurred, -1, kernel)
      cv.addWeighted(mat, p.sh, blurred, -(p.sh - 1), 0, mat)
      kernel.delete()
      blurred.delete()
    }

    // CLAHE and white balance temporarily use stable RGB transforms only.
    // CLAHE on RGBA requires explicit alpha handling.
    if (p.clahe > 0) {
      const rgb = new cv.Mat()
      const lab = new cv.Mat()
      const channels = new cv.MatVector()
      cv.cvtColor(mat, rgb, cv.COLOR_RGBA2RGB)
      cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab)
      cv.split(lab, channels)
      const light = channels.get(0)
      const clahe = new cv.CLAHE(Math.max(0.1, p.clahe), new cv.Size(8, 8))
      clahe.apply(light, light)
      channels.set(0, light)
      cv.merge(channels, lab)
      cv.cvtColor(lab, rgb, cv.COLOR_Lab2RGB)
      cv.cvtColor(rgb, mat, cv.COLOR_RGB2RGBA)
      clahe.delete()
      light.delete()
      channels.delete()
      lab.delete()
      rgb.delete()
    }

    cv.imshow(dstCanvas, mat)
    mat.delete()
    return true
  } catch (error) {
    console.error('OpenCV adjustment failed', error)
    if (mat) mat.delete()
    return false
  }
}
