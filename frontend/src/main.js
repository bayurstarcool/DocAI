import './app.css'
import App from './App.svelte'
import { mount } from 'svelte'
import { createIcons } from 'lucide'
import {
  Activity, ArrowLeft, CheckCircle, ChevronLeft, ChevronRight,
  Cpu, Database, Download, Eye, File, Folder, FolderOpen,
  Grid3x3, Home, Image, ImagePlus, Info, Layers, List, LogOut,
  Menu, Play, RefreshCw, ScanText, ScrollText, Search, Settings,
  Sparkles, Square, Upload, UploadCloud, User, X
} from 'lucide'

const icons = {
  Activity, ArrowLeft, CheckCircle, ChevronLeft, ChevronRight,
  Cpu, Database, Download, Eye, File, Folder, FolderOpen,
  Grid3x3, Home, Image, ImagePlus, Info, Layers, List, LogOut,
  Menu, Play, RefreshCw, ScanText, ScrollText, Search, Settings,
  Sparkles, Square, Upload, UploadCloud, User, X
}

function initIcons() {
  try { createIcons({ icons }) } catch(e) {
    console.warn('lucide icons init failed:', e)
  }
}

function boot() {
  // Svelte 5 mount() appends to target, doesn't replace.
  // Clear placeholder (.boot div) so "Loading DocAI..." disappears.
  const target = document.getElementById('app')
  if (target) {
    while (target.firstChild) target.removeChild(target.firstChild)
  }

  try {
    mount(App, { target })
  } catch(e) {
    const el = document.getElementById('boot-error')
    if (el) {
      el.style.display = 'block'
      el.textContent = 'Mount Error: ' + (e.message || e)
    }
    throw e
  }

  initIcons()
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot)
} else {
  boot()
}

export default null
