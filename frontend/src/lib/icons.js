export function refreshIcons() {
  try { window.lucide?.createIcons() } catch(e) {}
}
