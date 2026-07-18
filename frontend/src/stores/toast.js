import { writable } from 'svelte/store'

export const toast = writable({ message: '', type: 'success' })

let timer
export function showToast(message, type = 'success') {
  clearTimeout(timer)
  toast.set({ message, type })
  timer = setTimeout(() => toast.set({ message: '', type: 'success' }), 3000)
}
