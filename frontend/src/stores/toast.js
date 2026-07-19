import { writable } from 'svelte/store'

export const toast = writable({ message: '', type: 'success' })

export function showToast(message, type = 'success') {
  toast.set({ message, type })
}

export function closeToast() {
  toast.set({ message: '', type: 'success' })
}
