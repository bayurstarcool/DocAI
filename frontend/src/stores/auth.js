import { writable, get } from 'svelte/store'

function createAuthStore() {
  const storedToken = typeof localStorage !== 'undefined' ? localStorage.getItem('docai_token') || '' : ''
  const storedUser = typeof localStorage !== 'undefined' ? localStorage.getItem('docai_user') || '' : ''

  const { subscribe, set } = writable({ token: storedToken, user: storedUser })

  return {
    subscribe,

    login(t, u) {
      localStorage.setItem('docai_token', t)
      localStorage.setItem('docai_user', u)
      set({ token: t, user: u })
    },

    logout() {
      localStorage.removeItem('docai_token')
      localStorage.removeItem('docai_user')
      set({ token: '', user: '' })
      fetch('/api/auth/logout')
      window.location.href = '/login'
    },

    getToken() {
      return localStorage.getItem('docai_token') || ''
    }
  }
}

export const auth = createAuthStore()

export async function apiFetch(url, opts = {}) {
  const token = localStorage.getItem('docai_token') || ''
  const headers = { ...opts.headers }
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(url, { ...opts, headers })
  if (res.status === 401) {
    window.location.href = '/login'
    throw new Error('Unauthorized')
  }
  return res
}

export async function apiJson(url, opts = {}) {
  const res = await apiFetch(url, opts)
  if (!res.ok) {
    const d = await res.json().catch(() => ({}))
    throw new Error(d.detail || d.error || 'Request failed')
  }
  return res.json()
}
