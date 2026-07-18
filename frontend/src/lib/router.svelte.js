/**
 * SPA router using a writable store + history.pushState for URL sync.
 * Navigation updates the store AND the browser history so back/forward works.
 */
import { writable } from 'svelte/store'

const VALID_ROUTES = ['/', '/test', '/train', '/image-tests', '/datasets']

export const currentRoute = writable('/')

export function navigate(href) {
  if (!href || href === '#') return
  if (!VALID_ROUTES.includes(href)) return

  // Update URL via history API (no page reload)
  if (window.location.pathname !== href) {
    window.history.pushState({ route: href }, '', href)
  }

  currentRoute.set(href)
}

export function initRouter() {
  // Read initial route from URL pathname
  let initial = window.location.pathname
  if (!VALID_ROUTES.includes(initial)) initial = '/'
  currentRoute.set(initial)

  // Handle browser back/forward navigation
  window.addEventListener('popstate', (e) => {
    const route = e.state?.route || window.location.pathname
    if (VALID_ROUTES.includes(route)) {
      currentRoute.set(route)
    }
  })
}
