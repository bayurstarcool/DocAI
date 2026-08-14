import { writable } from 'svelte/store'

const VALID_ROUTES = ["/", "/test", "/train", "/docres", "/image-tests", "/datasets", "/dataset-manager", "/synthetic-shadow", "/synthetic-dewarp", "/magang", "/magang-review"]

export const currentRoute = writable('/')

export function navigate(href) {
  if (!href || href === '#') return

  const isStatic = VALID_ROUTES.includes(href)
  const isDynamic = (href.startsWith('/dataset-manager/') || href.startsWith('/datasets/')) && href.split('/').length === 3
  if (!isStatic && !isDynamic) return

  if (window.location.pathname !== href) {
    window.history.pushState({ route: href }, '', href)
  }

  currentRoute.set(href)
}

export function initRouter() {
  let initial = window.location.pathname
  const isStatic = VALID_ROUTES.includes(initial)
  const isDynamic = (initial.startsWith('/dataset-manager/') || initial.startsWith('/datasets/')) && initial.split('/').length === 3
  if (!isStatic && !isDynamic) initial = '/'
  currentRoute.set(initial)

  window.addEventListener('popstate', (e) => {
    const route = e.state?.route || window.location.pathname
    const isStatic = VALID_ROUTES.includes(route)
    const isDynamic = (route.startsWith('/dataset-manager/') || route.startsWith('/datasets/')) && route.split('/').length === 3
    if (isStatic || isDynamic) {
      currentRoute.set(route)
    }
  })
}
