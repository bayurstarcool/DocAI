<script>
  import { auth } from '../stores/auth.js'
  import { refreshIcons } from './icons.js'
  import { navigate, currentRoute } from './router.svelte.js'

  let mobileOpen = false

  const navItems = [
    { href: '/', icon: 'layout-dashboard', label: 'Dashboard' },
    { href: '/test', icon: 'flask-conical', label: 'Test Model' },
    { href: '/train', icon: 'brain', label: 'Training' },
    { href: '/image-tests', icon: 'images', label: 'Image Tests' },
    { href: '/datasets', icon: 'database', label: 'Datasets' },
  ]

  function isActive(href) {
    return $currentRoute === href
  }

  function go(href) {
    navigate(href)
    mobileOpen = false
    refreshIcons()
  }
</script>

<nav class="navbar">
  <div class="nav-inner">
    <a href="/" onclick={(e) => { e.preventDefault(); go('/') }} class="logo">
      <div class="logo-mark"><i data-lucide="scan-text"></i></div>
      <span class="logo-text">DocAI</span>
    </a>

    <div class="nav-links">
      {#each navItems as item}
        <a
          href={item.href}
          onclick={(e) => { e.preventDefault(); go(item.href) }}
          class:active={isActive(item.href)}
        >
          <i data-lucide={item.icon}></i>
          {item.label}
        </a>
      {/each}
    </div>

    <div class="nav-right">
      <div class="user-chip">
        <div class="user-avatar"><i data-lucide="user"></i></div>
        <span class="user-name">{$auth.user || 'admin'}</span>
      </div>
      <button class="btn-logout" onclick={() => auth.logout()} aria-label="Logout">
        <i data-lucide="log-out"></i>
      </button>
      <button class="nav-toggle" onclick={() => mobileOpen = !mobileOpen} aria-label="Menu">
        <i data-lucide="menu"></i>
      </button>
    </div>
  </div>

  {#if mobileOpen}
    <div class="nav-mobile open">
      {#each navItems as item}
        <a
          href={item.href}
          onclick={(e) => { e.preventDefault(); go(item.href) }}
          class:active={isActive(item.href)}
        >
          <i data-lucide={item.icon}></i>
          {item.label}
        </a>
      {/each}
      <div class="divider"></div>
      <a href="/login" onclick={(e) => { e.preventDefault(); auth.logout() }}>
        <i data-lucide="log-out"></i>
        Logout
      </a>
    </div>
  {/if}
</nav>

<style>
  .navbar { position: sticky; top: 0; z-index: 100; background: rgba(9,9,11,0.75); backdrop-filter: blur(16px); border-bottom: 1px solid var(--border); }
  .nav-inner { max-width: 1280px; margin: 0 auto; padding: 0 1.5rem; height: 64px; display: flex; align-items: center; justify-content: space-between; }
  .logo { display: flex; align-items: center; gap: 0.625rem; }
  .logo-mark { width: 36px; height: 36px; background: linear-gradient(135deg,#6366f1,#06b6d4); border-radius: 10px; display: grid; place-items: center; color: white; }
  .logo-text { font-size: 1.125rem; font-weight: 700; }
  .nav-links { display: flex; gap: 0.25rem; }
  .nav-links a { color: var(--text2); font-weight: 500; font-size: 0.875rem; padding: 0.5rem 0.875rem; border-radius: var(--radius-xs); transition: all 0.15s; display: flex; align-items: center; gap: 0.5rem; text-decoration: none; }
  .nav-links a:hover { color: var(--text); background: var(--bg3); }
  .nav-links a.active { color: var(--accent2); background: rgba(6,182,212,0.1); }
  .nav-right { display: flex; align-items: center; gap: 0.75rem; }
  .user-chip { display: flex; align-items: center; gap: 0.5rem; padding: 0.375rem 0.75rem 0.375rem 0.375rem; background: var(--bg3); border: 1px solid var(--border); border-radius: 999px; }
  .user-avatar { width: 28px; height: 28px; background: linear-gradient(135deg, #8b5cf6, #ec4899); border-radius: 50%; display: grid; place-items: center; color: white; }
  .btn-logout { display: flex; align-items: center; padding: 0.5rem; background: transparent; border: 1px solid var(--border); border-radius: var(--radius-xs); color: var(--text2); transition: all 0.15s; }
  .btn-logout:hover { border-color: var(--error); color: var(--error); }
  .nav-toggle { display: none; background: none; border: 1px solid var(--border); color: var(--text2); padding: 0.5rem; border-radius: var(--radius-xs); cursor: pointer; }
  .nav-mobile { display: none; position: absolute; top: 64px; left: 0; right: 0; background: var(--bg2); border-bottom: 1px solid var(--border); padding: 0.75rem; z-index: 200; }
  .nav-mobile.open { display: block; }
  .nav-mobile a { display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1rem; color: var(--text2); border-radius: var(--radius-xs); font-weight: 500; font-size: 0.9rem; text-decoration: none; }
  .nav-mobile a:hover { background: var(--bg3); color: var(--text); }
  .nav-mobile a.active { color: var(--accent2); background: rgba(6,182,212,0.1); }
  .divider { height: 1px; background: var(--border); margin: 0.5rem 0; }
  @media (max-width: 768px) {
    .nav-links { display: none !important; }
    .nav-toggle { display: flex; }
    .user-name { display: none; }
  }
</style>
