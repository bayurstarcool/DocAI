<script>
  import { auth } from '../stores/auth.js'
  import { refreshIcons } from './icons.js'
  import { navigate, currentRoute } from './router.svelte.js'

  let mobileOpen = false
  let isAdmin = false

  import { onMount } from 'svelte'
  onMount(async () => {
    const token = localStorage.getItem('docai_token')
    if (token) {
      try {
        const payload = JSON.parse(atob(token))
        isAdmin = payload.user === 'admin'
      } catch(e) {}
    }
  })

  const navItems = [
    { href: '/', icon: 'layout-dashboard', label: 'Dashboard', admin: true },
    { href: '/test', icon: 'flask-conical', label: 'Test Model', admin: true },
    { href: '/train', icon: 'brain', label: 'DocAI Training', admin: true },
    { href: '/docres', icon: 'cpu', label: 'DocRes', admin: true },
    { href: '/image-tests', icon: 'images', label: 'Image Tests', admin: true },
    { href: '/datasets', icon: 'database', label: 'Datasets', admin: true },
    { href: '/dataset-manager', icon: 'settings-2', label: 'Dataset Manager', admin: true },
    { href: '/synthetic-shadow', icon: 'sun-dim', label: 'Shadow Gen', admin: true },
    { href: '/magang', icon: 'graduation-cap', label: 'Magang' },
    { href: '/magang-review', icon: 'clipboard-check', label: 'Review', admin: true },
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
        {#if item.admin && !isAdmin}
        {:else}
          <a
            href={item.href}
            onclick={(e) => { e.preventDefault(); go(item.href) }}
            class:active={isActive(item.href)}
          >
            <i data-lucide={item.icon}></i>
            {item.label}
          </a>
        {/if}
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
        {#if item.admin && !isAdmin}
        {:else}
          <a
            href={item.href}
            onclick={(e) => { e.preventDefault(); go(item.href) }}
            class:active={isActive(item.href)}
          >
            <i data-lucide={item.icon}></i>
            {item.label}
          </a>
        {/if}
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
  .navbar { position: sticky; top: 0; z-index: 100; background: rgba(8,9,10,.82); backdrop-filter: blur(20px) saturate(135%); border-bottom: 1px solid rgba(255,255,255,.065); }
  .nav-inner { width: min(100%,1320px); margin: 0 auto; padding: 0 clamp(1rem,3vw,2rem); height: 62px; display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
  .logo { display:flex; align-items:center; gap:.65rem; flex-shrink:0; }
  .logo-mark { width:34px; height:34px; background:linear-gradient(145deg,#6d72dd,#5359b7); border:1px solid rgba(255,255,255,.14); border-radius:9px; display:grid; place-items:center; color:white; box-shadow:0 7px 22px rgba(69,75,190,.23); }
  .logo-text { font-size:1rem; font-weight:590; letter-spacing:-.035em; }
  .nav-links { display:flex; align-items:center; gap:.18rem; padding:.25rem; background:rgba(255,255,255,.018); border:1px solid rgba(255,255,255,.055); border-radius:9px; }
  .nav-links a { color:var(--text2); font-weight:510; font-size:.82rem; padding:.46rem .72rem; border-radius:6px; display:flex; align-items:center; gap:.45rem; min-height:36px; }
  .nav-links a:hover { color:var(--text); background:rgba(255,255,255,.045); }
  .nav-links a.active { color:#e8e9ff; background:rgba(113,112,255,.14); box-shadow:inset 0 0 0 1px rgba(130,143,255,.14); }
  .nav-right { display:flex; align-items:center; gap:.55rem; flex-shrink:0; }
  .user-chip { display:flex; align-items:center; gap:.45rem; padding:.3rem .65rem .3rem .3rem; background:rgba(255,255,255,.025); border:1px solid var(--border); border-radius:999px; }
  .user-avatar { width:27px; height:27px; background:linear-gradient(145deg,#7170ff,#4f54a8); border-radius:50%; display:grid; place-items:center; color:white; }
  .user-name { max-width:110px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:.78rem; }
  .btn-logout,.nav-toggle { display:flex; align-items:center; justify-content:center; width:38px; height:38px; min-height:38px; background:rgba(255,255,255,.02); border:1px solid var(--border); border-radius:7px; color:var(--text2); }
  .btn-logout:hover { border-color:rgba(239,68,68,.5); color:#f87171; }
  .nav-toggle { display:none; }
  .nav-mobile { display:none; position:absolute; top:62px; left:.65rem; right:.65rem; background:rgba(15,16,17,.98); border:1px solid var(--border); border-radius:0 0 12px 12px; padding:.6rem; z-index:200; box-shadow:0 20px 50px rgba(0,0,0,.5); }
  .nav-mobile.open { display:block; }
  .nav-mobile a { display:flex; align-items:center; gap:.75rem; min-height:46px; padding:.7rem .85rem; color:var(--text2); border-radius:7px; font-weight:510; font-size:.9rem; }
  .nav-mobile a:hover,.nav-mobile a.active { background:rgba(113,112,255,.1); color:var(--text); }
  .divider { height:1px; background:var(--border); margin:.45rem 0; }
  @media(max-width:900px){ .nav-links{display:none}.nav-toggle{display:flex}.user-chip{display:none}.nav-inner{height:58px}.nav-mobile{top:58px} }
  @media(max-width:430px){ .btn-logout{display:none}.logo-mark{width:32px;height:32px}.nav-inner{padding:0 .85rem} }
</style>
