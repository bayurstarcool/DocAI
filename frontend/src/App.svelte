<script>
  import { onMount } from 'svelte'
  import { refreshIcons } from './lib/icons.js'
  import { currentRoute, initRouter } from './lib/router.svelte.js'
  import Navbar from './lib/Navbar.svelte'
  import Toast from './lib/Toast.svelte'
  import Dashboard from './pages/Dashboard.svelte'
  import TestModel from './pages/TestModel.svelte'
  import Training from './pages/Training.svelte'
  import ImageTests from './pages/ImageTests.svelte'
  import Datasets from './pages/Datasets.svelte'

  onMount(() => {
    initRouter()
    refreshIcons()
  })
</script>

<div class="bg-grid"></div>
<Navbar />
<main class="container">
  {#if $currentRoute === '/'}
    <Dashboard />
  {:else if $currentRoute === '/test'}
    <TestModel />
  {:else if $currentRoute === '/train'}
    <Training />
  {:else if $currentRoute === '/image-tests'}
    <ImageTests />
  {:else if $currentRoute === '/datasets'}
    <Datasets />
  {:else}
    <Dashboard />
  {/if}
</main>
<Toast />

<style>
  .container { max-width: 1280px; margin: 0 auto; padding: 2rem 1.5rem; position: relative; z-index: 1; }
</style>
