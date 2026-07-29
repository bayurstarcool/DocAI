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
  import DatasetManager from './pages/DatasetManager.svelte'
  import DatasetDetail from './pages/DatasetDetail.svelte'
  import DocResTraining from './pages/DocResTraining.svelte'
  import SyntheticShadow from './pages/SyntheticShadow.svelte'
import Magang from './pages/Magang.svelte'
import MagangReview from './pages/MagangReview.svelte'

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
  {:else if $currentRoute === '/docres'}
    <DocResTraining />
  {:else if $currentRoute === '/image-tests'}
    <ImageTests />
  {:else if $currentRoute === '/datasets' || $currentRoute.startsWith('/datasets/')}
    <Datasets />
  {:else if $currentRoute === '/dataset-manager'}
    <DatasetManager />
  {:else if $currentRoute.startsWith('/dataset-manager/')}
    <DatasetDetail />
  {:else if $currentRoute === '/synthetic-shadow'}
    <SyntheticShadow />
  {:else if $currentRoute === '/magang'}
    <Magang />
  {:else if $currentRoute === '/magang-review'}
    <MagangReview />
  {:else}
    <Dashboard />
  {/if}
</main>
<Toast />

<style>
  .container { width: min(100%, 1320px); margin: 0 auto; padding: 2rem clamp(1rem,3vw,2rem) 4rem; position: relative; z-index: 1; }
  @media (max-width: 768px) { .container { padding: 1.25rem .85rem 5rem; } }
</style>
