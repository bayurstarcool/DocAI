<script>
  import { toast } from '../stores/toast.js'
</script>

{#if $toast.message}
  <div class="toast {$toast.type}" class:show={$toast.message}>
    {$toast.message}
  </div>
{/if}

<style>
  .toast {
    position: fixed; bottom: 2rem; right: 2rem; padding: 0.75rem 1.25rem;
    border-radius: var(--radius-xs); font-size: 0.85rem; font-weight: 500;
    transform: translateY(100px); opacity: 0; transition: all 0.3s; z-index: 999;
  }
  .toast.show { transform: translateY(0); opacity: 1; }
  .toast.success { background: var(--success); color: white; }
  .toast.error { background: var(--error); color: white; }
</style>
