<script>
  import { closeToast, toast } from '../stores/toast.js'
</script>

{#if $toast.message}
  <div class="toast {$toast.type}" class:show={$toast.message}>
    <div class="toast-message">{$toast.message}</div>
    <button class="toast-close" onclick={closeToast} aria-label="Tutup pesan">×</button>
  </div>
{/if}

<style>
  .toast {
    position: fixed; top: 1rem; left: 50%; width: min(720px, calc(100vw - 2rem)); padding: 0.875rem 1rem;
    border-radius: var(--radius-xs); font-size: 0.85rem; font-weight: 500;
    transform: translate(-50%, -120px); opacity: 0; transition: all 0.2s; z-index: 999;
    display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem;
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.25);
  }
  .toast.show { transform: translate(-50%, 0); opacity: 1; }
  .toast-message { line-height: 1.45; white-space: pre-wrap; }
  .toast-close { border: 0; background: transparent; color: currentColor; cursor: pointer; font-size: 1.25rem; line-height: 1; opacity: 0.85; padding: 0; }
  .toast-close:hover { opacity: 1; }
  .toast.success { background: var(--success); color: white; }
  .toast.error { background: var(--error); color: white; }
</style>
