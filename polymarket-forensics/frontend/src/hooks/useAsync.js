/**
 * Generic async data hook with abort-on-unmount, error state, and retry.
 *
 * Usage:
 *   const { data, loading, error, retry } = useAsync(
 *     (signal) => api.wallet(addr, signal),
 *     [addr],
 *   );
 */
import { useCallback, useEffect, useState } from 'react';

export function useAsync(fn, deps = [], { immediate = true } = {}) {
  const [state, setState] = useState({ data: null, loading: immediate, error: null });
  const [tick, setTick] = useState(0);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const memoFn = useCallback(fn, deps);

  useEffect(() => {
    if (!immediate && tick === 0) return undefined;
    const ctrl = new AbortController();
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: null }));
    memoFn(ctrl.signal)
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: null });
      })
      .catch((error) => {
        if (cancelled || error.name === 'AbortError') return;
        setState({ data: null, loading: false, error });
      });
    return () => {
      cancelled = true;
      ctrl.abort();
    };
  }, [memoFn, tick, immediate]);

  const retry = useCallback(() => setTick((n) => n + 1), []);
  return { ...state, retry };
}


export function usePolledAsync(fn, deps = [], { intervalMs = 15000 } = {}) {
  const { data, loading, error, retry } = useAsync(fn, deps);
  useEffect(() => {
    const id = setInterval(retry, intervalMs);
    return () => clearInterval(id);
  }, [retry, intervalMs]);
  return { data, loading, error, retry };
}
