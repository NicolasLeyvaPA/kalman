import { useCallback } from 'react';
import { api } from '../services/api';
import { useAsync } from './useAsync';

export function useWallet(address) {
  const fetcher = useCallback(
    (signal) => api.wallet(address, signal), [address],
  );
  return useAsync(fetcher, [address]);
}

export function useWalletTrades(address, limit = 50) {
  const fetcher = useCallback(
    (signal) => api.walletTrades(address, limit, signal), [address, limit],
  );
  return useAsync(fetcher, [address, limit]);
}

export function useWalletFunding(address) {
  const fetcher = useCallback(
    (signal) => api.walletFunding(address, signal), [address],
  );
  return useAsync(fetcher, [address]);
}
