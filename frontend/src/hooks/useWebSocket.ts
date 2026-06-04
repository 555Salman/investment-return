import { useEffect, useRef } from 'react'

export function useWebSocket(
  path: string,
  onMessage: (data: unknown) => void,
  enabled = true
) {
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!enabled) return

    const url = `ws://${window.location.hostname}:8000${path}`
    const ws  = new WebSocket(url)
    wsRef.current = ws

    ws.onmessage = (e) => {
      try {
        onMessage(JSON.parse(e.data))
      } catch {
        onMessage(e.data)
      }
    }

    ws.onerror = () => console.warn('WebSocket error on', path)

    return () => { ws.close() }
  }, [path, enabled])   // eslint-disable-line react-hooks/exhaustive-deps

  return wsRef
}
