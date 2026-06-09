import { useEffect, useRef } from 'react'

export function useWebSocket(
  path: string,
  onMessage: (data: unknown) => void,
  enabled = true
) {
  const wsRef        = useRef<WebSocket | null>(null)
  const onMessageRef = useRef(onMessage)

  // Keep the ref in sync with the latest callback without re-creating the socket.
  useEffect(() => {
    onMessageRef.current = onMessage
  }, [onMessage])

  useEffect(() => {
    if (!enabled) return

    // Derive protocol from the page so ws:// is used in HTTP dev and wss:// in
    // HTTPS production — removes the hardcoded port 8000 and insecure scheme.
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url      = `${protocol}//${window.location.host}${path}`
    const ws       = new WebSocket(url)
    wsRef.current  = ws

    ws.onmessage = (e) => {
      try {
        onMessageRef.current(JSON.parse(e.data))
      } catch {
        onMessageRef.current(e.data)
      }
    }

    ws.onerror = () => console.warn('WebSocket error on', path)

    return () => { ws.close() }
  }, [path, enabled])

  return wsRef
}
