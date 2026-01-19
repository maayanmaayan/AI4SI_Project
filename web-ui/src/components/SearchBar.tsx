import { useState } from 'react'

interface SearchBarProps {
  onLocationFound: (location: { lat: number; lng: number; displayName: string }) => void
}

interface NominatimResult {
  display_name: string
  lat: string
  lon: string
}

export function SearchBar({ onLocationFound }: SearchBarProps) {
  const [query, setQuery] = useState('')
  const [isSearching, setIsSearching] = useState(false)
  const [results, setResults] = useState<NominatimResult[]>([])
  const [showResults, setShowResults] = useState(false)

  const handleSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) {
      setResults([])
      setShowResults(false)
      return
    }

    setIsSearching(true)
    try {
      // Use Nominatim (OpenStreetMap geocoding) API
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
          searchQuery,
        )}&limit=5&bounded=1&viewbox=35.1,31.7,35.3,31.9&bounded=1`,
        {
          headers: {
            'User-Agent': '15-Minute-City-Explorer/1.0',
          },
        },
      )
      const data: NominatimResult[] = await response.json()
      setResults(data)
      setShowResults(true)
    } catch (error) {
      console.error('Search error:', error)
      setResults([])
    } finally {
      setIsSearching(false)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setQuery(value)
    if (value.trim()) {
      // Debounce search
      const timeoutId = setTimeout(() => handleSearch(value), 300)
      return () => clearTimeout(timeoutId)
    } else {
      setResults([])
      setShowResults(false)
    }
  }

  const handleSelectResult = (result: NominatimResult) => {
    const lat = parseFloat(result.lat)
    const lng = parseFloat(result.lon)
    onLocationFound({ lat, lng, displayName: result.display_name })
    setQuery(result.display_name)
    setShowResults(false)
  }

  return (
    <div
      style={{
        position: 'absolute',
        top: 16,
        left: 16,
        zIndex: 1000,
        width: '320px',
      }}
    >
      <div style={{ position: 'relative' }}>
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          placeholder="Search for a location..."
          style={{
            width: '100%',
            padding: '10px 40px 10px 14px',
            borderRadius: 999,
            border: 'none',
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            fontSize: 14,
            boxShadow: '0 4px 12px rgba(15, 23, 42, 0.15)',
            outline: 'none',
            fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, sans-serif',
          }}
          onFocus={() => {
            if (results.length > 0) setShowResults(true)
          }}
        />
        {isSearching && (
          <div
            style={{
              position: 'absolute',
              right: 12,
              top: '50%',
              transform: 'translateY(-50%)',
              width: 16,
              height: 16,
              border: '2px solid rgba(99, 102, 241, 0.3)',
              borderTop: '2px solid #6366f1',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
            }}
          />
        )}
        {!isSearching && query && (
          <div
            style={{
              position: 'absolute',
              right: 12,
              top: '50%',
              transform: 'translateY(-50%)',
              cursor: 'pointer',
              fontSize: 18,
              color: '#64748b',
              lineHeight: 1,
            }}
            onClick={() => {
              setQuery('')
              setResults([])
              setShowResults(false)
            }}
          >
            ×
          </div>
        )}
      </div>

      {showResults && results.length > 0 && (
        <div
          style={{
            marginTop: 8,
            backgroundColor: 'rgba(255, 255, 255, 0.98)',
            borderRadius: 12,
            boxShadow: '0 8px 24px rgba(15, 23, 42, 0.2)',
            overflow: 'hidden',
            maxHeight: '300px',
            overflowY: 'auto',
          }}
        >
          {results.map((result, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => handleSelectResult(result)}
              style={{
                width: '100%',
                padding: '12px 16px',
                border: 'none',
                borderBottom: idx < results.length - 1 ? '1px solid #e2e8f0' : 'none',
                backgroundColor: 'transparent',
                textAlign: 'left',
                cursor: 'pointer',
                fontSize: 13,
                color: '#0f172a',
                fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, sans-serif',
                transition: 'background-color 0.15s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#f8fafc'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent'
              }}
            >
              <div style={{ fontWeight: 500, marginBottom: 2 }}>
                {result.display_name.split(',')[0]}
              </div>
              <div style={{ fontSize: 11, color: '#64748b' }}>
                {result.display_name.split(',').slice(1).join(',').trim()}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
