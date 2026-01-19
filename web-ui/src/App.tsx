import { useState } from 'react'
import { MapView } from './components/MapView'
import { PredictionModal } from './components/PredictionModal'
import { InfoModal } from './components/InfoModal'
import { SearchBar } from './components/SearchBar'
import { getMockPrediction } from './hooks/useMockPrediction'
import type { PredictionResult } from './types/prediction'
import './index.css'

function App() {
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [modalVisible, setModalVisible] = useState(false)
  const [infoModalVisible, setInfoModalVisible] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [searchCenter, setSearchCenter] = useState<{ lat: number; lng: number } | null>(null)

  const handleLocationSelected = (location: { lat: number; lng: number }) => {
    if (isLoading) return
    setIsLoading(true)
    setModalVisible(true)

    // Simulate a short model inference time for realism
    setTimeout(() => {
      const result = getMockPrediction(location)
      setPredictionResult(result)
      setIsLoading(false)
    }, 1800)
  }

  const handleCategoryChange = (categoryId: PredictionResult['selectedCategoryId']) => {
    if (!predictionResult) return
    setPredictionResult({ ...predictionResult, selectedCategoryId: categoryId })
  }

  const handleClose = () => {
    setModalVisible(false)
    setIsLoading(false)
  }

  return (
    <div
      style={{
        position: 'relative',
        width: '100vw',
        height: '100vh',
        margin: 0,
        padding: 0,
        overflow: 'hidden',
      }}
    >
      <MapView
        onLocationSelected={handleLocationSelected}
        selectedLocation={predictionResult?.location ?? null}
        selectedCategoryId={predictionResult?.selectedCategoryId ?? null}
        searchCenter={searchCenter}
      />

      <SearchBar
        onLocationFound={(location) => {
          setSearchCenter({ lat: location.lat, lng: location.lng })
        }}
      />

      <button
        type="button"
        onClick={() => setInfoModalVisible(true)}
        style={{
          position: 'absolute',
          top: 16,
          right: 16,
          width: 40,
          height: 40,
          borderRadius: '50%',
          border: 'none',
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          color: '#6366f1',
          fontSize: 20,
          fontWeight: 600,
          cursor: 'pointer',
          boxShadow: '0 4px 12px rgba(15, 23, 42, 0.15)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          transition: 'all 0.2s ease',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = '#ffffff'
          e.currentTarget.style.transform = 'scale(1.05)'
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.95)'
          e.currentTarget.style.transform = 'scale(1)'
        }}
        aria-label="Learn how this system works"
      >
        i
      </button>

      <PredictionModal
        visible={modalVisible}
        predictionResult={predictionResult}
        isLoading={isLoading}
        onClose={handleClose}
        onCategoryChange={handleCategoryChange}
      />

      <InfoModal visible={infoModalVisible} onClose={() => setInfoModalVisible(false)} />
    </div>
  )
}

export default App
