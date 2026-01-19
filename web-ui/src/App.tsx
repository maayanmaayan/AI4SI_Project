import { useState } from 'react'
import { MapView } from './components/MapView'
import { PredictionModal } from './components/PredictionModal'
import { getMockPrediction } from './hooks/useMockPrediction'
import type { PredictionResult } from './types/prediction'
import { mapConfig } from './config/mapConfig'
import './index.css'

function App() {
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [modalVisible, setModalVisible] = useState(false)

  const handleLocationSelected = (location: { lat: number; lng: number }) => {
    const result = getMockPrediction(location)
    setPredictionResult(result)
    setModalVisible(true)
  }

  const handleCategoryChange = (categoryId: PredictionResult['selectedCategoryId']) => {
    if (!predictionResult) return
    setPredictionResult({ ...predictionResult, selectedCategoryId: categoryId })
  }

  const handleClose = () => {
    setModalVisible(false)
  }

  return (
    <div
      style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #eff6ff, #faf5ff)',
        padding: 24,
        boxSizing: 'border-box',
      }}
    >
      <div
        style={{
          maxWidth: 1200,
          margin: '0 auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 16,
        }}
      >
        <header
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'baseline',
          }}
        >
          <div>
            <h1
              style={{
                margin: 0,
                fontSize: 24,
                fontWeight: 600,
                color: '#0f172a',
                fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, sans-serif',
              }}
            >
              15‑Minute City Explorer
            </h1>
            <p
              style={{
                margin: '4px 0 0',
                fontSize: 14,
                color: '#475569',
                maxWidth: 640,
              }}
            >
              Explore {mapConfig.cityName} and click within the highlighted area to see a mock
              recommendation for which everyday service category could strengthen local 15‑minute
              access.
            </p>
          </div>
        </header>

        <main
          style={{
            flex: 1,
            minHeight: '70vh',
            borderRadius: 20,
            backgroundColor: '#ffffff',
            boxShadow: '0 24px 60px rgba(15, 23, 42, 0.12)',
            padding: 16,
          }}
        >
          <div
            style={{
              height: 'calc(80vh - 80px)',
              minHeight: 480,
              borderRadius: 16,
              overflow: 'hidden',
            }}
          >
            <MapView onLocationSelected={handleLocationSelected} />
          </div>
        </main>
      </div>

      <PredictionModal
        visible={modalVisible}
        predictionResult={predictionResult}
        onClose={handleClose}
        onCategoryChange={handleCategoryChange}
      />
    </div>
  )
}

export default App
