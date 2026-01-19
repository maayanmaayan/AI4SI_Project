import { useState } from 'react'
import { MapView } from './components/MapView'
import { PredictionModal } from './components/PredictionModal'
import { getMockPrediction } from './hooks/useMockPrediction'
import type { PredictionResult } from './types/prediction'
import './index.css'

function App() {
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [modalVisible, setModalVisible] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

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
      />

      <PredictionModal
        visible={modalVisible}
        predictionResult={predictionResult}
        isLoading={isLoading}
        onClose={handleClose}
        onCategoryChange={handleCategoryChange}
      />
    </div>
  )
}

export default App
