import { describe, expect, it } from 'vitest'
import { getMockPrediction } from '../hooks/useMockPrediction'
import { SERVICE_CATEGORIES } from '../constants/serviceCategories'

describe('getMockPrediction', () => {
  it('returns probabilities that sum approximately to 1', () => {
    const prediction = getMockPrediction({ lat: 0, lng: 0 })
    const sum = prediction.probabilities.reduce((acc, v) => acc + v, 0)
    expect(sum).toBeGreaterThan(0.99)
    expect(sum).toBeLessThan(1.01)
  })

  it('returns exactly 8 probabilities', () => {
    const prediction = getMockPrediction({ lat: 0, lng: 0 })
    expect(prediction.probabilities).toHaveLength(SERVICE_CATEGORIES.length)
  })

  it('selects a valid category id', () => {
    const prediction = getMockPrediction({ lat: 0, lng: 0 })
    const ids = SERVICE_CATEGORIES.map((c) => c.id)
    expect(ids).toContain(prediction.selectedCategoryId)
  })
})

