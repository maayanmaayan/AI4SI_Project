import { describe, expect, it, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { PredictionModal } from '../components/PredictionModal'
import { SERVICE_CATEGORIES } from '../constants/serviceCategories'

const basePrediction = {
  location: { lat: 31.78, lng: 35.22 },
  probabilities: Array.from({ length: SERVICE_CATEGORIES.length }, () => 1 / SERVICE_CATEGORIES.length),
  selectedCategoryId: SERVICE_CATEGORIES[0].id,
}

describe('PredictionModal', () => {
  it('renders 8 service categories when visible', () => {
    render(
      <PredictionModal
        visible
        predictionResult={basePrediction}
        onClose={() => {}}
        onCategoryChange={() => {}}
      />,
    )

    const buttons = screen.getAllByRole('button', { name: /%/ })
    expect(buttons).toHaveLength(SERVICE_CATEGORIES.length)
  })

  it('calls onCategoryChange when a category is clicked', () => {
    const onCategoryChange = vi.fn()

    render(
      <PredictionModal
        visible
        predictionResult={basePrediction}
        onClose={() => {}}
        onCategoryChange={onCategoryChange}
      />,
    )

    const targetCategory = SERVICE_CATEGORIES[1]
    fireEvent.click(screen.getByText(targetCategory.shortLabel))

    expect(onCategoryChange).toHaveBeenCalledWith(targetCategory.id)
  })
})

