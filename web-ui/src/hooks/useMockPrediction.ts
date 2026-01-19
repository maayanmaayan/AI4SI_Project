import { SERVICE_CATEGORIES, type ServiceCategoryId } from '../constants/serviceCategories'
import type { PredictionResult, SelectedLocation } from '../types/prediction'

function normalize(values: number[]): number[] {
  const sum = values.reduce((acc, v) => acc + v, 0)
  if (sum <= 0) {
    const equal = 1 / values.length
    return values.map(() => equal)
  }
  return values.map((v) => v / sum)
}

export function getMockPrediction(location: SelectedLocation): PredictionResult {
  const raw = Array.from({ length: SERVICE_CATEGORIES.length }, () => Math.random() + 0.01)
  const probabilities = normalize(raw)

  let bestIndex = 0
  for (let i = 1; i < probabilities.length; i += 1) {
    if (probabilities[i] > probabilities[bestIndex]) {
      bestIndex = i
    }
  }

  const selectedCategoryId: ServiceCategoryId = SERVICE_CATEGORIES[bestIndex].id

  return {
    location,
    probabilities,
    selectedCategoryId,
  }
}

