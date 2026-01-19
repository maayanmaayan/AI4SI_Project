import type { ServiceCategoryId } from '../constants/serviceCategories'

export interface SelectedLocation {
  lat: number
  lng: number
}

export interface PredictionResult {
  location: SelectedLocation
  probabilities: number[]
  selectedCategoryId: ServiceCategoryId
}

