import jsPDF from 'jspdf'
import type { PredictionResult } from '../types/prediction'
import { SERVICE_CATEGORIES, type ServiceCategoryId } from '../constants/serviceCategories'
import { mapConfig } from '../config/mapConfig'

function formatPercentage(value: number): string {
  return `${Math.round(value * 100)}%`
}

function formatCoordinates(lat: number, lng: number): string {
  return `${lat.toFixed(4)}, ${lng.toFixed(4)}`
}

function getExplanation(categoryId: ServiceCategoryId): string {
  switch (categoryId) {
    case 'education':
      return 'Education here can shorten daily school commutes and support families within a comfortable walking radius.'
    case 'entertainment':
      return 'Entertainment and cultural venues can strengthen local social life and reduce the need for long evening trips.'
    case 'grocery':
      return 'Everyday grocery options here help residents meet routine needs without car-based shopping.'
    case 'health':
      return 'Health services at this point bring basic care closer to older adults and families.'
    case 'posts_banks':
      return 'Posts and banks here keep administrative errands local instead of concentrating them in distant centers.'
    case 'parks':
      return 'A small park here offers daily access to greenery, shade, and calm within a short walk.'
    case 'sustenance':
      return 'Food venues at this location make streets more vibrant and create everyday meeting points.'
    case 'shops':
      return 'Shops here round out the local offer so most errands can be completed within 15 minutes on foot.'
    default:
      return ''
  }
}

export function exportRecommendationToPdf(prediction: PredictionResult): void {
  const doc = new jsPDF()

  const { location, probabilities, selectedCategoryId } = prediction
  const selectedCategory = SERVICE_CATEGORIES.find((c) => c.id === selectedCategoryId)

  doc.setFontSize(16)
  doc.text(
    `15-Minute City Recommendation – ${mapConfig.cityName}`,
    14,
    20,
  )

  doc.setFontSize(11)
  doc.text(`Location: ${formatCoordinates(location.lat, location.lng)}`, 14, 30)

  doc.setFontSize(12)
  doc.text('Service categories (mock probabilities)', 14, 42)

  const startY = 48
  let y = startY

  SERVICE_CATEGORIES.forEach((category, index) => {
    const label = category.shortLabel
    const pct = formatPercentage(probabilities[index] ?? 0)
    doc.text(`${label}: ${pct}`, 14, y)
    y += 6
  })

  if (selectedCategory) {
    y += 6
    doc.setFontSize(12)
    doc.text('Suggested focus', 14, y)
    y += 6

    doc.setFontSize(11)
    doc.text(`${selectedCategory.label}`, 14, y)
    y += 8

    const explanation = getExplanation(selectedCategory.id)
    const lines = doc.splitTextToSize(explanation, 180)
    doc.text(lines, 14, y)
  }

  const fileName = `recommendation_${mapConfig.cityName.toLowerCase()}_${location.lat.toFixed(
    4,
  )}_${location.lng.toFixed(4)}.pdf`

  doc.save(fileName)
}

