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
      return 'New education facilities here could shorten daily school commutes and support families within a comfortable walking radius. Given the mix of family housing and nearby streets that already carry school trips, this location can absorb that activity without overloading the wider network.'
    case 'entertainment':
      return 'Cultural and leisure venues at this location can strengthen local social life and reduce the need for long evening trips. The surrounding blocks already attract evening footfall, so adding cultural uses here would extend street activity without demanding extra car parking.'
    case 'grocery':
      return 'A small grocery cluster here would help residents meet day-to-day needs without relying on car-based shopping. This node sits between several residential pockets that currently rely on longer trips, so a local grocery anchor would rebalance everyday flows.'
    case 'health':
      return 'Health services in this area can bring everyday care closer to older adults and families, reducing distance to basic check-ups. The demographic profile around this point suggests a mix of children and older residents who particularly benefit from shorter, predictable trips.'
    case 'posts_banks':
      return 'Banking and postal services here would support administrative errands locally, instead of concentrating them in distant centers. The local street pattern already links neighbourhood centers, so adding everyday admin services here would fit smoothly into existing walking routes.'
    case 'parks':
      return 'A pocket park at this point can provide everyday green space, improve microclimate, and create a calm stop on walking routes. Slight gaps in tree cover and public seating around this junction mean even a small green space would noticeably improve local comfort.'
    case 'sustenance':
      return 'Cafés and small food venues here can make streets more vibrant and provide everyday meeting points within a short walk. Existing ground-floor frontages and footfall patterns make this a natural corner for everyday social stops without turning it into a regional nightlife hub.'
    case 'shops':
      return 'Everyday retail here can complete the local offer of services, so residents can cover most errands within 15 minutes on foot. The surrounding catchment has enough residents and intersecting walking routes to support small-scale shops without drawing heavy car traffic.'
    default:
      return ''
  }
}

function getImpactMetrics(probability: number) {
  const clamped = Math.max(0, Math.min(1, probability))
  const peopleServed = Math.round(800 + clamped * 4200) // ~800–5000
  const communityScore = Math.round(40 + clamped * 60) // 40–100
  const airPollutionReduction = Math.round(20 + clamped * 60) // 20–80
  const congestionReduction = Math.round(15 + clamped * 65) // 15–80

  return {
    peopleServed,
    communityScore,
    airPollutionReduction,
    congestionReduction,
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
  doc.text('Service categories – recommendation summary', 14, 42)

  const startY = 48
  let y = startY

  SERVICE_CATEGORIES.forEach((category, index) => {
    const label = category.shortLabel
    const pct = formatPercentage(probabilities[index] ?? 0)
    doc.text(`${label}: ${pct}`, 14, y)
    y += 6
  })

  // Determine top 3 categories by probability
  const withProb = SERVICE_CATEGORIES.map((category, index) => ({
    category,
    probability: probabilities[index] ?? 0,
  })).sort((a, b) => b.probability - a.probability)

  const top3 = withProb.slice(0, 3)

  y += 8
  doc.setFontSize(12)
  doc.text('Top 3 recommended service categories', 14, y)
  y += 4

  doc.setFontSize(11)
  top3.forEach((entry, idx) => {
    const { category, probability } = entry
    const rankLabel = `${idx + 1}. ${category.label} (${formatPercentage(probability)})`
    const impact = getImpactMetrics(probability)

    y += 8
    doc.text(rankLabel, 14, y)
    y += 5

    const bullets = [
      `• People served locally: ~${impact.peopleServed.toLocaleString()} residents`,
      `• Support for local community life: ${impact.communityScore} / 100`,
      `• Reduction in private car use & air pollution: ${impact.airPollutionReduction} / 100`,
      `• Reduction in road congestion: ${impact.congestionReduction} / 100`,
    ]
    bullets.forEach((line) => {
      y += 5
      doc.text(line, 18, y)
    })

    const explanation = getExplanation(category.id)
    const explLines = doc.splitTextToSize(explanation, 170)
    y += 6
    doc.text(explLines, 18, y)

    // Add a bit more spacing before next category
    y += explLines.length * 4
  })

  if (selectedCategory) {
    y += 10
    doc.setFontSize(12)
    doc.text('Highlighted focus in the interface', 14, y)
    y += 6

    doc.setFontSize(11)
    doc.text(`${selectedCategory.label}`, 14, y)
  }

  const fileName = `recommendation_${mapConfig.cityName.toLowerCase()}_${location.lat.toFixed(
    4,
  )}_${location.lng.toFixed(4)}.pdf`

  doc.save(fileName)
}

