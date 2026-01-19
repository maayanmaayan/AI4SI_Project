import type { PredictionResult } from '../types/prediction'
import { SERVICE_CATEGORIES, type ServiceCategoryId } from '../constants/serviceCategories'
import { mapConfig } from '../config/mapConfig'
import { exportRecommendationToPdf } from '../utils/pdfExport'

interface PredictionModalProps {
  visible: boolean
  predictionResult: PredictionResult | null
  onClose: () => void
  onCategoryChange: (categoryId: ServiceCategoryId) => void
}

function formatPercentage(value: number): string {
  return `${Math.round(value * 100)}%`
}

function formatCoordinates(lat: number, lng: number): string {
  return `${lat.toFixed(4)}, ${lng.toFixed(4)}`
}

function getExplanation(categoryId: ServiceCategoryId): string {
  switch (categoryId) {
    case 'education':
      return 'New education facilities here could shorten daily school commutes and support families within a comfortable walking radius.'
    case 'entertainment':
      return 'Cultural and leisure venues at this location can strengthen local social life and reduce the need for long evening trips.'
    case 'grocery':
      return 'A small grocery cluster here would help residents meet day-to-day needs without relying on car-based shopping.'
    case 'health':
      return 'Health services in this area can bring everyday care closer to older adults and families, reducing distance to basic check-ups.'
    case 'posts_banks':
      return 'Banking and postal services here would support administrative errands locally, instead of concentrating them in distant centers.'
    case 'parks':
      return 'A pocket park at this point can provide everyday green space, improve microclimate, and create a calm stop on walking routes.'
    case 'sustenance':
      return 'Cafés and small food venues here can make streets more vibrant and provide everyday meeting points within a short walk.'
    case 'shops':
      return 'Everyday retail here can complete the local offer of services, so residents can cover most errands within 15 minutes on foot.'
    default:
      return ''
  }
}

export function PredictionModal({
  visible,
  predictionResult,
  onClose,
  onCategoryChange,
}: PredictionModalProps) {
  if (!visible || !predictionResult) return null

  const { location, probabilities, selectedCategoryId } = predictionResult

  return (
    <div
      role="dialog"
      aria-modal="true"
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(15, 23, 42, 0.3)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: '#ffffff',
          borderRadius: 16,
          padding: 24,
          maxWidth: 720,
          width: '90%',
          boxShadow: '0 18px 45px rgba(15, 23, 42, 0.18)',
          color: '#0f172a',
          fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, sans-serif',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
          <div>
            <div
              style={{
                fontSize: 14,
                textTransform: 'uppercase',
                letterSpacing: 1,
                color: '#64748b',
              }}
            >
              15‑Minute City Recommendation
            </div>
            <h2
              style={{
                margin: '4px 0 0',
                fontSize: 20,
                fontWeight: 600,
              }}
            >
              {mapConfig.cityName} · {formatCoordinates(location.lat, location.lng)}
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              borderRadius: 999,
              border: 'none',
              padding: '6px 12px',
              backgroundColor: '#e2e8f0',
              cursor: 'pointer',
              fontSize: 13,
            }}
          >
            Close
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '2fr 3fr', gap: 24 }}>
          <div>
            <div
              style={{
                marginBottom: 8,
                fontSize: 13,
                fontWeight: 500,
                color: '#475569',
              }}
            >
              Service categories (mock probabilities)
            </div>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 8,
              }}
            >
              {SERVICE_CATEGORIES.map((category, index) => {
                const isSelected = category.id === selectedCategoryId
                const percentage = probabilities[index] ?? 0

                return (
                  <button
                    key={category.id}
                    type="button"
                    onClick={() => onCategoryChange(category.id)}
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'flex-start',
                      padding: 10,
                      borderRadius: 12,
                      border: isSelected ? `2px solid ${category.color}` : '1px solid #e2e8f0',
                      backgroundColor: isSelected ? `${category.color}22` : '#f8fafc',
                      cursor: 'pointer',
                    }}
                  >
                    <span style={{ fontSize: 18, marginBottom: 4 }}>{category.iconEmoji}</span>
                    <span
                      style={{
                        fontSize: 13,
                        fontWeight: 600,
                        marginBottom: 2,
                        color: '#0f172a',
                      }}
                    >
                      {category.shortLabel}
                    </span>
                    <span
                      style={{
                        fontSize: 12,
                        color: '#64748b',
                      }}
                    >
                      {formatPercentage(percentage)}
                    </span>
                  </button>
                )
              })}
            </div>
          </div>

          <div
            style={{
              padding: 16,
              borderRadius: 16,
              background: 'linear-gradient(135deg, #e0f2fe, #fefce8)',
            }}
          >
            {SERVICE_CATEGORIES.filter((c) => c.id === selectedCategoryId).map((category) => (
              <div key={category.id}>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontSize: 24, marginRight: 8 }}>{category.iconEmoji}</span>
                  <div>
                    <div
                      style={{
                        fontSize: 13,
                        textTransform: 'uppercase',
                        letterSpacing: 1,
                        color: '#64748b',
                      }}
                    >
                      Suggested focus
                    </div>
                    <div
                      style={{
                        fontSize: 17,
                        fontWeight: 600,
                        color: '#0f172a',
                      }}
                    >
                      {category.label}
                    </div>
                  </div>
                </div>
                <p
                  style={{
                    fontSize: 14,
                    color: '#475569',
                    lineHeight: 1.5,
                    margin: '8px 0 0',
                  }}
                >
                  {getExplanation(category.id)}
                </p>
                <button
                  type="button"
                  onClick={() => exportRecommendationToPdf(predictionResult)}
                  style={{
                    marginTop: 16,
                    borderRadius: 999,
                    border: 'none',
                    padding: '8px 16px',
                    background:
                      'linear-gradient(135deg, #6366f1, #8b5cf6)',
                    color: 'white',
                    fontSize: 13,
                    fontWeight: 500,
                    cursor: 'pointer',
                    boxShadow: '0 10px 25px rgba(79, 70, 229, 0.35)',
                  }}
                >
                  Export as PDF
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

