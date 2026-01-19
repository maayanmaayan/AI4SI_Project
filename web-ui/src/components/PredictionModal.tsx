import type { PredictionResult } from '../types/prediction'
import { SERVICE_CATEGORIES, type ServiceCategoryId } from '../constants/serviceCategories'
import { mapConfig } from '../config/mapConfig'
import { exportRecommendationToPdf } from '../utils/pdfExport'

interface PredictionModalProps {
  visible: boolean
  predictionResult: PredictionResult | null
  isLoading?: boolean
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
  // Simple, readable mock mapping from probability to impact metrics.
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

function ImpactBar({
  label,
  value,
  max = 100,
  suffix = '',
}: {
  label: string
  value: number
  max?: number
  suffix?: string
}) {
  const pct = Math.max(0, Math.min(1, value / max))
  return (
    <div style={{ marginBottom: 8 }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 12,
          color: '#475569',
          marginBottom: 4,
        }}
      >
        <span>{label}</span>
        <span>
          {value.toLocaleString()}
          {suffix}
        </span>
      </div>
      <div
        style={{
          width: '100%',
          height: 6,
          borderRadius: 999,
          backgroundColor: 'rgba(148, 163, 184, 0.25)',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${pct * 100}%`,
            height: '100%',
            borderRadius: 999,
            background:
              'linear-gradient(90deg, rgba(79,70,229,0.85), rgba(14,165,233,0.9))',
          }}
        />
      </div>
    </div>
  )
}

export function PredictionModal({
  visible,
  predictionResult,
  isLoading = false,
  onClose,
  onCategoryChange,
}: PredictionModalProps) {
  if (!visible) return null

  // Show loading state
  if (isLoading || !predictionResult) {
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
            padding: 48,
            maxWidth: 400,
            width: '90%',
            boxShadow: '0 18px 45px rgba(15, 23, 42, 0.18)',
            color: '#0f172a',
            fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, sans-serif',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div
            style={{
              width: 48,
              height: 48,
              border: '4px solid rgba(79, 70, 229, 0.2)',
              borderTop: '4px solid #6366f1',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              marginBottom: 16,
            }}
          />
          <div
            style={{
              fontSize: 14,
              color: '#64748b',
              textAlign: 'center',
            }}
          >
            Calculating recommendation for this location...
          </div>
        </div>
      </div>
    )
  }

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
            {SERVICE_CATEGORIES.filter((c) => c.id === selectedCategoryId).map((category) => {
              const idx = SERVICE_CATEGORIES.findIndex((c) => c.id === category.id)
              const probability = probabilities[idx] ?? 0
              const impact = getImpactMetrics(probability)

              return (
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
                      margin: '8px 0 4px',
                    }}
                  >
                    {getExplanation(category.id)}
                  </p>
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'flex-end',
                      marginBottom: 12,
                    }}
                  >
                    <button
                      type="button"
                      style={{
                        borderRadius: 999,
                        border: 'none',
                        padding: '6px 14px',
                        background:
                          'linear-gradient(135deg, #e2e8f0, #cbd5f5)',
                        fontSize: 12,
                        fontWeight: 500,
                        color: '#0f172a',
                        cursor: 'pointer',
                        boxShadow: '0 6px 14px rgba(148, 163, 184, 0.45)',
                      }}
                    >
                      Deeper analysis
                    </button>
                  </div>
                  <div style={{ marginBottom: 8, fontSize: 12, color: '#64748b' }}>
                    Estimated contribution within everyday walking/cycling distance:
                  </div>
                  <ImpactBar
                    label="People served locally"
                    value={impact.peopleServed}
                    max={5000}
                  />
                  <ImpactBar
                    label="Support for local community life"
                    value={impact.communityScore}
                    max={100}
                    suffix=" / 100"
                  />
                  <ImpactBar
                    label="Reduction in private car use & air pollution"
                    value={impact.airPollutionReduction}
                    max={100}
                    suffix=" / 100"
                  />
                  <ImpactBar
                    label="Reduction in road congestion"
                    value={impact.congestionReduction}
                    max={100}
                    suffix=" / 100"
                  />
                  <div
                    style={{
                      marginTop: 16,
                      display: 'flex',
                      justifyContent: 'flex-end',
                      gap: 0,
                    }}
                  >
                    <button
                      type="button"
                      onClick={() => exportRecommendationToPdf(predictionResult)}
                      style={{
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
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

