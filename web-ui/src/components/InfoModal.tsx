interface InfoModalProps {
  visible: boolean
  onClose: () => void
}

export function InfoModal({ visible, onClose }: InfoModalProps) {
  if (!visible) return null

  return (
    <div
      role="dialog"
      aria-modal="true"
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(15, 23, 42, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 2000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: '#ffffff',
          borderRadius: 16,
          padding: 32,
          maxWidth: 680,
          width: '90%',
          maxHeight: '85vh',
          overflowY: 'auto',
          boxShadow: '0 24px 60px rgba(15, 23, 42, 0.25)',
          color: '#0f172a',
          fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, sans-serif',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 24 }}>
          <h2
            style={{
              margin: 0,
              fontSize: 24,
              fontWeight: 600,
              color: '#0f172a',
            }}
          >
            How This System Works
          </h2>
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
              marginLeft: 16,
            }}
          >
            Close
          </button>
        </div>

        <div style={{ fontSize: 15, lineHeight: 1.7, color: '#475569' }}>
          <section style={{ marginBottom: 28 }}>
            <h3
              style={{
                fontSize: 18,
                fontWeight: 600,
                color: '#0f172a',
                marginBottom: 12,
                marginTop: 0,
              }}
            >
              Data-Driven Urban Planning
            </h3>
            <p style={{ margin: '0 0 12px 0' }}>
              This system uses advanced AI to help urban planners identify where everyday services would most benefit
              residents within a 15-minute walk or bike ride. Our recommendations are based on real spatial and
              demographic data, refined through a transparent machine learning process.
            </p>
          </section>

          <section style={{ marginBottom: 28 }}>
            <h3
              style={{
                fontSize: 18,
                fontWeight: 600,
                color: '#0f172a',
                marginBottom: 12,
                marginTop: 0,
              }}
            >
              Spatial and Demographic Data Refinement
            </h3>
            <p style={{ margin: '0 0 12px 0' }}>
              The system integrates multiple data sources to build a comprehensive picture of each location:
            </p>
            <ul style={{ margin: '0 0 12px 0', paddingLeft: 24 }}>
              <li style={{ marginBottom: 8 }}>
                <strong>OpenStreetMap (OSM) data:</strong> Real-world service locations, building footprints, street
                networks, and walkability features
              </li>
              <li style={{ marginBottom: 8 }}>
                <strong>Census and demographic data:</strong> Population density, socioeconomic indicators, age
                distribution, and mobility patterns
              </li>
              <li style={{ marginBottom: 8 }}>
                <strong>Spatial context:</strong> Network-based walking distances, intersection density, and local
                service density within 15-minute access zones
              </li>
            </ul>
            <p style={{ margin: '0 0 12px 0' }}>
              These data points are processed and normalized to create a consistent, comparable view of urban
              environments, ensuring recommendations account for both physical accessibility and community needs.
            </p>
          </section>

          <section style={{ marginBottom: 28 }}>
            <h3
              style={{
                fontSize: 18,
                fontWeight: 600,
                color: '#0f172a',
                marginBottom: 12,
                marginTop: 0,
              }}
            >
              AI Model Trained on Optimal Examples
            </h3>
            <p style={{ margin: '0 0 12px 0' }}>
              Our Spatial Graph Transformer model was trained exclusively on neighborhoods that successfully implement
              the 15-minute city model. These neighborhoods serve as exemplars—real-world examples where residents can
              access most daily needs within a short walk or bike ride.
            </p>
            <p style={{ margin: '0 0 12px 0' }}>
              By learning from these optimal examples, the model identifies patterns in how services are distributed
              relative to population, demographics, and urban form. Rather than following rigid rules, the AI learns
              flexible, context-aware patterns that adapt to different neighborhood characteristics.
            </p>
          </section>

          <section style={{ marginBottom: 28 }}>
            <h3
              style={{
                fontSize: 18,
                fontWeight: 600,
                color: '#0f172a',
                marginBottom: 12,
                marginTop: 0,
              }}
            >
              Diverse and Representative Training Data
            </h3>
            <p style={{ margin: '0 0 12px 0' }}>
              The training neighborhoods represent a diverse range of urban contexts—different population densities,
              building types, street patterns, and demographic profiles. This diversity ensures the model can recognize
              successful 15-minute city patterns across varied urban environments, not just one specific neighborhood
              type.
            </p>
            <p style={{ margin: '0 0 12px 0' }}>
              Each training neighborhood was verified as meeting 15-minute city principles through established urban
              planning criteria, ensuring the model learns from genuinely successful implementations.
            </p>
          </section>

          <section style={{ marginBottom: 28 }}>
            <h3
              style={{
                fontSize: 18,
                fontWeight: 600,
                color: '#0f172a',
                marginBottom: 12,
                marginTop: 0,
              }}
            >
              How Predictions Are Generated
            </h3>
            <p style={{ margin: '0 0 12px 0' }}>
              When you click a location on the map, the system:
            </p>
            <ol style={{ margin: '0 0 12px 0', paddingLeft: 24 }}>
              <li style={{ marginBottom: 8 }}>
                Analyzes the spatial context—nearby services, population density, and walkability features
              </li>
              <li style={{ marginBottom: 8 }}>
                Considers demographic patterns—who lives nearby and what their likely needs are
              </li>
              <li style={{ marginBottom: 8 }}>
                Compares this location to patterns learned from successful 15-minute city neighborhoods
              </li>
              <li style={{ marginBottom: 8 }}>
                Generates a probability distribution over 8 service categories, indicating which interventions would
                most improve local accessibility
              </li>
            </ol>
            <p style={{ margin: '0 0 12px 0' }}>
              The model provides probability scores rather than binary yes/no answers, allowing planners to see the
              relative priority of different service types and make informed decisions based on local context.
            </p>
          </section>

          <section style={{ marginBottom: 20 }}>
            <h3
              style={{
                fontSize: 18,
                fontWeight: 600,
                color: '#0f172a',
                marginBottom: 12,
                marginTop: 0,
              }}
            >
              Transparency and Trust
            </h3>
            <p style={{ margin: '0 0 12px 0' }}>
              We believe in transparent, explainable AI for urban planning. This system:
            </p>
            <ul style={{ margin: '0 0 12px 0', paddingLeft: 24 }}>
              <li style={{ marginBottom: 8 }}>
                Shows probability distributions, not just single recommendations, so you can see the model's confidence
                levels
              </li>
              <li style={{ marginBottom: 8 }}>
                Provides impact estimates (people served, community benefits, environmental improvements) based on
                established urban planning metrics
              </li>
              <li style={{ marginBottom: 8 }}>
                Explains recommendations in planning language, connecting AI outputs to real-world urban design
                principles
              </li>
              <li style={{ marginBottom: 8 }}>
                Uses open data sources (OSM, Census) that you can verify independently
              </li>
            </ul>
            <p style={{ margin: '0 0 12px 0' }}>
              The model is a tool to support planning decisions, not replace professional judgment. Recommendations
              should be considered alongside local knowledge, community input, and feasibility constraints.
            </p>
          </section>
        </div>
      </div>
    </div>
  )
}
