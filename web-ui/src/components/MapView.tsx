import { useEffect, useMemo, useState } from 'react'
import { MapContainer, TileLayer, Polygon, Marker, useMapEvents } from 'react-leaflet'
import L from 'leaflet'
import type { LatLngExpression, LatLngTuple } from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { mapConfig } from '../config/mapConfig'
import { SERVICE_CATEGORIES, type ServiceCategoryId } from '../constants/serviceCategories'

interface MapViewProps {
  onLocationSelected: (location: { lat: number; lng: number }) => void
}

interface ServiceFeature {
  geometry: {
    type: 'Point'
    coordinates: [number, number]
  }
  properties: {
    categoryId: ServiceCategoryId
  }
}

interface ServiceGeoJson {
  type: 'FeatureCollection'
  features: ServiceFeature[]
}

// Fix default marker icon paths for Leaflet in bundlers
const defaultIcon = new L.Icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
})

L.Marker.prototype.options.icon = defaultIcon

function pointInPolygon(point: LatLngTuple, polygon: LatLngTuple[]): boolean {
  // Simple ray casting algorithm for point-in-polygon
  const [x, y] = point
  let inside = false

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i]
    const [xj, yj] = polygon[j]

    const intersects =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + Number.EPSILON) + xi
    if (intersects) inside = !inside
  }

  return inside
}

function ClickHandler({ onInsideClick }: { onInsideClick: (latlng: LatLngTuple) => void }) {
  const [message, setMessage] = useState<string | null>(null)

  useMapEvents({
    click(e) {
      const latlng: LatLngTuple = [e.latlng.lat, e.latlng.lng]
      if (pointInPolygon(latlng, mapConfig.coveragePolygon)) {
        setMessage(null)
        onInsideClick(latlng)
      } else {
        setMessage('This location is outside the current coverage area.')
      }
    },
  })

  if (!message) return null

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 16,
        left: '50%',
        transform: 'translateX(-50%)',
        background: 'rgba(0,0,0,0.7)',
        color: 'white',
        padding: '8px 12px',
        borderRadius: 8,
        fontSize: 12,
      }}
    >
      {message}
    </div>
  )
}

function categoryColor(categoryId: ServiceCategoryId): string {
  return SERVICE_CATEGORIES.find((c) => c.id === categoryId)?.color ?? '#666666'
}

export function MapView({ onLocationSelected }: MapViewProps) {
  const [services, setServices] = useState<ServiceGeoJson | null>(null)

  useEffect(() => {
    fetch('/data/jerusalem_services.geojson')
      .then((res) => res.json())
      .then((data: ServiceGeoJson) => setServices(data))
      .catch(() => {
        // In mock mode, failure to load services should not break the app
        setServices(null)
      })
  }, [])

  const coveragePolygonLatLngs: LatLngExpression[] = useMemo(
    () => mapConfig.coveragePolygon.map((p) => [p[0], p[1]] as LatLngTuple),
    [],
  )

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <MapContainer
        center={mapConfig.center}
        zoom={mapConfig.zoom}
        minZoom={mapConfig.minZoom}
        maxZoom={mapConfig.maxZoom}
        style={{ width: '100%', height: '100%', borderRadius: 16, overflow: 'hidden' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <Polygon
          positions={coveragePolygonLatLngs}
          pathOptions={{
            color: '#7BA6FF',
            weight: 2,
            fillColor: '#7BA6FF',
            fillOpacity: 0.1,
          }}
        />

        {services?.features.map((feature, idx) => {
          const [lng, lat] = feature.geometry.coordinates
          const color = categoryColor(feature.properties.categoryId)

          const icon = new L.DivIcon({
            className: 'service-marker',
            html: `<div style="
              background:${color};
              width:18px;
              height:18px;
              border-radius:50%;
              border:2px solid white;
              box-shadow:0 0 4px rgba(0,0,0,0.3);
            "></div>`,
          })

          return <Marker key={idx} position={[lat, lng]} icon={icon} />
        })}

        <ClickHandler
          onInsideClick={(latlng) =>
            onLocationSelected({ lat: latlng[0], lng: latlng[1] })
          }
        />
      </MapContainer>
    </div>
  )
}

