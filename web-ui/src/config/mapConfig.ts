import type { LatLngExpression, LatLngTuple } from 'leaflet'

export interface MapConfig {
  cityName: string
  center: LatLngExpression
  zoom: number
  minZoom: number
  maxZoom: number
  coveragePolygon: LatLngTuple[]
}

export const mapConfig: MapConfig = {
  cityName: 'Jerusalem',
  center: [31.778, 35.235] as LatLngTuple,
  zoom: 12,
  minZoom: 10,
  maxZoom: 18,
  // Simple rectangular coverage around central Jerusalem (not exact, but sufficient for mock)
  coveragePolygon: [
    [31.82, 35.18],
    [31.82, 35.29],
    [31.74, 35.29],
    [31.74, 35.18],
  ],
}

