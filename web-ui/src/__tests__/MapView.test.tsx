import { describe, expect, it, vi } from 'vitest'
import { render } from '@testing-library/react'
import { MapView } from '../components/MapView'

// Note: Leaflet relies on DOM APIs that jsdom only partially implements.
// This test focuses on verifying that the component mounts without crashing
// and wires the callback, rather than full interaction.

describe('MapView', () => {
  it('renders without crashing and accepts a callback', () => {
    const onLocationSelected = vi.fn()
    const { container } = render(
      <MapView
        onLocationSelected={onLocationSelected}
        selectedLocation={{ lat: 31.78, lng: 35.22 }}
        selectedCategoryId={null}
      />,
    )

    expect(container.querySelector('.leaflet-container')).toBeTruthy()
  })
})

