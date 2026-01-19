import '@testing-library/jest-dom'

// Ensure DOM globals are available for components and Leaflet in tests.
if (typeof window === 'undefined') {
  // @ts-expect-error – we are defining window/document for the test environment.
  global.window = {} as unknown as Window & typeof globalThis
}

