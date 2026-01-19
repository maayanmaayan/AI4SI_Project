# Feature: Urban Planner Web Interface (Mock 15-Minute City Explorer)

The following plan should be complete, but its important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils types and models. Import from the right files etc.

## Feature Description

Build a **desktop-only, local web interface** targeted at **urban planners**, allowing them to:

- Explore a map of a **new city** (e.g., Jerusalem) within a defined **coverage boundary**.
- See existing **OSM-based services** rendered as **cute, category-specific icons**.
- Click **any point inside the coverage boundary** to get a **mock prediction**: a probability distribution over the 8 NEXI service categories.
- View a **modal** displaying:
  - The 8 categories with **percentages** and icons.
  - A **selected category** with a clear explanation of why that service could be beneficial there (mocked, but written in urban-planning language).
- **Switch between categories** in the modal to update the explanation and the icon on the map.
- Optionally **export** the current recommendation view as a simple **PDF report**.
- Keep the interface visually **clean, professional, pastel, and friendly**, with rounded modals and clear iconography.

For MVP, the interface runs **locally**, uses a **mock model** (random predictions), and uses **real OSM service locations** only as map decoration/context.

## User Story

As an **urban planner**  
I want to **explore a city on a map, click locations, and see recommended service categories with clear explanations**  
So that I can **understand and communicate potential 15-minute city interventions in an intuitive, visual way**.

## Problem Statement

The current ML project produces model predictions and evaluation metrics, but **urban planners do not have an intuitive, visual interface** to:
- Explore new cities (beyond the training neighborhoods),
- Interactively select points of interest,
- See recommended service categories in a way that is **visually grounded in geography** and **explained in domain terms**.

Without such an interface, model outputs remain **technical artifacts** rather than a **planning tool**.

## Solution Statement

Create a **map-based web interface** that:

- Shows a **base map** of Jerusalem (or any configured city) with a clearly delineated **coverage area**.
- Renders **OSM-derived points of interest** for the 8 service categories as **distinct icons**, conveying what data was collected.
- Allows clicking **any coordinate inside the coverage area**, triggering:
  - A **modal** that shows a generated **mock probability distribution** over the 8 categories.
  - A **highlighted “suggested service”**, chosen as the category with highest probability.
  - A **planner-friendly explanation** of why this service might help (population, community, health, car use).
- Lets the user **change the active service category** via icons inside the modal, updating the map icon and explanation.
- Provides a **simple “Export as PDF”** action that captures the current recommendation (location + category + explanation + probabilities) into a one-page PDF for report inclusion.
- Runs **locally** (e.g. `npm run dev` or `python -m ...` with a minimal backend if needed), with no real-time dependency on the ML training pipeline (future integration possible via an inference API).

For now, **predictions are random but nicely presented**, making this a **mock/demo interface** that can later be wired to the actual Spatial Graph Transformer.

## Feature Metadata

**Feature Type**: New Capability  
**Estimated Complexity**: Medium–High (full new UI + light backend for PDF export; no model integration yet)  
**Primary Systems Affected**:
- New: Web UI (frontend app, likely separate from existing `src/`).
- Possible: A thin Python/Node backend for PDF generation and eventually model inference.

**Dependencies**:
- Frontend: React (with Vite or Next.js), map library (Leaflet or Mapbox GL JS).
- Geospatial tiles: OpenStreetMap/Mapbox/Carto tiles.
- PDF export: client-side JS library (e.g., jsPDF) or simple backend using Python (ReportLab) or Node (pdfkit).
- OSM data: pre-extracted GeoJSON or similar for Jerusalem services (prepared offline, not in this feature’s scope, but define expected format).

---

## CONTEXT REFERENCES

### Relevant Codebase Files IMPORTANT: YOU MUST READ THESE FILES BEFORE IMPLEMENTING!

Although this feature is **out-of-scope** of the core ML pipeline and likely lives as a **separate web app**, the implementer should understand the project’s **domain** and **conventions**:

- `CURSOR.md`  
  - Why: Project conventions, PIV loop process, architectural principles (15-minute city, service categories, distance-based logic, star graphs).

- `PRD.md`  
  - Why: Deep domain context (service categories, 15-minute city principles, exemplar-based learning, distance-based supervision). Use this language in the UI copy and explanations.

- `docs/graph-transformer-architecture.md`  
  - Why: Understand the model’s conceptual behavior to inform how we **explain** recommendations in planning terms later (even in mock).

- `docs/plotting-guide.md`  
  - Why: Shows existing visualization tone/styles (colors, clarity) that we can loosely mirror in the web UI.

- `src/training/model.py`  
  - Why: If later we wire real predictions, we’ll expose this via an API. For now, just understand the model’s outputs (probability vector over 8 classes).

- `models/config.yaml`  
  - Why: Includes the 8 service categories, and relevant hyperparameters (temperature, etc.) that may influence eventual **real** predictions.

None of these files will be **modified** by this feature in mock mode, but they steer **copywriting** and **UX language**.

### New Files to Create

Assuming we create a separate web frontend in a sibling directory (to keep the Python training repo clean):

- `web-ui/package.json` (or `web-ui/pyproject.toml` if Python-based)  
  - Web app configuration + dependencies.

- `web-ui/src/App.tsx` (or `App.jsx`)  
  - Main React application shell.

- `web-ui/src/components/MapView.tsx`  
  - Map display, coverage polygon, click handling, service icons.

- `web-ui/src/components/PredictionModal.tsx`  
  - Modal with 8 categories, probabilities, explanations, export button.

- `web-ui/src/components/CategoryIcon.tsx`  
  - Reusable icon component for the 8 categories.

- `web-ui/src/hooks/useMockPrediction.ts`  
  - Hook that, given a coordinate, returns a random-but-normalized probability vector and a suggested category.

- `web-ui/src/styles/theme.ts` (or CSS/SCSS files)  
  - Pastel color palette, typography, border radius and shadows.

- `web-ui/public/data/jerusalem_services.geojson`  
  - Static file with OSM service locations for the 8 categories (dummy or real, but fixed), used purely for visualization.

- `web-ui/src/utils/pdfExport.ts`  
  - Utility for constructing and triggering a simple PDF export of current recommendation.

- `web-ui/tests/` (or `__tests__/`)  
  - Unit tests for UI logic (mock prediction, modal behavior) and at least one integration-style test (click → modal → export).

If you prefer a pure-Python stack (e.g. Flask + Jinja + Leaflet), analogous files would be:

- `web-ui/app.py` (Flask app)  
- `web-ui/templates/index.html`  
- `web-ui/static/js/map.js`  
- `web-ui/static/js/mock_prediction.js`  
- `web-ui/static/js/pdf_export.js`  
- `web-ui/static/css/styles.css`  
- `web-ui/tests/test_routes.py`  

### Relevant Documentation YOU SHOULD READ THESE BEFORE IMPLEMENTING!

Even though we’re not calling external docs from within this repo, the execution agent should be prepared to consult:

- [Leaflet Documentation](https://leafletjs.com/reference.html)  
  - Specific section: Map, TileLayer, Marker, Circle/Polygon, and click events.  
  - Why: We need robust map handling (zoom/pan, click, bounding polygon).

- [React Leaflet Guide](https://react-leaflet.js.org/docs/start-installation/)  
  - Specific section: Handling map events, rendering markers, and custom icons.  
  - Why: Establishes best practices for React-based map components.

- [jsPDF Documentation](https://artskydj.github.io/jsPDF/docs/jsPDF.html)  
  - Specific section: Adding text, basic layout, saving files.  
  - Why: For a simple client-side PDF export in the browser.

Alternative: If Python backend chosen:

- [ReportLab User Guide](https://www.reportlab.com/docs/reportlab-userguide.pdf)  
  - Specific section: Simple PDF pages and writing text.  
  - Why: Backend PDF generation endpoint.

### Patterns to Follow

This feature is a **new stack**, but we should **carry over project principles**:

**Naming Conventions:**

- Use **clear, domain-driven names**:
  - `ServiceCategory`, `PredictionResult`, `SelectedLocation`.
  - Components: `MapView`, `PredictionModal`, `CategoryList`, `ExportButton`.
- Keep camelCase for JS/TS, snake_case for Python if backend is used.

**Error Handling:**

- Use **graceful fallbacks**:
  - If user clicks outside bounds: show a small tooltip or toast, not a noisy error.
  - If PDF generation fails: show non-blocking message “Couldn’t export PDF. Please try again.”

**Logging Pattern:**

- In frontend, rely on browser DevTools; keep console logs minimal in production builds.
- If a backend is used, follow Python `logging` pattern similar to `src/utils/logging.py` (info-level logs for key events like PDF export, not for every click).

**Other Relevant Patterns:**

- **Config-driven**:
  - Extract city bounds and coverage polygon into a config file (e.g., `config/mapConfig.ts`), rather than hardcoding coordinates in components.
- **Separation of responsibilities**:
  - `MapView` handles display and events; `PredictionModal` handles visualization of predictions.
  - `useMockPrediction` encapsulates random generation logic and normalization, making it easily swappable for a real API in the future.
- **Future API-ready design**:
  - Design `useMockPrediction` with an interface similar to what a real model API would return: `{ probabilities: number[], categories: ServiceCategory[], selectedCategory: ServiceCategory }`.

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation

Establish the **project structure**, dependencies, and core domain concepts.

**Tasks:**

- **CREATE** `web-ui` project (React + Vite or similar) separate from the Python ML code.  
  - Choose TypeScript if comfortable to improve robustness.  
  - Install dependencies: React, React DOM, React-Leaflet (or Mapbox), jsPDF (or equivalent), a design system or simple CSS-in-JS approach.

- **DEFINE** domain models and constants:  
  - `ServiceCategory` enum / constant list with:
    - ID, label, icon name, color.  
  - `PredictionResult` type:
    - `location: { lat, lng }`, `probabilities: number[8]`, `selectedCategoryId: string`.

- **CONFIGURE** map settings:  
  - `mapConfig` containing:
    - City center coordinates (Jerusalem).  
    - Zoom levels.  
    - Coverage polygon or bounding box coordinates.

- **ADD** static data for services:  
  - Place `jerusalem_services.geojson` (or a simple JSON) into `public/` or `src/data/`.  
  - Define expected schema: `{ type: "FeatureCollection", features: [ { geometry: Point, properties: { categoryId } } ] }`.

### Phase 2: Core Implementation

Implement the **map UI**, **prediction logic**, and **modal**.

**Tasks:**

- **CREATE** `MapView`:  
  - Render base map centered on Jerusalem.  
  - Draw coverage polygon / rectangle with subtle color overlay.  
  - Load and render service markers for 8 categories with distinct icons.  
  - Handle click events:
    - If click inside coverage polygon → call `onLocationSelected({ lat, lng })` prop.  
    - If outside → show tooltip/notification.

- **CREATE** `useMockPrediction`:  
  - Accept `{ lat, lng }`.  
  - Generate 8 random positive values, normalize to sum to 1.  
  - Associate them with the 8 service categories.  
  - Determine `selectedCategoryId` as argmax.  
  - Return structured `PredictionResult`.

- **CREATE** `PredictionModal`:  
  - Props: `visible`, `onClose`, `predictionResult`, `onCategoryChange`.  
  - Layout:
    - Header: location info (coordinates, optional address).  
    - Category grid:
      - Each category shows icon + name + percentage.  
      - Clickable to select that category.  
    - Explanation panel:
      - For selected category, show templated explanation:
        - “Estimated X residents within 15-minute walk…”  
        - “Potential benefits for local community, health, and car use reduction…”  
      - Use consistent voice aligned with `PRD.md`.

- **LINK** modal to map:  
  - In `App`:
    - Maintain `selectedLocation` + `predictionResult` state.  
    - On map click inside bounds:
      - Call `useMockPrediction` (or a function using it).  
      - Update state and show modal.  
    - When category changes in modal:
      - Update `selectedCategoryId` in `predictionResult`.  
      - Optionally update a highlighted icon at the clicked point on the map.

### Phase 3: Integration

Connect all pieces and ensure **future extensibility** for real model integration.

**Tasks:**

- **INTEGRATE** PDF export:  
  - **CREATE** `pdfExport.ts`:
    - Function `exportRecommendationToPdf(predictionResult, selectedCategory, cityName)` that:
      - Builds a simple A4 PDF with:
        - Title (city + coordinates).  
        - Table or list of 8 categories with percentages.  
        - Highlighted section for selected category with explanation text.  
      - Triggers browser download.  
  - Add “Export as PDF” button in `PredictionModal`:
    - On click, call `exportRecommendationToPdf(...)`.  
    - Show small success or error message.

- **ABSTRACT** prediction source:  
  - Wrap `useMockPrediction` behind an interface (e.g., `usePredictionSource`).  
  - Clearly document how to later replace random predictions with real API calls (e.g., `fetch('/api/predict', { lat, lng })`).

- **STYLE** the UI:  
  - Implement a small theme:
    - Colors: soft pastels, high-contrast text.  
    - Radii: 8–12px for modals and cards.  
    - Typography: simple sans-serif, using a limited scale of sizes.  
  - Ensure:
    - Clear clickable affordances.  
    - No unnecessary legends (icons should be self-explanatory with labels).

### Phase 4: Testing & Validation

Design a **small but meaningful** test suite and manual checks, focused on **behavior** and **UX sanity**.

**Tasks:**

- **Unit Tests**:
  - Test `useMockPrediction`:
    - Returns 8 probabilities.  
    - Probabilities sum to ~1 (within small epsilon).  
    - Always selects a valid category ID.  
  - Test `pdfExport`:
    - Runs without throwing given a valid `PredictionResult`.  
    - (Optional) uses snapshot or spy to ensure `save` is called.

- **Component Tests**:
  - Test `PredictionModal`:
    - Renders 8 categories.  
    - Clicking different categories updates selected state and explanation content.  
  - Test `MapView` (where feasible):
    - Clicking inside bounds triggers selection callback.  
    - Clicking outside bounds does not.

- **Manual Validation**:
  - Ensure:
    - Map zoom/pan and search (if search implemented) feel smooth.  
    - Clicking around coverage area yields immediate modal with diverse, “believable enough” random distributions.  
    - PDF export creates readable, not ugly reports.

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### Task Format Guidelines

Use information-dense keywords for clarity:

- **CREATE**: New files or components  
- **UPDATE**: Modify existing files  
- **ADD**: Insert new functionality into existing code  
- **REMOVE**: Delete deprecated code  
- **REFACTOR**: Restructure without changing behavior  
- **MIRROR**: Copy pattern from elsewhere in codebase  

### CREATE web-ui project

- **IMPLEMENT**: Initialize a separate `web-ui` project using React (TypeScript) + Vite (or CRA/Next.js) in the repo root.  
- **PATTERN**: Keep this separate from existing `src/` Python package; treat as a sibling app (`web-ui`).  
- **IMPORTS**: React, React DOM, React-Leaflet, Leaflet, jsPDF (or equivalent), testing library (Jest/RTL or Vitest).  
- **GOTCHA**: Ensure `web-ui` dependencies don’t interfere with Python environment; keep Node modules scoped to `web-ui`.  
- **VALIDATE**: `cd web-ui && npm install && npm run dev` (check default app loads).

### CREATE Domain constants & types

- **UPDATE**: `web-ui/src/` with `constants/serviceCategories.ts` and `types/prediction.ts`.  
- **IMPLEMENT**:
  - `ServiceCategory` list with 8 entries (IDs matching NEXI categories).  
  - `PredictionResult` type.  
- **PATTERN**: Mirror service category naming and ordering as in `PRD.md`.  
- **IMPORTS**: None beyond TS types.  
- **GOTCHA**: Keep category IDs stable; they must be used across icons, GeoJSON, and predictions.  
- **VALIDATE**: Run TypeScript build / linter (e.g., `npm run build`).

### CREATE mapConfig

- **CREATE**: `web-ui/src/config/mapConfig.ts`.  
- **IMPLEMENT**:
  - City center: Jerusalem coordinates.  
  - Zoom, minZoom, maxZoom.  
  - Coverage polygon coordinates (e.g. simple rectangle or more detailed polygon).  
- **PATTERN**: Config pattern similar in spirit to `models/config.yaml`: centralize parameters.  
- **IMPORTS**: Types for coordinates if desired.  
- **GOTCHA**: Ensure polygon covers enough area for meaningful clicking.  
- **VALIDATE**: Temporary log or unit test to confirm polygon is valid (non-empty).

### CREATE MapView component

- **CREATE**: `web-ui/src/components/MapView.tsx`.  
- **IMPLEMENT**:
  - Render Leaflet map with tile layer.  
  - Draw coverage polygon with semi-transparent fill.  
  - Load `jerusalem_services.geojson` and render markers with category-specific icons.  
  - On map click:
    - Determine if inside coverage polygon (Leaflet or turf.js point-in-polygon).  
    - If inside: call `props.onLocationSelected({ lat, lng })`.  
    - Else: show tooltip/notification (e.g. small div or toast).  
- **PATTERN**: Clear separation of presentation and side effects; onLocationSelected prop drives state in parent.  
- **IMPORTS**: `react-leaflet` components, `mapConfig`, service categories, GeoJSON loader.  
- **GOTCHA**: Make sure to set Leaflet icon path correctly in React setups.  
- **VALIDATE**: `npm run dev`, manually click map, log selected locations.

### CREATE useMockPrediction hook

- **CREATE**: `web-ui/src/hooks/useMockPrediction.ts`.  
- **IMPLEMENT**:
  - Export a function `getMockPrediction(location)`:
    - Generate 8 random floats.  
    - Normalize them to sum to 1.  
    - Associate each with a `ServiceCategory`.  
    - Choose the highest as `selectedCategoryId`.  
    - Return `PredictionResult`.  
- **PATTERN**: Keep this pure and deterministic for a given random seed if you wish; or simple random each time.  
- **IMPORTS**: `ServiceCategory` list, `PredictionResult` type.  
- **GOTCHA**: Handle edge cases: sum of zero (unlikely but guard).  
- **VALIDATE**: Unit test: `npm test` for hook logic.

### CREATE PredictionModal component

- **CREATE**: `web-ui/src/components/PredictionModal.tsx`.  
- **IMPLEMENT**:
  - Props: `visible`, `predictionResult`, `onClose`, `onCategoryChange`, `cityName`.  
  - Layout:
    - Header with city + coordinates.  
    - Grid/list of 8 categories with icons and percentages.  
    - Explanation section with templated text for selected category.  
    - “Export as PDF” button.  
- **PATTERN**: Use simple, accessible modal pattern (focus trap, ESC to close, overlay click to close).  
- **IMPORTS**: `ServiceCategory`, theme/styles, `exportRecommendationToPdf` once implemented.  
- **GOTCHA**: Avoid modals appearing off-screen; desktop-only but still use reasonable responsive width.  
- **VALIDATE**: Storybook or component test verifying categories and selection.

### LINK MapView and PredictionModal in App

- **UPDATE**: `web-ui/src/App.tsx`.  
- **IMPLEMENT**:
  - Hold `selectedLocation` and `predictionResult` in state.  
  - Pass `onLocationSelected` to `MapView`:
    - When called, compute `predictionResult = getMockPrediction(location)` and open modal.  
  - Pass `predictionResult` and handlers to `PredictionModal`.  
- **PATTERN**: Keep `App` as orchestrator/container.  
- **IMPORTS**: `MapView`, `PredictionModal`, `getMockPrediction`.  
- **GOTCHA**: Handle quickly repeated clicks gracefully (overwrite old prediction).  
- **VALIDATE**: Manual: click map → modal appears with random predictions.

### CREATE pdfExport utility

- **CREATE**: `web-ui/src/utils/pdfExport.ts`.  
- **IMPLEMENT**:
  - Function `exportRecommendationToPdf(predictionResult, selectedCategory, cityName)` using jsPDF:
    - Title: “15-Minute City Recommendation – {cityName}”.  
    - Subheading: coordinates.  
    - Section: table/list of categories with percentages.  
    - Highlighted section: selected category + explanation text.  
- **PATTERN**: Encapsulate PDF logic so modal just calls a single function.  
- **IMPORTS**: jsPDF, `ServiceCategory`, `PredictionResult`.  
- **GOTCHA**: Keep layout simple to avoid spending huge time on design; this is for internal/planner use.  
- **VALIDATE**: Manual: click export, open PDF, verify content.

### WIRE export button in PredictionModal

- **UPDATE**: `PredictionModal` to call `exportRecommendationToPdf`.  
- **IMPLEMENT**:
  - On button click, call export with current state.  
  - Optionally show simple “Exported!” toast.  
- **PATTERN**: Keep UI responsive; exporting should not block main thread for long (PDF is small).  
- **IMPORTS**: `exportRecommendationToPdf`.  
- **GOTCHA**: Guard against missing `predictionResult` or selected category.  
- **VALIDATE**: Manual + small unit test that export call is invoked.

### ADD basic tests

- **CREATE**: `web-ui/src/__tests__/useMockPrediction.test.ts`.  
- **IMPLEMENT**: Test normalization, length = 8, valid category IDs.

- **CREATE**: `web-ui/src/__tests__/PredictionModal.test.tsx`.  
- **IMPLEMENT**: 
  - Renders category list.  
  - Clicking different category changes explanation text.

- **CREATE**: `web-ui/src/__tests__/MapView.test.tsx` (if feasible with jsdom + mock leaflet).  
- **IMPLEMENT**: Ensure `onLocationSelected` is called on simulated click inside bounds.

- **VALIDATE**: Run `npm test`.

---

## TESTING STRATEGY

### Unit Tests

- **Pure logic**:
  - `useMockPrediction`:
    - Probability vector properties.  
    - Determinism with fixed random seed (optional).  
  - `pdfExport`:
    - Does not throw and calls jsPDF APIs as expected (spy).

- **Component behavior**:
  - `PredictionModal`:
    - Displays 8 categories.  
    - Updates selected category and explanation correctly on click.

### Integration Tests

- App-level test (if using Cypress or Playwright later, optional for now):
  - Load app.  
  - Simulate a click inside coverage area.  
  - Verify modal appears with non-zero probabilities.  
  - Click export and confirm file is downloaded (or stubbed).

### Edge Cases

- Click just outside coverage boundary → no modal; show subtle “outside coverage” message.  
- Very high zoom levels or map panning away from city center.  
- Repeated quick clicks before closing modal.  
- PDF export when some fields are missing (should fail gracefully, but this should not happen in normal flow).

---

## VALIDATION COMMANDS

Adjust based on chosen stack; example assumes `web-ui` with npm + Vitest/Jest:

### Level 1: Syntax & Style

```bash
cd web-ui
npm run lint
npm run build
```

### Level 2: Unit Tests

```bash
cd web-ui
npm test
```

### Level 3: Integration Tests (Optional)

If Cypress/Playwright is added:

```bash
cd web-ui
npm run test:e2e
```

### Level 4: Manual Validation

- Start dev server: `cd web-ui && npm run dev`.  
- In browser:
  - Confirm:
    - Map loads Jerusalem.  
    - Coverage area visible.  
    - Service icons visible.  
  - Click multiple locations inside bounds:
    - Modal appears.  
    - Probabilities look sensible (sum ~100%).  
    - Explanations read clearly for planners.  
  - Export PDF:
    - File downloads.  
    - Contents are readable and correct.

### Level 5: Additional Validation (Optional)

- Present the mock interface to a **non-technical urban planner**:
  - Ask if:
    - The UI is intuitive.  
    - Explanations feel meaningful and not too technical.  
    - Visual design feels professional and friendly.

---

## ACCEPTANCE CRITERIA

- [ ] User can **zoom, pan, and search** within the Jerusalem map (if search included).  
- [ ] Coverage area is clearly visible; clicks **inside** open modal, clicks **outside** are rejected gracefully.  
- [ ] Modal shows:
  - [ ] 8 service categories with icons and percentages.  
  - [ ] One recommended category selected by default.  
  - [ ] Explanations written in **urban planning language** (not ML jargon).  
- [ ] User can **switch categories** in the modal and see updated explanations and map icon.  
- [ ] Predictions are generated **on-the-fly** per click using random probabilities (mock mode).  
- [ ] “Export as PDF” produces a simple, readable PDF capturing location, probabilities, and chosen category explanation.  
- [ ] All linting and build commands pass.  
- [ ] Tests for prediction logic and modal behavior pass.  
- [ ] No changes are made to the core ML training codepath (only optional future integration hooks).

---

## COMPLETION CHECKLIST

- [ ] `web-ui` project created and runs locally.  
- [ ] Map displays Jerusalem with coverage area and service icons.  
- [ ] Click-to-prediction flow works with random probabilities.  
- [ ] Modal UX is smooth and visually aligned with desired style (pastels, rounded corners).  
- [ ] PDF export works reliably for multiple locations and categories.  
- [ ] Unit tests and any integration tests pass.  
- [ ] Manual UX review with at least one non-developer confirms clarity.  
- [ ] Plan for future model integration documented (API shape and hook abstraction).

---

## NOTES

- This is explicitly a **mock/demo feature**: it should **not block or interfere** with the core ML training experiments.  
- Keep the architecture **API-ready**: design prediction logic as if it were calling a `/predict` endpoint, so later wiring to the real Spatial Graph Transformer is straightforward.  
- Prioritize **clarity and narrative** over technical detail in explanations; the goal is to help urban planners reason about interventions, not to expose internals of the model.

