# Paris Neighborhoods Boundaries Validation Report

## Executive Summary

This report validates the boundaries defined in `paris_neighborhoods.geojson` against online sources and official Paris administrative data. **Key findings:**

1. **All boundaries are simple rectangular polygons** - Real neighborhood boundaries in Paris are irregular polygons following streets, rivers, and administrative lines
2. **Several coordinate ranges appear inaccurate** - Some neighborhoods have coordinates that don't match their actual locations
3. **Boundary extents may be too narrow** - Some neighborhoods appear to cover smaller areas than their actual boundaries

## Detailed Validation by Neighborhood

### ✅ Compliant Neighborhoods

#### 1. Paris Rive Gauche (13e)
- **GeoJSON Coordinates**: [2.354, 48.828] to [2.364, 48.838]
- **Verified Location**: 13th arrondissement, Left Bank of Seine
- **Verified Coordinates**: Approximately 48.8262°N to 48.8367°N, 2.3533°E to 2.3834°E
- **Status**: ⚠️ **PARTIALLY ACCURATE**
  - Latitude range is approximately correct
  - **Longitude range is too narrow**: GeoJSON shows 2.354-2.364, but actual extends to 2.3834°E
  - Boundary should extend further east

#### 2. Clichy-Batignolles (17e)
- **GeoJSON Coordinates**: [2.307, 48.88] to [2.317, 48.89]
- **Verified Location**: 17th arrondissement, northeast area
- **Verified Coordinates**: 17th arrondissement center ~48.8836°N, 2.3217°E
- **Status**: ⚠️ **NEEDS VERIFICATION**
  - Coordinates appear to be in the general area
  - Longitude (2.307-2.317) is slightly west of arrondissement center (2.3217)
  - Should verify exact boundaries with Clichy-Batignolles development zone (54 hectares)

#### 3. Beaugrenelle / Front de Seine (15e)
- **GeoJSON Coordinates**: [2.282, 48.835] to [2.292, 48.845]
- **Verified Location**: 15th arrondissement, along Seine River
- **Verified Coordinates**: Center ~48.8506°N, 2.2846°E
- **Status**: ⚠️ **LATITUDE TOO LOW**
  - Longitude (2.282-2.292) is approximately correct (center at 2.2846)
  - **Latitude range (48.835-48.845) is lower than actual center (48.8506)**
  - Boundary should extend further north

#### 4. Saint-Vincent-de-Paul (14e)
- **GeoJSON Coordinates**: [2.332, 48.828] to [2.347, 48.838]
- **Verified Location**: 14th arrondissement, Avenue Denfert-Rochereau area
- **Verified Coordinates**: Hospital site ~48.8378°N, 2.3336°E
- **Status**: ✅ **APPROXIMATELY CORRECT**
  - Coordinates align with the general area
  - Latitude range includes the hospital location

#### 5. Maine–Montparnasse (6e/14e/15e)
- **GeoJSON Coordinates**: [2.306, 48.835] to [2.332, 48.852]
- **Verified Location**: Straddles 6th, 14th, and 15th arrondissements
- **Verified Coordinates**: Center ~48.8412°N, 2.3192°E
- **Status**: ✅ **APPROXIMATELY CORRECT**
  - Coordinates encompass the verified center point
  - Boundary spans appropriate area across three arrondissements

#### 6. Bartholomé–Brancion (15e)
- **GeoJSON Coordinates**: [2.286, 48.825] to [2.305, 48.836]
- **Verified Location**: 15th arrondissement
- **Status**: ⚠️ **NEEDS VERIFICATION**
  - No specific coordinates found online
  - Should verify with official PLU bioclimatique documents

#### 7. Paul Bourget (13e)
- **GeoJSON Coordinates**: [2.343, 48.814] to [2.364, 48.824]
- **Verified Location**: 13th arrondissement
- **Status**: ⚠️ **NEEDS VERIFICATION**
  - No specific coordinates found online
  - Should verify with official PLU bioclimatique documents

#### 8. Olympiades / Villa d'Este / Place de Vénétie (13e)
- **GeoJSON Coordinates**: [2.357, 48.82] to [2.372, 48.833]
- **Verified Location**: 13th arrondissement
- **Verified Coordinates**: 
  - Villa d'Este: ~48.8223°N, 2.3664°E
  - Olympiades: ~48.8237°N, 2.3531°E
- **Status**: ✅ **APPROXIMATELY CORRECT**
  - Coordinates encompass both Villa d'Este and Olympiades locations
  - Boundary appears to cover the area appropriately

#### 9. Bédier–Oudiné (13e)
- **GeoJSON Coordinates**: [2.368, 48.81] to [2.389, 48.822]
- **Verified Location**: 13th arrondissement
- **Status**: ⚠️ **NEEDS VERIFICATION**
  - No specific coordinates found online
  - Should verify with official PLU bioclimatique documents

### ❌ Non-Compliant Neighborhoods

#### 10. Montmartre (18e)
- **GeoJSON Coordinates**: [2.336, 48.876] to [2.346, 48.886]
- **Verified Location**: 18th arrondissement
- **Verified Coordinates**: Center ~48.8870°N, 2.3388°E
- **Status**: ⚠️ **BOUNDARY ISSUES**
  - Center coordinates are within the range
  - **Boundary is rectangular, but Montmartre has irregular boundaries** defined by specific streets
  - Should follow actual street boundaries (Rue Caulaincourt, Rue Custine, Rue de Clignancourt, Boulevard de Clichy, Boulevard de Rochechouart)

#### 11. Belleville (10e/11e/19e/20e)
- **GeoJSON Coordinates**: [2.366, 48.864] to [2.376, 48.874]
- **Verified Location**: Spans 10th, 11th, 19th, and 20th arrondissements
- **Verified Coordinates**: Center ~48.8710°N, 2.3845°E
- **Status**: ❌ **SIGNIFICANTLY INACCURATE**
  - **Longitude is completely wrong**: GeoJSON shows 2.366-2.376, but actual center is 2.3845°E
  - **Boundary is approximately 0.8-1.0 km too far west**
  - Latitude range is approximately correct
  - **CRITICAL: This boundary needs correction**

#### 12. Bastille / Oberkampf (11e)
- **GeoJSON Coordinates**: [2.363, 48.853] to [2.373, 48.863]
- **Verified Location**: 11th arrondissement
- **Verified Coordinates**: 
  - Rue Oberkampf center: ~48.8654°N, 2.3763°E
  - 11th arrondissement center: ~48.8586°N, 2.3794°E
- **Status**: ❌ **INACCURATE**
  - **Longitude is too far west**: GeoJSON shows 2.363-2.373, but actual is ~2.3763-2.3794°E
  - **Boundary is approximately 0.3-0.4 km too far west**
  - Latitude may also be slightly low

#### 13. La Défense
- **GeoJSON Coordinates**: [2.24, 48.878] to [2.25, 48.888]
- **Verified Location**: West of Paris, in Puteaux/Courbevoie/Nanterre
- **Verified Coordinates**: Center ~48.8897°N, 2.2418°E
- **Status**: ✅ **APPROXIMATELY CORRECT**
  - Coordinates align with verified location
  - Note: La Défense is technically outside Paris city limits

#### 14. Buttes-Chaumont / 19th Arr. (19e)
- **GeoJSON Coordinates**: [2.393, 48.88] to [2.403, 48.89]
- **Verified Location**: 19th arrondissement
- **Status**: ⚠️ **NEEDS VERIFICATION**
  - No specific coordinates found online
  - Should verify with official arrondissement boundaries

## Critical Issues Identified

### 1. **All Boundaries Are Rectangular**
   - Real Paris neighborhood boundaries follow streets, rivers, and administrative lines
   - They are irregular polygons, not rectangles
   - **Recommendation**: Obtain actual polygon boundaries from:
     - Paris Open Data portal (opendata.paris.fr)
     - Apur (Atelier Parisien d'Urbanisme)
     - OpenStreetMap administrative boundaries

### 2. **Significant Coordinate Errors**
   - **Belleville**: Longitude off by ~0.8-1.0 km (too far west)
   - **Bastille/Oberkampf**: Longitude off by ~0.3-0.4 km (too far west)
   - **Paris Rive Gauche**: Longitude range too narrow (missing eastern extent)
   - **Beaugrenelle**: Latitude range too low (missing northern extent)

### 3. **Missing Verification Data**
   - Several neighborhoods (Bartholomé–Brancion, Paul Bourget, Bédier–Oudiné, Buttes-Chaumont) need verification against official PLU bioclimatique documents

## Recommendations

1. **Obtain Official Boundaries**: Download actual polygon boundaries from Paris Open Data or Apur
2. **Correct Coordinate Errors**: Fix Belleville and Bastille/Oberkampf boundaries immediately
3. **Expand Boundaries**: Adjust Paris Rive Gauche and Beaugrenelle to match actual extents
4. **Replace Rectangles**: Convert all rectangular boundaries to actual polygon shapes
5. **Cross-Reference with PLU**: Verify all OAP (Orientation d'Aménagement et de Programmation) boundaries against official PLU bioclimatique documents

## Sources Consulted

- Paris Open Data portal (opendata.paris.fr)
- Wikipedia articles on Paris neighborhoods and arrondissements
- PLU bioclimatique documentation
- Geodatos.net coordinate databases
- Various Paris urban planning documents

---

**Report Generated**: Based on online validation searches
**Next Steps**: Obtain official GeoJSON boundaries from Paris administrative sources

---

## UPDATE: Official Boundaries Obtained (2024-12-19)

### Successfully Updated with Official Polygons

**8 out of 14 neighborhoods** have been updated with official polygon boundaries from the **PLU bioclimatique OAP** dataset (Paris Open Data):

1. ✅ **Paris Rive Gauche** - 93 points (official polygon)
2. ✅ **Beaugrenelle / 15th Arr.** - 33 points (official polygon)
3. ✅ **Saint-Vincent-de-Paul (14e)** - 17 points (official polygon)
4. ✅ **Maine–Montparnasse (6e/14e/15e)** - 101 points (official polygon)
5. ✅ **Bartholomé–Brancion (15e)** - 33 points (official polygon)
6. ✅ **Paul Bourget (13e)** - 30 points (official polygon)
7. ✅ **Olympiades / Villa d'Este / Place de Vénétie (13e)** - 11 points (official polygon)
8. ✅ **Bédier–Oudiné (13e)** - 80 points (official polygon)

**Data Source**: PLU bioclimatique OAP boundaries downloaded from Paris Open Data portal (opendata.paris.fr)
- Dataset: `plub_oap_perim`
- Format: GeoJSON with actual polygon coordinates
- License: Open Database Licence (ODbL)

### Still Using Rectangular Approximations

**6 neighborhoods** still need official polygon boundaries:

1. ⚠️ **Clichy-Batignolles** - Needs verification of OSM relation or arrondissement boundaries
2. ⚠️ **Montmartre** - Not an official administrative quartier; may need custom boundary definition
3. ⚠️ **Belleville** - Spans multiple arrondissements; needs quartier-level boundaries
4. ⚠️ **Bastille / Oberkampf** - Needs 11th arrondissement quartier boundaries
5. ⚠️ **La Défense** - Located outside Paris city limits; needs separate data source
6. ⚠️ **Buttes-Chaumont / 19th Arr.** - Needs 19th arrondissement quartier boundaries

### Recommendations for Remaining Neighborhoods

1. **For Clichy-Batignolles**: The OSM relation ID 152279 may be incorrect. Consider using arrondissement boundaries or verifying the correct relation ID.

2. **For Montmartre, Belleville, Bastille/Oberkampf**: These are cultural/historical neighborhoods rather than official administrative quartiers. Options:
   - Use arrondissement quartier boundaries (admin_level=10 in OSM)
   - Define custom boundaries based on historical/cultural definitions
   - Use OpenStreetMap place=neighbourhood or place=suburb tags

3. **For La Défense**: This is in the suburbs (Puteaux/Courbevoie/Nanterre). May need:
   - Data from those municipalities
   - OSM relation for the business district
   - Custom boundary definition

4. **For Buttes-Chaumont**: Use 19th arrondissement quartier boundaries from Paris Open Data or OSM admin_level=10.

### Files Updated

- `paris_neighborhoods.geojson` - Updated with official polygon boundaries for 8 neighborhoods
- All updated neighborhoods include metadata:
  - `boundary_source`: "PLU bioclimatique OAP (Paris Open Data)"
  - `boundary_updated`: "2024-12-19"
  - `boundary_type`: "official_polygon"
