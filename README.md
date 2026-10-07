# Smart Local Food, Restaurant & Hotel Finder

**Module:** ITLPA701 – Python and Fundamentals of AI (CAT2 Practical)  
**Approach:** TF-IDF search over attributed OpenStreetMap (OSM) listings  
**Tagline:** Find nearby meals and places to stay across Rwanda.  
**Domain:** Nearby food, restaurant, hotel, motel, lodge, and accommodation discovery

## Data integrity

`data/restaurants.csv` is a Rwanda-wide snapshot of named OSM nodes and ways
mapped as a restaurant, cafe, fast-food restaurant, food court, guest house,
hotel, motel, or lodge, queried on 2026-10-03. Each row retains its OSM object
type and ID, mapped coordinates, and a direct link to the source object. A
named location is assigned only when the listing itself has an OSM `addr:city`
tag. Other rows are explicitly grouped as **Location not recorded in OSM**;
the application no longer guesses a city from nearby map points. Their
coordinates still identify the mapped point and can be checked through its OSM
source link.

Service and facility details are limited to values actually recorded in the
OSM listing. The current snapshot includes place type, cuisine where tagged,
contact details, and some accommodation details such as stars, rooms, and
internet access. It does not record a complete list of services offered. For
example, an empty delivery, takeaway, reservation, or accessibility field means
that OSM has no such tag in this snapshot; it does not prove the business
doesn't provide that service. The app labels unrecorded details instead of
inventing them. Verify business details with the business before visiting.

Every mapped restaurant is searchable against a broad food catalog covering
Rwandan dishes, African and international cuisines, common meals, drinks, and
desserts. Hotels and restaurants are also searchable against relevant common
services. When a requested food or service is not confirmed by the business's
OSM tags, the app returns nearby matching types of places as **places to ask**,
not as businesses known to provide it. If a related cuisine or service tag is
mapped, that tag is shown separately and is still not proof that a particular
dish or service is currently available. The complete search catalogs are
visible on each listing under labels stating that menus and service
availability are not verified. Only values in the separate OSM-recorded fields
are mapped business facts.

Data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
available under the Open Database License (ODbL).
Map data is community-maintained and may be incomplete, stale, or incorrect.
An OSM listing is not independent confirmation that a business is open or
operating.

Choose a locality from the OSM place catalog and a nearby search radius
(5–50 km), **Location not recorded in OSM**, or **All Rwanda**, then browse or
search restaurants, eateries, accommodation, or all places. Nearby filtering
uses straight-line distance from the locality's mapped point and the listing's
coordinates; when permission is granted, the browser can instead use the
device's current GPS coordinates. GPS permission is requested only after the
user clicks the location button. Browser GPS requires a secure context:
localhost for development or HTTPS when deployed. Coordinates are sent to this
app's Streamlit server to filter results and are not persisted by the app. If
the user opens a Google Maps directions link, the link includes the current
and destination coordinates. GPS and locality distances are straight-line, not
road distances.
Well-known choices include Kigali, Musanze, Gisenyi
(Rubavu), Huye (Butare), Rusizi (Cyangugu), and Karongi (Kibuye). Nationwide
browsing/search includes every listing in the snapshot. Press Enter after
typing a search query; no separate search button is needed. Search results
include a Google Maps directions link to each mapped listing.
“Cuisine / food recorded” uses only the OSM `cuisine` tag; it is not a full
menu. Service/facility details are shown only for service tags present in the
snapshot. The app does not invent ratings, menus, prices, opening hours, or
distances; nearby distances are calculated from mapped coordinates as
straight-line estimates. Search matches mapped names, place types, cuisine,
and recorded service tags. Common foods and services in the search catalogs
also return nearby restaurants/hotels to contact when that item is not mapped;
results may still be empty if no relevant listing is in the selected area.
Street addresses may be absent. Accommodation
listings do not represent confirmed live room availability or booking inventory.

## Architecture

```text
Selected OSM locality and nearby radius
        |
Choose a food/accommodation category and type a query
        |
Browse nearby or all Rwanda, or search by name/type/cuisine
        |
Coordinate proximity + TF-IDF retrieval over OSM name/type/cuisine tags
        |
Known food/service query -> nearby relevant businesses, clearly marked as
verified map tags or unverified places to contact
        |
Mapped locations, map links, and directions
        |
Answer cites the original OpenStreetMap object
```

## Files

| File | Purpose |
|---|---|
| `data/restaurants.csv` | Attributed OSM snapshot |
| `data/locations.csv` | OSM locality coordinates for nearby search; not used to infer street addresses |
| `recommender.py` | Data loading, retrieval, ranking, and source-grounded answers |
| `evaluate.py` | Provenance and no-fabrication checks; not relevance benchmarking |
| `app.py` | Streamlit interface |
| `config.json` | Retrieval settings |
| `artifacts/` | Saved TF-IDF vectorizer, matrix, and validation output |

## Run

```bash
pip install -r requirements.txt
python recommender.py
python evaluate.py
streamlit run app.py
```

The Streamlit interface adapts its layout for phone, tablet, and desktop
screens. On smaller screens, preference and result panels stack vertically,
and long result tables can be scrolled horizontally.

The CSV is a dated nationwide snapshot and does not update automatically.
Refresh it from OpenStreetMap before using it as current business information.
Do not replace missing map tags with guesses or present search scores as ratings.
