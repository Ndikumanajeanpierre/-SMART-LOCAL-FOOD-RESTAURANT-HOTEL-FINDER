import streamlit as st

from gps_component import get_browser_location
from recommender import ALL_LOCATIONS, LOCATION_UNRECORDED, FoodRecommender

st.set_page_config(
    page_title="Smart Local Food, Restaurant & Hotel Finder",
    page_icon="🍽️",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1600px;
        padding: clamp(1rem, 3vw, 2rem) clamp(.75rem, 3vw, 3rem);
    }
    .app-title {
        background: #1E88E5;
        border-radius: 12px;
        color: white;
        font-size: clamp(1.2rem, 4vw, 1.75rem);
        font-weight: 600;
        line-height: 1.3;
        padding: clamp(.75rem, 3vw, 1.375rem);
        text-align: center;
        overflow-wrap: anywhere;
    }
    [data-testid="stDataFrame"] {
        max-width: 100%;
    }
    @media (max-width: 768px) {
        [data-testid="stAppViewContainer"] .main .block-container {
            padding: 1rem .75rem;
        }
        [data-testid="stHorizontalBlock"] {
            align-items: stretch;
            flex-direction: column !important;
            gap: .75rem !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 100% !important;
            min-width: 100%;
            width: 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

GPS_LOCATION = "My current location"
KNOWN_RWANDA_LOCATIONS = {
    "Kigali": "Kigali — Rwanda's capital",
    "Musanze": "Musanze — Volcanoes National Park gateway",
    "Gisenyi": "Gisenyi (Rubavu) — Lake Kivu",
    "Huye": "Huye (Butare)",
    "Nyagatare": "Nyagatare",
    "Rusizi": "Rusizi (Cyangugu)",
    "Karongi": "Karongi (Kibuye) — Lake Kivu",
    "Muhanga": "Muhanga",
    "Rwamagana": "Rwamagana",
    "Nyanza": "Nyanza",
    "Kayonza": "Kayonza",
}


def directions_url(row, origin=None):
    origin_parameter = (
        f"&origin={origin[0]}%2C{origin[1]}" if origin else ""
    )
    return (
        "https://www.google.com/maps/dir/?api=1&destination="
        f"{row['lat']}%2C{row['lon']}{origin_parameter}"
    )


@st.cache_resource
def get_recommender():
    return FoodRecommender()


rec = get_recommender()


st.markdown(
    "<div class='app-title'>"
    "🍽️ SMART LOCAL FOOD, RESTAURANT & HOTEL FINDER</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center'>Find nearby meals and places to stay across Rwanda.</p>",
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([1, 2.4], gap="large")

with left:
    with st.container(border=True):
        st.subheader("⚙️ Preferences")
        st.caption("Search from your current position or a well-known Rwanda town.")
        location_choice = st.selectbox(
            "📍 Search near",
            [
                GPS_LOCATION,
                ALL_LOCATIONS,
                *[
                    name
                    for name in KNOWN_RWANDA_LOCATIONS
                    if name in rec.locality_names
                ],
                *[
                    name
                    for name in rec.locality_names
                    if name not in KNOWN_RWANDA_LOCATIONS
                ],
                LOCATION_UNRECORDED,
            ],
            format_func=lambda name: (
                KNOWN_RWANDA_LOCATIONS.get(name, name)
            ),
        )
        browser_location = get_browser_location()
        gps_coordinates = None
        if location_choice == GPS_LOCATION:
            if browser_location and browser_location.get("status") == "success":
                gps_coordinates = (
                    float(browser_location["latitude"]),
                    float(browser_location["longitude"]),
                )
                st.success(
                    "Using your live browser location "
                    f"(accuracy about {browser_location['accuracy']:.0f} m)."
                )
            elif browser_location and browser_location.get("status") == "error":
                st.warning(browser_location["message"])
            else:
                st.info(
                    "Allow location access in your browser when prompted to "
                    "find places around you."
                )
        location = (
            ("gps", *gps_coordinates)
            if gps_coordinates
            else None
            if location_choice == GPS_LOCATION
            else location_choice
        )
        radius_km = st.selectbox(
            "Nearby search radius",
            [5, 10, 25, 50],
            index=2,
            format_func=lambda value: f"{value} km",
            disabled=location_choice in (
                ALL_LOCATIONS,
                LOCATION_UNRECORDED,
            ),
        )
        category = st.selectbox(
            "Category",
            ["All places", "Restaurants & eateries", "Accommodation"],
        )
        view = st.radio("Show", ["Browse all", "Search"])

with right:
    if view == "Browse all":
        results = (
            rec.list_location(location, category, radius_km)
            if location
            else rec.df.iloc[0:0]
        )
        area_label = (
            "RWANDA"
            if location_choice == ALL_LOCATIONS
            else "YOUR CURRENT LOCATION"
            if location_choice == GPS_LOCATION
            else KNOWN_RWANDA_LOCATIONS.get(location_choice, location_choice).upper()
        )
        heading = (
            f"NEAR {area_label}"
            if location_choice not in (ALL_LOCATIONS, LOCATION_UNRECORDED)
            else area_label
        )
        st.markdown(f"#### 🍽️ {category.upper()} {heading}")
        st.caption(f"{len(results)} mapped listings")
        if location_choice == GPS_LOCATION and not gps_coordinates:
            st.info(
                "Allow browser location access above to browse places nearby, "
                "or choose a well-known Rwanda locality."
            )
        elif results.empty:
            st.info(
                "No named listings in this category are currently present in "
                "the OpenStreetMap snapshot for this location."
            )
        if "location_source" in results:
            st.caption(
                "A named location is shown only when the listing has an OSM "
                "addr:city tag. Coordinates identify the mapped point."
            )
        st.caption(
            "OSM-recorded cuisine and services are factual tags. Columns marked "
            "unverified are general ideas to ask the business about—not claims "
            "that it serves or offers them."
        )
        distance_columns = ["distance_km"] if "distance_km" in results else []
        if category == "Restaurants & eateries":
            listings = results[
                [
                    "name",
                    "listing_type",
                    "food_served",
                    "food_search_catalog",
                    "services_recorded",
                    "service_search_catalog",
                    "lat",
                    "lon",
                    *distance_columns,
                    "location_source",
                    "osm_url",
                ]
            ].rename(
                columns={
                    "name": "Name",
                    "listing_type": "Type",
                    "food_served": "Cuisine / food recorded",
                    "food_search_catalog": "Searchable food types (menu not verified)",
                    "services_recorded": "Services recorded in OSM",
                    "service_search_catalog": "Searchable service types (availability not verified)",
                    "lat": "Latitude",
                    "lon": "Longitude",
                    "distance_km": "Distance from selected locality (km)",
                    "location_source": "Location assignment",
                    "osm_url": "OpenStreetMap source",
                }
            )
            listings["Cuisine / food recorded"] = listings[
                "Cuisine / food recorded"
            ].replace("", "Not recorded in OpenStreetMap")
            listings["Services recorded in OSM"] = listings[
                "Services recorded in OSM"
            ].replace("", "No service tags recorded")
        elif category == "Accommodation":
            listings = results[
                [
                    "name",
                    "listing_type",
                    "accommodation_details",
                    "services_recorded",
                    "food_search_catalog",
                    "service_search_catalog",
                    "lat",
                    "lon",
                    *distance_columns,
                    "location_source",
                    "osm_url",
                ]
            ].rename(
                columns={
                    "name": "Name",
                    "listing_type": "Accommodation type",
                    "accommodation_details": "Accommodation details recorded",
                    "services_recorded": "Services recorded in OSM",
                    "food_search_catalog": "Searchable food types (menu not verified)",
                    "service_search_catalog": "Searchable service types (availability not verified)",
                    "lat": "Latitude",
                    "lon": "Longitude",
                    "distance_km": "Distance from selected locality (km)",
                    "location_source": "Location assignment",
                    "osm_url": "OpenStreetMap source",
                }
            )
            listings["Accommodation details recorded"] = listings[
                "Accommodation details recorded"
            ].replace("", "No extra details recorded in OpenStreetMap")
            listings["Services recorded in OSM"] = listings[
                "Services recorded in OSM"
            ].replace("", "No service tags recorded")
        else:
            listings = results[
                [
                    "name",
                    "category",
                    "listing_type",
                    "food_served",
                    "food_search_catalog",
                    "accommodation_details",
                    "services_recorded",
                    "service_search_catalog",
                    "lat",
                    "lon",
                    *distance_columns,
                    "location_source",
                    "osm_url",
                ]
            ].rename(
                columns={
                    "name": "Name",
                    "category": "Category",
                    "listing_type": "Type",
                    "food_served": "Cuisine / food recorded",
                    "food_search_catalog": "Searchable food types (menu not verified)",
                    "accommodation_details": "Accommodation details recorded",
                    "services_recorded": "Services recorded in OSM",
                    "service_search_catalog": "Searchable service types (availability not verified)",
                    "lat": "Latitude",
                    "lon": "Longitude",
                    "distance_km": "Distance from selected locality (km)",
                    "location_source": "Location assignment",
                    "osm_url": "OpenStreetMap source",
                }
            )
            listings["Cuisine / food recorded"] = listings[
                "Cuisine / food recorded"
            ].replace("", "Not recorded in OpenStreetMap")
            listings["Accommodation details recorded"] = listings[
                "Accommodation details recorded"
            ].replace("", "Not applicable or not recorded in OpenStreetMap")
            listings["Services recorded in OSM"] = listings[
                "Services recorded in OSM"
            ].replace("", "No service tags recorded")
        listings["Location assignment"] = listings["Location assignment"].replace(
            {
                "OpenStreetMap addr:city": "OSM address city",
                "Not recorded in OpenStreetMap": LOCATION_UNRECORDED,
            }
        )
        listings["Type" if "Type" in listings else "Accommodation type"] = (
            listings["Type" if "Type" in listings else "Accommodation type"]
            .str.replace("_", " ")
        )
        listings["Directions"] = results.apply(
            lambda row: directions_url(row, gps_coordinates), axis=1
        ).to_numpy()
        st.dataframe(
            listings,
            hide_index=True,
            width="stretch",
            column_config={
                "OpenStreetMap source": st.column_config.LinkColumn(
                    "Source listing", display_text="Open map"
                ),
                "Directions": st.column_config.LinkColumn("Directions"),
            },
        )
    else:
        query = st.text_input(
            "🍴 Search food, cuisine, restaurant, hotel, motel, or lodge",
            placeholder="e.g. brochettes, cafe, hotel, or a place name",
        )

        search_area = (
            "RWANDA"
            if location_choice == ALL_LOCATIONS
            else KNOWN_RWANDA_LOCATIONS.get(
                location_choice, location_choice
            ).upper()
        )
        st.markdown(f"#### 🔎 MATCHING LISTINGS IN {search_area}")
        if location_choice not in (ALL_LOCATIONS, LOCATION_UNRECORDED):
            st.caption(
                (
                    f"Searching within {radius_km} km of your live location."
                    if gps_coordinates
                    else (
                        f"Searching within {radius_km} km of "
                        f"{KNOWN_RWANDA_LOCATIONS.get(location_choice, location_choice)}'s "
                        "mapped locality point."
                    )
                )
            )
        results = (
            rec.recommend(
                query,
                location=location,
                category=category,
                radius_km=radius_km,
            )
            if query.strip() and location
            else rec.df.iloc[0:0]
        )

        if not query.strip():
            st.info(
                "Enter a food, cuisine, restaurant, hotel, motel, or lodge "
                "name and press Enter to search. No separate search button "
                "is needed."
            )
        elif not location:
            st.info(
                "Allow browser location access above, or choose a known "
                "Rwanda locality to search nearby."
            )
        elif results.empty:
            st.warning(rec.answer(query, results))
            if location not in (ALL_LOCATIONS, LOCATION_UNRECORDED):
                st.caption(
                    "Try a larger nearby radius or search all of Rwanda. "
                    "Only food and businesses recorded in the OSM snapshot "
                    "can be found."
                )
        else:
            st.info(rec.answer(query, results))
            for i, r in results.iterrows():
                with st.container(border=True):
                    info, match = st.columns([3, 1])
                    info.markdown(f"### {i + 1}. {r['name']}")
                    if "search_match_type" in r.index:
                        if r["search_match_type"] == "cuisine_tag_candidate":
                            info.info(
                                f"OpenStreetMap has a related cuisine tag for "
                                f"{r['requested_item']}; the exact dish is not "
                                "confirmed on the menu."
                            )
                        elif r["search_match_type"] == "osm_service_tag_candidate":
                            info.info(
                                f"A related OSM service tag was found for "
                                f"{r['requested_item']}; confirm details directly."
                            )
                        else:
                            info.warning(
                                f"Possible nearby place to ask about "
                                f"{r['requested_item']}. The menu/service is "
                                "not confirmed by the map data."
                            )
                    info.write(
                        f"📍 **Type:** {r['listing_type'].replace('_', ' ')}"
                    )
                    recorded_location = (
                        r["location"]
                        if r["location_source"] == "OpenStreetMap addr:city"
                        else "No street address or named locality recorded"
                    )
                    info.write(f"**Location recorded:** {recorded_location}")
                    if r["tourism"]:
                        details = (
                            r["accommodation_details"]
                            or "No extra details recorded in OpenStreetMap"
                        )
                        info.write(f"🏨 **Accommodation details:** {details}")
                    if r["amenity"]:
                        cuisine = r["food_served"] or "Not recorded in OpenStreetMap"
                        info.write(f"🥘 **Cuisine / food recorded:** {cuisine}")
                    services = (
                        r["services_recorded"] or "No service tags recorded"
                    )
                    info.write(f"🧾 **Services recorded in OSM:** {services}")
                    if r["food_examples_to_ask"] or r["service_examples_to_ask"]:
                        if r["food_search_catalog"]:
                            with info.expander(
                                "Searchable foods (menu not verified)"
                            ):
                                info.write(r["food_search_catalog"])
                        if r["service_search_catalog"]:
                            with info.expander(
                                "Searchable services (availability not verified)"
                            ):
                                info.write(r["service_search_catalog"])
                    if r["location_source"] != "OpenStreetMap addr:city":
                        info.caption(
                            "A street address or named locality is not recorded "
                            "for this listing. Use its mapped point for directions."
                        )
                    b1, b2, b3 = info.columns(3)
                    map_url = (
                        "https://www.openstreetmap.org/?mlat="
                        f"{r['lat']}&mlon={r['lon']}#map=18/{r['lat']}/{r['lon']}"
                    )
                    info.caption(f"Mapped coordinates: {r['lat']}, {r['lon']}")
                    if "distance_km" in r.index:
                        distance_origin = (
                            "your current location"
                            if gps_coordinates
                            else KNOWN_RWANDA_LOCATIONS.get(
                                location_choice, location_choice
                            )
                        )
                        info.caption(
                            f"{r['distance_km']:.1f} km from "
                            f"{distance_origin}"
                        )
                    b1.link_button(
                        "🧭 Get directions",
                        directions_url(r, gps_coordinates),
                        width="stretch",
                    )
                    b2.link_button("🗺️ View map", map_url, width="stretch")
                    b3.link_button("↗️ OSM listing", r["osm_url"], width="stretch")
                    if r["phone"]:
                        info.link_button("📞 Call", f"tel:{r['phone']}")
                    if "search_match_type" not in r.index:
                        match.metric("TEXT MATCH", f"{int(r['match_pct'])}%")

snapshot_date = rec.df["source_updated_at"].max()[:10]
st.markdown(
    f"Data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) "
    f"(ODbL), snapshot {snapshot_date}. Coverage is limited to mapped named "
    "listings; tags may be incomplete or out of date. Verify details before "
    "visiting. Exact street addresses and room availability may not be recorded. "
    "Nearby distances are straight-line estimates, not live GPS or road distances."
)
