"""Validate dataset provenance and that answers stay grounded in mapped records."""
import json
from pathlib import Path

from recommender import (
    ALL_LOCATIONS,
    FOOD_EXAMPLES_TO_ASK,
    LOCATION_UNRECORDED,
    FoodRecommender,
)


def evaluate():
    rec = FoodRecommender()
    required = {
        "id", "name", "location", "amenity", "tourism", "stars", "rooms",
        "internet_access", "lat", "lon", "osm_url", "location_source",
        "food_examples_to_ask", "service_examples_to_ask",
        "food_search_catalog", "service_search_catalog",
    }
    missing_columns = required - set(rec.df.columns)
    if missing_columns:
        raise ValueError(f"Missing required data columns: {sorted(missing_columns)}")
    non_address_locations = rec.df["location_source"].ne("OpenStreetMap addr:city")
    if not rec.df.loc[non_address_locations, "location"].eq(
        LOCATION_UNRECORDED
    ).all():
        raise AssertionError("A listing without addr:city has an inferred location.")
    if rec.df.loc[~non_address_locations, "location"].eq(
        LOCATION_UNRECORDED
    ).any():
        raise AssertionError("An explicit OSM address city was not retained.")
    if rec.locations != sorted(rec.df["location"].unique()):
        raise AssertionError("Location choices do not match recorded assignments.")

    expected_urls = rec.df.apply(
        lambda row: (
            f"https://www.openstreetmap.org/{row['osm_type']}/{row['osm_id']}"
        ),
        axis=1,
    )
    all_records_linked = bool(rec.df["osm_url"].eq(expected_urls).all())
    sample = rec.recommend("restaurant", top_k=1)
    citation_present = bool(
        not sample.empty and sample.iloc[0]["osm_url"] in rec.answer("restaurant", sample)
    )
    unknown = rec.recommend("zzqv-unmapped-query-9301")
    unknown_refused = unknown.empty and "couldn't find" in rec.answer(
        "zzqv-unmapped-query-9301", unknown
    )
    if not all_records_linked or not citation_present or not unknown_refused:
        raise AssertionError("Grounding/provenance validation failed.")

    location_counts = {
        location: int(count)
        for location, count in rec.df["location"].value_counts().items()
    }
    unrecorded_location_count_correct = len(
        rec.list_location(LOCATION_UNRECORDED)
    ) == int(rec.df["location"].eq(LOCATION_UNRECORDED).sum())
    if not unrecorded_location_count_correct:
        raise AssertionError("Unrecorded-location listings were omitted.")
    nationwide_listings = rec.list_location(ALL_LOCATIONS)
    all_rwanda_listing_count_correct = len(nationwide_listings) == len(rec.df)
    nationwide_search = rec.recommend("restaurant", location=ALL_LOCATIONS)
    unfiltered_search = rec.recommend("restaurant")
    nationwide_search_unfiltered = bool(
        nationwide_search["id"].tolist() == unfiltered_search["id"].tolist()
    )
    if not all_rwanda_listing_count_correct or not nationwide_search_unfiltered:
        raise AssertionError("Nationwide browse/search applied an unexpected location filter.")

    nearby = rec.list_location("Kigali", radius_km=10)
    nearby_distance_valid = bool(
        not nearby.empty and nearby["distance_km"].le(10).all()
    )
    nearby_search = rec.recommend(
        "restaurant", location="Kigali", radius_km=10
    )
    nearby_search_distance_valid = bool(
        not nearby_search.empty and nearby_search["distance_km"].le(10).all()
    )
    if not nearby_distance_valid or not nearby_search_distance_valid:
        raise AssertionError("Nearby results exceed the selected locality radius.")
    gps_nearby = rec.list_location(("gps", -1.9534357, 30.1140089), radius_km=10)
    gps_distance_valid = bool(
        not gps_nearby.empty and gps_nearby["distance_km"].le(10).all()
    )
    if not gps_distance_valid:
        raise AssertionError("Live-coordinate nearby results exceed the radius.")

    category_counts = {
        category: int(rec.df["category"].str.contains(category, regex=False).sum())
        for category in ("Restaurants & eateries", "Accommodation")
    }
    if rec.df["category"].eq("").any():
        raise AssertionError("One or more listings have no mapped place category.")
    businesses = rec.df["amenity"].ne("") | rec.df["tourism"].ne("")
    suggestions_present = bool(
        rec.df.loc[businesses, "food_examples_to_ask"].ne("").all()
        and rec.df.loc[businesses, "service_examples_to_ask"].ne("").all()
        and rec.df.loc[~businesses, "food_examples_to_ask"].eq("").all()
        and rec.df.loc[~businesses, "service_examples_to_ask"].eq("").all()
    )
    suggestions_separate_from_facts = bool(
        rec.df.loc[businesses, "food_examples_to_ask"].eq(
            FOOD_EXAMPLES_TO_ASK
        ).all()
        and rec.df.apply(
            lambda row: (
                row["food_examples_to_ask"] not in row["document"]
                and row["service_examples_to_ask"] not in row["document"]
            ),
            axis=1,
        )
        .all()
    )
    if not suggestions_present or not suggestions_separate_from_facts:
        raise AssertionError("General examples were not kept separate from OSM facts.")
    food_query_candidates = rec.recommend(
        "I want sushi", location=("gps", -1.9534357, 30.1140089), radius_km=25
    )
    food_query_returns_local_eateries = bool(
        not food_query_candidates.empty
        and (
            food_query_candidates["amenity"].ne("")
            | food_query_candidates["tourism"].ne("")
        ).all()
        and food_query_candidates["requested_item"].eq("sushi").all()
        and food_query_candidates["search_match_type"]
        .isin(("unverified_food_candidate", "cuisine_tag_candidate"))
        .all()
    )
    hotel_service_candidates = rec.recommend(
        "hotel with a swimming pool",
        location=("gps", -1.9534357, 30.1140089),
        radius_km=25,
        category="Accommodation",
    )
    hotel_service_search_works = bool(
        not hotel_service_candidates.empty
        and hotel_service_candidates["tourism"].ne("").all()
        and hotel_service_candidates["requested_item"].eq("swimming pool").all()
        and hotel_service_candidates["search_match_type"]
        .isin(("unverified_service_candidate", "osm_service_tag_candidate"))
        .all()
    )
    if not food_query_returns_local_eateries or not hotel_service_search_works:
        raise AssertionError("Food or hotel-service catalog search failed.")
    if rec.df["location"].eq("").any() or rec.df["location_source"].eq("").any():
        raise AssertionError("One or more listings have no locality assignment.")
    location_sources = {
        source: int(count)
        for source, count in rec.df["location_source"].value_counts().items()
    }

    return {
        "mapped_listings": len(rec.df),
        "available_location_choices": len(rec.locations),
        "listings_with_explicit_osm_address_city": int(
            (~non_address_locations).sum()
        ),
        "listings_without_recorded_osm_address_city": int(
            non_address_locations.sum()
        ),
        "all_rwanda_browse_count_correct": all_rwanda_listing_count_correct,
        "all_rwanda_search_unfiltered": nationwide_search_unfiltered,
        "nearby_listings_within_selected_radius": nearby_distance_valid,
        "nearby_search_within_selected_radius": nearby_search_distance_valid,
        "live_coordinate_search_within_selected_radius": gps_distance_valid,
        "locations_with_listings": len(location_counts),
        "listings_by_location": location_counts,
        "listings_by_category": category_counts,
        "location_assignment_sources": location_sources,
        "accommodation_types": {
            key: int(value)
            for key, value in rec.df["tourism"].value_counts().items()
        },
        "listings_with_cuisine_tags": int(rec.df["cuisine"].ne("").sum()),
        "listings_with_phone_tags": int(rec.df["phone"].ne("").sum()),
        "listings_with_recorded_service_tags": int(
            rec.df["services_recorded"].ne("").sum()
        ),
        "businesses_with_separate_food_and_service_ideas": int(businesses.sum()),
        "food_and_service_ideas_populated_separately": suggestions_present,
        "food_and_service_ideas_separate_from_osm_facts": suggestions_separate_from_facts,
        "unlisted_food_query_returns_nearby_places_to_ask": food_query_returns_local_eateries,
        "hotel_service_query_returns_nearby_hotels_to_ask": hotel_service_search_works,
        "all_records_link_to_openstreetmap": all_records_linked,
        "unrecorded_location_listing_count_correct": unrecorded_location_count_correct,
        "answer_includes_record_source": citation_present,
        "unknown_query_returns_no_fabricated_listing": unknown_refused,
    }


if __name__ == "__main__":
    report = evaluate()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    output = Path(__file__).parent / "artifacts" / "eval_results.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
