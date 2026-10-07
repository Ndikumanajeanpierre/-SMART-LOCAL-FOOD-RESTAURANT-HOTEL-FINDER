"""Restaurant search grounded in OpenStreetMap listings and tags."""
import json
import math
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
ARTIFACTS = BASE / "artifacts"
EATERY_CATEGORY = "Restaurants & eateries"
ACCOMMODATION_CATEGORY = "Accommodation"
ALL_LOCATIONS = "All Rwanda"
LOCATION_UNRECORDED = "Location not recorded in OSM"
SERVICE_TAG_COLUMNS = (
    "delivery",
    "takeaway",
    "drive_through",
    "outdoor_seating",
    "reservation",
    "wheelchair",
    "toilets",
    "internet_access",
    "smoking",
    "opening_hours",
)
FOOD_EXAMPLES_TO_ASK = (
    "Isombe (cassava leaves), ibihaza (pumpkin), matoke/ibitoki, "
    "igisafuriya, brochettes, akabenz, sambaza, tilapia, grilled fish, "
    "goat, beef, chicken, pork, beans, rice, ugali, cassava, sweet potatoes, "
    "Irish potatoes, plantain, vegetables, chapati, rolex, samosas, "
    "African dishes, Ethiopian injera, Nigerian jollof rice, Senegalese "
    "thieboudienne, South African braai, Indian curry, biryani, tikka "
    "masala, naan, Chinese dishes, noodles, fried rice, dumplings, sushi, "
    "ramen, Thai dishes, pad thai, Vietnamese pho, Korean barbecue, "
    "Mediterranean dishes, falafel, hummus, shawarma, kebab, pizza, pasta, "
    "lasagna, burgers, hot dogs, sandwiches, fried chicken, chips/fries, "
    "salads, soup, vegetarian dishes, vegan dishes, gluten-free dishes, "
    "coffee, tea, fresh juice, smoothies, cakes, pastries, ice cream, "
    "pancakes, waffles, and desserts"
)
EATERY_SERVICE_EXAMPLES_TO_ASK = (
    "dine-in, takeaway, delivery, curbside pickup, reservations, catering, "
    "outdoor seating, private dining, buffet, breakfast, lunch, dinner, "
    "vegetarian options, vegan options, halal options, allergy-aware meals, "
    "Wi-Fi, accessible entrance, accessible toilet, parking, toilets, "
    "cash, card, mobile money, and opening hours"
)
LODGING_SERVICE_EXAMPLES_TO_ASK = (
    "room availability, breakfast, restaurant, bar, room service, Wi-Fi, "
    "parking, airport transfer, shuttle, laundry, housekeeping, luggage "
    "storage, 24-hour reception, concierge, booking/cancellation terms, "
    "conference room, meeting facilities, event venue, swimming pool, spa, "
    "gym, garden, air conditioning, hot water, family rooms, cot/crib, "
    "accessible room, accessible entrance, accessible toilet, payment by "
    "cash/card/mobile money, and check-in/check-out times"
)

FOOD_ALIASES = {
    "isombe": ("isombe", "cassava leaves", "manioc leaves"),
    "ibihaza": ("ibihaza", "pumpkin"),
    "matoke": ("matoke", "ibitoki", "plantain", "cooking bananas"),
    "igisafuriya": ("igisafuriya",),
    "brochettes": ("brochette", "brochettes", "kebab", "kebabs", "skewer"),
    "akabenz": ("akabenz",),
    "sambaza": ("sambaza", "lake tanganyika sardine"),
    "tilapia": ("tilapia",),
    "grilled fish": ("grilled fish", "fish"),
    "goat": ("goat", "nyama y'ihene"),
    "beef": ("beef",),
    "chicken": ("chicken",),
    "pork": ("pork",),
    "beans": ("beans", "ibishyimbo"),
    "rice": ("rice", "umuceri"),
    "ugali": ("ugali", "posho", "fufu"),
    "cassava": ("cassava", "manioc"),
    "sweet potatoes": ("sweet potato", "sweet potatoes"),
    "Irish potatoes": ("irish potato", "irish potatoes", "potatoes"),
    "vegetables": ("vegetable", "vegetables", "greens"),
    "chapati": ("chapati", "roti"),
    "rolex": ("rolex", "chapati egg"),
    "samosas": ("samosa", "samosas", "sambusa", "sambusas"),
    "African food": ("african food", "african cuisine"),
    "Ethiopian food": ("ethiopian", "injera"),
    "Nigerian food": ("nigerian", "jollof rice", "jollof"),
    "Senegalese food": ("senegalese", "thieboudienne"),
    "South African food": ("south african", "braai"),
    "Indian food": ("indian food", "indian cuisine", "curry", "biryani", "tikka masala", "naan"),
    "Chinese food": ("chinese food", "chinese cuisine", "fried rice", "dumplings"),
    "noodles": ("noodle", "noodles"),
    "sushi": ("sushi", "sashimi"),
    "ramen": ("ramen",),
    "Thai food": ("thai food", "thai cuisine", "pad thai"),
    "Vietnamese food": ("vietnamese", "pho"),
    "Korean food": ("korean", "korean barbecue", "kimchi"),
    "Mediterranean food": ("mediterranean", "falafel", "hummus"),
    "shawarma": ("shawarma",),
    "Mexican food": ("mexican food", "mexican cuisine", "taco", "tacos", "burrito", "burritos", "quesadilla", "enchilada"),
    "Japanese food": ("japanese food", "japanese cuisine", "tempura", "teriyaki", "udon"),
    "Indonesian food": ("indonesian", "nasi goreng", "satay"),
    "Turkish food": ("turkish food", "turkish cuisine", "doner", "döner", "baklava"),
    "Greek food": ("greek food", "greek cuisine", "gyro", "gyros", "souvlaki"),
    "French food": ("french food", "french cuisine", "croissant", "crepe", "crepes"),
    "Spanish food": ("spanish food", "spanish cuisine", "paella", "tapas"),
    "Brazilian food": ("brazilian", "feijoada"),
    "Caribbean food": ("caribbean", "jerk chicken"),
    "seafood": ("seafood", "prawns", "shrimp", "crab", "lobster", "octopus"),
    "pizza": ("pizza",),
    "pasta": ("pasta", "spaghetti", "macaroni"),
    "lasagna": ("lasagna", "lasagne"),
    "burgers": ("burger", "burgers", "hamburger"),
    "hot dogs": ("hot dog", "hot dogs"),
    "sandwiches": ("sandwich", "sandwiches", "panini"),
    "fried chicken": ("fried chicken",),
    "chips/fries": ("chips", "fries", "french fries"),
    "salads": ("salad", "salads"),
    "soup": ("soup",),
    "vegetarian food": ("vegetarian", "vegetarian food"),
    "vegan food": ("vegan", "vegan food"),
    "gluten-free food": ("gluten free", "gluten-free"),
    "dairy-free food": ("dairy free", "dairy-free"),
    "halal food": ("halal food", "halal"),
    "kosher food": ("kosher food", "kosher"),
    "coffee": ("coffee", "espresso", "cappuccino", "latte"),
    "tea": ("tea",),
    "fresh juice": ("juice", "fresh juice"),
    "smoothies": ("smoothie", "smoothies"),
    "cakes": ("cake", "cakes"),
    "pastries": ("pastry", "pastries", "croissant"),
    "ice cream": ("ice cream", "gelato"),
    "pancakes": ("pancake", "pancakes", "crepe", "crepes"),
    "waffles": ("waffle", "waffles"),
    "doughnuts": ("doughnut", "doughnuts", "donut", "donuts"),
    "cookies": ("cookie", "cookies", "biscuit", "biscuits"),
    "chocolate": ("chocolate", "brownie", "brownies"),
    "desserts": ("dessert", "desserts", "pudding"),
}

HOTEL_SERVICE_ALIASES = {
    "room availability": ("room availability", "vacancy", "available room"),
    "breakfast": ("breakfast",),
    "on-site restaurant": ("on-site restaurant", "hotel dining"),
    "bar": ("bar",),
    "room service": ("room service",),
    "Wi-Fi": ("wi-fi", "wifi", "internet"),
    "parking": ("parking",),
    "airport transfer": ("airport transfer", "airport shuttle", "transfer"),
    "shuttle": ("shuttle",),
    "laundry": ("laundry",),
    "housekeeping": ("housekeeping", "cleaning"),
    "luggage storage": ("luggage storage", "baggage storage"),
    "24-hour reception": ("24 hour reception", "24-hour reception", "reception"),
    "concierge": ("concierge",),
    "booking terms": ("booking", "cancellation"),
    "conference facilities": ("conference", "meeting room", "meeting facilities"),
    "event venue": ("event venue", "wedding venue", "events"),
    "swimming pool": ("swimming pool", "pool"),
    "spa": ("spa", "massage"),
    "gym": ("gym", "fitness"),
    "garden": ("garden",),
    "air conditioning": ("air conditioning", "a/c", "ac"),
    "hot water": ("hot water",),
    "family rooms": ("family room", "family rooms"),
    "cot/crib": ("cot", "crib"),
    "accessible room": ("accessible room", "wheelchair access", "accessibility"),
    "toilet": ("toilet", "restroom", "bathroom"),
    "payment methods": ("cash", "credit card", "debit card", "mobile money", "momo"),
    "check-in/check-out": ("check in", "check-in", "check out", "check-out"),
}

RESTAURANT_SERVICE_ALIASES = {
    "dine-in": ("dine in", "dine-in", "eat in"),
    "takeaway": ("takeaway", "take out", "takeout"),
    "delivery": ("delivery", "deliver food"),
    "curbside pickup": ("curbside pickup", "curbside"),
    "reservations": ("reservation", "reservations", "book a table"),
    "catering": ("catering",),
    "outdoor seating": ("outdoor seating", "patio"),
    "private dining": ("private dining",),
    "buffet": ("buffet",),
    "breakfast": ("breakfast",),
    "lunch": ("lunch",),
    "dinner": ("dinner",),
    "vegetarian options": ("vegetarian options", "vegetarian"),
    "vegan options": ("vegan options", "vegan"),
    "halal options": ("halal options", "halal"),
    "allergy-aware meals": ("allergy", "allergen"),
    "Wi-Fi": ("wi-fi", "wifi", "internet"),
    "accessible entrance": ("accessible entrance", "wheelchair entrance"),
    "accessible toilet": ("accessible toilet", "accessible restroom"),
    "parking": ("parking",),
    "toilets": ("toilet", "restroom", "bathroom"),
    "cash": ("cash",),
    "card payments": ("card payment", "credit card", "debit card"),
    "mobile money": ("mobile money", "momo", "mtn mobile money", "airtel money"),
    "opening hours": ("opening hours", "open now"),
}

ALL_SERVICE_ALIASES = {**HOTEL_SERVICE_ALIASES, **RESTAURANT_SERVICE_ALIASES}
FOOD_EXAMPLES_TO_ASK = ", ".join(FOOD_ALIASES)
EATERY_SERVICE_EXAMPLES_TO_ASK = ", ".join(RESTAURANT_SERVICE_ALIASES)
LODGING_SERVICE_EXAMPLES_TO_ASK = ", ".join(HOTEL_SERVICE_ALIASES)


def load_config(path=BASE / "config.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_search_text(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def match_catalog_term(query, catalog):
    normalized_query = f" {_normalize_search_text(query)} "
    matches = []
    for label, aliases in catalog.items():
        for alias in aliases:
            normalized_alias = _normalize_search_text(alias)
            if normalized_alias and f" {normalized_alias} " in normalized_query:
                matches.append((len(normalized_alias), label))
    return max(matches, default=(0, None))[1]


def load_data(cfg):
    df = pd.read_csv(BASE / cfg["data_path"], keep_default_na=False)
    df = df.drop_duplicates("id")
    for column in (
        "tourism",
        "stars",
        "rooms",
        "internet_access",
        *SERVICE_TAG_COLUMNS,
    ):
        if column not in df:
            df[column] = ""
    has_osm_city = df["location_source"].eq("OpenStreetMap addr:city")
    df.loc[~has_osm_city, "location"] = ""
    df.loc[~has_osm_city, "location_source"] = "Not recorded in OpenStreetMap"
    df["location"] = df["location"].replace("", LOCATION_UNRECORDED)
    df["cuisine"] = df["cuisine"].fillna("").str.replace("_", " ", regex=False)
    df["food_served"] = df["cuisine"].str.replace(";", ", ", regex=False)
    df["category"] = ""
    has_eatery_tag = df["amenity"].ne("")
    has_lodging_tag = df["tourism"].ne("")
    df.loc[has_eatery_tag, "category"] = EATERY_CATEGORY
    df.loc[has_lodging_tag & ~has_eatery_tag, "category"] = ACCOMMODATION_CATEGORY
    df.loc[has_lodging_tag & has_eatery_tag, "category"] = (
        f"{EATERY_CATEGORY}; {ACCOMMODATION_CATEGORY}"
    )
    df["listing_type"] = df["amenity"].where(has_eatery_tag, df["tourism"])
    df.loc[has_lodging_tag & has_eatery_tag, "listing_type"] += (
        " / " + df["tourism"]
    )
    df["accommodation_details"] = df.apply(
        lambda row: ", ".join(
            value for value in (
                f"{row['stars']} stars (OSM)" if row["stars"] else "",
                f"{row['rooms']} rooms (OSM)" if row["rooms"] else "",
                f"internet access: {row['internet_access']}"
                if row["internet_access"] else "",
            ) if value
        ),
        axis=1,
    )
    service_columns = [
        column for column in SERVICE_TAG_COLUMNS if column in df
    ]
    service_columns.extend(
        column
        for column in df.columns
        if column.startswith(("payment:", "diet:"))
    )
    df["services_recorded"] = df.apply(
        lambda row: ", ".join(
            f"{column.replace('_', ' ')}: {row[column]}"
            for column in service_columns
            if row[column]
        ),
        axis=1,
    )
    df["food_examples_to_ask"] = df.apply(
        lambda row: (
            FOOD_EXAMPLES_TO_ASK
            if row["amenity"] or row["tourism"]
            else ""
        ),
        axis=1,
    )
    df["service_examples_to_ask"] = df.apply(
        lambda row: (
            "; ".join(
                suggestions
                for relevant, suggestions in (
                    (bool(row["amenity"]), EATERY_SERVICE_EXAMPLES_TO_ASK),
                    (bool(row["tourism"]), LODGING_SERVICE_EXAMPLES_TO_ASK),
                )
                if relevant
            )
        ),
        axis=1,
    )
    df["food_search_catalog"] = df.apply(
        lambda row: (
            ", ".join(FOOD_ALIASES)
            if row["amenity"] or row["tourism"]
            else ""
        ),
        axis=1,
    )
    df["service_search_catalog"] = df.apply(
        lambda row: (
            "; ".join(
                ", ".join(catalog)
                for relevant, catalog in (
                    (bool(row["amenity"]), RESTAURANT_SERVICE_ALIASES),
                    (bool(row["tourism"]), HOTEL_SERVICE_ALIASES),
                )
                if relevant
            )
        ),
        axis=1,
    )
    df["document"] = (
        df["name"] + ". " + df["listing_type"].str.replace("_", " ", regex=False)
        + ". " + df["tourism"].str.replace("_", " ", regex=False)
        + ". " + df["cuisine"].str.replace(";", " ", regex=False)
        + ". " + df["services_recorded"]
    ).str.lower()
    return df.reset_index(drop=True)


def load_locations(df):
    return sorted(df["location"].unique())


def distance_from_coordinates(df, place_latitude, place_longitude):
    """Return distances in kilometres from a mapped point."""
    listing_latitude = pd.to_numeric(
        df["lat"], errors="coerce"
    ).to_numpy(dtype=float)
    listing_longitude = pd.to_numeric(
        df["lon"], errors="coerce"
    ).to_numpy(dtype=float)
    place_latitude = math.radians(float(place_latitude))
    place_longitude = math.radians(float(place_longitude))
    listing_latitude = np.radians(listing_latitude)
    listing_longitude = np.radians(listing_longitude)
    delta_latitude = listing_latitude - place_latitude
    delta_longitude = listing_longitude - place_longitude
    haversine = (
        np.sin(delta_latitude / 2) ** 2
        + math.cos(place_latitude)
        * np.cos(listing_latitude)
        * np.sin(delta_longitude / 2) ** 2
    )
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))


class FoodRecommender:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.df = load_data(self.cfg)
        self.locations = load_locations(self.df)
        places_path = BASE / self.cfg.get("locations_path", "data/locations.csv")
        self.place_catalog = pd.read_csv(places_path, keep_default_na=False)
        self.locality_names = sorted(
            (set(self.place_catalog["name"]) | set(self.locations))
            - {ALL_LOCATIONS, LOCATION_UNRECORDED}
        )
        self._places_by_name = {
            row["name"]: row
            for _, row in self.place_catalog.iterrows()
        }
        r = self.cfg["retrieval"]
        self.vectorizer = TfidfVectorizer(
            ngram_range=tuple(r["ngram_range"]), stop_words="english"
        )
        self.matrix = self.vectorizer.fit_transform(self.df["document"])

    def save(self):
        ARTIFACTS.mkdir(exist_ok=True)
        joblib.dump(self.vectorizer, ARTIFACTS / "vectorizer.joblib")
        joblib.dump(self.matrix, ARTIFACTS / "tfidf_matrix.joblib")

    def retrieve(self, query):
        q = self.vectorizer.transform([query.lower()])
        sims = cosine_similarity(q, self.matrix).ravel()
        out = self.df.copy()
        out["match"] = sims
        return out[out["match"] >= self.cfg["retrieval"]["min_similarity"]]

    def _filter_location(self, listings, location, radius_km):
        if not location or location == ALL_LOCATIONS:
            return listings
        if location == LOCATION_UNRECORDED:
            return listings[listings["location"] == LOCATION_UNRECORDED]

        if isinstance(location, tuple) and location[0] == "gps":
            distances = distance_from_coordinates(
                listings, location[1], location[2]
            )
            nearby = listings.assign(distance_km=distances)
            return nearby[nearby["distance_km"] <= radius_km]

        place = self._places_by_name.get(location)
        if place is None:
            return listings[listings["location"] == location]

        distances = distance_from_coordinates(
            listings, place["lat"], place["lon"]
        )
        nearby = listings.assign(distance_km=distances)
        return nearby[nearby["distance_km"] <= radius_km]

    def list_location(self, location, category=None, radius_km=25):
        listings = self.df
        listings = self._filter_location(listings, location, radius_km)
        if category and category != "All places":
            listings = listings[
                listings["category"].str.contains(category, regex=False)
            ]
        if "distance_km" in listings:
            listings = listings.sort_values(["distance_km", "name"])
        else:
            listings = listings.sort_values("name")
        return listings.reset_index(drop=True)

    def recommend(
        self, query, location=None, top_k=None, category=None, radius_km=25
    ):
        top_k = top_k or self.cfg["retrieval"]["top_k"]
        requested_food = match_catalog_term(query, FOOD_ALIASES)
        service_catalog = (
            HOTEL_SERVICE_ALIASES
            if category == ACCOMMODATION_CATEGORY
            else RESTAURANT_SERVICE_ALIASES
            if category == EATERY_CATEGORY
            else ALL_SERVICE_ALIASES
        )
        requested_service = (
            match_catalog_term(query, service_catalog)
            if requested_food is None
            else None
        )
        if requested_food or requested_service:
            hits = self.df
            hits = self._filter_location(hits, location, radius_km)
            if requested_food:
                has_eatery = hits["amenity"].ne("")
                has_lodging = hits["tourism"].ne("")
                if category == ACCOMMODATION_CATEGORY:
                    hits = hits[has_lodging]
                elif category == EATERY_CATEGORY:
                    hits = hits[has_eatery]
                else:
                    hits = hits[has_eatery | has_lodging]
                target_aliases = tuple(
                    _normalize_search_text(alias)
                    for alias in FOOD_ALIASES[requested_food]
                )
                matching_cuisine = hits["cuisine"].map(
                    lambda cuisine: any(
                        alias in _normalize_search_text(cuisine)
                        for alias in target_aliases
                    )
                ) & hits["amenity"].ne("")
                exact_hits = hits[matching_cuisine].copy()
                search_type = (
                    "cuisine_tag_candidate"
                    if not exact_hits.empty
                    else "unverified_food_candidate"
                )
                hits = exact_hits if not exact_hits.empty else hits
            else:
                has_eatery = hits["amenity"].ne("")
                has_lodging = hits["tourism"].ne("")
                if category == ACCOMMODATION_CATEGORY:
                    hits = hits[has_lodging]
                elif category == EATERY_CATEGORY:
                    hits = hits[has_eatery]
                else:
                    hits = hits[has_eatery | has_lodging]
                service_aliases = tuple(
                    _normalize_search_text(alias)
                    for alias in service_catalog[requested_service]
                )
                exact_hits = hits[
                    hits["services_recorded"].map(
                        lambda services: any(
                            alias in _normalize_search_text(services)
                            for alias in service_aliases
                        )
                    )
                ].copy()
                search_type = (
                    "osm_service_tag_candidate"
                    if not exact_hits.empty
                    else "unverified_service_candidate"
                )
                hits = exact_hits if not exact_hits.empty else hits

            if hits.empty:
                return hits
            hits = hits.copy()
            hits["requested_item"] = requested_food or requested_service
            hits["search_domain"] = (
                "food"
                if requested_food
                else (
                    "hotel service"
                    if category == ACCOMMODATION_CATEGORY
                    else "restaurant service"
                    if category == EATERY_CATEGORY
                    else "place service"
                )
            )
            hits["search_match_type"] = search_type
            hits["match"] = 0.0
            hits["match_pct"] = 0.0
            order = ["name"]
            if "distance_km" in hits:
                order = ["distance_km", "name"]
            return hits.sort_values(order).head(top_k).reset_index(drop=True)

        hits = self.retrieve(query)
        hits = self._filter_location(hits, location, radius_km)
        if category and category != "All places":
            hits = hits[hits["category"].str.contains(category, regex=False)]
        if hits.empty:
            return hits

        hits["match_pct"] = (hits["match"] * 100).round(0)
        sort_columns = ["match", "name"]
        ascending = [False, True]
        if "distance_km" in hits:
            sort_columns = ["match", "distance_km", "name"]
            ascending = [False, True, True]
        return hits.sort_values(sort_columns, ascending=ascending).head(
            top_k
        ).reset_index(drop=True)

    def answer(self, query, results):
        """Explain whether results match mapped cuisine or are candidates to ask."""
        if results.empty:
            return (
                f"I couldn't find a mapped listing matching '{query}'. "
                "OpenStreetMap may not have this restaurant or cuisine recorded."
            )
        best = results.iloc[0]
        if "search_match_type" in best.index:
            requested_item = best["requested_item"]
            if best["search_match_type"] == "cuisine_tag_candidate":
                evidence = (
                    "These nearby places have a related cuisine tag in "
                    "OpenStreetMap; it does not confirm the exact dish is on "
                    "their menu."
                )
            elif best["search_match_type"] == "osm_service_tag_candidate":
                evidence = (
                    "These listings have a related service tag in "
                    "OpenStreetMap; confirm the details with the business."
                )
            else:
                evidence = (
                    "The available map data does not confirm this food or "
                    "service. These are nearby places to contact and ask."
                )
            return (
                f"Nearby {best['search_domain']} options for '{requested_item}'. "
                f"{evidence}"
            )
        cuisine = best["cuisine"]
        description = f" Cuisine recorded on the map: {cuisine}." if cuisine else ""
        listing_type = best["listing_type"].replace("_", " ")
        location_description = (
            "with no OSM address city recorded"
            if best["location"] == LOCATION_UNRECORDED
            else f"in {best['location']}"
        )
        if "distance_km" in best.index:
            location_description = (
                f"{best['distance_km']:.1f} km from the selected map locality"
            )
        return (
            f"For '{query}', {best['name']} is a mapped {listing_type} "
            f"{location_description}."
            f"{description} [OpenStreetMap: {best['osm_url']}]"
        )

    def timed_recommend(self, *args, **kwargs):
        t = time.perf_counter()
        res = self.recommend(*args, **kwargs)
        return res, time.perf_counter() - t


if __name__ == "__main__":
    rec = FoodRecommender()
    rec.save()
    res = rec.recommend("restaurant")
    print(
        res[
            [
                "name",
                "location",
                "listing_type",
                "food_served",
                "accommodation_details",
                "match_pct",
            ]
        ]
    )
    print(rec.answer("restaurant", res))
