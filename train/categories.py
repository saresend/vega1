"""Category (`choice`) questions for the vega1-cat head: "which of these?"
with one correct option, built from the same labels as the noul data.

Every kind has TRAIN and HELDOUT phrasings (held-out ones are only asked on
selection/test videos). Subjects are held out as in the noul data (UCF
classes, COCO categories, the bathroom scene, stuff), and one whole kind
(`time_of_day`) is test-only, to measure category questions the head has
never seen.

Options are plain labels (`"criteria": {"dog": null, ...}`), shuffled, with a
correct option and 1–7 distractors that the labels say are absent.
"""

import random
import re

import statements as st

# kind → (train phrasings, held-out phrasings)
PHRASINGS = {
    "activity": (st.ACTIVITY_CHOICE, st.HELDOUT_ACTIVITY_CHOICE),
    "object": (
        ["Which of these objects is in the video?", "What object appears in the video?", "Which object is visible?",
         "What can you see in the video?", "Pick the object that appears in the footage.", "Object in the video",
         "Which of these is shown?", "What thing is in this clip?"],
        ["Identify the object in the scene.", "Which item does the footage show?"],
    ),
    "group": None,  # per supercategory, below
    "supercategory": (
        ["What kind of thing is in the video?", "Which type of object appears?", "What category of object is shown?",
         "Object type", "What sort of thing can you see?"],
        ["Which kind of item does the clip show?", "Classify the thing in the video."],
    ),
    "people_count": (
        ["How many people are in the video?", "How many people are there?", "Number of people",
         "How many people can you see?", "Count the people in the video."],
        ["How many persons appear in the clip?", "What is the number of people shown?"],
    ),
    "place": (
        ["Where was this video filmed?", "What kind of place is this?", "Where does the video take place?",
         "Location", "Which setting is shown?", "Where is this?"],
        ["Identify the location of the clip.", "In what kind of place was this shot?"],
    ),
    "setting": (
        ["Is the video indoors or outdoors?", "Where is the scene: inside or outside?", "Indoor or outdoor?",
         "Is this inside a building or outside?"],
        ["Was this filmed inside or outside?", "Setting of the clip"],
    ),
    "surface": (
        ["What is on the ground?", "What is the ground made of?", "What surface can you see?", "Ground surface",
         "What covers the ground in the video?"],
        ["Which ground surface does the clip show?", "What are people or things standing on?"],
    ),
    "sport": (
        ["What sport is shown?", "Which sport is being played?", "What sport or game is this?", "Sport",
         "Which sport does the video show?"],
        ["Identify the sport in the clip.", "What sport is happening here?"],
    ),
    "video_type": (
        ["What kind of video is this?", "Is this real footage or computer graphics?", "Video type",
         "How was this video made?"],
        ["Was this clip filmed or generated?", "What type of footage is this?"],
    ),
    "color": (
        ["Is the video in color or black and white?", "What are the colors of the video?", "Color or grayscale?",
         "Colour of the footage"],
        ["Is this clip monochrome or in color?", "Which color mode does the video use?"],
    ),
    "overlay": (
        ["What text is on the screen?", "Is there text on top of the video?", "What is overlaid on the video?",
         "On-screen text"],
        ["Which overlay does the clip have?", "What appears on top of the footage?"],
    ),
    # Test-only kind: never trained on.
    "time_of_day": (
        [],
        ["What time of day is it?", "Was this filmed during the day or at night?", "Time of day",
         "Is it day or night in the video?"],
    ),
}
TEST_ONLY_KINDS = {"time_of_day"}

# Questions about one COCO supercategory: "which animal is in the video?"
GROUPS = {
    "animal": (["Which animal is in the video?", "What animal is this?", "What kind of animal appears?", "Animal"],
               ["Identify the animal in the clip.", "Which animal does the footage show?"]),
    "vehicle": (["Which vehicle is in the video?", "What vehicle is shown?", "What kind of vehicle appears?", "Vehicle"],
                ["Identify the vehicle in the clip.", "Which vehicle does the footage show?"]),
    "food": (["What food is in the video?", "Which food is shown?", "What is there to eat?", "Food"],
             ["Identify the food in the clip.", "Which food does the footage show?"]),
    "furniture": (["Which furniture is in the video?", "What piece of furniture is shown?", "Furniture"],
                  ["Identify the furniture in the clip.", "Which furniture does the room have?"]),
    "electronic": (["Which device is in the video?", "What electronic device is shown?", "Device"],
                   ["Identify the device in the clip.", "Which gadget does the footage show?"]),
    "kitchen": (["Which kitchen item is in the video?", "What tableware is shown?", "Kitchenware"],
                ["Identify the kitchen item in the clip.", "Which dish or utensil is visible?"]),
    "appliance": (["Which appliance is in the video?", "What kitchen appliance is shown?", "Appliance"],
                  ["Identify the appliance in the clip.", "Which appliance does the room have?"]),
    "accessory": (["Which accessory is in the video?", "What bag or accessory is shown?", "Accessory"],
                  ["Identify the accessory in the clip.", "Which accessory can you see?"]),
}

SUPER_LABELS = {"animal": "an animal", "vehicle": "a vehicle", "food": "food", "furniture": "furniture",
                "electronic": "an electronic device", "kitchen": "kitchenware", "sports": "sports equipment",
                "appliance": "an appliance"}
HELDOUT_SUPER_LABELS = {"electronic"}  # as an option, selection/test only

# Place → (label, attribute)
PLACES = {"kitchen": "a kitchen", "bedroom": "a bedroom", "bathroom": "a bathroom", "street": "a city street",
          "beach": "a beach", "snow": "a snowy place", "water": "on the water"}
# Surface label → (stuff name or attribute)
SURFACES = {"grass": "grass", "sand": "sand", "a road": "road", "dirt": "dirt", "a floor": "floor",
            "gravel": "gravel", "a sidewalk": "sidewalk", "snow": "attr:snow"}
# COCO sports objects → the sport they show
SPORTS = {"tennis racket": "tennis", "skis": "skiing", "snowboard": "snowboarding", "surfboard": "surfing",
          "skateboard": "skateboarding", "baseball bat": "baseball", "baseball glove": "baseball",
          "frisbee": "frisbee", "kite": "kite flying"}
PEOPLE = ["no people", "one person", "a group of people"]
OVERLAYS = {"cap": "subtitles", "wm": "a logo or watermark", None: "no subtitles or logo"}


def noun(cat: str) -> str:
    return st.COCO_NAMES.get(cat, cat)


class Choices:
    """Writes `choice` examples through a build_dataset Writer."""

    def __init__(self, w, rng: random.Random):
        self.w, self.rng = w, rng

    def ask(self, kind: str, correct: str, distractors: list[str], unseen: bool = False, phrasings=None,
            k: tuple[int, int] = (2, 6)) -> None:
        distractors = sorted(set(distractors) - {correct})
        if not distractors:
            return
        if kind in TEST_ONLY_KINDS:
            if self.w.split != "test":
                return
            unseen = True
        n = min(len(distractors), self.rng.randint(k[0], k[1]) - 1)
        options = [correct] + self.rng.sample(distractors, n)
        self.rng.shuffle(options)
        train, held = phrasings or PHRASINGS[kind]
        text, heldout_phrase = self.w.phrasing(train, held)
        self.w.add({"type": "choice", "instructions": text, "criteria": {o: None for o in options}},
                   [float(o == correct) for o in options], f"{kind}_choice", heldout_phrase, unseen)


def humanize(cls: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", cls).lower()


def ucf(c: Choices, clip: dict, classes: list[str], heldout: set[str]) -> None:
    cls = clip["cls"]
    pool = classes if c.w.split != "train" else [x for x in classes if x not in heldout]
    for _ in range(2):
        c.ask("activity", humanize(cls), [humanize(x) for x in pool], cls in heldout, k=(3, 8))
    out, inside = st.UCF_ATTRIBUTES["outdoors"][2], st.UCF_ATTRIBUTES["outdoors"][3]
    if cls in out or cls in inside:
        c.ask("setting", "outdoors" if cls in out else "indoors", ["indoors", "outdoors"])
    if cls in st.UCF_ATTRIBUTES["water"][2] and c.rng.random() < 0.5:
        c.ask("surface", "water", ["grass", "a road", "a floor", "sand", "snow", "dirt"])


def coco(c: Choices, img: dict, allowed: set[str], unseen_cats: set[str], supercat: dict) -> None:
    rng, split = c.rng, c.w.split
    present = [x for x in img["present"] if x in allowed and x != "person"]
    absent = [x for x in allowed if x not in img["seen"] and x != "person"]

    # Which object: one present category against absent ones, half of them from
    # the same supercategory when possible (hard negatives).
    for cat in rng.sample(present, min(2, len(present))):
        hard = [x for x in absent if supercat[x] == supercat[cat]]
        pool = rng.sample(hard, min(3, len(hard))) + rng.sample(absent, min(5, len(absent)))
        c.ask("object", noun(cat), [noun(x) for x in pool], cat in unseen_cats, k=(2, 8))

    # Which <animal/vehicle/...>: exactly one category of the group is present.
    for group, phrasings in GROUPS.items():
        members = [x for x in allowed if supercat[x] == group]
        here = [x for x in members if x in img["seen"]]
        if len(here) == 1 and here[0] in img["present"] and rng.random() < 0.6:
            others = [noun(x) for x in members if x not in img["seen"]]
            c.ask("group", noun(here[0]), others, here[0] in unseen_cats, phrasings=phrasings, k=(2, 6))

    # What kind of thing: one supercategory present, the others not seen at all.
    labels = {g: l for g, l in SUPER_LABELS.items() if g not in HELDOUT_SUPER_LABELS or split != "train"}
    groups_seen = {supercat[x] for x in img["seen"]}
    groups_present = [g for g in labels if any(supercat[x] == g for x in img["present"])]
    if len(groups_present) == 1 and rng.random() < 0.5:
        g = groups_present[0]
        c.ask("supercategory", labels[g], [l for h, l in labels.items() if h not in groups_seen],
              g in HELDOUT_SUPER_LABELS)

    # How many people (only clear-cut cases).
    n, big, crowd = img["persons"], img["big_persons"], img["crowd"]
    count = 0 if n == 0 else 1 if (n == 1 and not crowd and big == 1) else 2 if (big >= 3 or crowd) else None
    if count is not None and rng.random() < 0.6:
        c.ask("people_count", PEOPLE[count], PEOPLE)

    attrs = img["attributes"]
    place_true = [p for p in PLACES if attrs.get(p) is True and (p != "bathroom" or split == "test")]
    place_false = [PLACES[p] for p in PLACES if attrs.get(p) is False and (p != "bathroom" or split == "test")]
    if len(place_true) == 1:
        c.ask("place", PLACES[place_true[0]], place_false, place_true[0] == "bathroom", k=(2, 5))

    if attrs.get("indoors") is not None or attrs.get("outdoors") is not None:
        inside, outside = attrs.get("indoors"), attrs.get("outdoors")
        if inside is True and outside is not True:
            c.ask("setting", "indoors", ["outdoors"])
        elif outside is True and inside is not True:
            c.ask("setting", "outdoors", ["indoors"])

    stuff = img["stuff"]
    def surface(key):
        return attrs.get(key[5:]) if key.startswith("attr:") else stuff.get(key)
    surf_true = [s for s, key in SURFACES.items() if surface(key) is True]
    surf_false = [s for s, key in SURFACES.items() if surface(key) is False]
    if len(surf_true) == 1:
        c.ask("surface", surf_true[0], surf_false, k=(2, 5))

    sports_here = {SPORTS[x] for x in img["present"] if x in SPORTS}
    sports_seen = {SPORTS[x] for x in img["seen"] if x in SPORTS}
    if len(sports_here) == 1 and len(sports_seen) == 1:
        c.ask("sport", next(iter(sports_here)), [s for s in set(SPORTS.values()) if s not in sports_seen], k=(2, 6))

    if attrs.get("night") is True:
        c.ask("time_of_day", "night", ["day"])
    elif attrs.get("night") is False and attrs.get("outdoors") is True:
        c.ask("time_of_day", "day", ["night"])


def synth(c: Choices) -> None:
    c.ask("video_type", "computer-generated graphics", ["real camera footage"])
    c.ask("people_count", "no people", PEOPLE)


def common(c: Choices, source: str, edits: frozenset[str], gray: bool) -> None:
    """Questions about the video itself (any source)."""
    rng = c.rng
    if source != "synth" and rng.random() < 0.1:
        c.ask("video_type", "real camera footage", ["computer-generated graphics"])
    if gray or "bw" in edits or rng.random() < 0.15:
        c.ask("color", "black and white" if gray or "bw" in edits else "color", ["color", "black and white"])
    overlay = [e for e in ("cap", "wm") if e in edits]
    if len(overlay) == 1 or (not overlay and rng.random() < 0.1):
        c.ask("overlay", OVERLAYS[overlay[0] if overlay else None], list(OVERLAYS.values()))
