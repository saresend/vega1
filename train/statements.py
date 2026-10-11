"""Phrasings and label tables for build_dataset.py.

Every statement kind has TRAIN phrasings and HELDOUT phrasings; held-out ones
are only ever asked on selection/test videos, to measure whether the head
learned the task rather than the templates. Kinds whose subject varies
({a} activity, {o} object, {c} caption) also hold out whole subjects
(UCF classes, COCO categories, UCF attributes) — the "unseen statement" slices.
"""

ACTIVITY_CHOICE = ["What activity is shown in the video?", "Which action is being performed?",
                   "What is happening in this video?", "What sport or activity does this video show?",
                   "What is the person doing?", "Which of these best describes the video?",
                   "What are they doing in this clip?", "Pick the activity in the footage.",
                   "What kind of activity is this?", "Classify the action in the video.",
                   "What exercise, sport or task is being done?", "Main activity"]
HELDOUT_ACTIVITY_CHOICE = ["Identify the activity in this clip.", "What does the footage show someone doing?",
                           "Which action appears here?"]

NOUL = {
    "activity": (
        ["The video shows someone doing {a}", "Someone in the video is doing {a}", "This is a video of {a}",
         "A person is doing {a}", "The activity is {a}", "This footage shows {a}", "{a}",
         "Is someone doing {a}?", "The main action in the video is {a}", "We see {a} in this video",
         "Video of people doing {a}", "Does the video show {a}?"],
        ["The clip contains {a}", "Someone is shown doing {a}", "Is this a video of {a}?"],
    ),
    "caption": (
        ["The video is captioned", "There are subtitles in the video", "Text captions are shown on screen",
         "The video has subtitles", "Does the video have captions?", "Subtitles appear at some point",
         "There is subtitle text on screen", "Is the video subtitled?", "Captions are burned into the video",
         "The video shows on-screen captions", "Words appear on screen as subtitles", "Has captions"],
        ["The clip has burned-in subtitles", "Are there captions in this clip?", "On-screen text transcribes speech"],
    ),
    "watermark": (
        ["There is a watermark", "The video has a watermark", "A channel logo is overlaid on the video",
         "There is a logo in the corner", "Is there a watermark on the video?", "Watermarked",
         "A broadcaster's logo is visible", "The footage carries a watermark or logo", "Is the video branded with a logo?",
         "There is overlaid text branding the video"],
        ["The clip shows a station bug", "Does the footage have a watermark?", "A logo is superimposed on the picture"],
    ),
    "grayscale": (
        ["The video is in black and white", "The footage is grayscale", "Is the video black and white?",
         "There is no color in the video", "Monochrome footage", "The video is black-and-white",
         "The picture has no colour", "Is this footage in grayscale?"],
        ["The clip is monochrome", "Was this filmed in black and white?", "Shades of gray only"],
    ),
    # Asked with the grayscale label flipped: a positive phrasing whose truth is "not grayscale".
    "color": (
        ["The video is in color", "The footage is in colour", "Is the video in color?", "Full-color footage"],
        ["The clip is in colour", "Is this a color video?"],
    ),
    "people_any": (
        ["There are people in the video", "Someone is in the video", "A person appears in the video",
         "There is a person", "Is anyone in the video?", "People are visible", "The video shows a person",
         "Are there any people?", "At least one person is visible", "Humans appear in the footage"],
        ["Somebody is on screen", "Does the video contain people?", "There is at least one human"],
    ),
    "people_one": (
        ["There is exactly one person", "Only one person is visible", "A single person appears",
         "Is there just one person?", "There is one person in the video", "One person, alone"],
        ["Exactly one human is shown", "Is only a single person present?"],
    ),
    "people_group": (
        ["There is a group of people", "Several people are visible", "There are at least three people",
         "Is there a crowd?", "Many people appear in the video", "A group of people is shown"],
        ["Three or more people are present", "Are there lots of people?"],
    ),
    "object": (
        ["There is {o}", "{O} is visible", "The video shows {o}", "Is there {o}?", "I can see {o} in the video",
         "{o_bare}", "Somewhere in the video there is {o}", "The scene contains {o}", "Does the video show {o}?",
         "{O} appears in the footage", "There is {o} in the video", "Can you see {o}?"],
        ["The clip features {o}", "Is {o} anywhere in the footage?", "{O} is in the scene"],
    ),
    # Negated statements, label flipped. CLIP-style text encoders largely
    # ignore "no", so the head has to learn this from the question embedding.
    "object_negated": (
        ["There is no {o_bare}", "No {o_bare} appears in the video", "The video does not show {o}",
         "There isn't {o}", "{O} is nowhere to be seen"],
        ["The clip has no {o_bare}", "Nothing in the video is {o}"],
    ),
    "caption_match": (
        ["{c}", "The video shows {c_lower}", "This video: {c_lower}", "In the video, {c_lower}", "Footage of {c_lower}"],
        ["The clip depicts {c_lower}", "We see {c_lower}"],
    ),
    "attribute": (
        ["{t}"], ["{t}"],  # attribute statements carry their own phrasings below
    ),
}

# UCF-101 class attributes, labelled only where they hold for (nearly) every
# clip of the class. Each attribute: (train phrasings, held-out phrasings,
# positive classes, negative classes); classes in neither are skipped.
# Negatives `None` means every class that is neither positive nor AMBIGUOUS.

UCF_ATTRIBUTES = {
    "water": (
        ["There is water in the video", "The video takes place in or on water", "Someone is in the water",
         "Is there water?", "The scene involves a pool, lake, river or sea"],
        ["Water is visible", "Does this happen on the water?"],
        {"BreastStroke", "CliffDiving", "Diving", "FrontCrawl", "Kayaking", "Rafting", "Rowing", "Skijet", "Surfing"},
        None,
    ),
    "ball": (
        ["There is a ball", "A ball is being used", "Someone is playing with a ball", "Is there a ball in the video?",
         "The game involves a ball"],
        ["A ball is visible", "Is a ball involved?"],
        {"Basketball", "BasketballDunk", "BaseballPitch", "Bowling", "CricketBowling", "CricketShot",
         "FieldHockeyPenalty", "GolfSwing", "JugglingBalls", "SoccerJuggling", "SoccerPenalty", "TableTennisShot",
         "TennisSwing", "VolleyballSpiking", "Billiards"},
        None,
    ),
    "animal": (
        ["There is an animal", "An animal appears in the video", "Is there an animal?", "The video includes an animal",
         "There is a horse or a dog"],
        ["Some animal is visible", "Does an animal appear?"],
        {"HorseRace", "HorseRiding", "WalkingWithDog"},
        None,
    ),
    "outdoors": (
        ["The video is outdoors", "The scene is outdoors", "This takes place outside", "Is it outdoors?",
         "The video was filmed outside", "Outdoor scene"],
        ["The setting is outside", "Is this in the open air?"],
        {"BandMarching", "BaseballPitch", "Biking", "CliffDiving", "FieldHockeyPenalty", "FrisbeeCatch", "GolfSwing",
         "HammerThrow", "HighJump", "HorseRace", "HorseRiding", "JavelinThrow", "Kayaking", "LongJump",
         "MilitaryParade", "PoleVault", "Rafting", "Rowing", "Skiing", "Skijet", "SkyDiving", "SoccerPenalty",
         "Surfing", "ThrowDiscus", "WalkingWithDog", "SkateBoarding"},
        {"ApplyEyeMakeup", "ApplyLipstick", "BalanceBeam", "BenchPress", "Billiards", "BlowDryHair", "BlowingCandles",
         "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BrushingTeeth", "CuttingInKitchen", "Fencing",
         "FloorGymnastics", "Haircut", "HeadMassage", "IceDancing", "Knitting", "Mixing", "MoppingFloor",
         "ParallelBars", "PlayingPiano", "PommelHorse", "RockClimbingIndoor", "ShavingBeard", "StillRings",
         "SumoWrestling", "TableTennisShot", "Typing", "UnevenBars", "WritingOnBoard"},
    ),
    "indoors": (
        ["The video is indoors", "This takes place inside a building", "Is it indoors?", "Indoor scene",
         "The scene is inside"],
        ["The setting is a room or hall", "Was this filmed inside?"],
        None, None,  # filled below as the complement of "outdoors"
    ),
    "sport": (
        ["Someone is playing a sport", "This is a sports video", "Is a sport being played?", "The video shows a sport",
         "An athlete is competing", "Sports footage"],
        ["Someone is doing a sport", "Is this athletics or a sport?"],
        {"Archery", "BalanceBeam", "BaseballPitch", "Basketball", "BasketballDunk", "Bowling", "BreastStroke",
         "CliffDiving", "CricketBowling", "CricketShot", "Diving", "Fencing", "FieldHockeyPenalty", "FloorGymnastics",
         "FrontCrawl", "GolfSwing", "HammerThrow", "HighJump", "HorseRace", "IceDancing", "JavelinThrow", "Kayaking",
         "LongJump", "ParallelBars", "PoleVault", "PommelHorse", "Rowing", "Rafting", "Shotput", "Skiing",
         "SoccerPenalty", "StillRings", "SumoWrestling", "Surfing", "TableTennisShot", "TennisSwing", "ThrowDiscus",
         "UnevenBars", "VolleyballSpiking", "SkateBoarding"},
        {"ApplyEyeMakeup", "ApplyLipstick", "BlowDryHair", "BlowingCandles", "BrushingTeeth", "CuttingInKitchen",
         "Haircut", "Hammering", "HeadMassage", "Knitting", "Mixing", "MoppingFloor", "Drumming", "PlayingCello",
         "PlayingDaf", "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar", "PlayingTabla",
         "PlayingViolin", "ShavingBeard", "Typing", "WritingOnBoard", "BabyCrawling", "MilitaryParade", "BandMarching",
         "WalkingWithDog"},
    ),
    "exercise": (
        ["Someone is working out", "A person is exercising", "Is someone doing a workout?", "Fitness exercise",
         "The video shows strength training or calisthenics"],
        ["Someone is training at the gym", "Is this a workout video?"],
        {"BenchPress", "BodyWeightSquats", "CleanAndJerk", "HandstandPushups", "JumpingJack", "JumpRope", "Lunges",
         "PullUps", "PushUps", "WallPushups", "BoxingPunchingBag", "BoxingSpeedBag", "RopeClimbing"},
        {"ApplyEyeMakeup", "ApplyLipstick", "BlowDryHair", "BlowingCandles", "BrushingTeeth", "CuttingInKitchen",
         "Haircut", "Hammering", "HeadMassage", "Knitting", "Mixing", "MoppingFloor", "Drumming", "PlayingCello",
         "PlayingDaf", "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar", "PlayingTabla",
         "PlayingViolin", "ShavingBeard", "Typing", "WritingOnBoard", "BabyCrawling", "MilitaryParade", "Billiards",
         "Bowling", "Archery", "GolfSwing", "HorseRiding", "WalkingWithDog", "Kayaking", "Rowing"},
    ),
    "boat": (
        ["There is a boat", "Someone is in a boat", "Is there a boat?", "A boat is visible"],
        ["A boat appears", "Is anyone in a boat?"],
        {"Kayaking", "Rafting", "Rowing"},
        None,
    ),
    # Held out: never trained on, asked only on test videos.
    "instrument": (
        [],
        ["Someone is playing a musical instrument", "Is a musical instrument being played?", "Music is being played",
         "There is a musical instrument"],
        {"Drumming", "PlayingCello", "PlayingDaf", "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano",
         "PlayingSitar", "PlayingTabla", "PlayingViolin", "BandMarching"},
        None,
    ),
    "food": (
        [],
        ["There is food in the video", "Someone is preparing food", "Is there food?", "Food is visible"],
        {"CuttingInKitchen", "Mixing", "PizzaTossing"},
        None,
    ),
}
HELDOUT_ATTRIBUTES = {"instrument", "food"}
AMBIGUOUS = {
    "water": {"Skiing", "IceDancing"},
    "ball": {"Shotput", "Punch", "BoxingSpeedBag", "PizzaTossing", "HulaHoop", "YoYo", "Nunchucks"},
    "animal": {"MilitaryParade", "BandMarching"},
    "boat": {"Skijet", "Surfing", "CliffDiving", "Diving"},
    "food": {"BlowingCandles"},
    "instrument": {"MilitaryParade", "SalsaSpin", "IceDancing"},
}

# COCO categories whose statements are never trained on: half are used for
# model selection, half only for the final test.
SELECTION_COCO_CATEGORIES = ["giraffe", "umbrella", "pizza", "laptop", "kite", "bench", "banana", "cow"]
TEST_COCO_CATEGORIES = ["zebra", "clock", "toilet", "skateboard", "suitcase", "fire hydrant", "teddy bear", "scissors"]

COCO_NAMES = {"sports ball": "ball", "dining table": "table", "tv": "television", "cell phone": "phone",
              "hair drier": "hair dryer", "mouse": "computer mouse", "remote": "remote control"}

WATERMARK_TEXTS = ["SAMPLE CHANNEL", "NEWS 24", "@daily_clips", "LIVE", "HD", "TV5", "SportsNow", "© MediaCorp",
                   "WATCH ON STREAMIT", "Channel 9", "EXCLUSIVE", "viralvids.com", "ESPN-ish", "FOX-like", "BBC Two?"]


def article(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def finish_attributes(classes: list[str]) -> None:
    """Resolves `None` negative sets to "every class not positive" (and
    indoors as outdoors flipped), given the full UCF class list."""
    for name, (train, held, pos, neg) in list(UCF_ATTRIBUTES.items()):
        if name == "indoors":
            out_pos, out_neg = UCF_ATTRIBUTES["outdoors"][2], UCF_ATTRIBUTES["outdoors"][3]
            UCF_ATTRIBUTES[name] = (train, held, set(out_neg), set(out_pos))
        elif neg is None:
            UCF_ATTRIBUTES[name] = (train, held, pos, set(classes) - pos - AMBIGUOUS.get(name, set()))


# COCO scene attributes decided by caption consensus. Each: (train phrasings,
# held-out phrasings, positive regex, negative rule). An image is positive when
# at least 2 of its ~5 captions match the positive regex; negative when no
# caption matches `avoid` (the positive regex plus related words) and, if
# `negative` is given, at least 2 captions match it.
_INDOOR = (r"\broom\b|kitchen|bathroom|bedroom|office|indoors|\binside\b|restaurant|\bdesk\b|couch|sofa|"
           r"living room|\bbed\b|counter|table")
_OUTDOOR = (r"street|\broad\b|field|beach|\bpark\b|outside|outdoors|ocean|snow|\bsky\b|grass|sidewalk|\blake\b|"
            r"river|mountain|\bski|runway|\bhill|\btrees?\b|\bcity\b|parking")

COCO_ATTRIBUTES = {
    "night": (
        ["It is nighttime", "The video was filmed at night", "Is it night?", "It's dark outside", "A night scene"],
        ["The scene takes place after dark", "Was this shot at night?"],
        r"\bnight|nighttime|at dusk|after dark",
        {"avoid": r"night|dark|dusk|evening|\blit\b|lights|sunset|lamp"},
    ),
    "snow": (
        ["There is snow", "The ground is covered in snow", "Is there snow?", "A snowy scene", "It is winter with snow"],
        ["Snow is visible", "Does the scene have snow?"],
        r"\bsnow",
        {"avoid": r"snow|\bski|winter|\bice\b|icy"},
    ),
    "beach": (
        ["The video is at a beach", "There is a beach", "Is this at the beach?", "People are on the sand by the sea"],
        ["A beach is shown", "Was this filmed on a beach?"],
        r"\bbeach|\bshore",
        {"avoid": r"beach|shore|sand|ocean|\bsea\b|surf|water|wave"},
    ),
    "water": (
        UCF_ATTRIBUTES["water"][0][:4] + ["There is a lake, river or ocean"],
        UCF_ATTRIBUTES["water"][1],
        r"\bwater\b|\bocean|\blake\b|\briver|\bsea\b|\bpool\b|\bharbor|\bwaves?\b|\bbay\b",
        {"avoid": r"water|ocean|lake|river|\bsea\b|pool|harbor|wave|boat|beach|surf|swim|shore|\bbay\b|canal|pond"},
    ),
    "street": (
        ["The video shows a city street", "There is a road", "Is this on a street?", "A street scene",
         "Cars or people on a city street"],
        ["The scene is a road in town", "Is a street visible?"],
        r"\bstreet|\broad\b|sidewalk|intersection|highway|\bcity\b|downtown",
        {"avoid": r"street|road|sidewalk|intersection|highway|city|downtown|traffic|\bcars?\b|\bbus\b|truck|town|"
                  r"parking|crosswalk"},
    ),
    "kitchen": (
        ["The video takes place in a kitchen", "There is a kitchen", "Is this a kitchen?", "A kitchen scene"],
        ["The room is a kitchen", "Was this filmed in a kitchen?"],
        r"\bkitchen",
        {"avoid": r"kitchen|stove|oven|counter|refrigerator|fridge|\bsink|cook|microwave|cabinet"},
    ),
    "bedroom": (
        ["There is a bed", "The video is in a bedroom", "Is this a bedroom?", "A bed is visible"],
        ["The room is a bedroom", "Is there a bed in the scene?"],
        r"\bbedroom|\bbed\b",
        {"avoid": r"\bbeds?\b|bedroom|hotel|sleep|pillow|blanket"},
    ),
    "bathroom": (
        [],
        ["The video is in a bathroom", "Is this a bathroom?", "There is a toilet or shower", "A bathroom scene"],
        r"\bbathroom|\btoilet|\bshower|bathtub",
        {"avoid": r"bathroom|toilet|shower|bath|\bsink|restroom|urinal"},
    ),
    "indoors": (
        UCF_ATTRIBUTES["indoors"][0], UCF_ATTRIBUTES["indoors"][1], _INDOOR,
        {"avoid": _INDOOR, "negative": _OUTDOOR},
    ),
    "outdoors": (
        UCF_ATTRIBUTES["outdoors"][0], UCF_ATTRIBUTES["outdoors"][1], _OUTDOOR,
        {"avoid": _OUTDOOR, "negative": _INDOOR},
    ),
    "sport": (
        UCF_ATTRIBUTES["sport"][0], UCF_ATTRIBUTES["sport"][1],
        r"(play|throw|hit|swing|catch|kick|serv)\w* .{0,25}(tennis|baseball|soccer|frisbee|football|basketball|"
        r"volleyball|\bball\b)|\bski(s|ing|er|ers)?\b|\bsurf(s|ing|er|ers)?\b|skateboard\w*|snowboard\w*|\btennis\b|"
        r"\bbaseball\b|\bsoccer\b",
        {"avoid": r"play|ball|tennis|baseball|soccer|frisbee|\bski|surf|skate|snowboard|game|sport|kite|racket|"
                  r"\bbat\b|athlete|match|player|team|swing|pitch|race|horse|riding"},
    ),
    "eating": (
        ["Someone is eating", "A person is eating food", "Is someone eating?", "People are having a meal"],
        ["Somebody takes a bite of food", "Is anyone eating?"],
        r"\beat(s|ing)?\b|\bbites? (into|of)|having (a meal|lunch|dinner|breakfast)",
        {"avoid": r"\beat|bite|meal|lunch|dinner|breakfast|chew|food|feed"},
    ),
    "food": (
        ["There is food in the video", "Food is visible", "Is there food?", "There is something to eat"],
        ["The scene shows food", "Is any food visible?"],
        r"\bfood|pizza|sandwich|donut|doughnut|cake|\bmeal|plate of|broccoli|vegetable|fruit|banana|apple|orange|"
        r"hot dog|dessert|salad|bowl of|bread|burger|fries|carrot",
        {"avoid": r"food|pizza|sandwich|donut|doughnut|cake|meal|plate|broccoli|vegetable|fruit|banana|apple|orange|"
                  r"hot dog|dessert|salad|bowl|bread|burger|fries|carrot|eat|lunch|dinner|breakfast|snack|kitchen|"
                  r"restaurant|cook|dish|drink|coffee|wine"},
    ),
    "sitting": (
        ["Someone is sitting", "A person is sitting down", "Is anyone seated?", "People are sitting"],
        ["Somebody is seated", "Is a person sitting?"],
        r"\bsit(s|ting)?\b|\bseated",
        {"avoid": r"\bsit|seat|bench|couch|chair|sofa", "negative": r"\bstand(s|ing)?\b|\bwalk|\brun|\bski|surf|"
                                                                    r"skateboard|\bplay|\bswing"},
    ),
    "standing": (
        ["Someone is standing", "A person is standing up", "Is anyone standing?", "People are standing"],
        ["Somebody is on their feet", "Is a person standing?"],
        r"\bstand(s|ing)?\b",
        {"avoid": r"\bstand", "negative": r"\bsit(s|ting)?\b|\bseated|\blying|\blaying|\bsleep"},
    ),
    "riding": (
        ["Someone is riding something", "A person is riding a bike, horse or board", "Is anyone riding?",
         "Someone rides a vehicle or animal"],
        ["Somebody is riding", "Is a person riding something?"],
        r"\brid(e|es|ing)\b",
        {"avoid": r"\brid|horse|bike|bicycle|motorcycle|skateboard|surf|\bski|scooter|wave"},
    ),
    "phone": (
        ["Someone is using a phone", "There is a cell phone", "Is anyone on their phone?", "A phone is visible"],
        ["Somebody holds a smartphone", "Is a mobile phone shown?"],
        r"\bphone|cellphone|smartphone|\bcell\b",
        {"avoid": r"phone|cell|smartphone|texting|\btext|call"},
    ),
    "black_and_white": (
        NOUL["grayscale"][0], NOUL["grayscale"][1],
        r"black and white (photo|image|picture|shot|photograph|scene|view|film)|(photo|image|picture|photograph) .{0,30}"
        r"black and white|monochrome|sepia|in black and white|grayscale|greyscale",
        {"avoid": r"black and white|black-and-white|monochrome|sepia|grayscale|greyscale|old photo|vintage|antique"},
    ),
}
HELDOUT_COCO_ATTRIBUTES = {"bathroom"}

# Statements about whole COCO supercategories ("there is an animal").
SUPERCATEGORIES = {
    "animal": (["There is an animal", "An animal appears in the video", "Is there an animal?", "The video shows an animal"],
               ["Some animal is visible", "Does an animal appear?"]),
    "vehicle": (["There is a vehicle", "A vehicle is visible", "Is there a vehicle?", "The video shows a car, bus, bike or other vehicle"],
                ["Some kind of vehicle appears", "Are there any vehicles?"]),
    "furniture": (["There is furniture", "Furniture is visible", "Is there a chair, couch, bed or table?"],
                  ["Some furniture appears", "Does the room have furniture?"]),
    "electronic": ([], ["There is an electronic device", "A screen or computer is visible", "Is there a TV, laptop or phone?"]),
}
HELDOUT_SUPERCATEGORIES = {"electronic"}

# "Stuff" from COCO panoptic segmentation: background things that aren't COCO
# object categories. Name → (noun phrase, plural?, panoptic categories).
STUFF = {
    "grass": ("grass", False, ["grass-merged"]),
    "trees": ("trees", True, ["tree-merged"]),
    "building": ("a building", False, ["building-other-merged", "house"]),
    "sky": ("the sky", False, ["sky-other-merged"]),
    "road": ("a road", False, ["road"]),
    "sidewalk": ("a sidewalk", False, ["pavement-merged"]),
    "mountains": ("mountains", True, ["mountain-merged"]),
    "dirt": ("dirt", False, ["dirt-merged"]),
    "rocks": ("rocks", True, ["rock-merged"]),
    "sand": ("sand", False, ["sand"]),
    "fence": ("a fence", False, ["fence-merged"]),
    "ceiling": ("a ceiling", False, ["ceiling-merged"]),
    "wall": ("a wall", False, ["wall-brick", "wall-stone", "wall-tile", "wall-wood", "wall-other-merged"]),
    "floor": ("a floor", False, ["floor-wood", "floor-other-merged"]),
    "window": ("a window", False, ["window-blind", "window-other"]),
    "cabinets": ("cabinets", True, ["cabinet-merged"]),
    "curtains": ("curtains", True, ["curtain"]),
    "door": ("a door", False, ["door-stuff"]),
    "stairs": ("stairs", True, ["stairs"]),
    "bridge": ("a bridge", False, ["bridge"]),
    "railroad": ("railroad tracks", True, ["railroad"]),
    "flowers": ("flowers", True, ["flower"]),
    "roof": ("a roof", False, ["roof"]),
    "tent": ("a tent", False, ["tent"]),
    "sports field": ("a sports field", False, ["playingfield"]),
    "mirror": ("a mirror", False, ["mirror-stuff"]),
    "pillow": ("a pillow", False, ["pillow"]),
    "towel": ("a towel", False, ["towel"]),
    "blanket": ("a blanket", False, ["blanket"]),
    "shelves": ("shelves", True, ["shelf"]),
    "counter": ("a counter", False, ["counter"]),
    "rug": ("a rug", False, ["rug-merged"]),
    "gravel": ("gravel", False, ["gravel"]),
    "banner": ("a banner", False, ["banner"]),
    "net": ("a net", False, ["net"]),
}
SELECTION_STUFF = ["fence", "flowers", "bridge", "curtains"]
TEST_STUFF = ["mountains", "rug", "tent", "stairs"]

NOUL["stuff_plural"] = (
    ["There are {o}", "{O} are visible", "The video shows {o}", "Are there {o}?", "I can see {o} in the video",
     "{o_bare}", "Somewhere in the video there are {o}", "The scene contains {o}", "{O} appear in the footage"],
    ["The clip features {o}", "Are {o} anywhere in the footage?", "{O} are in the scene"],
)
NOUL["stuff_plural_negated"] = (
    ["There are no {o}", "No {o} appear in the video", "The video does not show {o}", "There aren't any {o}"],
    ["The clip has no {o}", "{O} are nowhere to be seen"],
)

# Synthetic graphics (fractals, cellular automata, test patterns, noise): true
# for "graphics" statements, false for the "camera" ones, and a negative for
# everything about the real world.
NOUL["graphics"] = (
    ["The video is computer-generated graphics", "This is an animation or abstract pattern",
     "The video shows a test pattern or abstract graphics", "Is this computer graphics?", "Abstract animated pattern",
     "The video is a screen of synthetic graphics"],
    ["The clip is a digital animation", "Is this rendered rather than filmed?"],
)
NOUL["camera"] = (
    ["This is real camera footage", "The video was filmed with a camera", "Is this footage of the real world?",
     "A real scene filmed by a camera"],
    ["The clip shows the real world", "Was this shot on camera?"],
)
