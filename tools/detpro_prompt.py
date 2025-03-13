import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import torch
from torch import nn
import torch.nn.functional as F





bases=(
        'person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat',
        'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra', 'giraffe',
        'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'kite',
        'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair',
        'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'microwave',
        'oven', 'toaster', 'refrigerator', 'book', 'clock', 'vase',
        'toothbrush'
    )
novels=(
    'airplane', 'bus', 'cat', 'dog', 'cow', 'elephant', 'umbrella', 'tie',
    'snowboard', 'skateboard', 'cup', 'knife', 'cake', 'couch', 'keyboard',
    'sink', 'scissors'
)

COCO_CLASSES = bases + novels

LVIS_CLASSES = (
        'aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol',
        'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna',
        'apple', 'applesauce', 'apricot', 'apron', 'aquarium',
        'arctic_(type_of_shoe)', 'armband', 'armchair', 'armoire', 'armor',
        'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer',
        'avocado', 'award', 'awning', 'ax', 'baboon', 'baby_buggy',
        'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel',
        'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon',
        'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banjo',
        'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow',
        'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap',
        'baseball_glove', 'basket', 'basketball', 'bass_horn', 'bat_(animal)',
        'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'batter_(food)',
        'battery', 'beachball', 'bead', 'bean_curd', 'beanbag', 'beanie',
        'bear', 'bed', 'bedpan', 'bedspread', 'cow', 'beef_(food)', 'beeper',
        'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt',
        'belt_buckle', 'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor',
        'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath',
        'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card',
        'pirate_flag', 'black_sheep', 'blackberry', 'blackboard', 'blanket',
        'blazer', 'blender', 'blimp', 'blinker', 'blouse', 'blueberry',
        'gameboard', 'boat', 'bob', 'bobbin', 'bobby_pin', 'boiled_egg',
        'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'bookcase',
        'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle',
        'bottle_opener', 'bouquet', 'bow_(weapon)', 'bow_(decorative_ribbons)',
        'bow-tie', 'bowl', 'pipe_bowl', 'bowler_hat', 'bowling_ball', 'box',
        'boxing_glove', 'suspenders', 'bracelet', 'brass_plaque', 'brassiere',
        'bread-bin', 'bread', 'breechcloth', 'bridal_gown', 'briefcase',
        'broccoli', 'broach', 'broom', 'brownie', 'brussels_sprouts',
        'bubble_gum', 'bucket', 'horse_buggy', 'bull', 'bulldog', 'bulldozer',
        'bullet_train', 'bulletin_board', 'bulletproof_vest', 'bullhorn',
        'bun', 'bunk_bed', 'buoy', 'burrito', 'bus_(vehicle)', 'business_card',
        'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car',
        'cabinet', 'locker', 'cake', 'calculator', 'calendar', 'calf',
        'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)',
        'can', 'can_opener', 'candle', 'candle_holder', 'candy_bar',
        'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup',
        'canteen', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino',
        'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car',
        'car_battery', 'identity_card', 'card', 'cardigan', 'cargo_ship',
        'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton',
        'cash_register', 'casserole', 'cassette', 'cast', 'cat', 'cauliflower',
        'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone',
        'chain_mail', 'chair', 'chaise_longue', 'chalice', 'chandelier',
        'chap', 'checkbook', 'checkerboard', 'cherry', 'chessboard',
        'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'chime',
        'chinaware', 'crisp_(potato_chip)', 'poker_chip', 'chocolate_bar',
        'chocolate_cake', 'chocolate_milk', 'chocolate_mousse', 'choker',
        'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cider',
        'cigar_box', 'cigarette', 'cigarette_case', 'cistern', 'clarinet',
        'clasp', 'cleansing_agent', 'cleat_(for_securing_rope)', 'clementine',
        'clip', 'clipboard', 'clippers_(for_plants)', 'cloak', 'clock',
        'clock_tower', 'clothes_hamper', 'clothespin', 'clutch_bag', 'coaster',
        'coat', 'coat_hanger', 'coatrack', 'cock', 'cockroach',
        'cocoa_(beverage)', 'coconut', 'coffee_maker', 'coffee_table',
        'coffeepot', 'coil', 'coin', 'colander', 'coleslaw',
        'coloring_material', 'combination_lock', 'pacifier', 'comic_book',
        'compass', 'computer_keyboard', 'condiment', 'cone', 'control',
        'convertible_(automobile)', 'sofa_bed', 'cooker', 'cookie',
        'cooking_utensil', 'cooler_(for_food)', 'cork_(bottle_plug)',
        'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet',
        'cornice', 'cornmeal', 'corset', 'costume', 'cougar', 'coverall',
        'cowbell', 'cowboy_hat', 'crab_(animal)', 'crabmeat', 'cracker',
        'crape', 'crate', 'crayon', 'cream_pitcher', 'crescent_roll', 'crib',
        'crock_pot', 'crossbar', 'crouton', 'crow', 'crowbar', 'crown',
        'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch',
        'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup',
        'cupboard', 'cupcake', 'hair_curler', 'curling_iron', 'curtain',
        'cushion', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'dartboard',
        'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk',
        'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table', 'tux',
        'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher',
        'dishwasher_detergent', 'dispenser', 'diving_board', 'Dixie_cup',
        'dog', 'dog_collar', 'doll', 'dollar', 'dollhouse', 'dolphin',
        'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'dove', 'dragonfly',
        'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit',
        'dresser', 'drill', 'drone', 'dropper', 'drum_(musical_instrument)',
        'drumstick', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumbbell',
        'dumpster', 'dustpan', 'eagle', 'earphone', 'earplug', 'earring',
        'easel', 'eclair', 'eel', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater',
        'eggplant', 'electric_chair', 'refrigerator', 'elephant', 'elk',
        'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon', 'fan',
        'faucet', 'fedora', 'ferret', 'Ferris_wheel', 'ferry', 'fig_(fruit)',
        'fighter_jet', 'figurine', 'file_cabinet', 'file_(tool)', 'fire_alarm',
        'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace',
        'fireplug', 'first-aid_kit', 'fish', 'fish_(food)', 'fishbowl',
        'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap',
        'flash', 'flashlight', 'fleece', 'flip-flop_(sandal)',
        'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal',
        'folding_chair', 'food_processor', 'football_(American)',
        'football_helmet', 'footstool', 'fork', 'forklift', 'freight_car',
        'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice',
        'frying_pan', 'fudge', 'funnel', 'futon', 'gag', 'garbage',
        'garbage_truck', 'garden_hose', 'gargle', 'gargoyle', 'garlic',
        'gasmask', 'gazelle', 'gelatin', 'gemstone', 'generator',
        'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture',
        'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles',
        'goldfish', 'golf_club', 'golfcart', 'gondola_(boat)', 'goose',
        'gorilla', 'gourd', 'grape', 'grater', 'gravestone', 'gravy_boat',
        'green_bean', 'green_onion', 'griddle', 'grill', 'grits', 'grizzly',
        'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet',
        'hairpin', 'halter_top', 'ham', 'hamburger', 'hammer', 'hammock',
        'hamper', 'hamster', 'hair_dryer', 'hand_glass', 'hand_towel',
        'handcart', 'handcuff', 'handkerchief', 'handle', 'handsaw',
        'hardback_book', 'harmonium', 'hat', 'hatbox', 'veil', 'headband',
        'headboard', 'headlight', 'headscarf', 'headset',
        'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet',
        'heron', 'highchair', 'hinge', 'hippopotamus', 'hockey_stick', 'hog',
        'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'hookah',
        'hornet', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce',
        'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear',
        'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate',
        'igniter', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board',
        'jacket', 'jam', 'jar', 'jean', 'jeep', 'jelly_bean', 'jersey',
        'jet_plane', 'jewel', 'jewelry', 'joystick', 'jumpsuit', 'kayak',
        'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono',
        'kitchen_sink', 'kitchen_table', 'kite', 'kitten', 'kiwi_fruit',
        'knee_pad', 'knife', 'knitting_needle', 'knob', 'knocker_(on_a_door)',
        'koala', 'lab_coat', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)',
        'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard',
        'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather',
        'legging_(clothing)', 'Lego', 'legume', 'lemon', 'lemonade', 'lettuce',
        'license_plate', 'life_buoy', 'life_jacket', 'lightbulb',
        'lightning_rod', 'lime', 'limousine', 'lion', 'lip_balm', 'liquor',
        'lizard', 'log', 'lollipop', 'speaker_(stereo_equipment)', 'loveseat',
        'machine_gun', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)',
        'mallard', 'mallet', 'mammoth', 'manatee', 'mandarin_orange', 'manger',
        'manhole', 'map', 'marker', 'martini', 'mascot', 'mashed_potato',
        'masher', 'mask', 'mast', 'mat_(gym_equipment)', 'matchbox',
        'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine',
        'melon', 'microphone', 'microscope', 'microwave_oven', 'milestone',
        'milk', 'milk_can', 'milkshake', 'minivan', 'mint_candy', 'mirror',
        'mitten', 'mixer_(kitchen_tool)', 'money',
        'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor',
        'motor_scooter', 'motor_vehicle', 'motorcycle', 'mound_(baseball)',
        'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom',
        'music_stool', 'musical_instrument', 'nailfile', 'napkin',
        'neckerchief', 'necklace', 'necktie', 'needle', 'nest', 'newspaper',
        'newsstand', 'nightshirt', 'nosebag_(for_animals)',
        'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'nutcracker',
        'oar', 'octopus_(food)', 'octopus_(animal)', 'oil_lamp', 'olive_oil',
        'omelet', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich',
        'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'inkpad',
        'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas',
        'palette', 'pan_(for_cooking)', 'pan_(metal_container)', 'pancake',
        'pantyhose', 'papaya', 'paper_plate', 'paper_towel', 'paperback_book',
        'paperweight', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol',
        'parchment', 'parka', 'parking_meter', 'parrot',
        'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport',
        'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter',
        'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'wooden_leg',
        'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box',
        'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)',
        'pepper', 'pepper_mill', 'perfume', 'persimmon', 'person', 'pet',
        'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano',
        'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow',
        'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball',
        'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)',
        'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat',
        'plate', 'platter', 'playpen', 'pliers', 'plow_(farm_equipment)',
        'plume', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)',
        'pole', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)',
        'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato',
        'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel',
        'printer', 'projectile_(weapon)', 'projector', 'propeller', 'prune',
        'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin', 'puncher',
        'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt', 'rabbit',
        'race_car', 'racket', 'radar', 'radiator', 'radio_receiver', 'radish',
        'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry', 'rat',
        'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt',
        'recliner', 'record_player', 'reflector', 'remote_control',
        'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat', 'road_map',
        'robe', 'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade',
        'rolling_pin', 'root_beer', 'router_(computer_equipment)',
        'rubber_band', 'runner_(carpet)', 'plastic_bag',
        'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin',
        'sail', 'salad', 'salad_plate', 'salami', 'salmon_(fish)',
        'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)',
        'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse',
        'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf',
        'school_bus', 'scissors', 'scoreboard', 'scraper', 'screwdriver',
        'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane',
        'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark',
        'sharpener', 'Sharpie', 'shaver_(electric)', 'shaving_cream', 'shawl',
        'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt',
        'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shot_glass',
        'shoulder_bag', 'shovel', 'shower_head', 'shower_cap',
        'shower_curtain', 'shredder_(for_paper)', 'signboard', 'silo', 'sink',
        'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole',
        'skirt', 'skullcap', 'sled', 'sleeping_bag', 'sling_(bandage)',
        'slipper_(footwear)', 'smoothie', 'snake', 'snowboard', 'snowman',
        'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'softball',
        'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon',
        'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)',
        'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'crawfish',
        'sponge', 'spoon', 'sportswear', 'spotlight', 'squid_(food)',
        'squirrel', 'stagecoach', 'stapler_(stapling_machine)', 'starfish',
        'statue_(sculpture)', 'steak_(food)', 'steak_knife', 'steering_wheel',
        'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew', 'stirrer',
        'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer',
        'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign',
        'streetlight', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl',
        'sugarcane_(plant)', 'suit_(clothing)', 'sunflower', 'sunglasses',
        'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband',
        'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword',
        'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table',
        'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag', 'taillight',
        'tambourine', 'army_tank', 'tank_(storage_vessel)',
        'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure',
        'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup',
        'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth',
        'telephone_pole', 'telephoto_lens', 'television_camera',
        'television_set', 'tennis_ball', 'tennis_racket', 'tequila',
        'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread',
        'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil',
        'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven',
        'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush',
        'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel',
        'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light',
        'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'trampoline',
        'tray', 'trench_coat', 'triangle_(musical_instrument)', 'tricycle',
        'tripod', 'trousers', 'truck', 'truffle_(chocolate)', 'trunk', 'vat',
        'turban', 'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)',
        'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn',
        'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest',
        'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture',
        'waffle', 'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick',
        'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe',
        'washbasin', 'automatic_washer', 'watch', 'water_bottle',
        'water_cooler', 'water_faucet', 'water_heater', 'water_jug',
        'water_gun', 'water_scooter', 'water_ski', 'water_tower',
        'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake',
        'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream',
        'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)',
        'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket',
        'wineglass', 'blinder_(for_horses)', 'wok', 'wolf', 'wooden_spoon',
        'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt',
        'yoke_(animal_equipment)', 'zebra', 'zucchini')

OBJ_CLASSES = ['whole entity']
 
COCO_FULL_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

OBJECT365v2_CLASSES = (
        'Person', 'Sneakers', 'Chair', 'Other Shoes', 'Hat', 'Car', 'Lamp',
        'Glasses', 'Bottle', 'Desk', 'Cup', 'Street Lights', 'Cabinet/shelf',
        'Handbag/Satchel', 'Bracelet', 'Plate', 'Picture/Frame', 'Helmet',
        'Book', 'Gloves', 'Storage box', 'Boat', 'Leather Shoes', 'Flower',
        'Bench', 'Potted Plant', 'Bowl/Basin', 'Flag', 'Pillow', 'Boots',
        'Vase', 'Microphone', 'Necklace', 'Ring', 'SUV', 'Wine Glass', 'Belt',
        'Moniter/TV', 'Backpack', 'Umbrella', 'Traffic Light', 'Speaker',
        'Watch', 'Tie', 'Trash bin Can', 'Slippers', 'Bicycle', 'Stool',
        'Barrel/bucket', 'Van', 'Couch', 'Sandals', 'Bakset', 'Drum',
        'Pen/Pencil', 'Bus', 'Wild Bird', 'High Heels', 'Motorcycle', 'Guitar',
        'Carpet', 'Cell Phone', 'Bread', 'Camera', 'Canned', 'Truck',
        'Traffic cone', 'Cymbal', 'Lifesaver', 'Towel', 'Stuffed Toy',
        'Candle', 'Sailboat', 'Laptop', 'Awning', 'Bed', 'Faucet', 'Tent',
        'Horse', 'Mirror', 'Power outlet', 'Sink', 'Apple', 'Air Conditioner',
        'Knife', 'Hockey Stick', 'Paddle', 'Pickup Truck', 'Fork',
        'Traffic Sign', 'Ballon', 'Tripod', 'Dog', 'Spoon',
        'Clock', 'Pot', 'Cow', 'Cake', 'Dinning Table', 'Sheep', 'Hanger',
        'Blackboard/Whiteboard', 'Napkin', 'Other Fish', 'Orange/Tangerine',
        'Toiletry', 'Keyboard', 'Tomato', 'Lantern',
        'Machinery Vehicle', 'Fan', 'Green Vegetables', 'Banana',
        'Baseball Glove', 'Airplane', 'Mouse', 'Train', 'Pumpkin', 'Soccer',
        'Skiboard', 'Luggage', 'Nightstand', 'Tea pot', 'Telephone', 'Trolley',
        'Head Phone', 'Sports Car', 'Stop Sign', 'Dessert', 'Scooter',
        'Stroller', 'Crane', 'Remote', 'Refrigerator', 'Oven', 'Lemon', 'Duck',
        'Baseball Bat', 'Surveillance Camera', 'Cat', 'Jug', 'Broccoli',
        'Piano', 'Pizza', 'Elephant', 'Skateboard', 'Surfboard', 'Gun',
        'Skating and Skiing shoes', 'Gas stove', 'Donut', 'Bow Tie', 'Carrot',
        'Toilet', 'Kite', 'Strawberry', 'Other Balls', 'Shovel', 'Pepper',
        'Computer Box', 'Toilet Paper', 'Cleaning Products', 'Chopsticks',
        'Microwave', 'Pigeon', 'Baseball', 'Cutting/chopping Board',
        'Coffee Table', 'Side Table', 'Scissors', 'Marker', 'Pie', 'Ladder',
        'Snowboard', 'Cookies', 'Radiator', 'Fire Hydrant', 'Basketball',
        'Zebra', 'Grape', 'Giraffe', 'Potato', 'Sausage', 'Tricycle', 'Violin',
        'Egg', 'Fire Extinguisher', 'Candy', 'Fire Truck', 'Billards',
        'Converter', 'Bathtub', 'Wheelchair', 'Golf Club', 'Briefcase',
        'Cucumber', 'Cigar/Cigarette ', 'Paint Brush', 'Pear', 'Heavy Truck',
        'Hamburger', 'Extractor', 'Extention Cord', 'Tong', 'Tennis Racket',
        'Folder', 'American Football', 'earphone', 'Mask', 'Kettle', 'Tennis',
        'Ship', 'Swing', 'Coffee Machine', 'Slide', 'Carriage', 'Onion',
        'Green beans', 'Projector', 'Frisbee',
        'Washing Machine/Drying Machine', 'Chicken', 'Printer', 'Watermelon',
        'Saxophone', 'Tissue', 'Toothbrush', 'Ice cream', 'Hotair ballon',
        'Cello', 'French Fries', 'Scale', 'Trophy', 'Cabbage', 'Hot dog',
        'Blender', 'Peach', 'Rice', 'Wallet/Purse', 'Volleyball', 'Deer',
        'Goose', 'Tape', 'Tablet', 'Cosmetics', 'Trumpet', 'Pineapple',
        'Golf Ball', 'Ambulance', 'Parking meter', 'Mango', 'Key', 'Hurdle',
        'Fishing Rod', 'Medal', 'Flute', 'Brush', 'Penguin', 'Megaphone',
        'Corn', 'Lettuce', 'Garlic', 'Swan', 'Helicopter', 'Green Onion',
        'Sandwich', 'Nuts', 'Speed Limit Sign', 'Induction Cooker', 'Broom',
        'Trombone', 'Plum', 'Rickshaw', 'Goldfish', 'Kiwi fruit',
        'Router/modem', 'Poker Card', 'Toaster', 'Shrimp', 'Sushi', 'Cheese',
        'Notepaper', 'Cherry', 'Pliers', 'CD', 'Pasta', 'Hammer', 'Cue',
        'Avocado', 'Hamimelon', 'Flask', 'Mushroon', 'Screwdriver', 'Soap',
        'Recorder', 'Bear', 'Eggplant', 'Board Eraser', 'Coconut',
        'Tape Measur/ Ruler', 'Pig', 'Showerhead', 'Globe', 'Chips', 'Steak',
        'Crosswalk Sign', 'Stapler', 'Campel', 'Formula 1 ', 'Pomegranate',
        'Dishwasher', 'Crab', 'Hoverboard', 'Meat ball', 'Rice Cooker', 'Tuba',
        'Calculator', 'Papaya', 'Antelope', 'Parrot', 'Seal', 'Buttefly',
        'Dumbbell', 'Donkey', 'Lion', 'Urinal', 'Dolphin', 'Electric Drill',
        'Hair Dryer', 'Egg tart', 'Jellyfish', 'Treadmill', 'Lighter',
        'Grapefruit', 'Game board', 'Mop', 'Radish', 'Baozi', 'Target',
        'French', 'Spring Rolls', 'Monkey', 'Rabbit', 'Pencil Case', 'Yak',
        'Red Cabbage', 'Binoculars', 'Asparagus', 'Barbell', 'Scallop',
        'Noddles', 'Comb', 'Dumpling', 'Oyster', 'Table Teniis paddle',
        'Cosmetics Brush/Eyeliner Pencil', 'Chainsaw', 'Eraser', 'Lobster',
        'Durian', 'Okra', 'Lipstick', 'Cosmetics Mirror', 'Curling',
        'Table Tennis ')

class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
    
    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):

    def __init__(self, classnames, clip_model, random_init, ctx_init = None, bg_class = False, ctx=8, cls_token_position='end'):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = ctx
        self.class_token_position = cls_token_position
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, \
            f'cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})'
        
        if random_init:
            # random init
            print('Initializing a generic context')
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = ' '.join(['X'] * n_ctx)
            print(f'Initial context: "{prompt_prefix}"')
            print(f'Number of context words (tokens): {n_ctx}')
        else:
            # use given words to initialize context vectors
            # ctx_init = "This is a photo of"
            # n_ctx = len(ctx_init.split(' '))
            # prompt = clip.tokenize(ctx_init)
            # with torch.no_grad():
            #     embedding = clip_model.token_embedding(prompt).type(dtype)
            # ctx_vectors = embedding[0, 1:1+n_ctx, :]
            # prompt_prefix = ctx_init
            print('Load context')
            ctx_vectors = ctx_init
            n_ctx = ctx_init.shape[0]
            prompt_prefix = ' '.join(['X'] * n_ctx)
            print(f'Initial context: "{prompt_prefix}"')
            print(f'Number of context words (tokens): {n_ctx}')


        

        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        # classnames = [name.replace('_', ' ') for name in classnames]
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]

        # tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        # with torch.no_grad():
        #     embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # # print('debug?? ', tokenized_prompts[:, 1+n_ctx:], embedding[:, 1+n_ctx:, :])
        # # These token vectors will be saved when in save_model(),
        # # but they should be ignored in load_model() as we want to use
        # # those computed using the current class names
        # self.register_buffer('token_prefix', embedding[:, :1, :]) # SOS
        # self.register_buffer('token_suffix', embedding[:, 1+n_ctx:, :]) # CLS, EOS
        # print("DEBUG")
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        # self.tokenized_prompts = tokenized_prompts # torch.Tensor
        # self.name_lens = name_lens

        self.clip_model = clip_model

        self.bg_class = bg_class
        bg_prompt = torch.empty(10, ctx_dim, dtype=dtype)
        nn.init.normal_(bg_prompt, std=0.02)
        self.bg_prompt = nn.Parameter(bg_prompt)

        tokenized_bg_prompt = ' '.join(['X'] * 10)#(self.n_ctx+2))
        self.tokenized_bg_prompt = clip.tokenize(tokenized_bg_prompt)
    
    def get_bg_prompt(self):
        # print(self.token_prefix.device)
        # print(self.ctx.device)
        # print(self.bg_prompt.device)
        # with torch.no_grad():
        #     embedding = self.clip_model.token_embedding(tokenized_prompts).type(dtype)
        prompt = torch.cat([
            self.token_prefix[0], # (1, dim)
            # self.ctx, # (n_ctx, dim)
            self.bg_prompt, 
            self.token_suffix[2][2:] # (*, dim)
        ], dim=0)
        return prompt, self.tokenized_bg_prompt
    
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.class_token_position == 'end':
            prompts = torch.cat([
                prefix, # (n_cls, 1, dim)
                ctx, # (n_cls, n_ctx, dim)
                suffix # (n_cls, *, dim)
            ], dim=1)
        elif self.class_token_position == 'middle':
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1, :, :]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i_half1 = ctx[i:i+1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i:i+1, half_n_ctx:, :]
                prompt = torch.cat([
                    prefix_i, # (1, 1, dim)
                    ctx_i_half1, # (1, n_ctx//2, dim)
                    class_i, # (1, name_len, dim)
                    ctx_i_half2, # (1, n_ctx//2, dim)
                    suffix_i # (1, *, dim)
                ], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == 'front':
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1, :, :]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i = ctx[i:i+1, :, :]
                prompt = torch.cat([
                    prefix_i, # (1, 1, dim)
                    class_i, # (1, name_len, dim)
                    ctx_i, # (1, n_ctx, dim)
                    suffix_i # (1, *, dim)
                ], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        
        else:
            raise ValueError
        

        if self.bg_class:
            bg_prompt, bg_token = self.get_bg_prompt()
            prompts = torch.cat([prompts, bg_prompt[None]])
            tokenized_prompts = torch.cat([self.tokenized_prompts, bg_token])

            return prompts, tokenized_prompts
        else:
            return prompts, self.tokenized_prompts
    
    def forward_for_classes(self, classes):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(len(classes), -1, -1)
        
        n_ctx = self.n_ctx
        prompt_prefix = ' '.join(['X'] * n_ctx)
        prompts = [prompt_prefix + ' ' + name + '.' for name in classes]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        # tokenized_prompts = tokenized_prompts.cuda()
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts.cuda()).type(self.clip_model.dtype)
        prefix = embedding[:, :1, :] # SOS
        suffix = embedding[:, 1+n_ctx:, :] # CLS, EOS

        prompts = torch.cat([
            prefix, # (n_cls, 1, dim)
            ctx, # (n_cls, n_ctx, dim)
            suffix # (n_cls, *, dim)
        ], dim=1)
        
        if self.bg_class:
            bg_prompt, bg_token = self.get_bg_prompt()
            prompts = torch.cat([prompts, bg_prompt[None]])
            tokenized_prompts = torch.cat([tokenized_prompts, bg_token])

        return prompts, tokenized_prompts



def get_detpro_text_embedding(class_names, prompt_learners, text_encoder):
    emb = []
    for prompt_learner in prompt_learners:
        prompts, tokenized_prompts = prompt_learner.forward_for_classes(class_names)
        text_features = text_encoder(prompts, tokenized_prompts)
        emb.append(text_features)
    emb = [x / x.norm(dim=-1, keepdim = True) for x in emb]
    emb = sum(emb)
    emb = emb / emb.norm(dim = -1, keepdim = True)
    return emb



prompt_paths = [
    '../my_OVD_get_pseudo_box/detpro_prompt/fg_bg_5_5_6_r1_prompt.pth',
    '../my_OVD_get_pseudo_box/detpro_prompt/fg_bg_5_6_7_r1_prompt.pth',
    '../my_OVD_get_pseudo_box/detpro_prompt/fg_bg_5_7_8_r1_prompt.pth',
    '../my_OVD_get_pseudo_box/detpro_prompt/fg_bg_5_8_9_r1_prompt.pth',
    '../my_OVD_get_pseudo_box/detpro_prompt/fg_bg_5_9_10_r1_prompt.pth',
]
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_text_model, _ = clip.load('CLIP_ViT-B-32.pt', device=device)
clip_text_model = clip_text_model.float()
for p in clip_text_model.parameters():
    p.requires_grad = False
text_encoder = TextEncoder(clip_text_model)
prompt_learners = nn.ModuleList()
for path in prompt_paths:
    prompt = torch.load(path, device)
    prompt_learners.append(PromptLearner(['classnames'], clip_text_model, False, prompt))
with torch.no_grad():
    names = [name.replace('_', ' ') for name in OBJ_CLASSES]
    emb = get_detpro_text_embedding(names, prompt_learners, text_encoder)
    torch.save(emb, 'obj_detpro_category_embeddings_vit-b-32.pt')


