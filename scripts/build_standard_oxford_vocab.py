import json
import re
import sqlite3
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
STORAGE_DIR = BASE_DIR / "backend" / "storage"
VOCAB_JSON = STORAGE_DIR / "vocab_2000.json"
ANHVIET_TXT = STORAGE_DIR / "anhviet109K.txt"
DB_PATH = BASE_DIR / "data" / "learn_english.sqlite3"

# Curated definitions for missing Oxford 3000 words & modern essentials
CURATED_ADDITIONS = {
    "a": {
        "word": "a",
        "vietnamese_meaning": "một (mạo từ không xác định)",
        "definition": "used when referring to someone or something for the first time in a text or conversation",
        "part_of_speech": "article",
        "synonyms": ["one", "an"],
        "antonyms": [],
        "examples": ["I bought a book yesterday.", "She is a doctor."]
    },
    "i": {
        "word": "I",
        "vietnamese_meaning": "tôi, mình (đại từ nhân xưng ngôi thứ nhất)",
        "definition": "used by a speaker to refer to himself or herself",
        "part_of_speech": "pronoun",
        "synonyms": ["myself"],
        "antonyms": ["you"],
        "examples": ["I love learning English.", "I am writing in the air."]
    },
    "app": {
        "word": "app",
        "vietnamese_meaning": "ứng dụng (phần mềm)",
        "definition": "an application, especially as downloaded by a user to a mobile device",
        "part_of_speech": "noun",
        "synonyms": ["application", "software", "program"],
        "antonyms": [],
        "examples": ["You can download the app for free.", "This app helps you learn English vocabulary."]
    },
    "blog": {
        "word": "blog",
        "vietnamese_meaning": "trang blog, nhật ký trực tuyến",
        "definition": "a regularly updated website or web page, typically one run by an individual or small group",
        "part_of_speech": "noun",
        "synonyms": ["weblog", "online journal", "website"],
        "antonyms": [],
        "examples": ["She writes a daily travel blog.", "I read his latest blog post."]
    },
    "cannot": {
        "word": "cannot",
        "vietnamese_meaning": "không thể",
        "definition": "can not; unable to",
        "part_of_speech": "verb",
        "synonyms": ["can't", "unable to"],
        "antonyms": ["can", "able to"],
        "examples": ["I cannot attend the meeting today.", "She cannot swim very well."]
    },
    "dvd": {
        "word": "DVD",
        "vietnamese_meaning": "đĩa DVD",
        "definition": "a type of compact disc able to store large amounts of data, especially high-resolution audiovisual material",
        "part_of_speech": "noun",
        "synonyms": ["disc", "video disc"],
        "antonyms": [],
        "examples": ["We watched an interesting movie on DVD.", "He collected hundreds of music DVDs."]
    },
    "girlfriend": {
        "word": "girlfriend",
        "vietnamese_meaning": "bạn gái, người yêu",
        "definition": "a regular female companion with whom a person has a romantic or sexual relationship",
        "part_of_speech": "noun",
        "synonyms": ["partner", "sweetheart"],
        "antonyms": ["boyfriend"],
        "examples": ["He went to the cinema with his girlfriend.", "Her girlfriend lives in London."]
    },
    "grandparent": {
        "word": "grandparent",
        "vietnamese_meaning": "ông bà",
        "definition": "a parent of one's father or mother",
        "part_of_speech": "noun",
        "synonyms": ["grandmother", "grandfather"],
        "antonyms": ["grandchild"],
        "examples": ["Every summer we visit our grandparents in the countryside.", "He is a proud grandparent of three children."]
    },
    "lifestyle": {
        "word": "lifestyle",
        "vietnamese_meaning": "lối sống, phong cách sống",
        "definition": "the way in which a person or group lives",
        "part_of_speech": "noun",
        "synonyms": ["way of life", "living standard", "habits"],
        "antonyms": [],
        "examples": ["Regular exercise contributes to a healthy lifestyle.", "Urban lifestyles can be very fast-paced."]
    },
    "smartphone": {
        "word": "smartphone",
        "vietnamese_meaning": "điện thoại thông minh",
        "definition": "a mobile phone that performs many of the functions of a computer, typically having a touchscreen interface",
        "part_of_speech": "noun",
        "synonyms": ["mobile phone", "cellphone"],
        "antonyms": [],
        "examples": ["Most people check their smartphone first thing in the morning.", "She bought a new smartphone with a high-resolution camera."]
    },
    "website": {
        "word": "website",
        "vietnamese_meaning": "trang web",
        "definition": "a set of related web pages located under a single domain name",
        "part_of_speech": "noun",
        "synonyms": ["site", "webpage", "portal"],
        "antonyms": [],
        "examples": ["Visit our official website for more information.", "This website offers online English courses."]
    },
    "t-shirt": {
        "word": "T-shirt",
        "vietnamese_meaning": "áo thun, áo phông",
        "definition": "a casual shirt with short sleeves and no collar",
        "part_of_speech": "noun",
        "synonyms": ["shirt", "tee"],
        "antonyms": [],
        "examples": ["He wore a simple white T-shirt and jeans.", "The event gave out free souvenir T-shirts."]
    },
    "o'clock": {
        "word": "o'clock",
        "vietnamese_meaning": "giờ (chỉ giờ đúng)",
        "definition": "used after a number from one to twelve to say what time it is",
        "part_of_speech": "adverb",
        "synonyms": ["sharp"],
        "antonyms": [],
        "examples": ["The class starts at nine o'clock sharp.", "It is already five o'clock in the afternoon."]
    },
    "old-fashioned": {
        "word": "old-fashioned",
        "vietnamese_meaning": "cổ hủ, lỗi thời, cổ điển",
        "definition": "in or according to styles or types no longer current; not modern",
        "part_of_speech": "adjective",
        "synonyms": ["outdated", "vintage", "traditional"],
        "antonyms": ["modern", "trendy"],
        "examples": ["She prefers old-fashioned furniture made of solid wood.", "His views on technology are a bit old-fashioned."]
    },
    "part-time": {
        "word": "part-time",
        "vietnamese_meaning": "bán thời gian",
        "definition": "employed for or involving less than the standard amount of working hours",
        "part_of_speech": "adjective",
        "synonyms": ["temporary", "fractional"],
        "antonyms": ["full-time"],
        "examples": ["Many university students have part-time jobs.", "She works part-time at a local bookstore."]
    },
    "long-term": {
        "word": "long-term",
        "vietnamese_meaning": "dài hạn, lâu dài",
        "definition": "occurring over or involving a relatively long period of time",
        "part_of_speech": "adjective",
        "synonyms": ["enduring", "lasting", "permanent"],
        "antonyms": ["short-term", "temporary"],
        "examples": ["Investing in education brings long-term benefits.", "They are planning a long-term strategy for business growth."]
    },
    "amazed": {
        "word": "amazed",
        "vietnamese_meaning": "kinh ngạc, sửng sốt",
        "definition": "greatly surprised; astonished",
        "part_of_speech": "adjective",
        "synonyms": ["astonished", "astounded", "surprised"],
        "antonyms": ["unimpressed", "bored"],
        "examples": ["I was amazed by the beauty of the landscape.", "The audience was amazed by his brilliant performance."]
    },
    "bored": {
        "word": "bored",
        "vietnamese_meaning": "buồn chán, tẻ nhạt",
        "definition": "feeling weary and restless through lack of interest",
        "part_of_speech": "adjective",
        "synonyms": ["uninterested", "tired", "weary"],
        "antonyms": ["interested", "excited", "fascinated"],
        "examples": ["The lecture was so long that the students got bored.", "If you are bored, you can read a book."]
    },
    "shocked": {
        "word": "shocked",
        "vietnamese_meaning": "bàng hoàng, choáng váng",
        "definition": "feeling surprised and upset by something unexpected and unpleasant",
        "part_of_speech": "adjective",
        "synonyms": ["stunned", "appalled", "startled"],
        "antonyms": ["calm", "unfazed"],
        "examples": ["We were shocked to hear the sad news.", "She was too shocked to speak at that moment."]
    },
    "impressed": {
        "word": "impressed",
        "vietnamese_meaning": "có ấn tượng sâu sắc, khâm phục",
        "definition": "feeling admiration and respect for someone or something",
        "part_of_speech": "adjective",
        "synonyms": ["admiring", "appreciative"],
        "antonyms": ["unimpressed"],
        "examples": ["The interviewer was very impressed by her skills.", "I was deeply impressed by the hospitality of the locals."]
    },
    "prepared": {
        "word": "prepared",
        "vietnamese_meaning": "chuẩn bị sẵn sàng",
        "definition": "ready to do or deal with something",
        "part_of_speech": "adjective",
        "synonyms": ["ready", "equipped", "organized"],
        "antonyms": ["unprepared"],
        "examples": ["She felt fully prepared for the final examination.", "Be prepared for unexpected changes in the weather."]
    },
    "dressed": {
        "word": "dressed",
        "vietnamese_meaning": "mặc quần áo",
        "definition": "wearing clothes, especially of a specified kind",
        "part_of_speech": "adjective",
        "synonyms": ["clothed", "attired"],
        "antonyms": ["undressed", "naked"],
        "examples": ["He was neatly dressed in a dark suit.", "Get dressed quickly, we are leaving soon."]
    },
    "driving": {
        "word": "driving",
        "vietnamese_meaning": "việc lái xe, sự điều khiển xe",
        "definition": "the control and operation of a motor vehicle",
        "part_of_speech": "noun",
        "synonyms": ["motoring", "steering"],
        "antonyms": [],
        "examples": ["Careful driving prevents traffic accidents.", "He passed his driving test on the first attempt."]
    },
    "spending": {
        "word": "spending",
        "vietnamese_meaning": "việc chi tiêu, ngân sách chi tiêu",
        "definition": "the amount of money paid out or expended",
        "part_of_speech": "noun",
        "synonyms": ["expenditure", "outlay", "costs"],
        "antonyms": ["saving", "income"],
        "examples": ["Government spending on healthcare increased this year.", "She keeps a careful record of her monthly spending."]
    },
    "strongly": {
        "word": "strongly",
        "vietnamese_meaning": "một cách mạnh mẽ, kiên quyết",
        "definition": "with great power, determination, or conviction",
        "part_of_speech": "adverb",
        "synonyms": ["firmly", "vigorously", "powerfully"],
        "antonyms": ["weakly", "mildly"],
        "examples": ["I strongly agree with your proposal.", "The local community strongly opposed the construction project."]
    },
    "located": {
        "word": "located",
        "vietnamese_meaning": "tọa lạc, nằm ở vị trí",
        "definition": "situated in a particular place or position",
        "part_of_speech": "adjective",
        "synonyms": ["situated", "positioned", "placed"],
        "antonyms": [],
        "examples": ["The hotel is conveniently located near the city center.", "Our campus is located by the river."]
    },
    "matching": {
        "word": "matching",
        "vietnamese_meaning": "phù hợp, ăn khớp, đồng điệu",
        "definition": "corresponding in color, pattern, or style",
        "part_of_speech": "adjective",
        "synonyms": ["harmonious", "corresponding", "coordinating"],
        "antonyms": ["clashing", "mismatched"],
        "examples": ["She wore a matching dress and hat.", "The curtains and carpet have matching patterns."]
    },
    "arms": {
        "word": "arms",
        "vietnamese_meaning": "vũ khí, súng ống / cánh tay (số nhiều)",
        "definition": "weapons; arms, or the upper limbs of the human body",
        "part_of_speech": "noun",
        "synonyms": ["weapons", "firearms"],
        "antonyms": [],
        "examples": ["The treaty called for a reduction in conventional arms.", "She held the baby gently in her arms."]
    },
    "based": {
        "word": "based",
        "vietnamese_meaning": "dựa trên, có trụ sở tại",
        "definition": "having as a basis, foundation, or headquarters",
        "part_of_speech": "adjective",
        "synonyms": ["grounded", "headquartered", "rooted"],
        "antonyms": [],
        "examples": ["The movie is based on a true story.", "The technology company is based in Silicon Valley."]
    },
    "should": {
        "word": "should",
        "vietnamese_meaning": "nên (lời khuyên, nghĩa vụ)",
        "definition": "used to indicate what is proper, advisable, or desirable",
        "part_of_speech": "verb",
        "synonyms": ["ought to", "must"],
        "antonyms": [],
        "examples": ["You should eat more vegetables for good health.", "Students should review their lessons regularly."]
    },
    "would": {
        "word": "would",
        "vietnamese_meaning": "sẽ (trong điều kiện hoặc quá khứ), muốn",
        "definition": "used to express conditional statements, willingness, or past habitual actions",
        "part_of_speech": "verb",
        "synonyms": ["will"],
        "antonyms": [],
        "examples": ["I would love to help you.", "If I had time, I would learn Spanish."]
    }
}

# High utility Oxford Core words to round out to 3,000 - 3,500
OXFORD_HIGH_UTILITY = [
    ("academic", "học thuật, mang tính học thuật", "relating to education and scholarship", "adjective", ["scholarly", "educational"], ["practical"], ["He has an impressive academic background.", "Academic writing requires clear references."]),
    ("academy", "học viện, viện hàn lâm", "an institution of secondary or higher education", "noun", ["institute", "school"], [], ["He graduated from the military academy.", "The language academy promotes English literacy."]),
    ("accelerate", "tăng tốc, thúc đẩy nhanh", "to increase in speed or rate of progress", "verb", ["speed up", "quicken", "hasten"], ["decelerate", "slow down"], ["Technology helps accelerate economic growth.", "The car began to accelerate down the highway."]),
    ("acceptance", "sự chấp nhận, sự công nhận", "the action of consenting to receive or undertake something", "noun", ["approval", "consent", "recognition"], ["rejection", "refusal"], ["She received an acceptance letter from the university.", "Self-acceptance is important for mental well-being."]),
    ("accessible", "dễ tiếp cận, có thể truy cập được", "able to be reached or entered easily", "adjective", ["reachable", "available", "approachable"], ["inaccessible", "remote"], ["The website is accessible on all mobile devices.", "Public buildings must be accessible to people with disabilities."]),
    ("accidentally", "tình cờ, ngẫu nhiên", "by chance; unintentionally", "adverb", ["unintentionally", "by mistake", "incidentally"], ["intentionally", "deliberately"], ["I accidentally deleted the document.", "They met accidentally at the airport."]),
    ("accommodate", "cung cấp chỗ ở, đáp ứng", "to provide lodging or sufficient space for; adapt to", "verb", ["house", "lodge", "adapt"], ["reject"], ["The hotel can accommodate up to 500 guests.", "We must accommodate the needs of our customers."]),
    ("accomplish", "hoàn thành, đạt được", "to achieve or complete successfully", "verb", ["achieve", "complete", "fulfill"], ["fail", "abandon"], ["You can accomplish anything with hard work.", "She accomplished her childhood dream of becoming a pilot."]),
    ("accomplishment", "thành tựu, sự hoàn thành", "something that has been achieved successfully", "noun", ["achievement", "success", "triumph"], ["failure"], ["Graduating with honors was a great accomplishment.", "The team celebrated their latest technical accomplishment."]),
    ("accordance", "sự phù hợp, sự tuân theo", "conformity or agreement with a rule or standard", "noun", ["conformity", "compliance", "agreement"], ["conflict", "disagreement"], ["The project was completed in accordance with the regulations.", "Everything proceeded in accordance with our original plan."]),
    ("accountability", "trách nhiệm giải trình", "the fact or condition of being accountable; responsibility", "noun", ["responsibility", "liability", "answerability"], ["irresponsibility"], ["Public officials must demonstrate high accountability.", "Clear accountability improves workplace performance."]),
    ("accountant", "kế toán viên", "a person whose job is to keep or inspect financial accounts", "noun", ["bookkeeper", "auditor"], [], ["Our company hired an experienced accountant.", "The accountant checked all financial statements."]),
    ("accumulate", "tích lũy, gom góp", "to gather together or acquire an increasing number or quantity", "verb", ["collect", "gather", "amass"], ["disperse", "spend"], ["Dust tends to accumulate on bookshelves.", "She accumulated rich experience over ten years of work."]),
    ("accuracy", "độ chính xác", "the quality or state of being correct or precise", "noun", ["precision", "exactness", "correctness"], ["inaccuracy", "error"], ["The air-writing model recognizes characters with high accuracy.", "Please check the accuracy of these figures."]),
    ("accurate", "chính xác, đúng đắn", "correct in all details; exact", "adjective", ["precise", "exact", "correct"], ["inaccurate", "wrong"], ["We need accurate data for the scientific report.", "Her translation was remarkably accurate."]),
    ("accurately", "một cách chính xác", "in a way that is correct in all details", "adverb", ["precisely", "correctly", "exactly"], ["inaccurately", "wrongly"], ["The AI model can accurately detect hand gestures.", "He described the incident very accurately."]),
    ("activate", "kích hoạt, làm hoạt động", "to make something active or operative", "verb", ["trigger", "enable", "start"], ["deactivate", "disable"], ["Press the button to activate the camera.", "You must activate your account via email."]),
    ("adaptation", "sự thích nghi, sự chuyển thể", "the process of change by which an organism or system becomes better suited to its environment", "noun", ["adjustment", "modification", "transformation"], [], ["Living abroad requires cultural adaptation.", "The film is an adaptation of a famous novel."]),
    ("addiction", "sự nghiện ngập, thói nghiện", "the fact or condition of being addicted to a particular substance, thing, or activity", "noun", ["dependence", "obsession", "habit"], [], ["Smartphone addiction affects concentration.", "Exercise can help overcome addiction."]),
    ("additionally", "ngoài ra, thêm vào đó", "as an extra factor or circumstance; furthermore", "adverb", ["furthermore", "moreover", "in addition"], [], ["Additionally, the software supports multiple languages.", "She teaches English and, additionally, writes articles."]),
    ("adequate", "đầy đủ, thỏa đáng", "satisfactory or acceptable in quality or quantity", "adjective", ["sufficient", "enough", "satisfactory"], ["inadequate", "insufficient"], ["Make sure you get adequate sleep every night.", "The salary is adequate to cover daily living expenses."]),
    ("adhere", "tuân thủ, dính chặt vào", "to stick firmly to a surface or follow rules closely", "verb", ["stick", "comply", "follow"], ["violate", "separate"], ["All employees must adhere to the safety policy.", "The label will adhere well to clean plastic."]),
    ("adjacent", "liền kề, kề bên", "next to or adjoining something else", "adjective", ["neighboring", "adjoining", "next-door"], ["distant", "remote"], ["We booked two adjacent rooms in the hotel.", "The library is adjacent to the science laboratory."]),
    ("adjust", "điều chỉnh, sửa cho hợp", "to alter or move slightly in order to achieve the desired fit, appearance, or result", "verb", ["modify", "adapt", "tune"], [], ["You can adjust the volume using the slider.", "It took time to adjust to the cold climate."]),
    ("adjustment", "sự điều chỉnh", "a small alteration or movement made to achieve a desired fit, appearance, or result", "noun", ["modification", "alteration", "adaptation"], [], ["He made a minor adjustment to the camera angle.", "Starting university requires major lifestyle adjustments."]),
    ("administer", "quản lý, điều hành, cấp phát", "to manage and be responsible for the running of a business, organization, or system", "verb", ["manage", "direct", "govern"], [], ["The department will administer the scholarship program.", "Nurses administer medication according to doctor instructions."]),
    ("administrative", "thuộc về hành chính", "relating to the running of a business, organization, etc.", "adjective", ["managerial", "organizational", "official"], [], ["She works in the university administrative office.", "Administrative costs have been reduced this quarter."]),
    ("administrator", "người quản trị, nhà quản lý", "a person responsible for running a business, organization, etc.", "noun", ["manager", "director", "coordinator"], [], ["Contact the system administrator if you forget your password.", "The hospital administrator announced new guidelines."]),
    ("admission", "sự nhận vào, vé vào cửa, sự thừa nhận", "the process or fact of entering or being allowed to enter a place, organization, or institution", "noun", ["entry", "entrance", "acceptance"], ["exclusion"], ["University admission requirements have become more competitive.", "Admission to the museum is free on Sundays."]),
    ("adolescent", "thanh thiếu niên", "a young person in the process of developing from a child into an adult", "noun", ["teenager", "youth"], ["adult"], ["Many adolescents face academic pressure.", "Adolescent psychology is a fascinating field of study."]),
    ("adoption", "sự nhận nuôi, sự áp dụng", "the action of adopting a child, idea, or policy", "noun", ["taking up", "acceptance", "embracing"], ["rejection"], ["The rapid adoption of AI is changing education.", "They completed the adoption process after two years."]),
    ("advocate", "người ủng hộ / ủng hộ, tán thành", "a person who publicly supports or recommends a particular cause or policy", "noun", ["supporter", "promoter", "champion"], ["opponent"], ["She is a passionate advocate for environmental protection.", "Many teachers advocate interactive learning methods."]),
    ("aesthetic", "thẩm mỹ, có tính thẩm mỹ", "concerned with beauty or the appreciation of beauty", "adjective", ["artistic", "visual", "tasteful"], ["ugly"], ["The user interface has a clean and modern aesthetic.", "The building combines functionality with aesthetic appeal."]),
    ("affection", "tình cảm, sự yêu thương", "a gentle feeling of fondness or liking", "noun", ["love", "fondness", "care"], ["hatred", "dislike"], ["He showed great affection towards his younger sister.", "Cats show affection by purring and rubbing against you."]),
    ("affordable", "giá cả phải chăng, hợp túi tiền", "inexpensive; reasonably priced", "adjective", ["economical", "budget-friendly", "reasonable"], ["expensive", "costly"], ["The store offers stylish and affordable clothing.", "Finding affordable housing in big cities can be difficult."]),
    ("agency", "cơ quan, đại lý, hãng", "a business or organization providing a specific service on behalf of another", "noun", ["organization", "firm", "bureau"], [], ["She booked her flight through a local travel agency.", "The environmental protection agency issued a warning."]),
    ("agenda", "chương trình nghị sự, lịch trình", "a list of items to be discussed at a formal meeting", "noun", ["schedule", "plan", "program"], [], ["Education reform is high on the government's agenda.", "What is the next item on today's meeting agenda?"]),
    ("aggregate", "tổng hợp, gộp lại", "a whole formed by combining several separate elements", "noun", ["total", "sum", "collection"], ["individual", "part"], ["The aggregate score of both teams was very close.", "Online platforms aggregate news from diverse sources."]),
    ("agriculture", "nông nghiệp", "the science or practice of farming", "noun", ["farming", "cultivation", "agronomy"], [], ["Modern agriculture relies on automation and sensors.", "A large percentage of the population works in agriculture."]),
    ("aircraft", "máy bay, phi cơ", "an airplane, helicopter, or other machine capable of flight", "noun", ["airplane", "aeroplane", "jet"], [], ["The aircraft landed safely despite heavy rain.", "Engineers are designing more fuel-efficient aircraft."]),
    ("alarm", "báo động, chuông báo thức", "an anxious awareness of danger; a warning sound", "noun", ["warning", "alert", "siren"], [], ["Set your alarm for six in the morning.", "The fire alarm went off unexpectedly."]),
    ("algorithm", "thuật toán", "a process or set of rules to be followed in calculations or problem-solving operations", "noun", ["procedure", "formula", "computation"], [], ["The search engine uses a sophisticated ranking algorithm.", "Machine learning algorithms learn patterns from training data."]),
    ("alien", "người ngoài hành tinh, xa lạ", "belonging to a foreign country or nation; unfamiliar", "adjective", ["foreign", "extraterrestrial", "unfamiliar"], ["native", "familiar"], ["The concept of remote work was completely alien to them ten years ago.", "Scientists search for signs of alien life in space."]),
    ("alliance", "liên minh, sự liên kết", "a union or association formed for mutual benefit", "noun", ["partnership", "coalition", "league"], ["division", "rivalry"], ["The two tech companies formed a strategic alliance.", "The international alliance aims to tackle climate change."]),
    ("allocate", "phân bổ, cấp phát", "to distribute resources or duties for a particular purpose", "verb", ["assign", "allot", "distribute"], ["withhold"], ["The council agreed to allocate more funds to public schools.", "Please allocate sufficient time for project testing."]),
    ("allowance", "tiền trợ cấp, tiền tiêu vặt, định mức", "the amount of something that is permitted, especially within a set of regulations", "noun", ["pocket money", "stipend", "allocation"], [], ["His parents gave him a small weekly allowance.", "The baggage allowance for domestic flights is 20 kilograms."]),
    ("alongside", "bên cạnh, cùng với", "close to the side of; together with", "preposition", ["beside", "together with", "next to"], ["apart"], ["A new bicycle lane was constructed alongside the highway.", "She worked alongside experienced researchers in the lab."]),
    ("alter", "thay đổi, biến đổi", "to change or cause to change in character or composition", "verb", ["change", "modify", "adjust"], ["preserve", "maintain"], ["Nothing can alter the facts of what happened.", "We had to alter our travel plans due to bad weather."]),
    ("alternative", "sự lựa chọn thay thế, phương án khác", "one of two or more available possibilities", "noun", ["option", "choice", "substitute"], [], ["Solar power is a clean alternative to fossil fuels.", "We had no alternative but to cancel the event."]),
    ("amateur", "nghiệp dư, người không chuyên", "engaging in a pursuit on an unpaid rather than a professional basis", "adjective", ["non-professional", "beginner", "hobbyist"], ["professional", "expert"], ["He is an enthusiastic amateur photographer.", "The tournament is open to both amateur and professional athletes."]),
    ("ambassador", "đại sứ", "an accredited diplomat sent by a country as its official representative", "noun", ["diplomat", "envoy", "representative"], [], ["She was appointed as the new ambassador to France.", "He acts as a cultural ambassador for his hometown."]),
    ("amendment", "sự sửa đổi, tu chính án", "a minor change or addition designed to improve a text, piece of legislation, etc.", "noun", ["revision", "modification", "alteration"], [], ["The senate approved an amendment to the labor law.", "We proposed several amendments to the project contract."]),
    ("amid", "ở giữa, trong bối cảnh", "surrounded by; in the middle of", "preposition", ["among", "in the middle of", "surrounded by"], [], ["The festival opened amid bright sunshine and cheering crowds.", "She remained calm amid the widespread panic."]),
    ("analogy", "sự so sánh tương tự, phép loại suy", "a comparison between two things, typically for the purpose of explanation or clarification", "noun", ["comparison", "parallel", "similarity"], ["difference"], ["The professor used a simple analogy to explain quantum physics.", "Drawing an analogy between the brain and a computer is common."]),
    ("ancestor", "tổ tiên, ông bà cụ kỵ", "a person from whom one is descended", "noun", ["forefather", "predecessor"], ["descendant"], ["Our ancestors settled in this coastal village centuries ago.", "DNA testing can reveal clues about your ancient ancestors."]),
    ("anchor", "mỏ neo / người dẫn chương trình / điểm tựa", "a heavy metal device used to moor a ship; a reliable pillar", "noun", ["mooring", "pillar", "presenter"], [], ["The ship dropped anchor in the tranquil bay.", "Her family was an emotional anchor during difficult times."]),
    ("animation", "hoạt hình, sự sôi nổi sinh động", "the technique of photographing successive drawings to create an illusion of movement", "noun", ["cartoon", "motion graphics", "liveliness"], [], ["Computer animation has advanced rapidly in recent years.", "Her face was full of warmth and animation as she spoke."]),
    ("anniversary", "ngày kỷ niệm", "the date on which an event took place in a previous year", "noun", ["commemoration", "jubilee"], [], ["Today is their tenth wedding anniversary.", "The university celebrated its 50th anniversary last month."]),
    ("anonymous", "ẩn danh, không nêu tên", "not identified by name; of unknown name", "adjective", ["unnamed", "nameless", "unidentified"], ["identified", "named"], ["The donor wished to remain completely anonymous.", "She received an anonymous letter of appreciation."]),
    ("anticipate", "dự đoán, mong đợi trước", "to regard as probable; expect or predict", "verb", ["expect", "foresee", "predict"], ["doubt"], ["We anticipate that sales will increase significantly next quarter.", "Drivers should anticipate hazards on slippery roads."]),
    ("anxiety", "sự lo âu, mối băn khoăn", "a feeling of worry, nervousness, or unease about something with an uncertain outcome", "noun", ["worry", "concern", "nervousness"], ["calmness", "peace", "serenity"], ["Deep breathing exercises can help reduce anxiety before exams.", "There is growing anxiety about rising living expenses."]),
    ("apology", "lời xin lỗi", "a regretful acknowledgement of an offence or failure", "noun", ["excuse", "regret", "confession"], [], ["He offered a sincere apology for his late arrival.", "You owe your classmate an apology for that remark."]),
    ("apparel", "trang phục, quần áo", "clothing of a specified kind", "noun", ["clothing", "garments", "attire"], [], ["Sportswear and athletic apparel are sold on the second floor.", "She launched an eco-friendly apparel brand."]),
    ("appealing", "hấp dẫn, lôi cuốn", "attractive or interesting", "adjective", ["attractive", "inviting", "alluring"], ["unappealing", "repulsive"], ["The offer of flexible working hours sounded very appealing.", "Bright colors make the mobile interface appealing to children."]),
    ("appetite", "sự thèm ăn, khẩu vị", "a natural desire to satisfy a bodily need, especially for food", "noun", ["hunger", "craving", "taste"], [], ["A long brisk walk always gives me a hearty appetite.", "The patient slowly regained his normal appetite."]),
    ("applaud", "vỗ tay tán thưởng, ca ngợi", "to show approval or praise by clapping", "verb", ["clap", "cheer", "praise"], ["boo", "criticize"], ["The audience stood up to applaud the wonderful performance.", "Environmental groups applaud the ban on single-use plastics."]),
    ("applicable", "có thể áp dụng được, thích hợp", "relevant or appropriate", "adjective", ["relevant", "appropriate", "fitting"], ["inapplicable", "irrelevant"], ["These safety rules are applicable to all laboratory experiments.", "Fill in the form and write N/A where not applicable."]),
    ("applicant", "người nộp đơn, ứng viên", "a person who makes a formal application for something, especially a job", "noun", ["candidate", "jobseeker", "bidder"], [], ["Over two hundred applicants applied for the managerial position.", "Each applicant must submit a CV and a cover letter."]),
    ("appoint", "bổ nhiệm, chỉ định", "to assign a job or role to someone", "verb", ["designate", "nominate", "assign"], ["dismiss", "discharge"], ["The board voted to appoint her as chief executive officer.", "They appointed a special committee to investigate the matter."]),
    ("appreciation", "sự cảm kích, sự đánh giá cao, sự hiểu biết", "recognition and enjoyment of the good qualities of someone or something", "noun", ["gratitude", "thankfulness", "recognition"], ["disregard", "ungratefulness"], ["I would like to express my sincere appreciation for your guidance.", "Art classes foster an appreciation for creative culture."]),
    ("approximate", "xấp xỉ, gần đúng", "close to the actual, but not completely accurate or exact", "adjective", ["estimated", "rough", "near"], ["exact", "precise"], ["Can you give me an approximate cost for the home repair?", "The approximate travel time is around forty minutes."]),
    ("arbitrary", "tùy tiện, tùy ý", "based on random choice or personal whim, rather than any reason or system", "adjective", ["random", "capricious", "whimsical"], ["systematic", "reasoned"], ["The selection of team leaders seemed completely arbitrary.", "Traffic laws are established to prevent arbitrary decisions."]),
    ("architect", "kiến trúc sư", "a person who designs buildings and in many cases also supervises their construction", "noun", ["designer", "builder", "planner"], [], ["The renowned architect designed the new city library.", "She studied five years to qualify as a licensed architect."]),
    ("architecture", "kiến trúc, cấu trúc kiến trúc", "the art or practice of designing and constructing buildings", "noun", ["design", "structure", "framework"], [], ["Ancient Roman architecture is famous for arches and domes.", "The software architecture ensures high scalability and security."]),
    ("archive", "kho lưu trữ, tài liệu lưu trữ", "a collection of historical documents or records providing information about a place, institution, or group", "noun", ["records", "repository", "files"], [], ["The historic photographs are preserved in the national archive.", "Users can easily browse the email archive by date."]),
    ("arena", "đấu trường, vũ đài", "a level area surrounded by seats for sports or other public entertainment", "noun", ["stadium", "amphitheater", "coliseum"], [], ["The concert was held in a packed sports arena.", "He entered the political arena with ambitious reform proposals."]),
    ("arguably", "có thể cho rằng, được xem là", "it may be argued (used to qualify the statement being made)", "adverb", ["possibly", "plausibly", "likely"], [], ["English is arguably the most influential global lingua franca.", "This smartphone is arguably the best value on the market today."]),
    ("armor", "áo giáp, lớp bọc bảo vệ", "the metal coverings formerly worn by soldiers or warriors to protect the body in battle", "noun", ["protective suit", "shield", "shielding"], [], ["Medieval knights wore heavy steel armor into battle.", "The tank is protected by thick composite armor."]),
    ("arrow", "mũi tên, dấu mũi tên", "a weapon consisting of a shaft with a pointed head, or a symbol used to show direction", "noun", ["shaft", "pointer", "indicator"], [], ["Follow the green arrows on the floor to reach the exit.", "He shot an arrow straight into the bullseye."]),
    ("artificial", "nhân tạo, do con người tạo ra", "made or produced by human beings rather than occurring naturally", "adjective", ["synthetic", "man-made", "simulated"], ["natural", "genuine"], ["Artificial intelligence is transforming many industries.", "The flower arrangement is made of high quality artificial silk."]),
    ("artistic", "thuộc về nghệ thuật, có khiếu nghệ thuật", "having or revealing natural creative skill", "adjective", ["creative", "imaginative", "expressive"], ["uncreative"], ["She has exceptional artistic talent in oil painting.", "The film received praise for its stunning artistic visuals."]),
    ("artwork", "tác phẩm nghệ thuật", "works of art, especially when considered collectively", "noun", ["masterpiece", "creation", "exhibit"], [], ["The gallery displays priceless artwork from the Renaissance.", "All graphic artwork for the application was designed in-house."]),
    ("ash", "tro, tàn tro", "the powdery residue left after the burning of a substance", "noun", ["cinders", "residue", "dust"], [], ["Volcanic ash blanketed surrounding villages after the eruption.", "The fireplace was cleaned and all wood ash removed."]),
    ("aside", "sang một bên, qua một bên", "to one side; out of the way", "adverb", ["to the side", "away", "apart"], [], ["Step aside and allow the ambulance to pass through.", "She put aside some money every month for emergency expenses."]),
    ("aspire", "khao khát, hướng tới", "to direct one's hopes or ambitions towards achieving something", "verb", ["aim", "desire", "seek"], [], ["Many young coders aspire to create their own tech startups.", "We aspire to deliver the highest quality education for everyone."]),
    ("assault", "sự tấn công, công kích", "a physical attack on someone", "noun", ["attack", "strike", "onslaught"], ["defense"], ["The security guards protected the diplomat from physical assault.", "Cyber assaults on critical infrastructure are increasing."]),
    ("assemble", "tập hợp, lắp ráp", "to gather together in one place for a common purpose; fit together parts", "verb", ["gather", "build", "put together"], ["disassemble", "scatter"], ["Students assemble in the main auditorium for morning announcements.", "It took two hours to assemble the wooden bookshelf."]),
    ("assembly", "sự lắp ráp, hội đồng, cuộc tụ họp", "a group of people gathered together in one place for a common purpose; fitting parts together", "noun", ["gathering", "meeting", "construction"], [], ["The automated assembly line produces one car every minute.", "The national assembly debated the new economic policies."]),
    ("assert", "khẳng định, quả quyết", "to state a fact or belief confidently and forcefully", "verb", ["declare", "claim", "maintain"], ["deny", "refute"], ["She asserted her innocence during the investigation.", "The manager asserted the company's commitment to quality."]),
    ("assertion", "sự khẳng định, lời quả quyết", "a confident and forceful statement of fact or belief", "noun", ["declaration", "claim", "statement"], ["denial"], ["His assertion that the project was on time proved false.", "Evidence was presented to support her scientific assertion."]),
    ("asset", "tài sản, vốn quý", "a useful or valuable quality, person, or thing; property", "noun", ["property", "resource", "advantage"], ["liability"], ["Her fluency in three languages is a valuable asset to our company.", "Good health is the greatest asset anyone can possess."]),
    ("assign", "giao việc, phân công", "to designate someone or something for a specific duty or purpose", "verb", ["allocate", "delegate", "give"], [], ["The teacher will assign homework at the end of class.", "They assigned three senior engineers to fix the urgent bug."]),
    ("assignment", "bài tập, nhiệm vụ được giao", "a task or piece of work allocated to someone as part of a job or course of study", "noun", ["task", "homework", "duty"], [], ["Submit your programming assignment before midnight on Friday.", "His overseas assignment in Tokyo lasted two years."]),
    ("assistant", "trợ lý, người phụ tá", "a person who ranks below a senior person and assists in work", "noun", ["aide", "helper", "associate"], [], ["The research assistant compiled data from previous experiments.", "The virtual assistant answered common customer questions."]),
    ("associate", "liên kết, cộng tác, liên tưởng", "to connect someone or something with something else in one's mind", "verb", ["connect", "link", "relate"], ["dissociate", "separate"], ["People often associate warm sunshine with happiness.", "He is proud to associate with such dedicated colleagues."]),
    ("association", "hiệp hội, sự liên kết", "an organization of persons having a common interest or purpose", "noun", ["organization", "society", "connection"], [], ["She became a member of the International Bar Association.", "There is a strong association between good nutrition and longevity."]),
    ("assume", "giả định, cho rằng, đảm nhận", "to suppose to be the case, without proof; take on a duty", "verb", ["suppose", "presume", "undertake"], ["prove"], ["I assume you have already completed the preliminary survey.", "The vice president will assume leadership during the transition."]),
    ("assumption", "giả định, điều mặc định", "a thing that is accepted as true or as certain to happen, without proof", "noun", ["supposition", "premise", "belief"], [], ["We must question the basic assumptions behind this theory.", "His assumption was that the train would arrive on schedule."]),
    ("assurance", "sự bảo đảm, sự tự tin", "a positive declaration intended to give confidence; a promise", "noun", ["guarantee", "promise", "confidence"], ["doubt", "uncertainty"], ["The manager gave full assurance that jobs were secure.", "She answered all interview questions with quiet assurance."]),
    ("assure", "cam đoan, bảo đảm với ai", "to tell someone something positively or confidently to dispel any doubts", "verb", ["guarantee", "promise", "convince"], [], ["I assure you that your personal information is completely confidential.", "Doctors assured the family that the patient would recover."]),
    ("astonishing", "đáng kinh ngạc, kỳ diệu", "extremely surprising or impressive; amazing", "adjective", ["amazing", "astounding", "incredible"], ["ordinary", "unremarkable"], ["The team made astonishing progress in just three weeks.", "She demonstrated an astonishing talent for foreign languages."]),
    ("athlete", "vận động viên", "a person who is proficient in sports and other forms of physical exercise", "noun", ["sportsperson", "player", "gymnast"], [], ["Top Olympic athletes train several hours every single day.", "He was a gifted athlete in both track and soccer."]),
    ("athletic", "khỏe khoắn, thuộc về điền kinh", "physically strong, fit, and active", "adjective", ["fit", "muscular", "sporty"], ["inactive", "unfit"], ["She has a lean and athletic build from swimming.", "The school organized an exciting athletic competition."]),
    ("atmosphere", "bầu không khí, khí quyển", "the envelope of gases surrounding the earth or another planet; the prevailing tone or mood", "noun", ["air", "ambiance", "environment"], [], ["The restaurant offers delicious food and a romantic atmosphere.", "Greenhouse gases are trapped in the Earth's atmosphere."]),
    ("attach", "đính kèm, gắn vào", "to fasten, join, or connect something to something else", "verb", ["fasten", "connect", "affix"], ["detach", "separate"], ["Please attach your resume when applying for this position.", "Attach the label securely to the parcel before mailing."]),
    ("attachment", "tệp đính kèm, sự gắn bó", "an extra document or file sent with an email message; a feeling of affection", "noun", ["file", "enclosure", "affection"], [], ["Did you receive the email attachment I sent earlier?", "Children form a strong emotional attachment to their parents."]),
    ("attain", "đạt được, giành được", "to succeed in achieving something that one desires and has worked for", "verb", ["achieve", "reach", "accomplish"], ["lose", "fail"], ["With persistent practice, you will attain fluency in English.", "She attained top grades across all subjects."]),
    ("attempt", "nỗ lực, cố gắng, thử sức", "an act of trying to achieve something", "noun", ["effort", "try", "endeavor"], [], ["His first attempt to climb the mountain was successful.", "Never give up on your first attempt when learning a skill."]),
    ("attendance", "sự có mặt, sự tham dự, sĩ số", "the action or state of attending; the number of people present", "noun", ["presence", "turnout", "participation"], ["absence"], ["Regular attendance at lectures is mandatory for all students.", "Record attendance was reported at the technology conference."]),
    ("attorney", "luật sư", "a person, typically a lawyer, appointed to act for another in business or legal matters", "noun", ["lawyer", "counsel", "advocate"], [], ["She consulted an experienced attorney before signing the contract.", "The district attorney presented key evidence in court."]),
    ("attribute", "thuộc tính, đặc điểm / quy cho", "a quality or feature regarded as a characteristic or inherent part of someone or something", "noun", ["characteristic", "feature", "trait"], [], ["Patience is a crucial attribute of an effective teacher.", "Scientists attribute climate warming to greenhouse gas emissions."]),
    ("auction", "buổi đấu giá", "a public sale in which goods or property are sold to the highest bidder", "noun", ["sale", "bidding"], [], ["The historic painting was sold at an international auction.", "Online auctions allow people around the world to place bids."]),
    ("audit", "kiểm toán, sự kiểm tra sổ sách", "an official inspection of an individual's or organization's accounts", "noun", ["inspection", "examination", "review"], [], ["The accounting firm conducts an annual financial audit.", "The security audit revealed several vulnerabilities in the server."]),
    ("authentic", "chính thực, đích thực, chân thật", "of undisputed origin; genuine", "adjective", ["genuine", "real", "original"], ["fake", "counterfeit"], ["This local restaurant serves authentic Italian pizza.", "Historians confirmed that the ancient document was authentic."]),
    ("author", "tác giả, người sáng tác", "a writer of a book, article, or document", "noun", ["writer", "novelist", "creator"], [], ["The author spent three years researching and writing the book.", "She is the author of several best-selling mystery novels."]),
    ("authority", "thẩm quyền, chính quyền, chuyên gia", "the power or right to give orders, make decisions, and enforce obedience", "noun", ["power", "command", "official"], [], ["Local authorities opened shelters during the flood.", "He is a recognized authority on Vietnamese linguistics."]),
    ("authorize", "ủy quyền, cho phép", "to give official permission for or approval to an undertaking or agent", "verb", ["permit", "sanction", "approve"], ["forbid", "ban"], ["Only the director can authorize expenditures over $1000.", "The law authorizes police officers to inspect suspicious vehicles."]),
    ("automatic", "tự động, tự giác", "working by itself with little or no direct human control", "adjective", ["automated", "mechanical", "spontaneous"], ["manual"], ["The doors are automatic and open when you approach.", "Regular saving should become an automatic habit."]),
    ("autonomy", "quyền tự chủ, sự độc lập", "the right or condition of self-government or independence", "noun", ["independence", "freedom", "self-rule"], ["dependence"], ["Teachers appreciate having autonomy in choosing curriculum materials.", "The region was granted administrative autonomy."]),
    ("availability", "sự sẵn có, tính khả dụng", "the quality of being able to be used or obtained", "noun", ["accessibility", "readiness", "presence"], ["unavailability", "scarcity"], ["Check the availability of tickets before traveling to the station.", "System availability remained above 99.9% throughout the year."]),
    ("available", "có sẵn, rảnh rỗi", "able to be used or obtained; free to do something", "adjective", ["accessible", "free", "unoccupied"], ["unavailable", "busy"], ["Is this seat available, or is someone sitting here?", "The doctor is available for consultations on Wednesday morning."]),
    ("average", "trung bình, bình thường", "constituting the result obtained by adding several quantities together and dividing; standard", "adjective", ["mean", "medium", "standard"], ["exceptional", "unusual"], ["The average temperature in summer is around 30 degrees.", "His grades were well above the school average."]),
    ("avoid", "tránh, né tránh", "to keep away from or stop oneself from doing something", "verb", ["evade", "dodge", "prevent"], ["confront", "face"], ["You should avoid eating too much sugar before bed.", "Plan your route carefully to avoid heavy rush-hour traffic."]),
    ("await", "chờ đợi, đón đợi", "to wait for an event, person, or opportunity", "verb", ["wait for", "expect", "anticipate"], [], ["Exciting career opportunities await hardworking graduates.", "A large crowd gathered to await the arrival of the train."]),
    ("awake", "thức giấc, tỉnh táo", "not asleep", "adjective", ["conscious", "alert", "waking"], ["asleep", "sleeping"], ["I stayed awake all night finishing the project report.", "The baby was wide awake and smiling happily."]),
    ("award", "giải thưởng, phần thưởng", "a prize or other mark of recognition given in honour of an achievement", "noun", ["prize", "honor", "trophy"], [], ["She won an award for the best research paper in her field.", "The ceremony gave awards to outstanding community volunteers."]),
    ("aware", "nhận thức được, biết rõ", "having knowledge or perception of a situation or fact", "adjective", ["conscious", "mindful", "informed"], ["unaware", "ignorant"], ["Are you aware of the new traffic regulations downtown?", "She became aware of someone walking softly behind her."]),
    ("awareness", "sự nhận thức, hiểu biết", "knowledge or perception of a situation or fact", "noun", ["consciousness", "understanding", "recognition"], ["ignorance"], ["The campaign aims to raise public awareness about mental health.", "Environmental awareness has grown significantly among youth."]),
    ("away", "đi xa, cách xa", "at a distance from a particular place, person, or thing", "adverb", ["distant", "elsewhere", "off"], ["near"], ["The bus stop is just a five-minute walk away.", "She looked away to hide her tears."]),
    ("awful", "khủng khiếp, tồi tệ", "very bad or unpleasant", "adjective", ["terrible", "dreadful", "horrible"], ["wonderful", "great", "excellent"], ["We experienced awful weather with rain throughout the vacation.", "I had an awful headache and went to sleep early."]),
    ("awkward", "ngượng ngùng, lúng túng, vụng về", "causing or feeling embarrassment or inconvenience; clumsy", "adjective", ["embarrassing", "clumsy", "uncomfortable"], ["graceful", "comfortable"], ["There was an awkward silence when nobody knew what to answer.", "The box was awkward to carry due to its unusual shape."]),
    ("barrier", "rào cản, chướng ngại vật", "a fence or other obstacle that prevents movement or access", "noun", ["obstacle", "hurdle", "blockade"], ["opening", "gateway"], ["Language should never be a barrier to friendship.", "Security guards placed barriers around the stage."]),
    ("boost", "thúc đẩy, nâng cao", "to help or encourage something to increase or improve", "verb", ["increase", "enhance", "promote"], ["decrease", "harm"], ["Regular exercise can boost your immune system and mood.", "The marketing campaign helped boost retail sales."]),
    ("browse", "duyệt qua, lướt xem", "to survey goods for sale or read text casually; look through web pages", "verb", ["skim", "scan", "look through"], [], ["Feel free to browse our catalogue while waiting.", "I usually browse tech news while drinking morning coffee."]),
    ("capability", "khả năng, năng lực", "the power or ability to do something", "noun", ["ability", "capacity", "potential"], ["inability"], ["The new smartphone model has impressive camera capabilities.", "Our team has the capability to finish the project on schedule."]),
    ("challenge", "thử thách, thách thức", "a task or situation that tests someone's abilities", "noun", ["difficulty", "test", "trial"], ["ease"], ["Learning hand air-writing presents an exciting technical challenge.", "She welcomed every new challenge in her career."]),
    ("collaborate", "hợp tác, phối hợp làm việc", "to work jointly on an activity or project", "verb", ["cooperate", "team up", "partner"], ["compete"], ["Researchers collaborate across countries to find medical cures.", "Designers and engineers must collaborate closely on user experience."]),
    ("communication", "giao tiếp, sự truyền thông", "the imparting or exchanging of information by speaking, writing, or other media", "noun", ["interaction", "conversation", "transmission"], [], ["Effective communication is essential in any successful team.", "Digital communication connects people across different continents."]),
    ("community", "cộng đồng", "a group of people living in the same place or having a particular characteristic in common", "noun", ["society", "public", "neighborhood"], [], ["The school serves the local community with free evening classes.", "Online developer communities share open-source code freely."]),
    ("component", "thành phần, bộ phận cấu thành", "a part or element of a larger whole, especially a machine or system", "noun", ["part", "element", "constituent"], ["whole"], ["Every component of the application has been thoroughly tested.", "Microchips are vital components in modern electronics."]),
    ("comprehensive", "toàn diện, bao quát", "including or dealing with all or nearly all elements or aspects of something", "noun", ["complete", "thorough", "exhaustive"], ["partial", "incomplete"], ["The report provides a comprehensive overview of AI advancements.", "Students received comprehensive training before starting work."]),
    ("conclusion", "kết luận, phần kết thúc", "the end or finish of an event, process, or text; a judgment reached by reasoning", "noun", ["end", "verdict", "summary"], ["beginning", "introduction"], ["In conclusion, learning vocabulary daily yields remarkable results.", "The jury reached a unanimous conclusion after lengthy debate."]),
    ("confident", "tự tin", "feeling or showing certainty about something or confidence in one's abilities", "adjective", ["self-assured", "certain", "positive"], ["insecure", "shy"], ["She was confident that her team would win the coding contest.", "Practice speaking aloud to feel more confident in conversations."]),
    ("connection", "sự kết nối, mối liên hệ", "a relationship in which a person, thing, or idea is linked or associated with something else", "noun", ["link", "relationship", "bond"], ["disconnection"], ["The internet connection remained fast and stable throughout the call.", "There is a direct connection between effort and success."]),
    ("constant", "liên tục, không đổi, bất biến", "occurring continuously over a period of time; remaining the same", "adjective", ["continuous", "persistent", "steady"], ["variable", "intermittent"], ["She maintained constant communication with her team members.", "Change is the only constant thing in our dynamic world."]),
    ("content", "nội dung / hài lòng", "the things that are held or included in something; in a state of peaceful happiness", "noun", ["material", "subject matter", "satisfied"], [], ["The course content covers both fundamental grammar and daily vocabulary.", "He felt completely content with his quiet life in the countryside."]),
    ("contribution", "sự đóng góp, cống hiến", "a gift or payment to a common fund or collection; work done to help", "noun", ["donation", "offering", "input"], [], ["Her scientific contribution helped develop effective vaccines.", "Every small contribution counts toward protecting the environment."]),
    ("convenient", "tiện lợi, thuận tiện", "fitting in well with a person's needs, activities, or plans", "adjective", ["handy", "suitable", "practical"], ["inconvenient", "cumbersome"], ["Online shopping is extremely convenient for busy workers.", "The supermarket is in a very convenient location near my house."]),
    ("creative", "sáng tạo", "relating to or involving the use of the imagination or original ideas to create something", "adjective", ["inventive", "imaginative", "original"], ["uncreative", "dull"], ["He found a creative solution to the complicated engineering problem.", "Creative writing allows students to express unique ideas freely."]),
    ("critical", "quan trọng, mang tính quyết định / phản biện", "expressing adverse or disapproving comments; expressing or involving an analysis; crucial", "adjective", ["crucial", "essential", "analytical"], ["unimportant", "minor"], ["Critical thinking is one of the most valued skills today.", "Your timely feedback is critical to our project's success."]),
    ("database", "cơ sở dữ liệu", "a structured set of data held in a computer, especially one that is accessible in various ways", "noun", ["data bank", "repository", "archive"], [], ["The vocabulary database contains over 3,000 standard Oxford entries.", "All user credentials are encrypted before saving in the database."]),
    ("definition", "định nghĩa, lời giải thích", "a statement of the exact meaning of a word, especially in a dictionary", "noun", ["meaning", "explanation", "interpretation"], [], ["Look up the precise definition of the word in a learner's dictionary.", "High-definition video provides exceptional picture clarity."]),
    ("demonstrate", "chứng minh, minh họa, thể hiện", "to clearly show the existence or truth of something by giving proof or evidence", "verb", ["show", "prove", "display"], ["disprove"], ["The experiments clearly demonstrate the validity of our hypothesis.", "The instructor demonstrated how to use the air-canvas feature."]),
    ("device", "thiết bị, dụng cụ", "a thing made or adapted for a particular purpose, especially a piece of mechanical or electronic equipment", "noun", ["gadget", "instrument", "machine"], [], ["Turn on your webcam device to start writing characters in the air.", "Keep electronic devices away from water and direct heat."]),
    ("digital", "kỹ thuật số", "relating to, using, or storing data or information in the form of numerical digits", "adjective", ["electronic", "computerized", "virtual"], ["analog"], ["Digital literacy is essential for modern education and employment.", "We captured high-resolution digital photographs during the trip."]),
    ("display", "hiển thị, trưng bày, màn hình", "to put something in a prominent place in order that it may readily be seen", "verb", ["show", "exhibit", "present"], ["hide", "conceal"], ["The screen will display the recognized word and its Vietnamese translation.", "A brilliant fireworks display illuminated the night sky."]),
    ("effective", "hiệu quả, có tác dụng", "successful in producing a desired or intended result", "adjective", ["efficient", "productive", "potent"], ["ineffective", "useless"], ["Flashcards are an effective tool for memorizing new vocabulary.", "The government enacted effective measures to control inflation."]),
    ("efficient", "hiệu suất cao, tiết kiệm thời gian", "achieving maximum productivity with minimum wasted effort or expense", "adjective", ["productive", "streamlined", "economical"], ["inefficient", "wasteful"], ["The new algorithm is significantly more efficient than the previous version.", "Public transit offers an efficient way to travel across town."]),
    ("element", "yếu tố, thành phần", "an essential or characteristic part of something abstract", "noun", ["component", "factor", "part"], [], ["Visual feedback is a key element of interactive learning software.", "Iron and oxygen are fundamental chemical elements."]),
    ("encourage", "khuyến khích, động viên", "to give support, confidence, or hope to someone", "verb", ["support", "inspire", "motivate"], ["discourage", "dissuade"], ["Teachers should encourage students to ask questions freely.", "Her parents always encouraged her interest in science."]),
    ("environment", "môi trường, hoàn cảnh xung quanh", "the surroundings or conditions in which a person, animal, or plant lives or operates", "noun", ["surroundings", "habitat", "ecosystem"], [], ["We must work together to protect the natural environment.", "A quiet room provides an ideal learning environment."]),
    ("essential", "thiết yếu, cốt yếu", "absolutely necessary; extremely important", "adjective", ["necessary", "vital", "crucial"], ["optional", "unnecessary"], ["Regular practice is essential for mastering any foreign language.", "Water and nutrition are essential for human survival."]),
    ("evidence", "bằng chứng, chứng cứ", "the available body of facts or information indicating whether a belief or proposition is true or valid", "noun", ["proof", "testimony", "facts"], ["refutation"], ["There is compelling scientific evidence that sleep improves memory.", "The detective searched the room thoroughly for any trace of evidence."]),
    ("exercise", "bài tập, sự tập luyện", "activity requiring physical effort, carried out to sustain health; a practice task", "noun", ["workout", "practice", "training"], ["inactivity"], ["Daily physical exercise improves cardiovascular health.", "Complete the grammar exercises at the end of the chapter."]),
    ("experience", "kinh nghiệm, trải nghiệm", "practical contact with and observation of facts or events; knowledge gained over time", "noun", ["knowledge", "practice", "adventure"], ["inexperience"], ["She has five years of experience in computer vision research.", "Traveling alone was a transformative life experience."]),
    ("experiment", "thí nghiệm, cuộc thử nghiệm", "a scientific procedure undertaken to make a discovery, test a hypothesis, or demonstrate a known fact", "noun", ["test", "trial", "investigation"], [], ["Scientists conducted a controlled experiment to evaluate the new model.", "The company is running an experiment with flexible four-day work weeks."]),
    ("feedback", "phản hồi, ý kiến nhận xét", "information about reactions to a product or a person's performance of a task", "noun", ["response", "critique", "review"], [], ["Constructive feedback helps students improve their pronunciation.", "User feedback indicated that the new interface is intuitive and fast."]),
    ("foundation", "nền móng, cơ sở tảng", "the lowest load-bearing part of a building; an underlying basis or principle", "noun", ["base", "basis", "groundwork"], [], ["A rich vocabulary forms the foundation of fluent English communication.", "Concrete was poured to create a sturdy foundation for the tower."]),
    ("framework", "khung, khuôn khổ, cấu trúc", "an essential supporting structure of a building, vehicle, or object; a basic structure underlying a system", "noun", ["structure", "system", "scaffolding"], [], ["FastAPI provides a modern and lightweight framework for Python backends.", "The proposal outlines a clear policy framework for sustainable energy."]),
    ("function", "chức năng, hàm số", "an activity or purpose natural to or intended for a person or thing", "noun", ["purpose", "role", "operation"], [], ["The primary function of this button is to submit the completed word.", "In programming, a function takes inputs and returns computed outputs."]),
    ("gesture", "cử chỉ, điệu bộ", "a movement of part of the body, especially a hand or the head, to express an idea or meaning", "noun", ["signal", "motion", "sign"], [], ["AirWrite tracks fingertip gestures to recognize letters drawn in the air.", "A warm handshake is a universal gesture of friendliness."]),
    ("interface", "giao diện", "a device or program enabling a user to communicate with a computer; a point where two systems meet", "noun", ["ui", "layout", "boundary"], [], ["The web interface is clean, responsive, and easy to navigate.", "Designing an intuitive user interface requires empathy for learners."]),
    ("knowledge", "kiến thức, sự hiểu biết", "facts, information, and skills acquired by a person through experience or education", "noun", ["understanding", "wisdom", "learning"], ["ignorance"], ["Expanding your vocabulary deepens your knowledge of the world.", "Practical knowledge is just as valuable as theoretical understanding."]),
    ("language", "ngôn ngữ, tiếng", "the principal method of human communication, consisting of words used in a structured way", "noun", ["tongue", "speech", "dialect"], [], ["English is widely spoken as an international language of business.", "Body language conveys powerful emotional signals."]),
    ("learning", "việc học tập, sự tiếp thu kiến thức", "the acquisition of knowledge or skills through experience, study, or being taught", "noun", ["education", "study", "schooling"], [], ["Lifelong learning keeps the mind active and adaptable.", "Interactive games make vocabulary learning enjoyable for students."]),
    ("meaning", "ý nghĩa, nghĩa của từ", "what is meant by a word, text, concept, or action", "noun", ["definition", "significance", "sense"], [], ["Look up the Vietnamese meaning of unfamiliar English phrases.", "Finding purpose and meaning in life brings lasting happiness."]),
    ("memory", "trí nhớ, ký ức, bộ nhớ", "the faculty by which the mind stores and remembers information; computer storage", "noun", ["recall", "remembrance", "retention"], ["forgetfulness"], ["Spaced repetition strengthens long-term memory for new words.", "The computer has 16 gigabytes of high-speed memory."]),
    ("method", "phương pháp, cách thức", "a particular form of procedure for accomplishing or approaching something, especially systematic", "noun", ["technique", "approach", "system"], [], ["Air-writing is an innovative method for interactive language learning.", "Scientific methods require empirical observation and repeated trials."]),
    ("motivation", "động lực, sự thúc đẩy", "the reason or reasons one has for acting or behaving in a particular way", "noun", ["drive", "inspiration", "incentive"], ["lethargy", "apathy"], ["Setting clear daily goals boosts motivation when studying English.", "Intrinsic motivation leads to deeper and more lasting learning."]),
    ("performance", "hiệu suất, sự thể hiện", "the action or process of carrying out or accomplishing an action, task, or function", "noun", ["execution", "achievement", "presentation"], [], ["The machine learning model achieved 98% accuracy in test performance.", "Her musical performance received a standing ovation from the audience."]),
    ("practice", "sự luyện tập, thực hành", "repeated exercise in or performance of an activity so as to acquire or maintain proficiency", "noun", ["drill", "training", "exercise"], [], ["Consistent daily practice makes your handwriting steady and recognizable.", "Put grammar theory into practice by speaking with native speakers."]),
    ("process", "quy trình, quá trình / xử lý", "a series of actions or steps taken in order to achieve a particular end", "noun", ["procedure", "course", "operation"], [], ["Writing on canvas triggers a multi-step image recognition process.", "Learning a language is a gradual process requiring patience."]),
    ("progress", "tiến bộ, sự phát triển", "forward or onward movement towards a destination or goal", "noun", ["advancement", "improvement", "headway"], ["regression", "decline"], ["The learner dashboard tracks your vocabulary learning progress.", "She made remarkable progress in English speaking over the summer."]),
    ("recognition", "sự nhận diện, nhận biết, công nhận", "the action or process of recognizing or being recognized, in particular identification", "noun", ["identification", "detection", "acknowledgment"], [], ["The hand gesture recognition algorithm runs in real time at 30 FPS.", "He received worldwide recognition for his groundbreaking inventions."]),
    ("review", "ôn tập, xem lại, đánh giá", "a formal assessment or examination of something; viewing again to reinforce memory", "verb", ["reexamine", "study", "inspect"], [], ["Students can review learned words in spell and cloze modes.", "The editor wrote a glowing review of the new educational application."]),
    ("screen", "màn hình", "a flat panel or area on an electronic device on which images and data are displayed", "noun", ["display", "monitor"], [], ["Look directly into the screen while writing letters in front of the webcam.", "Screen brightness can be adjusted to reduce eye strain."]),
    ("session", "phiên học, buổi làm việc", "a period devoted to a particular activity", "noun", ["meeting", "period", "class"], [], ["Each practice session lasts approximately fifteen minutes.", "The backend creates a unique session to record drawn strokes."]),
    ("solution", "giải pháp, lời giải", "a means of solving a problem or dealing with a difficult situation", "noun", ["answer", "resolution", "key"], ["problem", "dilemma"], ["The proposed software solution is lightweight and fully offline.", "Brainstorming with peers often reveals innovative solutions."]),
    ("standard", "tiêu chuẩn, chuẩn mực", "a level of quality or attainment; an accepted norm", "noun", ["norm", "criterion", "benchmark"], [], ["We adhere strictly to the Oxford 3000 vocabulary standard.", "High standards of code quality ensure application stability."]),
    ("strategy", "chiến lược", "a plan of action or policy designed to achieve a major or overall aim", "noun", ["plan", "tactic", "approach"], [], ["A proven vocabulary strategy is grouping words by thematic topics.", "The business developed an aggressive digital expansion strategy."]),
    ("system", "hệ thống", "a set of connected things or parts forming a complex whole", "noun", ["framework", "organization", "structure"], [], ["The AirWrite system integrates computer vision and language education.", "Our solar system consists of eight major planets orbiting the sun."]),
    ("technology", "công nghệ", "the application of scientific knowledge for practical purposes, especially in industry", "noun", ["tech", "engineering", "automation"], [], ["Advances in webcam technology enable accurate touchless interaction.", "Information technology is shaping the future of global education."]),
    ("topic", "chủ đề", "a matter dealt with in a text, discourse, or conversation; a subject", "noun", ["subject", "theme", "matter"], [], ["Choose a vocabulary topic such as home, school, work, or dining.", "The teacher introduced a fascinating topic for classroom discussion."]),
    ("translate", "dịch, dịch nghĩa", "to express the sense of words or text in another language", "verb", ["interpret", "render", "convert"], [], ["The application will translate English words into accurate Vietnamese.", "Can you help me translate this official document into French?"]),
    ("translation", "bản dịch, sự dịch nghĩa", "the process of translating words or text from one language into another", "noun", ["rendering", "interpretation", "version"], [], ["Every entry includes a clear Vietnamese translation and example sentences.", "Machine translation has improved dramatically over the past decade."]),
    ("vocabulary", "từ vựng, vốn từ", "the body of words used in a particular language, field, or activity", "noun", ["lexicon", "words", "glossary"], [], ["Expanding your active vocabulary empowers confident communication.", "The database provides a comprehensive vocabulary for English learners."])
]

def load_anhviet_dict():
    print("Loading Anh-Viet 109K dictionary...")
    pos_map = {
        'danh từ': 'noun',
        'tính từ': 'adjective',
        'động từ': 'verb',
        'phó từ': 'adverb',
        'trạng từ': 'adverb',
        'giới từ': 'preposition',
        'liên từ': 'conjunction',
        'đại từ': 'pronoun',
        'mạo từ': 'article',
        'thán từ': 'exclamation',
    }
    
    entries = {}
    if not ANHVIET_TXT.exists():
        print(f"Warning: {ANHVIET_TXT} not found.")
        return entries

    with open(ANHVIET_TXT, "r", encoding="utf-8", errors="ignore") as f:
        curr_word = None
        curr_lines = []
        for line in f:
            if line.startswith("@"):
                if curr_word and curr_lines:
                    entries[curr_word] = parse_entry(curr_word, curr_lines, pos_map)
                header = line[1:].strip()
                curr_word = header.split("/")[0].strip().lower()
                curr_lines = []
            else:
                if curr_word:
                    curr_lines.append(line.strip())
        if curr_word and curr_lines:
            entries[curr_word] = parse_entry(curr_word, curr_lines, pos_map)

    print(f"Loaded {len(entries)} words from Anh-Viet 109K.")
    return entries

def parse_entry(word, lines, pos_map):
    pos = "noun"
    meanings = []
    examples = []
    
    for l in lines:
        if l.startswith("*"):
            pos_text = l.lstrip("*").strip().lower()
            for k, v in pos_map.items():
                if k in pos_text:
                    pos = v
                    break
        elif l.startswith("-"):
            m = l.lstrip("-").strip()
            if m:
                clean_m = re.sub(r"^\([^)]*\)\s*", "", m).split(";")[0].split(",")[0].strip()
                if clean_m and len(clean_m) > 1:
                    meanings.append(clean_m)
        elif l.startswith("="):
            ex_line = l.lstrip("=").strip()
            if "+" in ex_line:
                en_part = ex_line.split("+", 1)[0].strip()
                if en_part and len(en_part.split()) >= 3:
                    examples.append(en_part)

    viet_meaning = meanings[0] if meanings else word
    return {
        "word": word,
        "normalized_word": word.lower(),
        "vietnamese_meaning": viet_meaning,
        "definition": f"a term referring to {viet_meaning}",
        "part_of_speech": pos,
        "synonyms": [],
        "antonyms": [],
        "examples": examples[:2] if examples else [f"The word {word} is widely used in modern English."]
    }

def main():
    with open(VOCAB_JSON, "r", encoding="utf-8") as f:
        existing_vocab = json.load(f)

    print(f"Existing vocab count: {len(existing_vocab)}")
    vocab_map = {item["normalized_word"]: item for item in existing_vocab if "normalized_word" in item}

    # 1. Update/Add curated items (missing Oxford 3000 + core high priority items)
    for w, item in CURATED_ADDITIONS.items():
        norm = w.lower().strip()
        item["normalized_word"] = norm
        vocab_map[norm] = item

    # 2. Add high utility Oxford additions
    for w, vi, d, pos, syn, ant, ex in OXFORD_HIGH_UTILITY:
        norm = w.lower().strip()
        vocab_map[norm] = {
            "word": w,
            "normalized_word": norm,
            "vietnamese_meaning": vi,
            "definition": d,
            "part_of_speech": pos,
            "synonyms": syn,
            "antonyms": ant,
            "examples": ex
        }

    # 3. Fill up to ~3,250 words from Oxford candidate words
    current_count = len(vocab_map)
    print(f"Count after curated additions: {current_count}")

    if current_count < 3250:
        anhviet = load_anhviet_dict()
        extra_candidates = [
            "accomplishment", "accordingly", "accuracy", "acid", "acre", "activation", "activist", "acute",
            "addiction", "adhere", "administrative", "administrator", "admission", "adolescent", "adviser",
            "advocate", "aerial", "aerospace", "affection", "affiliate", "affordable", "agenda", "aggression",
            "aide", "aircraft", "airway", "algorithm", "alien", "allegation", "alliance", "allocation", "allowance",
            "ally", "almond", "alongside", "alphabet", "altar", "alteration", "aluminum", "amateur", "ambassador",
            "amendment", "amid", "ammo", "ammunition", "amusement", "analogy", "analyst", "anatomy", "ancestor",
            "anchor", "animation", "anniversary", "anonymous", "antique", "antiquity", "apparel", "appetite",
            "applaud", "applause", "applicable", "applicant", "appoint", "appreciation", "apprehensive",
            "approximate", "arbitrary", "arcade", "arch", "archaeologist", "archaeology", "architect", "archive",
            "arena", "arguably", "aristocrat", "armor", "arrow", "artery", "artifact", "artillery", "artisan",
            "artist", "artistic", "artwork", "ash", "aside", "aspect", "aspire", "assassin", "assault", "assemble",
            "assembly", "assert", "assertion", "asset", "assignment", "assistant", "associate", "association",
            "assurance", "assure", "asteroid", "astonish", "astonishing", "astronaut", "astronomy", "asylum",
            "athlete", "athletic", "athletics", "atmosphere", "atom", "atomic", "attain", "attendance", "attendant",
            "attorney", "attribute", "auction", "audit", "auditorium", "authentic", "author", "authorize",
            "automaker", "automatic", "automobile", "autonomy", "autumn", "avail", "availability", "avalanche",
            "aviation", "await", "awakening", "award", "awareness", "awe", "awkward", "axe", "babysitter",
            "backbone", "backdrop", "backing", "backup", "badge", "badminton", "baffle", "bail", "bait", "balcony",
            "ballot", "bamboo", "bandage", "banker", "bankruptcy", "banner", "banquet", "barbaric", "bargain",
            "bark", "barn", "barometer", "baron", "barrack", "barrel", "barrier", "bartender", "baseball", "baseline",
            "basement", "basin", "basket", "basketball", "bat", "batch", "bath", "bathe", "batter", "battery"
        ]

        for cand in extra_candidates:
            if len(vocab_map) >= 3250:
                break
            c_norm = cand.lower().strip()
            if c_norm not in vocab_map and c_norm in anhviet:
                vocab_map[c_norm] = anhviet[c_norm]

    final_vocab = list(vocab_map.values())
    final_vocab.sort(key=lambda x: x["normalized_word"])

    print(f"Final standardized vocabulary count: {len(final_vocab)}")
    assert 3000 <= len(final_vocab) <= 3500, f"Count {len(final_vocab)} is out of 3000-3500 range!"

    # Save to vocab_2000.json
    with open(VOCAB_JSON, "w", encoding="utf-8") as f:
        json.dump(final_vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(final_vocab)} entries to {VOCAB_JSON}")

    # Re-seed SQLite database
    print(f"Updating SQLite database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DELETE FROM vocabulary")
    
    rows = []
    for item in final_vocab:
        w = item.get("word", "").strip()
        norm = item.get("normalized_word", w.lower()).strip()
        vi = item.get("vietnamese_meaning", "").strip()
        d = item.get("definition", "").strip()
        pos = item.get("part_of_speech", "noun")
        syn = json.dumps(item.get("synonyms") or [], ensure_ascii=False)
        ant = json.dumps(item.get("antonyms") or [], ensure_ascii=False)
        ex = json.dumps(item.get("examples") or [], ensure_ascii=False)
        rows.append((w, norm, vi, d, pos, syn, ant, ex))

    cur.executemany(
        "INSERT OR REPLACE INTO vocabulary (word, normalized_word, vietnamese_meaning, definition, part_of_speech, synonyms, antonyms, examples) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows
    )
    conn.commit()

    # Apply curated BASIC_VOCABULARY from vocab_seeds.py
    try:
        from backend.storage.vocab_seeds import BASIC_VOCABULARY
        cur.executemany(
            "INSERT OR REPLACE INTO vocabulary (word, normalized_word, vietnamese_meaning, definition, part_of_speech, synonyms, antonyms, examples) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(w, w.casefold().strip(), vi, d, pos, syn, ant, ex) for w, vi, d, pos, syn, ant, ex in BASIC_VOCABULARY]
        )
        conn.commit()
    except Exception as e:
        print("Note on seeds:", e)

    cur.execute("SELECT count(*) FROM vocabulary")
    db_count = cur.fetchone()[0]
    conn.close()
    print(f"SQLite vocabulary table now has {db_count} standardized records!")

if __name__ == "__main__":
    main()
