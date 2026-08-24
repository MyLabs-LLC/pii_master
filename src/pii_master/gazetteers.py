"""Compact public-domain name lists for the gazetteer detector.

These are not a census dump and not a claim of coverage. They exist so the
fast path can recognise a *first + last* pair (or a titled name) the way
Papadopoulou et al. 2023 combine a NER model with a person-term gazetteer
— without paying for a neural pass, and without tagging every capitalised
English word as a person.

Sources: SSA / Census given-name and surname frequency lists (public
domain). The lists are deliberately diverse across the most common
Anglo, Hispanic, East Asian, South Asian, and African-American names;
PRIOR_ART.md §5.6 records that off-the-shelf name maskers are worse on
Black and AAPI names, and a gazetteer that is only the Anglo top-100
would reproduce that.

A name that is also a common English word (Green, Young, May, ...) is
fine as a *last* name because the detector requires a gazetteer first
name next to it. ``Patient`` is *not* in the last-name list: it is the
word, not the surname, in almost every document this scanner will see.
The detector still accepts a capitalised last token after a first name
and a middle initial (``John Q. Patient``) so the frozen-corpus gold
row is reachable without putting the word in the list.
"""

from __future__ import annotations

# Given names, lowercased. Keep this a frozenset so membership is O(1) and
# the detector never mutates it.
FIRST_NAMES: frozenset[str] = frozenset({
    "aaron", "abigail", "adam", "adrian", "ahmed", "aisha", "alan", "albert",
    "alejandro", "alex", "alexander", "alexandra", "alexis", "ali", "alice",
    "alicia", "amanda", "amber", "amelia", "aminata", "amy", "ana", "andrea",
    "andrew", "angela", "angelica", "anita", "anna", "anne", "anthony",
    "antonio", "aria", "ariana", "arthur", "ashley", "aubrey", "austin",
    "ava", "avery", "barbara", "benjamin", "betty", "beverly", "billy",
    "brandon", "brenda", "brian", "brittany", "bruce", "bryan", "caleb",
    "camila", "carl", "carlos", "carol", "caroline", "carolyn", "catherine",
    "cynthia", "charles", "charlotte", "cheryl", "chloe", "chris",
    "christian", "christina", "christine", "christopher", "cindy", "claire",
    "clarence", "cody", "colin", "connor", "corey", "craig", "crystal",
    "curtis", "dakota", "dale", "damian", "daniel", "danielle", "danny",
    "david", "deborah", "debra", "dennis", "derek", "diana", "diane",
    "diego", "dominic", "donald", "donna", "doris", "dorothy", "douglas",
    "dylan", "eddie", "edward", "eleanor", "elena", "eli", "elias", "elijah",
    "elizabeth", "ella", "ellen", "ellie", "emily", "emma", "eric", "erica",
    "erik", "erin", "ernest", "ethan", "eugene", "eva", "evan", "evelyn",
    "faith", "fatima", "felix", "fernando", "fiona", "florence", "frances",
    "francis", "francisco", "frank", "gabriel", "gabriela", "gary", "gavin",
    "george", "georgia", "gerald", "gina", "gloria", "grace", "gregory",
    "hannah", "harold", "harper", "harry", "hassan", "hayden", "heather",
    "helen", "henry", "hunter", "ian", "ibrahim", "irene", "iris", "isaac",
    "isabella", "isabelle", "isaiah", "ivan", "jack", "jackson", "jacob",
    "jacqueline", "jade", "jake", "james", "jamie", "jane", "janet", "janice",
    "jasmine", "jason", "javier", "jayden", "jean", "jeff", "jeffrey",
    "jennifer", "jeremy", "jerome", "jerry", "jesse", "jessica", "jesus",
    "jill", "joan", "joanna", "jocelyn", "jodi", "joe", "joel", "john",
    "johnny", "jonathan", "jordan", "jorge", "jose", "joseph", "joshua",
    "josiah", "joy", "joyce", "juan", "judith", "judy", "julia", "julian",
    "julie", "julio", "justin", "kai", "karen", "katherine", "kathleen",
    "kathryn", "katie", "kayla", "keith", "kelly", "kelsey", "kenneth",
    "kevin", "kim", "kimberly", "kofi", "kristen", "kristin", "kyle",
    "larry", "laura", "lauren", "lawrence", "leah", "leo", "leon", "leonard",
    "leslie", "liam", "lillian", "lily", "linda", "lindsay", "lisa", "logan",
    "lois", "loretta", "louis", "louise", "lucas", "lucia", "lucy", "luis",
    "luke", "lydia", "lynn", "madison", "mae", "malik", "manuel", "marc",
    "marco", "marcus", "margaret", "maria", "mariah", "marie", "marilyn",
    "mario", "marisa", "mark", "martha", "martin", "marvin", "mary", "mason",
    "mateo", "mathew", "matthew", "maureen", "maurice", "max", "maya",
    "megan", "melissa", "mia", "michael", "michele", "michelle", "miguel",
    "mike", "mildred", "mohamed", "mohammed", "monica", "morgan", "nancy",
    "naomi", "natalie", "nathan", "nathaniel", "neil", "nicholas", "nicole",
    "noah", "nora", "norman", "olivia", "omar", "oscar", "owen", "pamela",
    "patricia", "patrick", "paul", "paula", "pedro", "peter", "philip",
    "phillip", "phyllis", "priya", "rachel", "rafael", "ralph", "ramon",
    "randy", "raymond", "rebecca", "regina", "rene", "ricardo", "richard",
    "rick", "ricky", "rita", "robert", "roberto", "robin", "roger", "ronald",
    "rosa", "rose", "roy", "ruben", "russell", "ruth", "ryan", "samantha",
    "samuel", "sandra", "sara", "sarah", "scott", "sean", "sebastian",
    "sergio", "seth", "shane", "shannon", "sharon", "shaun", "shawn",
    "sheila", "shirley", "sofia", "sophia", "stacey", "stacy", "stanley",
    "stephanie", "stephen", "steve", "steven", "susan", "syed", "tamara",
    "tammie", "tanya", "taylor", "teresa", "terry", "theodore", "theresa",
    "thomas", "tiffany", "tim", "timothy", "tina", "todd", "tom", "tony",
    "tracey", "tracy", "travis", "troy", "tyler", "tyrone", "valerie",
    "vanessa", "victor", "victoria", "vincent", "virginia", "walter",
    "wanda", "wayne", "wendy", "william", "willie", "wyatt", "xavier",
    "yasmin", "yolanda", "yusuf", "zachary", "zoe",
})

LAST_NAMES: frozenset[str] = frozenset({
    "adams", "adeyemi", "aguilar", "ahmed", "ali", "allen", "alvarez",
    "alvarado", "anderson", "andrews", "armstrong", "arnold", "bailey",
    "baker", "banks", "barnes", "bell", "bennett", "berry", "black",
    "boyd", "bradley", "brooks", "brown", "bryant", "burns", "butler",
    "campbell", "carroll", "carter", "castillo", "castro", "chavez",
    "chen", "choi", "clark", "coleman", "cole", "collins", "cook",
    "cooper", "cox", "crawford", "cruz", "cunningham", "daniels",
    "davis", "delgado", "diallo", "diaz", "dixon", "doe", "duncan",
    "dunn", "edwards", "elliott", "ellis", "evans", "ferguson",
    "fernandez", "fisher", "flores", "ford", "foster", "fox", "freeman",
    "garcia", "gardner", "garza", "gibson", "gomez", "gonzales",
    "gonzalez", "gordon", "graham", "grant", "gray", "green", "griffin",
    "gutierrez", "guzman", "hall", "hamilton", "hansen", "harris",
    "harrison", "hart", "hassan", "hawkins", "hayes", "henderson",
    "henry", "hernandez", "herrera", "hicks", "hill", "holmes", "howard",
    "huang", "hudson", "hughes", "hunt", "hunter", "ibrahim", "jackson",
    "james", "jefferson", "jenkins", "jimenez", "johnson", "jones",
    "jordan", "jung", "kamara", "kelley", "kelly", "kennedy", "khan",
    "kim", "king", "knight", "kumar", "lane", "lee", "lewis", "li",
    "liu", "long", "lopez", "marshall", "martin", "martinez", "mason",
    "mcdonald", "medina", "mendez", "mendoza", "meyer", "miller",
    "mills", "mitchell", "mohamed", "moore", "morales", "moreno",
    "morgan", "morris", "munoz", "murphy", "murray", "myers", "nelson",
    "nguyen", "nichols", "nwosu", "okonkwo", "olson", "ortiz", "osman",
    "owens", "palmer", "park", "parker", "patel", "patterson", "payne",
    "perez", "perkins", "perry", "peters", "peterson", "phillips",
    "pierce", "porter", "powell", "price", "ramirez", "ramos", "ray",
    "reed", "reyes", "reynolds", "rice", "richardson", "riley", "rivera",
    "roberts", "robertson", "robinson", "rodriguez", "rogers", "romero",
    "rose", "ross", "ruiz", "russell", "ryan", "salazar", "sanchez",
    "sanders", "santos", "schmidt", "scott", "shah", "shaw", "silva",
    "simmons", "simpson", "singh", "smith", "snyder", "soto", "spencer",
    "stephens", "stevens", "stewart", "stone", "sullivan", "taylor",
    "tesfaye", "thomas", "thompson", "torres", "tran", "tucker",
    "turner", "vargas", "vasquez", "vazquez", "wagner", "walker",
    "wallace", "wang", "ward", "warren", "washington", "watson",
    "weaver", "webb", "wells", "west", "white", "williams", "wilson",
    "wood", "woods", "wright", "wu", "yang", "young", "zhang", "zhou",
})

# Tokens that look like a last name after "John Q. ____" but are document
# furniture, not a surname. ``Patient`` is *not* here — see module docstring.
LAST_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "from", "with", "this", "that", "was", "are",
    "were", "been", "have", "has", "had", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "hospital",
    "clinic", "medical", "record", "number", "account", "social",
    "security", "summary", "discharge", "admission", "prescription",
    "diagnosis", "treatment", "medication", "physician", "doctor",
    "nurse", "today", "tomorrow", "yesterday", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "street",
    "avenue", "road", "drive", "lane", "court", "place", "circle",
})

TITLES: frozenset[str] = frozenset({
    "dr", "mr", "mrs", "ms", "mx", "prof", "sir", "dame",
})

STREET_SUFFIXES: frozenset[str] = frozenset({
    "street", "st", "st.", "avenue", "ave", "ave.", "boulevard", "blvd",
    "blvd.", "road", "rd", "rd.", "drive", "dr", "dr.", "lane", "ln",
    "ln.", "court", "ct", "ct.", "way", "parkway", "pkwy", "pkwy.",
    "place", "pl", "pl.", "circle", "cir", "cir.", "terrace", "ter",
    "ter.", "highway", "hwy", "hwy.", "trail", "trl",
})
