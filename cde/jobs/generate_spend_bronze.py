"""
Stage 1 - Generate (bronze)

Synthesises the six source systems a bank has to join to see one customer's
spending, and lands them raw in bronze, each with its own key:

  crm_customer   CRM / KYC master (CIF): home pincode, age band, income band,
                 segment (MASS, AFFLUENT, NRI, SME_OWNER), onboarding date
  cbs_accounts   core banking accounts (SA, SALARY, CA, NRE) keyed by account_id, with the CIF
  card_master    card switch: card_hash -> account_id (the raw card number is never landed)
  card_txn       card switch: MCC, merchant name and city, entry mode (CHIP, CONTACTLESS,
                 MAGSTRIPE, ECOM), amount
  vpa_registry   UPI switch: VPA -> account_id
  upi_txn        UPI switch: payer / payee VPA, free-text payee name (no MCC), payee
                 pincode when the QR carries one, P2P or P2M, remarks, amount
  nb_txn         net banking / bill pay: CIF, biller name and type, amount
  atm_txn        ATM switch: card_hash, atm_id, amount (multiples of 100)
  ref.pincode_geo, ref.mcc_category, ref.category_taxonomy, ref.atm_location,
  ref.merchant_seed, ref.calendar_weekly
  ref.archetype_truth  the hidden spend archetype per CIF: synthetic ground truth, read
                       only to score how well the personas recover it, never a feature

In production this job would read the CRM, CBS, card switch, UPI switch, net
banking and ATM switch extracts instead.

How the synthetic bank behaves. ~8,000 merchants (national brands and local
shops in 34 cities) each carry 2-5 noisy payee-name variants
("SWIGGY*BLR 5521", "swiggy instamart", "Swiggy Ltd"), so categorising UPI
payments has real work to do. Every customer belongs to one of 7 hidden spend
archetypes (metro digital foodie, rural cash-first, affluent traveller,
salary-week bill payer, family grocery and school fees, commuter, online
shopper on EMI) that set their category mix, channel preferences, spend level
and travel habits. Seasonality is planted: Diwali electronics and apparel,
summer and year-end travel, school fees in April and June, month-start salary
spend. Trips move spend to another city for a few days; ~1.5% of customers
relocate for good while the CRM keeps the old home pincode.

History is prefix-stable: the world is simulated day by day from a fixed seed
and --as-of only decides where the simulation stops. A later as-of date adds
new days without changing any earlier fact. 0.2% of CRM and transaction records are
re-sent (duplicates for silver to remove).

--inject-bad-data points 2% of UPI payments at payer VPAs the registry does not
know, so dq_bronze fails on cue and yesterday's insights stay live.

Writes (drop + recreate every run):
  <prefix>_bronze.{crm_customer, cbs_accounts, card_master, card_txn, vpa_registry, upi_txn,
                   nb_txn, atm_txn}
  <prefix>_ref.{pincode_geo, mcc_category, category_taxonomy, atm_location, merchant_seed,
                calendar_weekly, archetype_truth}

Usage:
  spark-submit generate_spend_bronze.py [--as-of YYYY-MM-DD] [--db-prefix P]
                                        [--customers N] [--seed S] [--inject-bad-data]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import random
import re
from datetime import date, datetime, timedelta, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _memo(fn):
    """A plain-dict cache. functools.lru_cache wrappers pickle by reference, and the job module is not
    importable on Spark workers, so the simulation closures could not be shipped with them."""
    cache: dict = {}

    def wrapper(*args):
        if args not in cache:
            cache[args] = fn(*args)
        return cache[args]
    wrapper.__doc__, wrapper.__name__ = fn.__doc__, fn.__name__
    return wrapper

DEFAULT_DB_PREFIX = "rsingh_spend_anl"
DEFAULT_SEED = 20250401
DEFAULT_CUSTOMERS = 200_000
MAX_CUSTOMERS = 260_000             # id space of _perm(idx * 8192 + seq)
HISTORY_START = date(2025, 4, 1)    # first day of the transaction extracts
WORLD_END = date(2027, 3, 31)
ONBOARD_START = date(2008, 1, 1)
DUPLICATE_RATE = 0.002
BAD_VPA_RATE = 0.02
CARD_SALT = "spend-demo-card-salt"
OWN_HANDLE = "dbnk"
N_DEMO = 3                          # units 0..2 are the planted demo customers
PURCHASE_SCALE = 0.68               # tunes the default run to about 25-30M events at 200k customers

CATEGORIES = ["GROCERY", "FOOD_DELIVERY", "DINING", "FUEL", "TRAVEL", "APPAREL", "ELECTRONICS", "UTILITIES",
              "TELECOM", "EMI_LOAN", "EDUCATION", "HEALTH", "ENTERTAINMENT", "OTHER"]
TAXONOMY = {
    "GROCERY": "grocery store, supermarket, kirana, provisions, vegetables and fruits, daily household needs",
    "FOOD_DELIVERY": "online food delivery app, order meals home, cloud kitchen, quick commerce food",
    "DINING": "restaurant, cafe, dhaba, bakery, sweets shop, biryani, eating out",
    "FUEL": "petrol pump, fuel station, diesel, CNG filling station",
    "TRAVEL": "travel, flights, airline, railway tickets, hotel stay, cab, taxi, bus tickets, holidays",
    "APPAREL": "clothing, textiles, garments, fashion, footwear, tailors, sarees, silks",
    "ELECTRONICS": "electronics, mobile phones, laptops, home appliances, television, electricals",
    "UTILITIES": "electricity bill, power company, water bill, cooking gas, utility payment",
    "TELECOM": "mobile recharge, prepaid, postpaid, broadband, DTH, cable, telecom operator",
    "EMI_LOAN": "loan EMI, instalment, finance company, NBFC, credit repayment, gold loan",
    "EDUCATION": "school fees, college, tuition classes, coaching institute, academy, books",
    "HEALTH": "pharmacy, medical store, chemist, hospital, clinic, doctor, diagnostics lab",
    "ENTERTAINMENT": "movies, cinema, streaming subscription, events, gaming",
    "OTHER": "miscellaneous general merchandise, online marketplace, traders, enterprises, services",
}
# (mcc, description, category); MCC_UNMAPPED appear on card transactions but not in ref.mcc_category
MCC_CATEGORY = [
    (5411, "Grocery stores, supermarkets", "GROCERY"), (5499, "Misc food stores", "GROCERY"),
    (5814, "Fast food, food delivery", "FOOD_DELIVERY"), (5812, "Restaurants", "DINING"),
    (5813, "Bars, cafes", "DINING"), (5541, "Service stations", "FUEL"), (5542, "Fuel dispensers", "FUEL"),
    (4511, "Airlines", "TRAVEL"), (7011, "Hotels", "TRAVEL"), (4112, "Passenger railways", "TRAVEL"),
    (4121, "Taxicabs", "TRAVEL"), (4722, "Travel agencies", "TRAVEL"), (5651, "Family clothing", "APPAREL"),
    (5691, "Clothing stores", "APPAREL"), (5661, "Shoe stores", "APPAREL"), (5732, "Electronics stores", "ELECTRONICS"),
    (5722, "Household appliances", "ELECTRONICS"), (4900, "Utilities", "UTILITIES"),
    (4814, "Telecommunication services", "TELECOM"), (4899, "Cable and pay TV", "TELECOM"),
    (6012, "Financial institutions", "EMI_LOAN"), (8211, "Schools", "EDUCATION"), (8220, "Colleges", "EDUCATION"),
    (8299, "Educational services", "EDUCATION"), (5912, "Pharmacies", "HEALTH"), (8062, "Hospitals", "HEALTH"),
    (8011, "Doctors", "HEALTH"), (7832, "Cinemas", "ENTERTAINMENT"), (5815, "Digital media", "ENTERTAINMENT"),
    (5999, "Misc retail", "OTHER"), (5311, "Department stores", "OTHER"),
]
MCC_UNMAPPED = [5310, 5331, 7299]
MCC_BY_CATEGORY: dict[str, list[int]] = {}
for _mcc, _, _cat in MCC_CATEGORY:
    MCC_BY_CATEGORY.setdefault(_cat, []).append(_mcc)

# (code, city, state, tier, lat, lon, pincode prefix)
CITIES = [
    ("MUM", "Mumbai", "Maharashtra", 1, 19.08, 72.88, "400"), ("DEL", "Delhi", "Delhi", 1, 28.61, 77.21, "110"),
    ("BLR", "Bengaluru", "Karnataka", 1, 12.97, 77.59, "560"), ("CHE", "Chennai", "Tamil Nadu", 1, 13.08, 80.27, "600"),
    ("KOL", "Kolkata", "West Bengal", 1, 22.57, 88.36, "700"), ("HYD", "Hyderabad", "Telangana", 1, 17.39, 78.49, "500"),
    ("PUN", "Pune", "Maharashtra", 1, 18.52, 73.86, "411"), ("AHD", "Ahmedabad", "Gujarat", 1, 23.02, 72.57, "380"),
    ("JAI", "Jaipur", "Rajasthan", 2, 26.91, 75.79, "302"), ("LKO", "Lucknow", "Uttar Pradesh", 2, 26.85, 80.95, "226"),
    ("KOC", "Kochi", "Kerala", 2, 9.93, 76.27, "682"), ("IDR", "Indore", "Madhya Pradesh", 2, 22.72, 75.86, "452"),
    ("CHD", "Chandigarh", "Chandigarh", 2, 30.73, 76.78, "160"), ("NAG", "Nagpur", "Maharashtra", 2, 21.15, 79.09, "440"),
    ("CBE", "Coimbatore", "Tamil Nadu", 2, 11.02, 76.96, "641"), ("BBS", "Bhubaneswar", "Odisha", 2, 20.30, 85.82, "751"),
    ("SRT", "Surat", "Gujarat", 2, 21.17, 72.83, "395"), ("VIZ", "Visakhapatnam", "Andhra Pradesh", 2, 17.69, 83.22, "530"),
    ("MYS", "Mysuru", "Karnataka", 3, 12.30, 76.64, "570"), ("NSK", "Nashik", "Maharashtra", 3, 20.00, 73.79, "422"),
    ("MDU", "Madurai", "Tamil Nadu", 3, 9.93, 78.12, "625"), ("VNS", "Varanasi", "Uttar Pradesh", 3, 25.32, 82.97, "221"),
    ("DDN", "Dehradun", "Uttarakhand", 3, 30.32, 78.03, "248"), ("UDR", "Udaipur", "Rajasthan", 3, 24.59, 73.71, "313"),
    ("GOA", "Panaji", "Goa", 3, 15.49, 73.83, "403"), ("SLG", "Siliguri", "West Bengal", 3, 26.73, 88.40, "734"),
    ("KRR", "Karur", "Tamil Nadu", 4, 10.96, 78.08, "639"), ("RWR", "Rewari", "Haryana", 4, 28.20, 76.62, "123"),
    ("BHL", "Bhilwara", "Rajasthan", 4, 25.35, 74.64, "311"), ("PKD", "Palakkad", "Kerala", 4, 10.79, 76.65, "678"),
    ("STP", "Sitapur", "Uttar Pradesh", 4, 27.57, 80.68, "261"), ("ATP", "Anantapur", "Andhra Pradesh", 4, 14.68, 77.60, "515"),
    ("BWN", "Bardhaman", "West Bengal", 4, 23.23, 87.86, "713"), ("DWD", "Dharwad", "Karnataka", 4, 15.46, 75.01, "580"),
]
CITY = {c[0]: c for c in CITIES}
PINCODES_PER_CITY = 5
ATMS_PER_TIER = {1: 12, 2: 8, 3: 5, 4: 4}
LOCAL_MERCHANTS_PER_TIER = {1: 400, 2: 250, 3: 180, 4: 110}
TRAVEL_DESTINATIONS = [("GOA", 0.16), ("UDR", 0.09), ("JAI", 0.09), ("KOC", 0.08), ("MUM", 0.1), ("DEL", 0.1),
                       ("BLR", 0.08), ("DDN", 0.07), ("MYS", 0.05), ("VNS", 0.06), ("CHE", 0.06), ("KOL", 0.06)]

FIRST = ["Aarav", "Vivaan", "Aditya", "Arjun", "Sai", "Ishaan", "Rohan", "Rahul", "Amit", "Vikram", "Suresh",
         "Ramesh", "Manoj", "Deepak", "Imran", "Joseph", "Harpreet", "Ananya", "Diya", "Priya", "Pooja", "Sneha",
         "Kavya", "Meera", "Lakshmi", "Sunita", "Anita", "Fatima", "Mary", "Simran", "Neha", "Divya", "Karthik",
         "Venkat", "Srinivas", "Ravi", "Ganesh", "Mohan", "Prakash", "Sanjay", "Kiran", "Swati", "Jyoti"]
LAST = ["Sharma", "Verma", "Gupta", "Singh", "Kumar", "Yadav", "Patel", "Shah", "Mehta", "Reddy", "Rao", "Iyer",
        "Nair", "Menon", "Das", "Ghosh", "Banerjee", "Khan", "Ansari", "Fernandes", "Gill", "Joshi", "Kulkarni",
        "Mishra", "Tiwari", "Chauhan", "Jain", "Agarwal", "Bose", "Sahu"]
DEITIES = ["Lakshmi", "Ganesh", "Sai", "Balaji", "Krishna", "Durga", "Murugan", "Venkateswara", "Hanuman", "Shiva",
           "Ambika", "Annapoorna", "Saraswati", "Ayyappa", "Jagannath"]
LOCAL_TEMPLATES = {
    "GROCERY": ["{p} Kirana Store", "Sri {d} Provisions", "{p} General Stores", "{p} Supermarket",
                "{p} Fruits and Vegetables", "{p} Dairy", "{d} Rice Traders", "{p} Grocery Mart"],
    "FOOD_DELIVERY": ["{p} Cloud Kitchen", "{p} Home Foods Delivery", "{p} Tiffin Service"],
    "DINING": ["Hotel {p}", "{p} Restaurant", "{p} Dhaba", "{p} Bakery", "{d} Sweets", "{p} Cafe",
               "{p} Biryani House", "Sri {d} Bhavan"],
    "FUEL": ["{p} Fuel Station", "{d} Petroleum", "{p} Filling Station", "{p} Service Station", "{p} Petrol Pump"],
    "TRAVEL": ["{p} Tours and Travels", "{d} Travels", "{p} Cabs", "Hotel {p} Residency", "{p} Holidays",
               "{d} Bus Service", "{p} Lodge"],
    "APPAREL": ["{p} Textiles", "{p} Garments", "{d} Silks", "{p} Tailors", "{p} Footwear", "{p} Fashions",
                "{d} Saree Centre"],
    "ELECTRONICS": ["{p} Mobiles", "{p} Electronics", "{d} Electricals", "{p} Mobile World", "{p} Computers",
                    "{p} Home Appliances"],
    "UTILITIES": ["{c} Municipal Water Supply", "{c} Gas Agency"],
    "TELECOM": ["{p} Mobile Recharge", "{p} Cable Network", "{p} Broadband"],
    "EMI_LOAN": ["{p} Finance", "{d} Finserv", "{p} Credit Cooperative", "{d} Chit Funds"],
    "EDUCATION": ["{p} Tuition Classes", "{d} Academy", "{p} Coaching Centre", "{d} Public School",
                  "{d} Vidyalaya", "{p} Book Depot"],
    "HEALTH": ["{p} Medicals", "{d} Pharmacy", "{p} Clinic", "{d} Hospital", "{p} Diagnostics", "{p} Chemists",
               "{p} Dental Care"],
    "ENTERTAINMENT": ["{p} Cinemas", "{d} Theatre", "{p} Gaming Zone", "{p} Events"],
    "OTHER": ["{p} Enterprises", "{d} Traders", "{p} Agencies", "{p} and Sons", "{p} Hardware", "{p} Stationery"],
}
AMBIGUOUS_TEMPLATES = ["{p} Enterprises", "{d} Traders", "{p} and Sons", "{d} Agencies"]
AMBIGUOUS_SHARE = 0.05
LOCAL_CATEGORY_MIX = [("GROCERY", 0.22), ("DINING", 0.18), ("HEALTH", 0.10), ("APPAREL", 0.10),
                      ("ELECTRONICS", 0.07), ("FUEL", 0.06), ("TRAVEL", 0.06), ("EDUCATION", 0.06), ("OTHER", 0.07),
                      ("TELECOM", 0.03), ("EMI_LOAN", 0.02), ("ENTERTAINMENT", 0.02), ("FOOD_DELIVERY", 0.01)]
# name, category, kind: ONLINE (no store; UPI pincode null, card = e-commerce), CHAIN (a store in every city),
# BILLER (net banking biller; states list = where it operates, None = national)
NATIONAL = [
    ("BigBasket", "GROCERY", "ONLINE"), ("Blinkit", "GROCERY", "ONLINE"), ("Zepto", "GROCERY", "ONLINE"),
    ("JioMart", "GROCERY", "ONLINE"), ("DMart", "GROCERY", "CHAIN"), ("Reliance Fresh", "GROCERY", "CHAIN"),
    ("More Supermarket", "GROCERY", "CHAIN"), ("Spencers Retail", "GROCERY", "CHAIN"), ("Star Bazaar", "GROCERY", "CHAIN"),
    ("Nilgiris", "GROCERY", "CHAIN"), ("Ratnadeep Supermarket", "GROCERY", "CHAIN"),
    ("Swiggy", "FOOD_DELIVERY", "ONLINE"), ("Zomato", "FOOD_DELIVERY", "ONLINE"), ("EatSure", "FOOD_DELIVERY", "ONLINE"),
    ("Dominos Pizza", "FOOD_DELIVERY", "ONLINE"), ("Box8", "FOOD_DELIVERY", "ONLINE"), ("Faasos", "FOOD_DELIVERY", "ONLINE"),
    ("FreshMenu", "FOOD_DELIVERY", "ONLINE"),
    ("Haldirams", "DINING", "CHAIN"), ("Barbeque Nation", "DINING", "CHAIN"), ("McDonalds", "DINING", "CHAIN"),
    ("KFC", "DINING", "CHAIN"), ("Starbucks", "DINING", "CHAIN"), ("Cafe Coffee Day", "DINING", "CHAIN"),
    ("Chaayos", "DINING", "CHAIN"), ("Saravana Bhavan", "DINING", "CHAIN"), ("Paradise Biryani", "DINING", "CHAIN"),
    ("Wow Momo", "DINING", "CHAIN"), ("Pizza Hut", "DINING", "CHAIN"),
    ("IndianOil", "FUEL", "CHAIN"), ("HP Petrol Pump", "FUEL", "CHAIN"), ("Bharat Petroleum", "FUEL", "CHAIN"),
    ("Shell", "FUEL", "CHAIN"), ("Nayara Energy", "FUEL", "CHAIN"), ("Jio-bp", "FUEL", "CHAIN"),
    ("IRCTC", "TRAVEL", "ONLINE"), ("MakeMyTrip", "TRAVEL", "ONLINE"), ("IndiGo", "TRAVEL", "ONLINE"),
    ("Air India", "TRAVEL", "ONLINE"), ("Akasa Air", "TRAVEL", "ONLINE"), ("Goibibo", "TRAVEL", "ONLINE"),
    ("Ola Cabs", "TRAVEL", "ONLINE"), ("Uber India", "TRAVEL", "ONLINE"), ("Rapido", "TRAVEL", "ONLINE"),
    ("OYO Rooms", "TRAVEL", "ONLINE"), ("Taj Hotels", "TRAVEL", "CHAIN"), ("RedBus", "TRAVEL", "ONLINE"),
    ("Cleartrip", "TRAVEL", "ONLINE"), ("Yatra", "TRAVEL", "ONLINE"), ("ixigo", "TRAVEL", "ONLINE"),
    ("Myntra", "APPAREL", "ONLINE"), ("Ajio", "APPAREL", "ONLINE"), ("Max Fashion", "APPAREL", "CHAIN"),
    ("Pantaloons", "APPAREL", "CHAIN"), ("Westside", "APPAREL", "CHAIN"), ("Lifestyle Stores", "APPAREL", "CHAIN"),
    ("Zudio", "APPAREL", "CHAIN"), ("FabIndia", "APPAREL", "CHAIN"), ("Bata", "APPAREL", "CHAIN"),
    ("Raymond", "APPAREL", "CHAIN"), ("Nalli Silks", "APPAREL", "CHAIN"), ("Manyavar", "APPAREL", "CHAIN"),
    ("Croma", "ELECTRONICS", "CHAIN"), ("Reliance Digital", "ELECTRONICS", "CHAIN"), ("Vijay Sales", "ELECTRONICS", "CHAIN"),
    ("Samsung Smart Plaza", "ELECTRONICS", "CHAIN"), ("Poorvika Mobiles", "ELECTRONICS", "CHAIN"),
    ("Sangeetha Mobiles", "ELECTRONICS", "CHAIN"), ("Apple India Online", "ELECTRONICS", "ONLINE"),
    ("Airtel", "TELECOM", "BILLER"), ("Jio", "TELECOM", "BILLER"), ("Vi Vodafone Idea", "TELECOM", "BILLER"),
    ("BSNL", "TELECOM", "BILLER"), ("ACT Fibernet", "TELECOM", "BILLER"), ("Tata Play", "TELECOM", "BILLER"),
    ("Bajaj Finance", "EMI_LOAN", "BILLER"), ("Tata Capital", "EMI_LOAN", "BILLER"), ("Muthoot Finance", "EMI_LOAN", "BILLER"),
    ("Manappuram Finance", "EMI_LOAN", "BILLER"), ("Shriram Finance", "EMI_LOAN", "BILLER"),
    ("Home Credit India", "EMI_LOAN", "BILLER"), ("L&T Finance", "EMI_LOAN", "BILLER"), ("IIFL Finance", "EMI_LOAN", "BILLER"),
    ("Byjus", "EDUCATION", "ONLINE"), ("Unacademy", "EDUCATION", "ONLINE"), ("Physics Wallah", "EDUCATION", "ONLINE"),
    ("Aakash Institute", "EDUCATION", "CHAIN"), ("FIITJEE", "EDUCATION", "CHAIN"), ("Allen Career Institute", "EDUCATION", "CHAIN"),
    ("Delhi Public School", "EDUCATION", "BILLER"), ("Amity University", "EDUCATION", "BILLER"),
    ("Apollo Pharmacy", "HEALTH", "CHAIN"), ("MedPlus", "HEALTH", "CHAIN"), ("Netmeds", "HEALTH", "ONLINE"),
    ("PharmEasy", "HEALTH", "ONLINE"), ("Tata 1mg", "HEALTH", "ONLINE"), ("Apollo Hospitals", "HEALTH", "CHAIN"),
    ("Fortis Healthcare", "HEALTH", "CHAIN"), ("Manipal Hospitals", "HEALTH", "CHAIN"), ("Dr Lal PathLabs", "HEALTH", "CHAIN"),
    ("Thyrocare", "HEALTH", "ONLINE"), ("Wellness Forever", "HEALTH", "CHAIN"),
    ("PVR INOX", "ENTERTAINMENT", "CHAIN"), ("BookMyShow", "ENTERTAINMENT", "ONLINE"), ("Netflix", "ENTERTAINMENT", "ONLINE"),
    ("JioHotstar", "ENTERTAINMENT", "ONLINE"), ("Spotify", "ENTERTAINMENT", "ONLINE"), ("SonyLIV", "ENTERTAINMENT", "ONLINE"),
    ("Cinepolis", "ENTERTAINMENT", "CHAIN"),
    ("Amazon", "OTHER", "ONLINE"), ("Flipkart", "OTHER", "ONLINE"), ("Meesho", "OTHER", "ONLINE"),
    ("Urban Company", "OTHER", "ONLINE"), ("IKEA India", "OTHER", "CHAIN"), ("Decathlon", "OTHER", "CHAIN"),
]
# state electricity / gas utilities: net banking billers for customers living in that state
UTILITY_BILLERS = {
    "Maharashtra": ["MSEDCL", "Adani Electricity Mumbai", "Mahanagar Gas"], "Delhi": ["BSES Rajdhani", "Tata Power DDL",
                                                                                    "Delhi Jal Board"],
    "Karnataka": ["BESCOM", "HESCOM", "BWSSB"], "Tamil Nadu": ["TANGEDCO", "Chennai Metrowater"],
    "West Bengal": ["CESC", "WBSEDCL"], "Telangana": ["TSSPDCL", "HMWSSB"], "Gujarat": ["Torrent Power", "Gujarat Gas"],
    "Rajasthan": ["JVVNL", "AVVNL"], "Uttar Pradesh": ["UPPCL", "Green Gas"], "Kerala": ["KSEB", "Kerala Water Authority"],
    "Madhya Pradesh": ["MPPKVVCL"], "Chandigarh": ["Chandigarh Electricity"], "Odisha": ["TPCODL"],
    "Andhra Pradesh": ["APSPDCL", "APEPDCL"], "Uttarakhand": ["UPCL"], "Goa": ["Goa Electricity Department"],
    "Haryana": ["DHBVN", "UHBVN"],
}
GAS_BILLERS = ["Indane Gas", "Bharat Gas", "HP Gas"]
AGGREGATORS = ["PAYU", "RAZORPAY", "PAYTM", "BHARATPE", "PHONEPE", "CCAVENUE"]
UPI_HANDLES = ["ybl", "okicici", "okaxis", "paytm", "okhdfcbank", OWN_HANDLE]
P2P_REMARKS = ["rent", "dinner", "split bill", "gift", "loan return", "school van", "maid salary", "trip share",
               "", "", "", "thanks"]

AGE_BANDS = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
SEGMENTS = ["MASS", "AFFLUENT", "NRI", "SME_OWNER"]
INCOME_BANDS = ["LOW", "MID", "HIGH", "VERY_HIGH"]
PURCHASE_CHANNELS = ["CARD_POS", "CARD_ECOM", "UPI_P2M", "NETBANKING"]
ECOM_OK = {"GROCERY", "FOOD_DELIVERY", "TRAVEL", "APPAREL", "ELECTRONICS", "UTILITIES", "TELECOM", "EMI_LOAN",
           "EDUCATION", "HEALTH", "ENTERTAINMENT", "OTHER"}
NB_OK = {"UTILITIES", "TELECOM", "EMI_LOAN", "EDUCATION", "TRAVEL"}
POS_OK = set(CATEGORIES) - {"FOOD_DELIVERY", "UTILITIES", "EMI_LOAN"}
BILL_CATEGORIES = {"UTILITIES", "TELECOM", "EMI_LOAN"}
AMOUNT_MEDIAN = {"GROCERY": 900, "FOOD_DELIVERY": 420, "DINING": 750, "FUEL": 1400, "TRAVEL": 2800, "APPAREL": 2100,
                 "ELECTRONICS": 8500, "UTILITIES": 1700, "TELECOM": 420, "EMI_LOAN": 6500, "EDUCATION": 4200,
                 "HEALTH": 850, "ENTERTAINMENT": 550, "OTHER": 1300}
AMOUNT_SIGMA = {"TRAVEL": 0.95, "ELECTRONICS": 0.8, "EDUCATION": 0.8, "APPAREL": 0.7}

# archetype -> spec. cat: category weights; ch: purchase channel prefs (CARD_POS, CARD_ECOM, UPI_P2M, NETBANKING);
# tiers: home tier weights; seg: segment weights; age: age band weights; buy: purchases per week;
# atm / p2p: per month; spend: spend level multiplier; trips: per year; salary: extra weight of days 1-7
ARCHETYPES = {
    "METRO_DIGITAL_FOODIE": dict(
        share=0.16, cat=dict(FOOD_DELIVERY=.24, DINING=.14, GROCERY=.11, ENTERTAINMENT=.09, APPAREL=.09, TELECOM=.05,
                             TRAVEL=.08, ELECTRONICS=.04, UTILITIES=.04, HEALTH=.04, OTHER=.08),
        ch=(.08, .22, .68, .02), tiers=(.78, .19, .03, 0), seg=(.72, .2, .03, .05),
        age=(.34, .46, .15, .05, 0, 0), buy=1.7, atm=0.3, p2p=1.9, spend=1.0, trips=1.5, salary=0.3),
    "RURAL_CASH_FIRST": dict(
        share=0.15, cat=dict(GROCERY=.36, FUEL=.09, HEALTH=.11, APPAREL=.09, TELECOM=.08, EMI_LOAN=.06,
                             EDUCATION=.05, DINING=.05, OTHER=.11),
        ch=(.10, .01, .87, .02), tiers=(0, .04, .30, .66), seg=(.85, 0, .02, .13),
        age=(.1, .25, .27, .22, .12, .04), buy=0.8, atm=3.6, p2p=0.8, spend=0.5, trips=0.3, salary=0.2),
    "AFFLUENT_TRAVELLER": dict(
        share=0.08, cat=dict(TRAVEL=.27, DINING=.18, APPAREL=.09, ELECTRONICS=.07, ENTERTAINMENT=.07, GROCERY=.09,
                             HEALTH=.05, UTILITIES=.04, OTHER=.14),
        ch=(.42, .31, .23, .04), tiers=(.74, .23, .03, 0), seg=(.05, .6, .25, .1),
        age=(0, .22, .38, .26, .12, .02), buy=1.9, atm=0.35, p2p=0.7, spend=2.6, trips=5.0, salary=0.2),
    "SALARY_WEEK_BILL_PAYER": dict(
        share=0.15, cat=dict(UTILITIES=.20, TELECOM=.15, EMI_LOAN=.19, GROCERY=.15, HEALTH=.05, EDUCATION=.04,
                             OTHER=.09, DINING=.06, FUEL=.07),
        ch=(.14, .07, .47, .32), tiers=(.38, .36, .2, .06), seg=(.8, .12, 0, .08),
        age=(.05, .35, .3, .22, .07, .01), buy=1.1, atm=0.9, p2p=1.0, spend=1.0, trips=0.6, salary=2.6),
    "FAMILY_GROCERY_EDUCATION": dict(
        share=0.18, cat=dict(GROCERY=.31, EDUCATION=.13, HEALTH=.1, APPAREL=.08, UTILITIES=.07, DINING=.06, FUEL=.06,
                             TELECOM=.05, OTHER=.09, ELECTRONICS=.05),
        ch=(.16, .08, .68, .08), tiers=(.3, .36, .24, .1), seg=(.74, .14, .04, .08),
        age=(0, .2, .5, .27, .03, 0), buy=1.45, atm=0.9, p2p=1.3, spend=1.1, trips=0.8, salary=0.5),
    "COMMUTER_FUEL": dict(
        share=0.13, cat=dict(FUEL=.31, DINING=.12, TELECOM=.07, GROCERY=.15, EMI_LOAN=.1, TRAVEL=.06, OTHER=.09,
                             HEALTH=.05, ENTERTAINMENT=.05),
        ch=(.50, .06, .41, .03), tiers=(.34, .4, .2, .06), seg=(.6, .15, .0, .25),
        age=(.08, .38, .32, .17, .05, 0), buy=1.45, atm=1.1, p2p=0.9, spend=1.2, trips=0.8, salary=0.4),
    "ONLINE_SHOPPER_EMI": dict(
        share=0.15, cat=dict(ELECTRONICS=.19, APPAREL=.22, OTHER=.16, EMI_LOAN=.12, FOOD_DELIVERY=.06,
                             ENTERTAINMENT=.06, GROCERY=.1, TELECOM=.04, HEALTH=.05),
        ch=(.09, .52, .35, .04), tiers=(.5, .36, .14, 0), seg=(.72, .18, .05, .05),
        age=(.25, .45, .2, .08, .02, 0), buy=1.5, atm=0.4, p2p=1.1, spend=1.3, trips=1.0, salary=0.4),
}
ARCHETYPE_NAMES = list(ARCHETYPES)
SEGMENT_SPEND = {"MASS": 1.0, "AFFLUENT": 2.4, "NRI": 1.9, "SME_OWNER": 1.7}
SEGMENT_INCOME = {"MASS": (.35, .5, .13, .02), "AFFLUENT": (0, .1, .5, .4), "NRI": (0, .2, .45, .35),
                  "SME_OWNER": (.05, .4, .4, .15)}
RELOCATE_SHARE = 0.015

FESTIVALS = {  # date -> name (a festival week is the week holding the date; Diwali also the week before)
    date(2025, 3, 31): "Eid al-Fitr", date(2025, 8, 9): "Raksha Bandhan", date(2025, 8, 27): "Ganesh Chaturthi",
    date(2025, 9, 5): "Onam", date(2025, 10, 2): "Dussehra", date(2025, 10, 20): "Diwali",
    date(2025, 12, 25): "Christmas", date(2026, 1, 14): "Pongal / Sankranti", date(2026, 3, 4): "Holi",
    date(2026, 3, 20): "Eid al-Fitr", date(2026, 8, 26): "Onam", date(2026, 8, 28): "Raksha Bandhan",
    date(2026, 9, 14): "Ganesh Chaturthi", date(2026, 10, 20): "Dussehra", date(2026, 11, 8): "Diwali",
    date(2026, 12, 25): "Christmas", date(2027, 1, 14): "Pongal / Sankranti", date(2027, 3, 10): "Eid al-Fitr",
    date(2027, 3, 22): "Holi", date(2027, 10, 29): "Diwali", date(2027, 12, 25): "Christmas",
}
DIWALI = [d for d, n in FESTIVALS.items() if n == "Diwali"]
HOLIDAYS = {date(2025, 4, 18), date(2025, 5, 1), date(2025, 8, 15), date(2025, 10, 2), date(2025, 10, 20),
            date(2025, 12, 25), date(2026, 1, 26), date(2026, 3, 4), date(2026, 3, 20), date(2026, 4, 3),
            date(2026, 5, 1), date(2026, 8, 15), date(2026, 10, 2), date(2026, 10, 20), date(2026, 11, 8),
            date(2026, 12, 25), date(2027, 1, 26), date(2027, 3, 10), date(2027, 3, 22), date(2027, 8, 15),
            date(2027, 10, 29), date(2027, 12, 25)}
CALENDAR_FIRST_WEEK = date(2025, 4, 4)
CALENDAR_LAST_WEEK = date(2027, 12, 31)


def _pick(rng: random.Random, weighted):
    x, acc = rng.random() * sum(w for _, w in weighted), 0.0
    for value, w in weighted:
        acc += w
        if x < acc:
            return value
    return weighted[-1][0]


def _perm(key: int, mult: int) -> int:
    """Bijection on [1, 2^31 - 2]: distinct keys give distinct, unordered-looking numbers."""
    return (key * mult) % 2147483647


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 30:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    k, p, limit = 0, 1.0, math.exp(-lam)
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def _rand_date(rng: random.Random, lo: date, hi: date) -> date:
    return lo + timedelta(days=rng.randint(0, max(0, (hi - lo).days)))


def week_end(d: date) -> date:
    """The Friday closing the Saturday-to-Friday week that holds d."""
    return d + timedelta(days=(4 - d.weekday()) % 7)


def pincodes(city_code: str) -> list[str]:
    prefix = CITY[city_code][6]
    return [f"{prefix}{k * 7 + 1:03d}" for k in range(PINCODES_PER_CITY)]


def card_hash(card_number: str) -> str:
    return hashlib.sha256(f"{CARD_SALT}|{card_number}".encode()).hexdigest()[:32]


def _base36(n: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while True:
        n, r = divmod(n, 36)
        out = digits[r] + out
        if not n:
            return out


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())[:14]


# ------------------------------------------------------------------ merchants

def _variants(rng: random.Random, name: str, city_code: str, national: bool) -> list[str]:
    """2-5 payee-name spellings as a UPI app or a QR sticker would show them."""
    up = name.upper()
    words = name.split()
    out = {name}
    styles = [
        lambda: f"{up}*{city_code} {rng.randint(1000, 9999)}",
        lambda: f"{rng.choice(AGGREGATORS)}*{up.replace(' ', '')[:18]}",
        lambda: f"{name} {'Pvt Ltd' if national else 'Store'}",
        lambda: f"{name.lower()} {rng.choice(['online', 'india', 'payments', 'upi', 'qr'])}",
        lambda: " ".join(w[:4] if len(w) > 5 and rng.random() < 0.6 else w for w in words).upper(),
        lambda: re.sub(r"(?<=\w)[aeiou]", "", up, count=2, flags=re.I),
        lambda: f"{up} {CITY[city_code][1].upper()}",
        lambda: "".join(words) if len(words) > 1 else f"{name}{rng.randint(10, 99)}",
        lambda: f"{name} {rng.choice(['Private Limited', 'Limited', 'India', 'Ltd'])}",
    ]
    want = rng.randint(2, 5)
    for style in rng.sample(styles, len(styles)):
        if len(out) > want:
            break
        v = style().strip()
        if v.lower() != name.lower():
            out.add(v)
    return sorted(out)


@_memo
def merchant_universe(seed: int) -> tuple[dict, ...]:
    """Every merchant, deterministic from the seed and independent of customers. Each merchant:
    id, name, category, kind, city (None for national), pincode, mcc, vpa, variants, seed (bool)."""
    rng = random.Random(seed * 7919 + 17)
    out = []
    for i, (name, cat, kind) in enumerate(NATIONAL):
        hq = rng.choice(["MUM", "BLR", "DEL", "HYD", "CHE", "PUN"])
        out.append(dict(merchant_id=f"M{i + 1:05d}", name=name, category=cat, kind=kind, city=None, hq=hq,
                        pincode=None, mcc=rng.choice(MCC_BY_CATEGORY[cat]), vpa=f"{_slug(name)}@{rng.choice(['icici', 'hdfcbank', 'axisbank', 'ybl'])}",
                        variants=_variants(rng, name, hq, True), seed=True))
    for state, billers in UTILITY_BILLERS.items():
        for name in billers:
            out.append(dict(merchant_id=f"M{len(out) + 1:05d}", name=name, category="UTILITIES", kind="BILLER",
                            city=None, hq=None, state=state, pincode=None, mcc=4900, vpa=f"{_slug(name)}@sbi",
                            variants=_variants(rng, name, "MUM", True), seed=True))
    for name in GAS_BILLERS:
        out.append(dict(merchant_id=f"M{len(out) + 1:05d}", name=name, category="UTILITIES", kind="BILLER",
                        city=None, hq=None, pincode=None, mcc=4900, vpa=f"{_slug(name)}@sbi",
                        variants=_variants(rng, name, "MUM", True), seed=True))
    for code, city, _state, tier, *_ in CITIES:
        pins = pincodes(code)
        for _ in range(LOCAL_MERCHANTS_PER_TIER[tier]):
            cat = _pick(rng, LOCAL_CATEGORY_MIX)
            ambiguous = rng.random() < AMBIGUOUS_SHARE and cat != "OTHER"
            tpl = rng.choice(AMBIGUOUS_TEMPLATES if ambiguous else LOCAL_TEMPLATES[cat])
            name = tpl.format(p=rng.choice(FIRST if rng.random() < 0.6 else LAST), d=rng.choice(DEITIES), c=city)
            mid = f"M{len(out) + 1:05d}"
            out.append(dict(merchant_id=mid, name=name, category=cat, kind="LOCAL", city=code, hq=code,
                            pincode=rng.choice(pins), mcc=rng.choice(MCC_BY_CATEGORY[cat]),
                            vpa=f"{_slug(name)}.{int(mid[1:]) % 9973}@{rng.choice(UPI_HANDLES[:-1])}",
                            variants=_variants(rng, name, code, False), seed=rng.random() < 0.3))
    return tuple(out)


@_memo
def merchant_index(seed: int) -> dict:
    """Lookups the simulation draws from: nationals by (category, kind) and locals by (city, category)."""
    by_nat: dict = {}
    by_local: dict = {}
    by_state_utility: dict = {}
    for m in merchant_universe(seed):
        if m["kind"] == "LOCAL":
            by_local.setdefault((m["city"], m["category"]), []).append(m)
        elif m["category"] == "UTILITIES" and m.get("state"):
            by_state_utility.setdefault(m["state"], []).append(m)
        else:
            by_nat.setdefault((m["category"], m["kind"]), []).append(m)
    return {"national": by_nat, "local": by_local, "utility": by_state_utility}


# ------------------------------------------------------------------ calendar

def _festival_window(d: date) -> str | None:
    for f, name in FESTIVALS.items():
        lo = f - timedelta(days=14 if name == "Diwali" else 4)
        if lo <= d <= f + timedelta(days=2):
            return name
    return None


@_memo
def day_multipliers(d: date, archetype: str) -> tuple[float, dict]:
    """(purchase-count multiplier, per-category weight multipliers) for one day and archetype."""
    arch = ARCHETYPES[archetype]
    count, cat = 1.0, {}
    fest = _festival_window(d)
    if fest == "Diwali":
        count *= 1.3
        cat.update(ELECTRONICS=2.8, APPAREL=2.5, OTHER=1.8, DINING=1.2)
        if archetype == "ONLINE_SHOPPER_EMI":
            cat.update(ELECTRONICS=3.6, APPAREL=3.0)
    elif fest:
        count *= 1.1
        cat.update(APPAREL=1.5, DINING=1.3, GROCERY=1.2)
    if (d.month == 5) or (d.month == 6 and d.day <= 20):
        cat["TRAVEL"] = cat.get("TRAVEL", 1.0) * 2.0
    if (d.month == 12 and d.day >= 20) or (d.month == 1 and d.day <= 3):
        cat["TRAVEL"] = cat.get("TRAVEL", 1.0) * 1.6
        cat["ENTERTAINMENT"] = cat.get("ENTERTAINMENT", 1.0) * 1.4
    if d.month in (7, 8):
        cat["TRAVEL"] = cat.get("TRAVEL", 1.0) * 0.8
    if d.month in (4, 6) and d.day <= 20:
        cat["EDUCATION"] = 5.0 if archetype == "FAMILY_GROCERY_EDUCATION" else 2.0
    if d.day <= 7:
        count *= 1.0 + arch["salary"] * 0.35
    elif d.day >= 26:
        count *= 0.85
    return count, cat


def calendar_rows() -> list[tuple]:
    rows, w = [], CALENDAR_FIRST_WEEK
    while w <= CALENDAR_LAST_WEEK:
        days = [w - timedelta(days=6 - i) for i in range(7)]
        fest = sorted({n for d in days for f, n in FESTIVALS.items() if d == f
                       or (n == "Diwali" and f - timedelta(days=7) <= d <= f)})
        salary = int(any(d.day <= 5 for d in days))
        month_end = int(any((d + timedelta(days=1)).day == 1 for d in days))
        school = int(any(d.month in (4, 6) and d.day <= 20 for d in days))
        rows.append((w, days[0], int(bool(fest)), ", ".join(fest) or None, salary, month_end,
                     sum(d in HOLIDAYS for d in days), school))
        w += timedelta(days=7)
    return rows


# ------------------------------------------------------------------ customers

class Unit:
    """Everything one customer produces up to WORLD_END (as-of independent)."""

    def __init__(self, idx: int, seed: int):
        self.idx, self.seed = idx, seed
        self.rng = random.Random(seed * 1_000_003 + idx)
        self.customer: dict = {}
        self.accounts: list[dict] = []
        self.cards: list[dict] = []
        self.vpas: list[dict] = []
        self.card_txn: list[tuple] = []
        self.upi_txn: list[tuple] = []
        self.nb_txn: list[tuple] = []
        self.atm_txn: list[tuple] = []
        self.trips: list[tuple] = []
        self._seq = 0

    def next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}{_perm(self.idx * 8192 + self._seq + 1, 48271):010d}"


def _profile(u: Unit) -> None:
    rng, idx = u.rng, u.idx
    demo = idx < N_DEMO
    if demo:
        arch = ["METRO_DIGITAL_FOODIE", "SALARY_WEEK_BILL_PAYER", "ONLINE_SHOPPER_EMI"][idx]
    else:
        arch = _pick(rng, [(k, v["share"]) for k, v in ARCHETYPES.items()])
    a = ARCHETYPES[arch]
    seg = "AFFLUENT" if idx == 0 else "MASS" if demo else _pick(rng, list(zip(SEGMENTS, a["seg"])))
    tier = _pick(rng, [(t + 1, w) for t, w in enumerate(a["tiers"])])
    cities = [c for c in CITIES if c[3] == tier]
    city = CITY[["BLR", "HYD", "PUN"][idx]] if demo else cities[rng.randrange(len(cities))]
    age = _pick(rng, list(zip(AGE_BANDS, a["age"])))
    if demo:
        onboard = date(2018, 6, 1) + timedelta(days=41 * idx)
    elif rng.random() < 0.86:
        onboard = _rand_date(rng, ONBOARD_START, HISTORY_START - timedelta(days=60))
    else:
        onboard = _rand_date(rng, HISTORY_START - timedelta(days=59), WORLD_END - timedelta(days=30))
    level = a["spend"] * SEGMENT_SPEND[seg] * math.exp(rng.gauss(0.0, 0.35))
    relocate = None
    if idx == 1:
        relocate = (date(2026, 8, 24), "NAG")
    elif not demo and rng.random() < RELOCATE_SHARE:
        others = [c[0] for c in CITIES if c[0] != city[0] and c[3] <= max(tier, 2)]
        relocate = (_rand_date(rng, date(2025, 7, 1), WORLD_END - timedelta(days=30)), rng.choice(others))
    weights = {c: w * math.exp(rng.gauss(0.0, 0.25)) for c, w in a["cat"].items()}
    ch = [max(0.0, w * math.exp(rng.gauss(0.0, 0.2))) for w in a["ch"]]
    u.customer = dict(
        cif=f"C{_perm(idx + 1, 16807) % 100_000_000:08d}", idx=idx, archetype=arch, demo=demo,
        customer_name=f"{rng.choice(FIRST)} {rng.choice(LAST)}", home_city=city[0], home_state=city[2],
        home_pincode=rng.choice(pincodes(city[0])), age_band=age, segment=seg,
        income_band=_pick(rng, list(zip(INCOME_BANDS, SEGMENT_INCOME[seg]))), onboarded_on=onboard,
        level=level, cat_weights=weights, ch=ch, relocate=relocate,
        buy=a["buy"] * math.exp(rng.gauss(0.0, 0.2)), atm=a["atm"] * math.exp(rng.gauss(0.0, 0.3)),
        p2p=a["p2p"] * math.exp(rng.gauss(0.0, 0.3)), trips=a["trips"],
        has_loan=arch in ("SALARY_WEEK_BILL_PAYER", "ONLINE_SHOPPER_EMI") or rng.random() < 0.18,
    )


def _open_products(u: Unit) -> None:
    rng, c = u.rng, u.customer
    product = {"NRI": "NRE", "SME_OWNER": "CA"}.get(c["segment"], "SALARY" if rng.random() < 0.55 else "SA")
    branch = f"{c['home_city']}{rng.randint(1, 3):02d}"
    k = 0

    def account(prod: str, opened: date) -> dict:
        nonlocal k
        a = dict(account_id=f"{prod[:2]}{_perm(u.idx * 16 + k + 1, 69621):010d}", cif=c["cif"], product=prod,
                 branch_code=branch, opened_on=opened, status="ACTIVE")
        k += 1
        u.accounts.append(a)
        return a

    primary = account(product, c["onboarded_on"])
    if rng.random() < 0.18:
        account("SA", _rand_date(rng, c["onboarded_on"], max(c["onboarded_on"], HISTORY_START)))
    for j in range(2 if c["segment"] in ("AFFLUENT", "NRI") and rng.random() < 0.6 else 1):
        number = f"{rng.choice(['4', '5', '6'])}{rng.randint(10**14, 10**15 - 1)}"
        u.cards.append(dict(card_hash=card_hash(number), account_id=primary["account_id"],
                            card_type="DEBIT" if j == 0 else "CREDIT",
                            network=rng.choice(["RUPAY", "VISA", "MASTERCARD"]),
                            issued_on=_rand_date(rng, c["onboarded_on"], c["onboarded_on"] + timedelta(days=90))
                            if j else c["onboarded_on"],
                            status="ACTIVE"))
    first, last = c["customer_name"].lower().split()
    for j in range(2 if rng.random() < 0.35 else 1):
        handle = OWN_HANDLE if j == 0 and rng.random() < 0.5 else rng.choice(UPI_HANDLES[:-1])
        u.vpas.append(dict(vpa=f"{first}.{last}.{_base36(_perm(u.idx * 4 + j + 1, 69621))}@{handle}",
                           account_id=primary["account_id"],
                           linked_on=max(c["onboarded_on"], date(2019, 1, 1)) + timedelta(days=rng.randint(0, 60) if j else 0)))


def _trips(u: Unit) -> None:
    rng, c = u.rng, u.customer
    if c["idx"] == 0:
        u.trips.append((date(2026, 9, 4), date(2026, 9, 10), "GOA"))
    d = max(HISTORY_START, c["onboarded_on"])
    rate = c["trips"] / 365.0
    while d < WORLD_END:
        season = 2.5 if d.month in (5, 6) else 2.0 if d.month in (12, 1) else 1.0
        if rng.random() < rate * season:
            dest = _pick(rng, [(k, w) for k, w in TRAVEL_DESTINATIONS if k != c["home_city"]])
            length = rng.randint(2, 7)
            if not (c["idx"] == 0 and d.year == 2026 and d.month in (8, 9)):
                u.trips.append((d + timedelta(days=1), d + timedelta(days=length), dest))
            d += timedelta(days=length + 3)
        d += timedelta(days=1)


def _where(u: Unit, d: date) -> str:
    for lo, hi, dest in u.trips:
        if lo <= d <= hi:
            return dest
    reloc = u.customer["relocate"]
    if reloc and d >= reloc[0]:
        return reloc[1]
    return u.customer["home_city"]


def _amount(rng: random.Random, cat: str, level: float) -> float:
    base = AMOUNT_MEDIAN[cat] * (level ** 0.8 if cat not in ("TELECOM", "UTILITIES") else level ** 0.4)
    return round(max(20.0, base * math.exp(rng.gauss(0.0, AMOUNT_SIGMA.get(cat, 0.55)))), 2)


def _ts(rng: random.Random, d: date, lo: int = 7, hi: int = 22) -> datetime:
    return datetime(d.year, d.month, d.day, rng.randint(lo, hi), rng.randint(0, 59), rng.randint(0, 59))


def _merchant(u: Unit, cat: str, channel: str, city: str) -> dict:
    rng, idx = u.rng, merchant_index(u.seed)
    online = idx["national"].get((cat, "ONLINE"), [])
    chains = idx["national"].get((cat, "CHAIN"), [])
    billers = idx["national"].get((cat, "BILLER"), [])
    locals_ = idx["local"].get((city, cat), [])
    if cat == "UTILITIES":
        state = CITY[city][2]
        pool = idx["utility"].get(state, []) + idx["national"].get(("UTILITIES", "BILLER"), [])
        return rng.choice(pool)
    if channel in ("CARD_ECOM",):
        pool = online or billers or chains
    elif channel == "NETBANKING":
        pool = billers or online or chains
    else:
        r = rng.random()
        if locals_ and (r < 0.62 or not (chains or online)):
            favs = u.customer.setdefault("fav", {}).setdefault((city, cat), [])
            if len(favs) < 3 and (not favs or rng.random() < 0.3):
                favs.append(rng.choice(locals_))
            return rng.choice(favs)
        pool = (chains + online if channel == "UPI_P2M" else chains or online) or billers or locals_
    if not pool:
        pool = [m for m in merchant_universe(u.seed) if m["category"] == cat]
    return rng.choice(pool)


def _purchase(u: Unit, d: date, cat: str, channel: str | None = None, amount: float | None = None) -> None:
    rng, c = u.rng, u.customer
    city = _where(u, d)
    if channel is None:
        prefs = dict(zip(PURCHASE_CHANNELS, c["ch"]))
        allowed = {"CARD_POS": cat in POS_OK, "CARD_ECOM": cat in ECOM_OK, "UPI_P2M": True, "NETBANKING": cat in NB_OK}
        if city != c["home_city"] and cat not in ("TRAVEL",):
            prefs["CARD_POS"] *= 1.5
        channel = _pick(rng, [(ch, w) for ch, w in prefs.items() if allowed[ch] and w > 0] or [("UPI_P2M", 1.0)])
    if channel.startswith("CARD") and not u.cards:
        channel = "UPI_P2M"
    m = _merchant(u, cat, channel, city)
    amt = amount if amount is not None else _amount(rng, cat, c["level"])
    ts = _ts(rng, d)
    if channel.startswith("CARD"):
        card = u.cards[-1] if (len(u.cards) > 1 and rng.random() < 0.7 and u.cards[-1]["issued_on"] <= d) else u.cards[0]
        ecom = channel == "CARD_ECOM"
        mcc = m["mcc"] if rng.random() > 0.004 else rng.choice(MCC_UNMAPPED)
        mcity = CITY[m["hq"]][1] if ecom and m.get("hq") else CITY[city][1]
        mode = "ECOM" if ecom else _pick(rng, [("CHIP", 0.45), ("CONTACTLESS", 0.45), ("MAGSTRIPE", 0.1)])
        u.card_txn.append((u.next_id("C"), card["card_hash"], ts, amt, mcc, m["name"].upper()[:24], mcity.upper(), mode))
    elif channel == "UPI_P2M":
        vpa = u.vpas[0] if len(u.vpas) == 1 or rng.random() < 0.8 or u.vpas[1]["linked_on"] > d else u.vpas[1]
        pin = None
        if m["kind"] == "LOCAL":
            pin = m["pincode"] if rng.random() < 0.85 else None
        elif m["kind"] == "CHAIN":
            pin = rng.choice(pincodes(city)) if rng.random() < 0.7 else None
        u.upi_txn.append((u.next_id("U"), vpa["vpa"], m["vpa"], rng.choice(m["variants"]), pin, "P2M", ts, amt, None))
    else:
        btype = {"UTILITIES": "UTILITY", "TELECOM": "TELECOM", "EMI_LOAN": "LOAN", "EDUCATION": "EDUCATION"}.get(
            cat, "MERCHANT")
        name = m["name"] if rng.random() < 0.6 else rng.choice(m["variants"])
        u.nb_txn.append((u.next_id("N"), c["cif"], name, btype, ts, amt))


# Purchases planted for the demo customers, so their signals fire whatever the random draws do:
# unit 0 books and takes a Goa trip in Sep 2026 (TRAVEL_SURGE), unit 2 buys electronics over Diwali 2025
# (FESTIVE_ELECTRONICS). Unit 1's relocation to Nagpur on 2026-08-24 is set in _profile.
PLANTED: dict[int, dict[date, list[tuple[str, str, float]]]] = {
    0: {date(2026, 8, 29): [("TRAVEL", "CARD_ECOM", 18_400.0)],
        date(2026, 9, 1): [("TRAVEL", "CARD_ECOM", 9_800.0)],
        date(2026, 9, 4): [("TRAVEL", "UPI_P2M", 1_250.0)],
        date(2026, 9, 5): [("TRAVEL", "CARD_POS", 26_500.0), ("DINING", "CARD_POS", 3_400.0)],
        date(2026, 9, 7): [("TRAVEL", "UPI_P2M", 2_200.0), ("DINING", "UPI_P2M", 2_750.0)],
        date(2026, 9, 9): [("TRAVEL", "CARD_POS", 12_900.0)],
        date(2026, 9, 18): [("TRAVEL", "CARD_ECOM", 21_300.0)]},
    2: {date(2025, 10, 8): [("ELECTRONICS", "CARD_ECOM", 41_990.0)],
        date(2025, 10, 16): [("ELECTRONICS", "UPI_P2M", 18_499.0)],
        date(2025, 10, 19): [("ELECTRONICS", "CARD_ECOM", 6_299.0)]},
}


def _simulate(u: Unit, until: date) -> None:
    """Day by day from the customer's first day to `until`. Every draw for a day only depends on
    earlier days, so stopping at the as-of date leaves all earlier history unchanged."""
    rng, c = u.rng, u.customer
    start = max(HISTORY_START, c["onboarded_on"])
    cats = list(c["cat_weights"].items())
    emi = round(AMOUNT_MEDIAN["EMI_LOAN"] * c["level"] ** 0.8 * math.exp(rng.gauss(0.0, 0.4)), -2)
    emi_day, bill_day = rng.randint(3, 7), rng.randint(1, 9)
    utility = _merchant(u, "UTILITIES", "NETBANKING", c["home_city"])
    telecom = rng.choice(merchant_index(u.seed)["national"][("TELECOM", "BILLER")])
    lender = rng.choice(merchant_index(u.seed)["national"][("EMI_LOAN", "BILLER")])
    bill_nb = c["ch"][3] / max(sum(c["ch"]), 1e-9) > 0.15
    atms = {}
    d = start
    while d <= min(until, WORLD_END):
        count_mult, cat_mult = day_multipliers(d, c["archetype"])
        away = _where(u, d) != c["home_city"]
        lam = PURCHASE_SCALE * c["buy"] / 7.0 * count_mult * (1.3 if away else 1.0)
        for cat, channel, amount in PLANTED.get(c["idx"], {}).get(d, ()):
            _purchase(u, d, cat, channel, amount)
        for _ in range(_poisson(rng, lam)):
            w = [(k, v * cat_mult.get(k, 1.0) * (4.0 if away and k in ("TRAVEL", "DINING") else 1.0)) for k, v in cats]
            _purchase(u, d, _pick(rng, w))
        # recurring bills (net banking or UPI autopay), month-start for bill payers
        if d.day == bill_day:
            for m, cat in ((utility, "UTILITIES"), (telecom, "TELECOM")):
                if c["cat_weights"].get(cat, 0) > 0.03 or rng.random() < 0.3:
                    amt = _amount(rng, cat, c["level"])
                    if bill_nb:
                        u.nb_txn.append((u.next_id("N"), c["cif"], m["name"], "UTILITY" if cat == "UTILITIES"
                                         else "TELECOM", _ts(rng, d, 6, 11), amt))
                    else:
                        u.upi_txn.append((u.next_id("U"), u.vpas[0]["vpa"], m["vpa"], rng.choice(m["variants"]), None,
                                          "P2M", _ts(rng, d, 6, 11), amt, None))
        if d.day == emi_day and c["has_loan"]:
            if bill_nb or rng.random() < 0.5:
                u.nb_txn.append((u.next_id("N"), c["cif"], lender["name"] + " EMI", "LOAN", _ts(rng, d, 5, 9), emi))
            else:
                u.upi_txn.append((u.next_id("U"), u.vpas[0]["vpa"], lender["vpa"], rng.choice(lender["variants"]),
                                  None, "P2M", _ts(rng, d, 5, 9), emi, "EMI autopay"))
        # cash and person-to-person
        city = _where(u, d)
        if u.cards and rng.random() < c["atm"] / 30.0 * (1.4 if d.day <= 7 else 1.0):
            atm = atms.setdefault(city, f"ATM{city}{rng.randint(1, ATMS_PER_TIER[CITY[city][3]]):02d}")
            if rng.random() < 0.25:
                atm = f"ATM{city}{rng.randint(1, ATMS_PER_TIER[CITY[city][3]]):02d}"
            amt = 100 * max(1, round(rng.choice([5, 10, 20, 20, 30, 50, 100]) * c["level"] ** 0.5))
            u.atm_txn.append((u.next_id("A"), u.cards[0]["card_hash"], atm, _ts(rng, d, 6, 23), float(amt)))
        if rng.random() < c["p2p"] / 30.0:
            f, l = rng.choice(FIRST), rng.choice(LAST)
            u.upi_txn.append((u.next_id("U"), u.vpas[0]["vpa"], f"{f.lower()}{rng.randint(1, 999)}@{rng.choice(UPI_HANDLES)}",
                              f"{f} {l}", None, "P2P", _ts(rng, d), round(200 * math.exp(rng.gauss(0.6, 0.9)), 2),
                              rng.choice(P2P_REMARKS) or None))
        d += timedelta(days=1)


def simulate_unit(idx: int, seed: int, until: date = WORLD_END, events: bool = True) -> Unit:
    """Everything one customer produces up to `until`. Pure Python, so it can be unit tested without Spark.
    Profile, products and trips are drawn for the whole world first, so they never depend on `until`;
    events=False stops after the products (all the master tables need)."""
    u = Unit(idx, seed)
    _profile(u)
    _open_products(u)
    if events:
        _trips(u)
        _simulate(u, until)
    return u


def unit_rows(u: Unit, as_of: date) -> tuple:
    """Bronze rows for one customer as the source systems would show them on as_of."""
    c = u.customer
    if c["onboarded_on"] > as_of:
        return [], [], [], [], [], [], [], [], []
    end = datetime(as_of.year, as_of.month, as_of.day, 23, 59, 59)
    crm = [(c["cif"], c["customer_name"], c["home_pincode"], c["age_band"], c["income_band"], c["segment"],
            c["onboarded_on"])]
    accounts = [(a["account_id"], a["cif"], a["product"], a["branch_code"], a["opened_on"], a["status"])
                for a in u.accounts if a["opened_on"] <= as_of]
    cards = [(k["card_hash"], k["account_id"], k["card_type"], k["network"], k["issued_on"], k["status"])
             for k in u.cards if k["issued_on"] <= as_of]
    vpas = [(v["vpa"], v["account_id"], v["linked_on"]) for v in u.vpas if v["linked_on"] <= as_of]
    card_txn = [t for t in u.card_txn if t[2] <= end]
    upi_txn = [t for t in u.upi_txn if t[6] <= end]
    nb_txn = [t for t in u.nb_txn if t[4] <= end]
    atm_txn = [t for t in u.atm_txn if t[3] <= end]
    truth = [(c["cif"], c["archetype"], c["home_city"], c["relocate"][1] if c["relocate"] else None,
              c["relocate"][0] if c["relocate"] else None)]
    return crm, accounts, cards, vpas, card_txn, upi_txn, nb_txn, atm_txn, truth


def unit_truth(u: Unit) -> dict:
    """Hidden truth, for tests only."""
    c = u.customer
    return {"cif": c["cif"], "archetype": c["archetype"], "home_city": c["home_city"], "relocate": c["relocate"],
            "trips": list(u.trips), "demo": c["demo"]}


# ------------------------------------------------------------------ reference tables

def ref_rows(seed: int) -> dict[str, list[tuple]]:
    geo = [(pin, city, state, tier, lat, lon, code) for code, city, state, tier, lat, lon, _ in CITIES
           for pin in pincodes(code)]
    atms = []
    for code, city, state, tier, *_ in CITIES:
        pins = pincodes(code)
        for n in range(1, ATMS_PER_TIER[tier] + 1):
            atms.append((f"ATM{code}{n:02d}", pins[n % len(pins)], city, state, tier))
    seeds = []
    for m in merchant_universe(seed):
        if not m["seed"]:
            continue
        seeds.append((m["merchant_id"], m["name"], m["category"], "CANONICAL"))
        seeds += [(m["merchant_id"], v, m["category"], "VARIANT") for v in m["variants"] if v != m["name"]]
    return {
        "pincode_geo": geo,
        "mcc_category": [(mcc, desc, cat) for mcc, desc, cat in MCC_CATEGORY],
        "category_taxonomy": [(i + 1, cat, TAXONOMY[cat]) for i, cat in enumerate(CATEGORIES)],
        "atm_location": atms,
        "merchant_seed": seeds,
        "calendar_weekly": calendar_rows(),
    }


REF_SCHEMAS = {
    "pincode_geo": "pincode string, city string, state string, city_tier int, lat double, lon double, city_code string",
    "mcc_category": "mcc int, description string, category string",
    "category_taxonomy": "category_id int, category string, description string",
    "atm_location": "atm_id string, pincode string, city string, state string, city_tier int",
    "merchant_seed": "merchant_id string, merchant_name string, category string, name_kind string",
    "calendar_weekly": "week_end date, week_start date, festival_week int, festival_name string, salary_week int, "
                       "month_end_week int, holiday_count int, school_fee_week int",
}


# ------------------------------------------------------------------ Spark

# (output index in unit_rows, bronze or ref table, key for re-sent duplicates, columns as (name, type))
OUTPUTS = [
    (0, "bronze.crm_customer", "cif", [("cif", "string"), ("customer_name", "string"), ("home_pincode", "string"),
                                       ("age_band", "string"), ("income_band", "string"), ("segment", "string"),
                                       ("onboarded_on", "date")]),
    (1, "bronze.cbs_accounts", None, [("account_id", "string"), ("cif", "string"), ("product", "string"),
                                      ("branch_code", "string"), ("opened_on", "date"), ("status", "string")]),
    (2, "bronze.card_master", None, [("card_hash", "string"), ("account_id", "string"), ("card_type", "string"),
                                     ("network", "string"), ("issued_on", "date"), ("status", "string")]),
    (3, "bronze.vpa_registry", None, [("vpa", "string"), ("account_id", "string"), ("linked_on", "date")]),
    (4, "bronze.card_txn", "txn_id", [("txn_id", "string"), ("card_hash", "string"), ("txn_ts", "timestamp"),
                                      ("amount", "double"), ("mcc", "int"), ("merchant_name", "string"),
                                      ("merchant_city", "string"), ("entry_mode", "string")]),
    (5, "bronze.upi_txn", "txn_id", [("txn_id", "string"), ("payer_vpa", "string"), ("payee_vpa", "string"),
                                     ("payee_name", "string"), ("payee_pincode", "string"), ("txn_type", "string"),
                                     ("txn_ts", "timestamp"), ("amount", "double"), ("remarks", "string")]),
    (6, "bronze.nb_txn", "txn_id", [("txn_id", "string"), ("cif", "string"), ("biller_name", "string"),
                                    ("biller_type", "string"), ("txn_ts", "timestamp"), ("amount", "double")]),
    (7, "bronze.atm_txn", "txn_id", [("txn_id", "string"), ("card_hash", "string"), ("atm_id", "string"),
                                     ("txn_ts", "timestamp"), ("amount", "double")]),
    (8, "ref.archetype_truth", None, [("cif", "string"), ("archetype", "string"), ("home_city", "string"),
                                      ("relocated_to", "string"), ("relocated_on", "date")]),
]
BRONZE_TABLES = [t.split(".")[1] for _, t, _, _ in OUTPUTS if t.startswith("bronze.")]
UNITS_PER_BATCH = 500


def is_resent(key: str) -> bool:
    """Deterministic 0.2% of records the source sends twice."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 1_000_000 < DUPLICATE_RATE * 1_000_000


def _arrow_partition(seed: int, as_of: date, out: int, cols: list, dupe_key: str | None):
    """mapInArrow body: simulate the units of each id batch and emit one bronze table as Arrow.
    Arrow timestamps carry the simulated wall-clock time as UTC, whatever the worker's time zone.
    Re-sent records come out twice with resent=1, so the plan never has to be scanned twice."""
    def run(batches):
        import pyarrow as pa
        types = {"string": pa.string(), "date": pa.date32(), "double": pa.float64(), "int": pa.int32(),
                 "timestamp": pa.timestamp("us", tz="UTC")}
        names = [n for n, _ in cols] + ["resent"]
        schema = pa.schema([(n, types[t]) for n, t in cols] + [("resent", pa.int32())])
        key = [n for n, _ in cols].index(dupe_key) if dupe_key else None
        for batch in batches:
            ids = batch.column(0).to_pylist()
            for lo in range(0, len(ids), UNITS_PER_BATCH):
                rows = []
                for i in ids[lo:lo + UNITS_PER_BATCH]:
                    unit = simulate_unit(int(i), seed, as_of, events=out >= 4)
                    for r in unit_rows(unit, as_of)[out]:
                        rows.append((*r, 0))
                        if key is not None and is_resent(r[key]):
                            rows.append((*r, 1))
                columns = list(zip(*rows)) if rows else [[] for _ in names]
                yield pa.RecordBatch.from_arrays(
                    [pa.array(list(c), type=f.type) for c, f in zip(columns, schema)], schema=schema)
    return run


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--as-of", default=None, help="last business date in the extracts, YYYY-MM-DD (default: yesterday UTC)")
    p.add_argument("--db-prefix", default=DEFAULT_DB_PREFIX)
    p.add_argument("--customers", type=int, default=DEFAULT_CUSTOMERS, help="customers in the world")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--inject-bad-data", action="store_true", help="point 2%% of UPI payments at unknown payer VPAs")
    args, _ = p.parse_known_args(argv)
    if args.customers > MAX_CUSTOMERS:
        raise SystemExit(f"--customers above {MAX_CUSTOMERS} overflows the transaction id space")
    return args


def resolve_as_of(value: str | None) -> date:
    if value:
        return datetime.strptime(value, "%Y-%m-%d").date()
    return datetime.now(timezone.utc).date() - timedelta(days=1)


def _months(col: str):
    try:
        from pyspark.sql.functions import partitioning   # Spark 4
        return partitioning.months(col)
    except ImportError:                                  # Spark 3.5 on CDE
        from pyspark.sql import functions as F
        return F.months(col)


def _write(df, name: str, partition_col=None) -> None:
    w = df.writeTo(name).using("iceberg").tableProperty("format-version", "2")
    if partition_col is not None:
        w = w.partitionedBy(partition_col)
    w.createOrReplace()


def run(spark, args: argparse.Namespace) -> None:
    from pyspark.sql import functions as F

    as_of = resolve_as_of(args.as_of)
    bronze_db, ref_db = f"{args.db_prefix}_bronze", f"{args.db_prefix}_ref"
    logger.info("Simulating %d customers, extracts %s -> %s, seed %d", args.customers, HISTORY_START, as_of, args.seed)
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {bronze_db}")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {ref_db}")
    stamp = F.current_timestamp().alias("ingested_at")
    for name, rows in ref_rows(args.seed).items():
        _write(spark.createDataFrame(rows, REF_SCHEMAS[name]).select("*", stamp), f"{ref_db}.{name}")

    parts = max(8, args.customers // 2000)
    ingested = F.lit(datetime.now(timezone.utc).replace(tzinfo=None)).cast("timestamp")
    for out, table, dupe_key, cols in OUTPUTS:
        schema = ", ".join(f"{n} {t}" for n, t in cols) + ", resent int"
        df = spark.range(args.customers, numPartitions=parts).mapInArrow(
            _arrow_partition(args.seed, as_of, out, cols, dupe_key), schema)
        layer, name = table.split(".")
        if layer == "ref":
            _write(df.drop("resent"), f"{ref_db}.{name}")
            continue
        df = df.select(*[n for n, _ in cols], F.lit(as_of).cast("date").alias("batch_as_of"),
                       (ingested + F.make_interval(secs=F.col("resent"))).alias("ingested_at"))
        if name == "upi_txn" and args.inject_bad_data:
            logger.warning("--inject-bad-data: pointing 2%% of UPI payments at unknown payer VPAs")
            bad = (F.abs(F.xxhash64("txn_id")) % 10_000) < BAD_VPA_RATE * 10_000
            df = df.withColumn("payer_vpa", F.when(bad, F.concat(F.lit("unknown"), F.substring("txn_id", 2, 6),
                                                                 F.lit("@ybl"))).otherwise(F.col("payer_vpa")))
        _write(df, f"{bronze_db}.{name}", _months("txn_ts") if "txn_ts" in dict(cols) else None)

    for t in BRONZE_TABLES:
        logger.info("%s.%s: %d rows", bronze_db, t, spark.table(f"{bronze_db}.{t}").count())
    logger.info("Bronze load as of %s complete", as_of)


def main(argv=None, spark=None) -> None:
    from pyspark.sql import SparkSession

    args = parse_args(argv)
    own = spark is None
    spark = spark or SparkSession.builder.appName("spend-generate-bronze").getOrCreate()
    try:
        run(spark, args)
    finally:
        if own:
            spark.stop()


if __name__ == "__main__":
    main()
