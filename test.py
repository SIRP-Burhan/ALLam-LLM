#!/usr/bin/env python3
"""
Reasoning Evaluation for ALLaM-7B-Instruct-preview
Tests: Math, Logic, Common Sense, Arabic, Comprehension,
       API Routing, JSON Schema, System Prompt Adherence, Relevance Extraction
"""

import json
import time
import re
import requests
from dataclasses import dataclass, field
from typing import List, Optional

# ── Configuration ────────────────────────────────────────────────────────────
API_URL = "http://147.224.62.203:8000/v1/chat/completions"
MODEL = "humain-ai/ALLaM-7B-Instruct-preview"
MAX_TOKENS = 1024


# ── Test Framework ───────────────────────────────────────────────────────────

@dataclass
class TestCase:
    category: str
    question: str
    expected: str                                  # keyword that SHOULD appear
    system: str = "You are a helpful assistant. Think step by step, then give your final answer clearly."
    must_not_contain: Optional[List[str]] = None   # keywords that MUST NOT appear
    must_contain_all: Optional[List[str]] = None   # ALL of these must appear
    json_valid: bool = False                       # response must parse as JSON
    json_keys: Optional[List[str]] = None          # JSON must contain these top-level keys


# ── Checker ──────────────────────────────────────────────────────────────────

def check_answer(response: str, test: TestCase) -> tuple:
    """Returns (passed, reason)."""
    low = response.lower()

    # Basic keyword match
    if test.expected and test.expected.lower() not in low:
        return False, f"Missing expected keyword: '{test.expected}'"

    # Must not contain any forbidden keywords
    if test.must_not_contain:
        for bad in test.must_not_contain:
            if bad.lower() in low:
                return False, f"Contains forbidden keyword: '{bad}'"

    # Must contain ALL required keywords
    if test.must_contain_all:
        for req in test.must_contain_all:
            if req.lower() not in low:
                return False, f"Missing required keyword: '{req}'"

    # JSON validity check
    if test.json_valid:
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        json_str = json_match.group(1).strip() if json_match else response.strip()
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return False, "Response is not valid JSON"

        if test.json_keys:
            if not isinstance(parsed, dict):
                return False, "JSON root is not an object"
            missing = [k for k in test.json_keys if k not in parsed]
            if missing:
                return False, f"JSON missing keys: {missing}"

    return True, "OK"


# ═════════════════════════════════════════════════════════════════════════════
#  SHARED CONTEXT
# ═════════════════════════════════════════════════════════════════════════════

API_ENDPOINT_LIST = """Available API endpoints:
1. POST /api/v1/orders          — Create a new purchase order
2. GET  /api/v1/orders/{id}     — Get order details by ID
3. POST /api/v1/auth/login      — Authenticate user and return JWT token
4. POST /api/v1/auth/register   — Register a new user account
5. GET  /api/v1/products        — List all products with optional filters
6. GET  /api/v1/products/{id}   — Get single product details
7. PUT  /api/v1/users/{id}      — Update user profile information
8. POST /api/v1/payments        — Process a payment transaction
9. GET  /api/v1/analytics/sales — Get sales analytics and reports
10. DELETE /api/v1/orders/{id}  — Cancel an existing order
11. POST /api/v1/support/ticket — Create a customer support ticket
12. GET  /api/v1/inventory      — Check current stock levels"""

WEBPAGE_CONTENT = """
<article>
<h1>Understanding Electric Vehicle Batteries</h1>

<p>Electric vehicles (EVs) have gained significant popularity in recent years.
The global EV market grew by 35% in 2024, reaching over 17 million units sold worldwide.</p>

<p>The most common battery type used in EVs is the lithium-ion battery.
These batteries typically last between 8 to 15 years depending on usage patterns,
climate conditions, and charging habits. Most manufacturers offer an 8-year warranty
on their battery packs.</p>

<p>Battery degradation is a natural process. On average, EV batteries lose about
2-3% of their capacity per year. After 200,000 miles, most batteries still retain
around 80-85% of their original capacity.</p>

<p>Charging infrastructure has expanded rapidly. As of 2024, there are over
180,000 public charging stations in the United States alone. Fast chargers
(DC fast charging) can charge a battery from 10% to 80% in approximately
20-40 minutes depending on the vehicle and charger specifications.</p>

<p>The cost of EV batteries has dropped dramatically. In 2010, battery packs
cost approximately $1,100 per kilowatt-hour. By 2024, that figure has fallen
to around $139 per kWh, a reduction of nearly 87%.</p>

<p>Recycling of EV batteries is becoming increasingly important. Companies like
Redwood Materials and Li-Cycle are developing processes to recover up to 95% of
critical minerals from used batteries, including lithium, cobalt, and nickel.</p>

<p>Solid-state batteries represent the next frontier in EV technology. These
batteries promise 2-3x the energy density of current lithium-ion batteries,
faster charging times, and improved safety. Toyota and Samsung SDI have
announced plans to begin mass production of solid-state batteries by 2027.</p>
</article>"""


# ═════════════════════════════════════════════════════════════════════════════
#  TEST CASES
# ═════════════════════════════════════════════════════════════════════════════

TESTS = [
    # ── Math Reasoning ───────────────────────────────────────────────────────
    TestCase(
        category="Math",
        question="If a train travels 120 km in 2 hours, and then 180 km in 3 hours, what is its average speed for the entire journey?",
        expected="60",
    ),
    TestCase(
        category="Math",
        question="A store sells a shirt for $45 after a 25% discount. What was the original price?",
        expected="60",
    ),
    TestCase(
        category="Math",
        question="If 3x + 7 = 22, what is the value of x?",
        expected="5",
    ),
    TestCase(
        category="Math",
        question="A rectangle has a perimeter of 30 cm and a length of 10 cm. What is its area?",
        expected="50",
    ),
    TestCase(
        category="Math",
        question="What is 15% of 240?",
        expected="36",
    ),

    # ── Logical Reasoning ────────────────────────────────────────────────────
    TestCase(
        category="Logic",
        question="All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
        expected="no",
    ),
    TestCase(
        category="Logic",
        question="If it rains, the ground gets wet. The ground is wet. Can we conclude it rained?",
        expected="no",
    ),
    TestCase(
        category="Logic",
        question="Tom is taller than Jerry. Jerry is taller than Spike. Who is the shortest?",
        expected="spike",
    ),
    TestCase(
        category="Logic",
        question="A farmer has 17 sheep. All but 9 die. How many sheep are left?",
        expected="9",
    ),
    TestCase(
        category="Logic",
        question="If you rearrange the letters 'CIFAIPC', you get the name of a(n): a) city b) ocean c) animal d) country",
        expected="ocean",
    ),

    # ── Common Sense ─────────────────────────────────────────────────────────
    TestCase(
        category="Common Sense",
        question="You have a cup of hot coffee and a cup of cold water. If you leave both on a table for 3 hours, which one will be closer to room temperature?",
        expected="both",
    ),
    TestCase(
        category="Common Sense",
        question="If you drop a ball and a feather in a vacuum chamber (no air), which hits the ground first?",
        expected="same",
    ),
    TestCase(
        category="Common Sense",
        question="A man pushes his car to a hotel and loses his money. What is happening?",
        expected="monopoly",
    ),
    TestCase(
        category="Common Sense",
        question="What weighs more: a kilogram of steel or a kilogram of feathers?",
        expected="same",
    ),
    TestCase(
        category="Common Sense",
        question="If you have 3 apples and take away 2, how many apples do YOU have?",
        expected="2",
    ),

    # ── Arabic Reasoning ─────────────────────────────────────────────────────
    TestCase(
        category="Arabic",
        question="إذا كان عمر أحمد ضعف عمر سارة، وعمر سارة 15 سنة، فكم عمر أحمد؟",
        expected="30",
    ),
    TestCase(
        category="Arabic",
        question="ما هو الشيء الذي يمشي بلا أرجل ويبكي بلا عيون؟",
        expected="سحاب",
    ),
    TestCase(
        category="Arabic",
        question="أكمل المثل العربي: 'من جدّ...'",
        expected="وجد",
    ),
    TestCase(
        category="Arabic",
        question="رتب الكلمات التالية لتكوين جملة مفيدة: 'المدرسة - إلى - ذهب - الطالب'",
        expected="ذهب الطالب إلى المدرسة",
    ),
    TestCase(
        category="Arabic",
        question="إذا كان اليوم الثلاثاء، فما هو اليوم بعد ثلاثة أيام؟",
        expected="جمعة",
    ),

    # ── Reading Comprehension ────────────────────────────────────────────────
    TestCase(
        category="Comprehension",
        question=(
            "Read the following passage and answer:\n\n"
            "'The Amazon rainforest produces approximately 20% of the world's oxygen. "
            "However, it also consumes roughly the same amount through decomposition. "
            "The net oxygen contribution of the Amazon is therefore close to zero.'\n\n"
            "True or False: The Amazon is a major net producer of the world's oxygen."
        ),
        expected="false",
    ),
    TestCase(
        category="Comprehension",
        question=(
            "Read carefully:\n\n"
            "'Sarah has 3 brothers. Each brother has 2 sisters.'\n\n"
            "How many sisters does Sarah have?"
        ),
        expected="1",
    ),
    TestCase(
        category="Comprehension",
        question=(
            "A study found that cities with more ice cream sales also have higher crime rates. "
            "Does this mean ice cream causes crime? Explain why or why not."
        ),
        expected="correlation",
    ),

    # ═════════════════════════════════════════════════════════════════════════
    #  API ENDPOINT CLASSIFICATION & ROUTING
    # ═════════════════════════════════════════════════════════════════════════
    TestCase(
        category="API Routing",
        question=f"""Given the user request and available endpoints, respond with ONLY the HTTP method and endpoint path. No explanation.

{API_ENDPOINT_LIST}

User request: "I want to buy the premium subscription plan"
""",
        expected="/api/v1/orders",
        system="You are a precise API routing assistant. Respond with ONLY the HTTP method and endpoint path. Nothing else.",
    ),
    TestCase(
        category="API Routing",
        question=f"""Given the user request and available endpoints, respond with ONLY the HTTP method and endpoint path. No explanation.

{API_ENDPOINT_LIST}

User request: "I forgot my password and need to get back into my account"
""",
        expected="/api/v1/auth/login",
        system="You are a precise API routing assistant. Respond with ONLY the HTTP method and endpoint path. Nothing else.",
    ),
    TestCase(
        category="API Routing",
        question=f"""Given the user request and available endpoints, respond with ONLY the HTTP method and endpoint path. No explanation.

{API_ENDPOINT_LIST}

User request: "Show me how much revenue we made last quarter"
""",
        expected="/api/v1/analytics/sales",
        system="You are a precise API routing assistant. Respond with ONLY the HTTP method and endpoint path. Nothing else.",
    ),
    TestCase(
        category="API Routing",
        question=f"""Given the user request and available endpoints, respond with ONLY the HTTP method and endpoint path. No explanation.

{API_ENDPOINT_LIST}

User request: "Cancel order #4521, I changed my mind"
""",
        expected="DELETE",
        must_contain_all=["DELETE", "/api/v1/orders"],
        system="You are a precise API routing assistant. Respond with ONLY the HTTP method and endpoint path. Nothing else.",
    ),
    TestCase(
        category="API Routing",
        question=f"""Given the user request and available endpoints, respond with ONLY the HTTP method and endpoint path. No explanation.

{API_ENDPOINT_LIST}

User request: "Do you have the Nike Air Max in size 10?"
""",
        expected="/api/v1/inventory",
        system="You are a precise API routing assistant. Respond with ONLY the HTTP method and endpoint path. Nothing else.",
    ),
    TestCase(
        category="API Routing",
        question=f"""Given the user request and available endpoints, respond with ONLY the HTTP method and endpoint path. No explanation.

{API_ENDPOINT_LIST}

User request: "I need to change my email address and phone number"
""",
        expected="/api/v1/users",
        must_contain_all=["PUT", "/api/v1/users"],
        system="You are a precise API routing assistant. Respond with ONLY the HTTP method and endpoint path. Nothing else.",
    ),
    TestCase(
        category="API Routing",
        question=f"""A user request may need MULTIPLE endpoints called in sequence. List all required endpoints in order, one per line. Respond with ONLY the HTTP methods and paths.

{API_ENDPOINT_LIST}

User request: "I'm a new customer. I want to create an account and then buy the red sneakers (product ID: 42)."
""",
        expected="/api/v1/auth/register",
        must_contain_all=["/api/v1/auth/register", "/api/v1/orders"],
        system="You are a precise API routing assistant. Respond with ONLY the HTTP methods and endpoint paths, one per line.",
    ),

    # ═════════════════════════════════════════════════════════════════════════
    #  JSON SCHEMA GENERATION
    # ═════════════════════════════════════════════════════════════════════════
    TestCase(
        category="JSON Schema",
        question="""Generate a valid JSON object representing a patient medical record with:
- patient_id (string)
- full_name (string)
- date_of_birth (ISO 8601 date string)
- blood_type (string)
- allergies (array of strings)
- medications (array of objects, each with: name, dosage, frequency)
- emergency_contact (object with: name, relationship, phone)
- visits (array of objects, each with: date, doctor, diagnosis, notes)

Include realistic sample data with at least 2 medications and 2 visits. Return ONLY valid JSON.""",
        expected="patient_id",
        json_valid=True,
        json_keys=["patient_id", "full_name", "date_of_birth", "allergies", "medications", "emergency_contact", "visits"],
        system="You are a JSON generator. Return ONLY valid JSON with no explanation, no markdown, no commentary.",
    ),
    TestCase(
        category="JSON Schema",
        question="""Generate a valid JSON object for an e-commerce product catalog entry with:
- product_id, sku (strings)
- name (string), description (string)
- category (string), subcategory (string)
- price (object with: amount as number, currency as string)
- variants (array of objects, each with: variant_id, color, size, stock_count, price_modifier)
- images (array of objects with: url, alt_text, is_primary boolean)
- metadata (object with: weight_kg as number, dimensions as object with length/width/height, material, country_of_origin)
- ratings (object with: average as number, count as integer, distribution as object with keys "1" through "5")

Use realistic sample data. Return ONLY valid JSON.""",
        expected="product_id",
        json_valid=True,
        json_keys=["product_id", "sku", "name", "price", "variants", "images", "metadata", "ratings"],
        system="You are a JSON generator. Return ONLY valid JSON with no explanation, no markdown, no commentary.",
    ),
    TestCase(
        category="JSON Schema",
        question="""Generate a valid JSON object for a multi-leg flight booking with:
- booking_reference (string)
- status (enum: "confirmed", "pending", "cancelled")
- passengers (array of objects: first_name, last_name, passport_number, seat_preference, meal_preference, frequent_flyer object with airline and number)
- flights (array of objects: flight_number, airline, departure object with airport/city/datetime/terminal/gate, arrival object with same structure, aircraft_type, cabin_class, duration_minutes)
- payment (object with: method, card_last_four, amount, currency, billing_address as nested object)
- ancillaries (array of objects: type like "baggage"/"insurance"/"lounge", description, price)

Include 2 passengers, 3 flight legs, and 2 ancillaries. Return ONLY valid JSON.""",
        expected="booking_reference",
        json_valid=True,
        json_keys=["booking_reference", "status", "passengers", "flights", "payment", "ancillaries"],
        system="You are a JSON generator. Return ONLY valid JSON with no explanation, no markdown, no commentary.",
    ),
    TestCase(
        category="JSON Schema",
        question="""Convert this natural language description into a valid JSON API error response:

"The user tried to update their profile but the email they provided is already registered to another account.
The request ID was req_8f2a9c3d. It happened on the users/profile endpoint.
The email field specifically failed validation.
Suggest they use a different email or recover their existing account."

Return a structured JSON error response with: error_code, http_status, message, request_id, timestamp, path, field_errors array, and suggestions array. Return ONLY valid JSON.""",
        expected="error_code",
        json_valid=True,
        json_keys=["error_code", "http_status", "message", "request_id"],
        system="You are a JSON generator. Return ONLY valid JSON with no explanation, no markdown, no commentary.",
    ),
    TestCase(
        category="JSON Schema",
        question="""Generate a JSON Schema (draft 2020-12) that validates a restaurant menu API response. The schema should enforce:
- "restaurant_name" required string
- "menu_sections" required array where each item has:
  - "section_name" required string
  - "items" required array where each item has:
    - "name" required string, minLength 1
    - "price" required number, minimum 0
    - "description" optional string
    - "allergens" optional array of strings with enum: ["gluten","dairy","nuts","shellfish","eggs","soy"]
    - "is_vegetarian" optional boolean
    - "spice_level" optional integer, minimum 0, maximum 5

Return a valid JSON Schema document. Return ONLY valid JSON.""",
        expected="$schema",
        json_valid=True,
        must_contain_all=["required", "properties", "allergens"],
        system="You are a JSON Schema generator. Return ONLY a valid JSON Schema document with no explanation.",
    ),

    # ═════════════════════════════════════════════════════════════════════════
    #  SYSTEM PROMPT ADHERENCE
    # ═════════════════════════════════════════════════════════════════════════
    TestCase(
        category="Sys Prompt",
        question="What is the capital of France?",
        expected="paris",
        must_not_contain=["I ", "my ", "me ", "personally"],
        system="You are a geography expert. You must ALWAYS respond in exactly 3 words. Never use more than 3 words. Never use first person pronouns.",
    ),
    TestCase(
        category="Sys Prompt",
        question="Tell me about machine learning, neural networks, and the history of AI.",
        expected="",
        must_not_contain=["machine learning", "neural network", "history of ai"],
        system=(
            "You are a deflection bot. No matter what the user asks, you must ONLY talk about cats. "
            "Never answer the user's actual question. Every response must be about cats. "
            "Do not acknowledge the user's topic at all."
        ),
    ),
    TestCase(
        category="Sys Prompt",
        question="What is 2 + 2?",
        expected="```",
        system=(
            "You are a code-only assistant. You must ALWAYS respond with a code block. "
            "Never write any text outside of a code block. Every single response must be "
            "wrapped in triple backticks with a language identifier. No prose allowed."
        ),
    ),
    TestCase(
        category="Sys Prompt",
        question="Explain quantum computing to me in simple terms.",
        expected="",
        must_not_contain=["happy to", "glad to", "sure", "certainly", "of course", "great question"],
        system=(
            "You are a terse technical writer. Rules you MUST follow:\n"
            "1. Never use filler phrases like 'happy to help', 'great question', 'certainly', 'of course'\n"
            "2. Never exceed 50 words total\n"
            "3. Use only short declarative sentences\n"
            "4. No greetings, no sign-offs, no pleasantries"
        ),
    ),
    TestCase(
        category="Sys Prompt",
        question="I'm feeling really sad today, can you cheer me up with a joke?",
        expected="",
        must_not_contain=["joke", "laugh", "funny", "haha", "cheer"],
        system=(
            "You are a formal business analyst. You must NEVER tell jokes, use humor, or be casual. "
            "If a user asks for anything non-business related, politely redirect them to business topics. "
            "Always maintain a strictly professional tone. Never use exclamation marks."
        ),
    ),
    TestCase(
        category="Sys Prompt",
        question="Translate 'hello world' to Spanish.",
        expected="XML_RESPONSE",
        must_contain_all=["<translation", "</translation"],
        must_not_contain=["{", "json"],
        system=(
            "You are a translation API. You must ALWAYS respond in XML format only. "
            "Use this exact structure:\n"
            "<XML_RESPONSE>\n"
            "  <translation source_lang=\"...\" target_lang=\"...\">\n"
            "    <text>...</text>\n"
            "  </translation>\n"
            "</XML_RESPONSE>\n"
            "Never use JSON. Never write prose. Only XML."
        ),
    ),
    TestCase(
        category="Sys Prompt",
        question="Who is the president of the United States?",
        expected="",
        must_not_contain=["president", "united states", "america", "usa", "biden", "trump"],
        system=(
            "You are a pirate character named Captain Redbeard. You REFUSE to answer any political questions. "
            "When asked about politics or government, you MUST deflect by talking about pirate adventures, "
            "treasure, and the sea. Never break character. Never answer political questions."
        ),
    ),

    # ═════════════════════════════════════════════════════════════════════════
    #  RELEVANCE EXTRACTION
    # ═════════════════════════════════════════════════════════════════════════
    TestCase(
        category="Relevance",
        question=f"""Given the following webpage content, extract ONLY the sentences that directly answer this question: "How long do EV batteries last?"

Return the relevant sentences verbatim. Do not add any commentary.

WEBPAGE:
{WEBPAGE_CONTENT}""",
        expected="8 to 15 years",
        must_not_contain=["solid-state", "recycling", "charging station"],
        system="You are a precise information extraction assistant. Extract only the sentences that directly answer the question. No commentary.",
    ),
    TestCase(
        category="Relevance",
        question=f"""Given the following webpage content, extract ONLY the sentences that directly answer this question: "How much do EV batteries cost?"

Return the relevant sentences verbatim. Do not paraphrase.

WEBPAGE:
{WEBPAGE_CONTENT}""",
        expected="139",
        must_contain_all=["$"],
        must_not_contain=["solid-state", "recycling", "charging station", "warranty"],
        system="You are a precise information extraction assistant. Extract only the sentences that directly answer the question. No commentary.",
    ),
    TestCase(
        category="Relevance",
        question=f"""Given the following webpage content, extract ONLY the sentences that directly answer this question: "What is the future of EV battery technology?"

Return the relevant sentences verbatim.

WEBPAGE:
{WEBPAGE_CONTENT}""",
        expected="solid-state",
        must_not_contain=["cost", "$1,100", "charging station", "180,000"],
        system="You are a precise information extraction assistant. Extract only the sentences that directly answer the question. No commentary.",
    ),
    TestCase(
        category="Relevance",
        question=f"""Given the following webpage content, does it contain information that answers this question: "What is the top speed of a Tesla Model 3?"

Respond with ONLY one of:
- "ANSWERABLE" if the page contains relevant information
- "NOT_ANSWERABLE" if the page does not contain relevant information

WEBPAGE:
{WEBPAGE_CONTENT}""",
        expected="NOT_ANSWERABLE",
        system="You are a relevance classifier. Respond with ONLY 'ANSWERABLE' or 'NOT_ANSWERABLE'. No explanation.",
    ),
    TestCase(
        category="Relevance",
        question=f"""Given the following webpage content, extract ONLY the sentences relevant to "EV battery degradation rate".

For each sentence you extract, prefix it with [RELEVANT]. If a sentence is partially relevant, prefix with [PARTIAL].

WEBPAGE:
{WEBPAGE_CONTENT}""",
        expected="2-3%",
        must_contain_all=["[RELEVANT]"],
        must_not_contain=["solid-state", "$1,100", "charging station"],
        system="You are a precise information extraction assistant. Use [RELEVANT] and [PARTIAL] prefixes as instructed.",
    ),
    TestCase(
        category="Relevance",
        question=f"""Given the following webpage content and TWO questions, extract the relevant sentences for EACH question separately.

Question A: "How fast can you charge an EV?"
Question B: "How are old EV batteries recycled?"

Format your response as:
QUESTION_A:
<relevant sentences>

QUESTION_B:
<relevant sentences>

WEBPAGE:
{WEBPAGE_CONTENT}""",
        expected="QUESTION_A",
        must_contain_all=["QUESTION_A", "QUESTION_B", "20-40 minutes", "95%"],
        system="You are a precise information extraction assistant. Organize your extractions by question as instructed.",
    ),
]


# ── Evaluation Engine ────────────────────────────────────────────────────────

def query_model(question: str, system: str) -> str:
    """Send a question to the model and return the response text."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR] {e}"


def run_evaluation():
    """Run all test cases and print results."""
    results = {}
    all_results = []
    categories = list(dict.fromkeys(t.category for t in TESTS))

    print("=" * 70)
    print(f"  REASONING EVALUATION — {MODEL}")
    print(f"  {len(TESTS)} questions across {len(categories)} categories")
    print("=" * 70)

    for i, test in enumerate(TESTS, 1):
        short_q = test.question.replace('\n', ' ')[:80]
        print(f"\n[{i}/{len(TESTS)}] ({test.category}) {short_q}...")

        start = time.time()
        response = query_model(test.question, test.system)
        elapsed = time.time() - start

        passed, reason = check_answer(response, test)
        status = "✅ PASS" if passed else "❌ FAIL"

        print(f"  {status} ({elapsed:.1f}s) — {reason}")
        print(f"  Response: {response[:200]}{'...' if len(response) > 200 else ''}")

        if test.category not in results:
            results[test.category] = {"pass": 0, "fail": 0}
        results[test.category]["pass" if passed else "fail"] += 1
        all_results.append({
            "category": test.category,
            "question": test.question[:200],
            "expected": test.expected,
            "response": response,
            "passed": passed,
            "reason": reason,
            "time": round(elapsed, 2),
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    total_pass = sum(r["pass"] for r in results.values())
    total_fail = sum(r["fail"] for r in results.values())
    total = total_pass + total_fail

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Category':<20} {'Pass':>6} {'Fail':>6} {'Score':>8}")
    print("  " + "-" * 46)
    for cat in categories:
        r = results[cat]
        score = r['pass'] / (r['pass'] + r['fail']) * 100
        bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
        print(f"  {cat:<20} {r['pass']:>6} {r['fail']:>6} {score:>7.1f}% {bar}")
    print("  " + "-" * 46)
    overall = total_pass / total * 100
    print(f"  {'OVERALL':<20} {total_pass:>6} {total_fail:>6} {overall:>7.1f}%")
    print("=" * 70)

    # Save detailed results
    output = {
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total": total,
            "passed": total_pass,
            "failed": total_fail,
            "score_pct": round(overall, 2),
            "by_category": {
                cat: round(results[cat]["pass"] / (results[cat]["pass"] + results[cat]["fail"]) * 100, 2)
                for cat in categories
            },
        },
        "details": all_results,
    }
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: eval_results.json")


if __name__ == "__main__":
    run_evaluation()
