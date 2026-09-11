"""
price_agent.py — An AI agent that searches the web for Australian retailer
prices (in AUD) AND checks reviews, using two different kinds of tools so
you can see both patterns.

THE TWO TOOL TYPES:
    1. web_search               — a SERVER-SIDE tool. Anthropic's own
                                   infrastructure runs the search; the
                                   result comes back already resolved
                                   inside the same API response. You never
                                   see stop_reason == "tool_use" for this
                                   one alone.
    2. ask_user_clarification,  — CLIENT-SIDE tools (plain Python
       check_reviews              functions). When Claude wants to call
                                   one, the API returns stop_reason ==
                                   "tool_use" and hands you a tool_use
                                   block. YOUR CODE has to run the function
                                   and send the result back before Claude
                                   continues. This is the classic "agent
                                   loop."

Mixing both in one request is intentional: it shows what happens when a
server tool and client tools all appear in the picture — the API resolves
web_search on its own, but pauses for you on the other two.

HANDLING VAGUE REQUESTS:
    Claude decides for itself whether a query is specific enough to search
    (e.g. "mechanical keyboard") or too vague (e.g. "keyboard"). For a vague
    query it calls ask_user_clarification, which prints a short question and
    a few concrete options at the console and blocks on input() until the
    human answers. Once the request is resolved to a category rather than a
    single product, Claude researches one specific model per price bracket
    (Budget / Mid-range / Premium) and prices each of those.

FEEDBACK LOOP:
    After each result, the console asks for optional feedback. Feedback
    isn't just logged for later — it's appended to the SAME conversation
    (run_agent_turn is called again on the same `messages` list) so Claude
    re-searches with full context of what it already found. E.g. it search-
    es an iPhone and defaults to the 256GB model; feedback like "I want the
    512GB one instead" makes it re-search that specific variant rather than
    starting over. This is also how you correct it — e.g. feedback saying a
    price looks stale prompts it to re-verify against the retailer's own
    product page. Every feedback string is still appended to feedback.log
    (query + timestamp) as a record, whether or not it changes anything.

RATE LIMITING:
    Every API request is wrapped in call_with_retry(), which catches 429
    (rate limit) errors and retries with backoff — honoring the server's
    retry-after header when present, or exponential backoff with jitter
    otherwise. Watch the console for [rate limit] messages if it kicks in.

REQUIREMENTS:
    pip install anthropic python-dotenv
    Put ANTHROPIC_API_KEY=your-key-here in a local .env file

USAGE:
    python price_agent.py "Sony WH-1000XM5 headphones"
    python price_agent.py "mechanical keyboard"   # will ask to narrow down
"""

import sys
import json
import time
import random
import anthropic
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY (and friends) from a local .env file

MODEL = "claude-sonnet-4-5"

# How many times to retry a single API call after a 429 before giving up.
MAX_RETRIES = 5

FEEDBACK_LOG = "feedback.log"


# ---------------------------------------------------------------------------
# STEP 1: Define the client-side tools as normal Python functions.
#
# ask_user_clarification talks to the human; check_reviews is a STUB that
# returns made-up data so you can see the loop work without needing a real
# reviews API. To make check_reviews real, replace its body with a call to
# something like a retailer API, a scraping service, or your own database
# of scraped reviews.
# ---------------------------------------------------------------------------
def ask_user_clarification(question: str, options: list) -> dict:
    """Pause the agent loop and ask the human at the console to narrow down
    a vague request. Claude decides the question and the option list; we
    just render it and collect the answer.
    """
    print(f"\n  [clarification needed] {question}")
    for i, option in enumerate(options, start=1):
        print(f"    {i}. {option}")
    print(f"    {len(options) + 1}. Something else (type your own)")

    choice = input("  Your choice: ").strip()

    if choice.isdigit() and 1 <= int(choice) <= len(options):
        answer = options[int(choice) - 1]
    else:
        # Either they picked "something else" or just typed free text —
        # either way, use exactly what they typed.
        answer = choice

    return {"question": question, "answer": answer}


def check_reviews(product: str, retailer: str) -> dict:
    print(f"    [tool call] check_reviews(product={product!r}, retailer={retailer!r})")
    # --- STUB DATA — replace with a real lookup in production ---
    fake_ratings = {
        "amazon au": {"rating": 4.6, "review_count": 18420},
        "jb hi-fi": {"rating": 4.5, "review_count": 5210},
        "the good guys": {"rating": 4.3, "review_count": 2894},
        "officeworks": {"rating": 4.2, "review_count": 1043},
        "harvey norman": {"rating": 4.1, "review_count": 1567},
    }
    data = fake_ratings.get(retailer.lower(), {"rating": None, "review_count": 0})
    return {
        "product": product,
        "retailer": retailer,
        "rating": data["rating"],
        "review_count": data["review_count"],
        "note": "Simulated data for demo purposes — swap in a real reviews source.",
    }


# ---------------------------------------------------------------------------
# STEP 2: Describe all three tools to Claude.
# web_search uses Anthropic's built-in schema (type + name). The other two
# are custom tools: we write their name, description, and the JSON schema
# for their inputs ourselves — this is how Claude knows when and how to
# call them.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "web_search_20250305",
        "name": "web_search",
        # Higher than a single-product lookup needs, since a resolved
        # category (e.g. "mechanical keyboard") means researching 3 models
        # across price brackets, each priced at multiple retailers.
        "max_uses": 12,
        # Bias/restrict results to Australia so retailers and prices found
        # are relevant to an AU shopper (e.g. amazon.com.au, jbhifi.com.au).
        "user_location": {
            "type": "approximate",
            "country": "AU",
        },
    },
    {
        "name": "ask_user_clarification",
        "description": (
            "Ask the human a clarifying question when their request is too "
            "vague to search effectively (e.g. just 'keyboard' or "
            "'headphones', with no type, use case, or budget given). "
            "Propose 3-5 concrete, mutually distinct options — the most "
            "common ways people narrow this kind of request (sub-types or "
            "use cases, not prices) — the human can also type a free-text "
            "answer instead of picking one. "
            "Use your own judgement about whether to call this: if the "
            "request already specifies a product type, brand, or use case "
            "(e.g. 'mechanical keyboard', 'noise cancelling headphones "
            "under $200', 'Sony WH-1000XM5'), it is specific enough — do "
            "NOT ask, just proceed and pick sensible options yourself. Call "
            "this at most once per request."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "A short question narrowing down what the user wants, e.g. 'What kind of keyboard are you after?'",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-5 concrete options, e.g. ['Mechanical keyboard', 'Wireless/Bluetooth keyboard', 'Gaming keyboard', 'Compact/60% keyboard']",
                },
            },
            "required": ["question", "options"],
        },
    },
    {
        "name": "check_reviews",
        "description": (
            "Look up the customer review rating and review count for a "
            "specific product at a specific Australian retailer. Use this "
            "after finding prices, to help judge which option is the best "
            "value."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product": {
                    "type": "string",
                    "description": "The product name, e.g. 'Sony WH-1000XM5 headphones'",
                },
                "retailer": {
                    "type": "string",
                    "description": "The Australian retailer name, e.g. 'Amazon AU', 'JB Hi-Fi', 'Officeworks'",
                },
            },
            "required": ["product", "retailer"],
        },
    },
]


def execute_client_tool(name: str, tool_input: dict):
    """Dispatch a client-side tool call to the right Python function."""
    if name == "ask_user_clarification":
        return ask_user_clarification(**tool_input)
    if name == "check_reviews":
        return check_reviews(**tool_input)
    raise ValueError(f"Unknown client-side tool: {name}")


# ---------------------------------------------------------------------------
# RATE LIMITING
#
# Anthropic enforces per-organization limits on requests-per-minute (RPM),
# input tokens-per-minute (ITPM), and output tokens-per-minute (OTPM). Exceed
# any one and the API returns a 429 with a `retry-after` header telling you
# how long to wait. The SDK itself already retries a couple of times by
# default, but this wrapper makes that behavior explicit and visible so you
# can watch it happen, and extends it with jitter so that if you were running
# many agents at once, they wouldn't all retry at the exact same moment.
# ---------------------------------------------------------------------------
def call_with_retry(client: anthropic.Anthropic, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            if attempt == MAX_RETRIES:
                print(f"  [rate limit] giving up after {MAX_RETRIES} attempts.")
                raise

            # Prefer the server's own retry-after header when it's present —
            # it knows exactly when your quota resets. Otherwise fall back to
            # exponential backoff (1s, 2s, 4s, 8s...) with a little jitter.
            retry_after = e.response.headers.get("retry-after")
            if retry_after is not None:
                wait_seconds = float(retry_after)
            else:
                wait_seconds = (2 ** (attempt - 1)) + random.uniform(0, 0.5)

            print(
                f"  [rate limit] hit 429 on attempt {attempt}/{MAX_RETRIES}, "
                f"waiting {wait_seconds:.1f}s before retrying..."
            )
            time.sleep(wait_seconds)


def build_initial_messages(product: str) -> list:
    return [
        {
            "role": "user",
            "content": (
                f"The user wants help buying: '{product}'.\n\n"
                "STEP 1 — Resolve the request.\n"
                "If this is too vague to search effectively (e.g. a bare "
                "category like 'keyboard' or 'headphones' with no type, "
                "use case, or budget), call ask_user_clarification with a "
                "short question and 3-5 concrete options. If it's already "
                "specific enough (a product type, model, or use case is "
                "given, e.g. 'mechanical keyboard' or 'Sony WH-1000XM5'), "
                "skip clarification entirely and use your own judgement to "
                "proceed — don't ask just for the sake of asking.\n\n"
                "STEP 2 — Research and price it.\n"
                "If the resolved request names one specific product, "
                "search the web and compare its current price across at "
                "least 3 different retailers.\n"
                "If it resolves to a category (e.g. 'mechanical "
                "keyboard'), first pick one well-regarded specific model "
                "for each of 3 price brackets — Budget, Mid-range, and "
                "Premium (use sensible AUD ranges for this category) — "
                "then find the current price for each of those 3 models "
                "across at least 2 retailers.\n"
                "In both cases: only consider retailers that actually "
                "ship within or operate in Australia (e.g. Amazon AU, JB "
                "Hi-Fi, Officeworks, The Good Guys, Harvey Norman) — "
                "ignore US-only or other international-only sellers. "
                "Report every price in AUD (convert if a source quotes "
                "another currency, and note that it was converted). Prefer "
                "pricing you can confirm on the retailer's own current "
                "product page over third-party comparison/aggregator "
                "sites (e.g. PriceMe, GetPrice) or old ad copy — those go "
                "stale; if a price only came from one of those, say so in "
                "the Notes column instead of presenting it as confirmed "
                "current pricing. If a product looks discontinued, out of "
                "stock everywhere, or superseded by a newer model, say so "
                "plainly in the Notes column (and name the newer model if "
                "you find one) instead of quoting a stale price for it as "
                "if it's still available. For each retailer you find a "
                "price at, also call check_reviews to get its rating.\n\n"
                "STEP 3 — Report back.\n"
                "Return a markdown table — add a 'Price Bracket' and "
                "'Product' column if you researched a category — with "
                "columns: [Price Bracket |] [Product |] Retailer | Price "
                "(AUD) | Rating | Link | Notes. The Link column must be a "
                "markdown link, e.g. [Buy](https://...), pointing to the "
                "exact product page a price came from — use the URL from "
                "the web_search result you read it off, not a guessed or "
                "made-up URL; if you can't find a direct product page URL "
                "for a row, put 'n/a' instead of inventing one. Then add "
                "one line per bracket (or one line overall for a single "
                "product) recommending the best value option, considering "
                "both price and rating."
            ),
        }
    ]


def run_agent_turn(client: anthropic.Anthropic, messages: list) -> str:
    """Run the tool-calling loop until Claude produces a final text answer.

    `messages` is mutated in place with every assistant/tool-result turn, so
    the caller can append a new user message (e.g. follow-up feedback) and
    call this again to continue the SAME conversation — Claude keeps the
    full history of what it already searched and found.
    """
    # Keep sending requests until Claude stops asking for tool calls.
    while True:
        response = call_with_retry(
            client,
            model=MODEL,
            max_tokens=3000,
            messages=messages,
            tools=TOOLS,
        )

        # Always append the assistant's turn to the running conversation,
        # whether it's a final answer or a request to use tools.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Claude is done. server_tool_use / web_search results (if any)
            # were already resolved inline earlier in this same turn.
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return final_text

        # response.stop_reason == "tool_use": Claude wants a CLIENT tool run.
        # Collect a tool_result for every tool_use block that is ours to run.
        # (server_tool_use blocks, like web_search, are already resolved by
        # the API and don't need a tool_result from us.)
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  -> Claude requested tool: {block.name}({block.input})")
                result = execute_client_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )

        # Send the tool results back as a new user turn, then loop again.
        messages.append({"role": "user", "content": tool_results})


def log_feedback(query: str, feedback: str) -> None:
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()}\tquery={query!r}\tfeedback={feedback!r}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python price_agent.py "product name"')
        sys.exit(1)

    product_query = " ".join(sys.argv[1:])
    print(f"Searching prices and reviews for: {product_query}\n")

    client = anthropic.Anthropic()
    messages = build_initial_messages(product_query)
    result = run_agent_turn(client, messages)
    print("\n" + result)

    # --- FEEDBACK LOOP ---
    # Feedback isn't just logged — it's fed back into the SAME conversation
    # so Claude can act on it (e.g. "get the 512GB model instead" or "that
    # price looks stale, check the retailer's own page") and re-search with
    # full context of what it already found, rather than starting over.
    while True:
        feedback = input(
            "\nAny feedback on this search? (press Enter to finish): "
        ).strip()
        if not feedback:
            break

        log_feedback(product_query, feedback)
        print(f"(feedback saved to {FEEDBACK_LOG}; adjusting search...)")
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Feedback on the results above: {feedback}\n\n"
                    "Adjust the search accordingly — re-search or "
                    "re-verify only what the feedback calls into question "
                    "(e.g. a different spec/model, a stale or unconfirmed "
                    "price) and reuse anything already found that's still "
                    "valid. Return an updated table in the same format as "
                    "before, plus an updated one-line recommendation."
                ),
            }
        )
        result = run_agent_turn(client, messages)
        print("\n" + result)

    print("\nSession ended.")