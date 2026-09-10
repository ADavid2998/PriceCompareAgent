"""
price_agent.py — An AI agent that searches the web for prices AND checks
reviews, using two different kinds of tools so you can see both patterns.

THE TWO TOOL TYPES:
    1. web_search       — a SERVER-SIDE tool. Anthropic's own infrastructure
                           runs the search; the result comes back already
                           resolved inside the same API response. You never
                           see stop_reason == "tool_use" for this one alone.
    2. check_reviews     — a CLIENT-SIDE tool (this is a plain Python
                           function). When Claude wants to call it, the API
                           returns stop_reason == "tool_use" and hands you a
                           tool_use block. YOUR CODE has to run the function
                           and send the result back before Claude continues.
                           This is the classic "agent loop."

Mixing both in one request is intentional: it shows what happens when a
server tool and a client tool both appear in the picture — the API resolves
web_search on its own, but pauses for you on check_reviews.

RATE LIMITING:
    Every API request is wrapped in call_with_retry(), which catches 429
    (rate limit) errors and retries with backoff — honoring the server's
    retry-after header when present, or exponential backoff with jitter
    otherwise. Watch the console for [rate limit] messages if it kicks in.

REQUIREMENTS:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key-here"

USAGE:
    python price_agent.py "Sony WH-1000XM5 headphones"
"""

import sys
import json
import time
import random
import anthropic
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY (and friends) from a local .env file

MODEL = "claude-sonnet-4-5"

# How many times to retry a single API call after a 429 before giving up.
MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# STEP 1: Define the client-side tool as a normal Python function.
#
# This is a STUB — it returns made-up data so you can see the loop work
# without needing a real reviews API. To make it real, replace the body
# with a call to something like a retailer API, a scraping service, or
# your own database of scraped reviews.
# ---------------------------------------------------------------------------
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
# STEP 2: Describe both tools to Claude.
# web_search uses Anthropic's built-in schema (type + name).
# check_reviews is a custom tool: we write its name, description, and the
# JSON schema for its inputs ourselves — this is how Claude knows when and
# how to call it.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 5,
        # Bias/restrict results to Australia so retailers and prices found
        # are relevant to an AU shopper (e.g. amazon.com.au, jbhifi.com.au).
        "user_location": {
            "type": "approximate",
            "country": "AU",
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


def run_price_agent(product: str) -> str:
    client = anthropic.Anthropic()

    messages = [
        {
            "role": "user",
            "content": (
                f"Search the web and compare current prices for '{product}' "
                "across at least 3 different retailers that sell to "
                "customers in Australia (e.g. Amazon AU, JB Hi-Fi, "
                "Officeworks, The Good Guys, Harvey Norman). Only consider "
                "retailers that actually ship within or operate in "
                "Australia — ignore US-only or other international-only "
                "sellers. Report every price in AUD (convert if a source "
                "quotes another currency, and note that it was converted). "
                "For each retailer you find a price at, also call "
                "check_reviews to get its rating. Return a markdown table: "
                "Retailer | Price (AUD) | Rating | Notes. Then add one line "
                "recommending the best value option, considering both price "
                "and rating."
            ),
        }
    ]

    # --- THE AGENT LOOP ---
    # Keep sending requests until Claude stops asking for tool calls.
    while True:
        response = call_with_retry(
            client,
            model=MODEL,
            max_tokens=1500,
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python price_agent.py "product name"')
        sys.exit(1)

    product_query = " ".join(sys.argv[1:])
    print(f"Searching prices and reviews for: {product_query}\n")
    result = run_price_agent(product_query)
    print("\n" + result)