import time

LAST_CALL = 0
CALL_COUNT = 0

COOLDOWN = 15      # seconds
MAX_CALLS = 30     # per session (free-tier style)

def rate_limiter():
    global LAST_CALL, CALL_COUNT

    if CALL_COUNT >= MAX_CALLS:
        raise Exception("Daily free-tier limit reached.")

    now = time.time()
    if now - LAST_CALL < COOLDOWN:
        raise Exception(
            f"Rate limit exceeded. Try after {int(COOLDOWN - (now - LAST_CALL))}s"
        )

    LAST_CALL = now
    CALL_COUNT += 1
