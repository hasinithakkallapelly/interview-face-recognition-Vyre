import time

last_alert_time = 0
ALERT_COOLDOWN = 1  # seconds

def alert_user(message):
    global last_alert_time
    now = time.time()
    if now - last_alert_time >= ALERT_COOLDOWN:
        print("[ALERT]", message)
        last_alert_time = now

def cancel_interview(reason):
    print("[CANCELLED]", reason)
