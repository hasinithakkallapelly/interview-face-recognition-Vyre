import logging
import time

logger = logging.getLogger("interview_proctor")

last_alert_time = 0
ALERT_COOLDOWN = 1  # seconds

def alert_user(message):
    global last_alert_time
    now = time.time()
    if now - last_alert_time >= ALERT_COOLDOWN:
        logger.warning("[ALERT] %s", message)
        last_alert_time = now

def cancel_interview(reason):
    logger.error("[CANCELLED] %s", reason)
