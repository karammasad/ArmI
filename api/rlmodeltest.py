import logging
import flask
from flask import request, jsonify
from terra.base_client import Terra
import datetime
import requests
import json
import os

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("app")

app = flask.Flask(__name__)

# ----------------------------------------------------
# ✅ Terra API Credentials
# ----------------------------------------------------
webhook_secret = "480c3fbabd1e6b21e77050c3b79fc4c8eedc15f573456fe8"
dev_id = "4actk-armyfit-testing-epyHNH57sh"
api_key = "J73tDf52rDh0g5DpFurIyACVm5Uw64fb"

terra = Terra(
    api_key=api_key,
    dev_id=dev_id,
    secret=webhook_secret
)

# ----------------------------------------------------
# ✅ JSON File to Store All Data
# ----------------------------------------------------
DATA_FILE = "biometric_data.json"

def load_biometric_data():
    """Load lines from local JSON file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return f.readlines()
    return []

def save_biometric_data(entry: dict):
    """Append a single JSON record to the file."""
    with open(DATA_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ----------------------------------------------------
# ✅ /authenticate: Terra Connect Flow
# ----------------------------------------------------
@app.route("/authenticate", methods=['GET'])
def authenticate():
    """
    Generates a Terra Connect Widget session link for the user to authorize Garmin.
    After logging in, you can backfill daily/activity/sleep data.
    """
    try:
        widget_response = terra.generate_widget_session(
            providers=['GARMIN'],
            reference_id=['1234']  # or your internal user ID
        )
        widget_resp_dict = widget_response.get_json()
        widget_url = widget_resp_dict['url']

        html_content = f"""
        <html>
            <body>
                <h1>Terra Garmin Authentication</h1>
                <p>Click the button to connect your Garmin account:</p>
                <button onclick="location.href='{widget_url}'">Authenticate with Garmin</button>
            </body>
        </html>
        """
        return flask.Response(html_content, mimetype='text/html')

    except Exception as e:
        return jsonify({"error": "Failed to generate widget session", "message": str(e)}), 500

# ----------------------------------------------------
# ✅ /backfillActivity: Store BPM, Steps, and ACTIVITY TYPE
# ----------------------------------------------------
@app.route("/backfillActivity", methods=["GET"])
def backfill_activity():
    """
    Fetches 'activity' data for a Terra user in a date range.
    Stores BPM (heart rate), steps for that workout, and the activity type (metadata.name or .type).
    """
    # Replace with real Terra user_id
    user_id = "0ba45a3a-f555-4982-8008-dacf718f5b8d"
    terra_user = terra.from_user_id(user_id)

    # Example date range
    start_date = datetime.datetime(2024, 1, 1)
    end_date   = datetime.datetime(2025, 1, 31)

    response = terra_user.get_activity(
        start_date=start_date,
        end_date=end_date,
        to_webhook=False,
        with_samples=True
    )
    data_json = response.get_json()

    for activity_entry in data_json.get("data", []):
        metadata        = activity_entry.get("metadata", {})
        heart_rate_data = activity_entry.get("heart_rate_data", {})
        distance_data   = activity_entry.get("distance_data", {})
        steps_data      = activity_entry.get("steps_data", {})

        # BPM for the workout
        avg_hr = heart_rate_data.get("summary", {}).get("avg_hr_bpm")

        # Steps for the workout
        steps_workout = distance_data.get("summary", {}).get("steps")
        if steps_workout is None:
            # Some devices store steps in steps_data.summary
            steps_workout = steps_data.get("summary", {}).get("steps")

        # Extract the 'activity_type' from Terra metadata
        # e.g. "Strength", or code "80", or fallback "Unknown"
        # Extract activity name (e.g., "Islington eBiking")
        # Extract the 'activity_type' from Terra metadata
        activity_name = metadata.get("name")

        last_word = activity_name.split()[-1]

        record = {
            "type": "activity",
            "timestamp": metadata.get("start_time"),
            "user_id": user_id,
            "bpm_activity": avg_hr,
            "steps_activity": steps_workout,
            "activity_name": last_word  # ✅ Now stores only the last word
        }


        save_biometric_data(record)

    return jsonify(data_json)

# ----------------------------------------------------
# ✅ /backfillDaily: Store BPM, Steps, Stress from Daily
# ----------------------------------------------------
@app.route("/backfillDaily", methods=['GET'])
def backfill_daily():
    """
    Fetches 'daily' data for a Terra user in a date range.
    Stores daily BPM (avg), daily steps, and daily stress.
    """
    # Replace with real Terra user_id
    user_id = "0ba45a3a-f555-4982-8008-dacf718f5b8d"
    terra_user = terra.from_user_id(user_id)

    # Example date range
    start_date = datetime.datetime(2024, 1, 1)
    end_date   = datetime.datetime(2025, 1, 31)

    response = terra_user.get_daily(
        start_date=start_date,
        end_date=end_date,
        with_samples=True,
        to_webhook=False
    )
    data_json = response.get_json()

    for daily_entry in data_json.get("data", []):
        hr_summary    = daily_entry.get("heart_rate_data", {}).get("summary", {})
        stress_data   = daily_entry.get("stress_data", {})
        distance_data = daily_entry.get("distance_data", {})

        # BPM (avg daily)
        bpm_daily = hr_summary.get("avg_hr_bpm")
        # Steps (daily total)
        steps_daily = distance_data.get("steps")
        # Stress (avg stress level)
        stress_daily = stress_data.get("avg_stress_level")

        record = {
            "type": "daily",
            "timestamp": daily_entry.get("metadata", {}).get("start_time"),
            "user_id": user_id,
            "bpm_daily": bpm_daily,
            "steps_daily": steps_daily,
            "stress_daily": stress_daily
        }
        save_biometric_data(record)

    return jsonify(data_json)

# ----------------------------------------------------
# ✅ /backfillSleep: Store Respiration from Sleep
# ----------------------------------------------------
@app.route("/backfillSleep", methods=['GET'])
def backfill_sleep():
    """
    Fetches 'sleep' data for a Terra user in a date range.
    Stores only the respiration metric (if available).
    """
    # Replace with real Terra user_id
    user_id = "0ba45a3a-f555-4982-8008-dacf718f5b8d"
    terra_user = terra.from_user_id(user_id)

    # Example date range
    start_date = datetime.datetime(2024, 1, 1)
    end_date   = datetime.datetime(2025, 1, 31)

    response = terra_user.get_sleep(
        start_date=start_date,
        end_date=end_date,
        with_samples=True,
        to_webhook=False
    )
    data_json = response.get_json()

    for sleep_entry in data_json.get("data", []):
        respiration_data = sleep_entry.get("respiration_data", {})
        # Possibly we check "oxygen_saturation_data.avg_saturation_percentage" or "respiration_rate"
        respiration_sleep = respiration_data.get("oxygen_saturation_data", {}).get("avg_saturation_percentage")

        record = {
            "type": "sleep",
            "timestamp": sleep_entry.get("metadata", {}).get("start_time"),
            "user_id": user_id,
            "respiration_sleep": respiration_sleep
        }
        save_biometric_data(record)

    return jsonify(data_json)

@app.route("/consumeTerraWebhook", methods=['POST'])
def consume_terra_webhook():
    """
    This function listens for incoming Terra webhook data
    and processes it asynchronously.
    """
    try:
        webhook_data = request.get_json()

        if not webhook_data:
            return jsonify({"error": "Empty request body"}), 400

        _LOGGER.info(f"Received webhook: {json.dumps(webhook_data, indent=2)}")

        # ✅ Send response immediately to avoid timeouts
        from threading import Thread
        Thread(target=process_webhook_data, args=(webhook_data,)).start()

        return jsonify({"status": "success", "message": "Webhook received"}), 200

    except Exception as e:
        _LOGGER.error(f"Error processing webhook: {str(e)}")
        return jsonify({"error": "Failed to process webhook", "message": str(e)}), 500


def process_webhook_data(webhook_data):
    """
    Processes webhook data asynchronously to avoid timeouts.
    Saves activity, daily, and sleep data to JSON.
    """
    try:
        user_id = webhook_data.get("user", {}).get("user_id")
        event_type = webhook_data.get("type")

        if event_type == "activity":
            for activity_entry in webhook_data.get("data", []):
                metadata = activity_entry.get("metadata", {})
                heart_rate_data = activity_entry.get("heart_rate_data", {})
                steps_data = activity_entry.get("steps_data", {})

                avg_hr = heart_rate_data.get("summary", {}).get("avg_hr_bpm")
                steps = steps_data.get("summary", {}).get("steps")

                activity_name = metadata.get("name", "Unknown")
                last_word = activity_name.split()[-1] if isinstance(activity_name, str) else "Unknown"

                record = {
                    "type": "activity",
                    "timestamp": metadata.get("start_time"),
                    "user_id": user_id,
                    "bpm_activity": avg_hr,
                    "steps_activity": steps,
                    "activity_name": last_word
                }
                _LOGGER.info(f"Saving activity record: {record}")
                save_biometric_data(record)

        elif event_type == "daily":
            for daily_entry in webhook_data.get("data", []):
                hr_summary = daily_entry.get("heart_rate_data", {}).get("summary", {})
                stress_data = daily_entry.get("stress_data", {})
                distance_data = daily_entry.get("distance_data", {})

                bpm_daily = hr_summary.get("avg_hr_bpm")
                steps_daily = distance_data.get("steps")
                stress_daily = stress_data.get("avg_stress_level")

                record = {
                    "type": "daily",
                    "timestamp": daily_entry.get("metadata", {}).get("start_time"),
                    "user_id": user_id,
                    "bpm_daily": bpm_daily,
                    "steps_daily": steps_daily,
                    "stress_daily": stress_daily
                }
                _LOGGER.info(f"Saving daily record: {record}")
                save_biometric_data(record)

        elif event_type == "sleep":
            for sleep_entry in webhook_data.get("data", []):
                respiration_data = sleep_entry.get("respiration_data", {})
                respiration_sleep = respiration_data.get("oxygen_saturation_data", {}).get("avg_saturation_percentage")

                record = {
                    "type": "sleep",
                    "timestamp": sleep_entry.get("metadata", {}).get("start_time"),
                    "user_id": user_id,
                    "respiration_sleep": respiration_sleep
                }
                _LOGGER.info(f"Saving sleep record: {record}")
                save_biometric_data(record)

    except Exception as e:
        _LOGGER.error(f"Error in async webhook processing: {str(e)}")





# ----------------------------------------------------
# ✅ /mergedData: Combine by Date, keep a list of workouts
# ----------------------------------------------------
@app.route("/mergedData", methods=['GET'])
def get_merged_data():
    """
    Reads 'activity', 'daily', 'sleep' entries from biometric_data.json,
    merges them by date (YYYY-MM-DD),
    returning an array of objects like:
      {
        "date": "2025-01-14",
        "activities": [
          {
             "bpm_activity": ...,
             "steps_activity": ...,
             "activity_type": ...
             "timestamp": ...
          },
          ... possibly more workouts ...
        ],
        "daily": {
          "bpm_daily": ...,
          "steps_daily": ...,
          "stress_daily": ...
        },
        "sleep": {
          "respiration_sleep": ...
        }
      }
    """
    lines = load_biometric_data()
    daily_map = {}  # key: date_str => dict with "activities", "daily", "sleep"

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Grab date from "timestamp"
        timestamp = entry.get("timestamp")
        if not timestamp:
            continue
        date_str = timestamp[:10]  # e.g. "2025-01-14"

        # Ensure we have a default structure for this date
        if date_str not in daily_map:
            daily_map[date_str] = {
                "activities": [],
                "daily": {},
                "sleep": {}
            }

        entry_type = entry.get("type")

        if entry_type == "activity":
            daily_map[date_str]["activities"].append({
                "timestamp": entry.get("timestamp"),
                "bpm_activity": entry.get("bpm_activity"),
                "steps_activity": entry.get("steps_activity"),
                "activity_name": entry.get("activity_name")  # ✅ Now included
            })


        elif entry_type == "daily":
            # Put daily metrics into daily_map[date_str]["daily"]
            daily_map[date_str]["daily"] = {
                "bpm_daily": entry.get("bpm_daily"),
                "steps_daily": entry.get("steps_daily"),
                "stress_daily": entry.get("stress_daily")
            }

        elif entry_type == "sleep":
            # Store sleep respiration
            daily_map[date_str]["sleep"] = {
                "respiration_sleep": entry.get("respiration_sleep")
            }

    # Convert daily_map to a list of objects
    merged_data = []
    for date_str, day_dict in daily_map.items():
        merged_data.append({
            "date": date_str,
            "activities": day_dict["activities"],
            "daily": day_dict["daily"],
            "sleep": day_dict["sleep"]
        })

    # Sort by date if desired
    merged_data.sort(key=lambda x: x["date"])
    return jsonify(merged_data)

# ----------------------------------------------------
# ✅ Start the Flask App
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(port=5000, debug=False)




