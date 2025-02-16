import logging
import flask
from flask import request, jsonify
from terra.base_client import Terra
import datetime
import requests
import json
import os
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("app")

app = flask.Flask(__name__)

@app.after_request
def add_cors_headers(response):
    """Manually add CORS headers to all responses."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response



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
# ✅ /mergedData: Combine by Date, keep a list of workouts
# ----------------------------------------------------
@app.route("/mergedData", methods=['GET'])
def get_merged_data():
    """Reads and merges biometric data by date."""
    lines = load_biometric_data()
    daily_map = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        timestamp = entry.get("timestamp")
        if not timestamp:
            continue
        date_str = timestamp[:10]  # Format: YYYY-MM-DD

        if date_str not in daily_map:
            daily_map[date_str] = {"activities": [], "daily": {}, "sleep": {}}

        entry_type = entry.get("type")

        if entry_type == "activity":
            daily_map[date_str]["activities"].append({
                "timestamp": entry.get("timestamp"),
                "bpm_activity": entry.get("bpm_activity"),
                "steps_activity": entry.get("steps_activity"),
                "activity_name": entry.get("activity_name")
            })
        elif entry_type == "daily":
            daily_map[date_str]["daily"] = {
                "bpm_daily": entry.get("bpm_daily"),
                "steps_daily": entry.get("steps_daily"),
                "stress_daily": entry.get("stress_daily")
            }
        elif entry_type == "sleep":
            daily_map[date_str]["sleep"] = {
                "respiration_sleep": entry.get("respiration_sleep")
            }

    merged_data = [{"date": date_str, "activities": day_dict["activities"],
                    "daily": day_dict["daily"], "sleep": day_dict["sleep"]}
                   for date_str, day_dict in daily_map.items()]

    merged_data.sort(key=lambda x: x["date"])
    return jsonify(merged_data)

# ----------------------------------------------------
# ✅ Reinforcement Learning Model Utilities
# ----------------------------------------------------
def preprocess_data(mergedData):
    """Preprocesses merged biometric data for the RL model."""
    data_list = []
    activity_mapping = {}
    user_history = {}

    for entry in mergedData:
        date = entry["date"]
        bpm_daily = pd.to_numeric(entry["daily"].get("bpm_daily", np.nan), errors="coerce")
        steps_daily = pd.to_numeric(entry["daily"].get("steps_daily", np.nan), errors="coerce")
        stress_daily = pd.to_numeric(entry["daily"].get("stress_daily", np.nan), errors="coerce")
        sleep_respiration = pd.to_numeric(entry["sleep"].get("respiration_sleep", np.nan), errors="coerce")

        for activity in entry["activities"]:
            activity_name = activity["activity_name"]
            bpm_activity = pd.to_numeric(activity["bpm_activity"], errors="coerce")
            timestamp = activity["timestamp"]
            time_of_day = pd.to_datetime(timestamp).hour

            if activity_name not in activity_mapping:
                activity_mapping[activity_name] = len(activity_mapping)

            activity_encoded = activity_mapping[activity_name]

            if activity_name not in user_history:
                user_history[activity_name] = {"bpm_effect": [], "stress_effect": [], "sleep_effect": []}

            if not np.isnan(bpm_daily):
                user_history[activity_name]["bpm_effect"].append(bpm_daily)
            if not np.isnan(stress_daily):
                user_history[activity_name]["stress_effect"].append(stress_daily)
            if not np.isnan(sleep_respiration):
                user_history[activity_name]["sleep_effect"].append(sleep_respiration)

            data_list.append([
                date, bpm_daily, steps_daily, stress_daily, sleep_respiration,
                activity_encoded, bpm_activity, time_of_day, activity_name
            ])

    df = pd.DataFrame(data_list, columns=[
        "date", "bpm_daily", "steps_daily", "stress_daily", "sleep_respiration",
        "activity_encoded", "bpm_activity", "time_of_day", "activity_name"
    ])

    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df["bpm_daily"] /= 200
    df["steps_daily"] /= 40000
    df["stress_daily"] /= 100
    df["sleep_respiration"] /= 100
    df["time_of_day"] /= 24
    df["fatigue_score"] = (df["bpm_daily"] * 0.4 + (1 - df["steps_daily"]) * 0.3 + (1 - df["sleep_respiration"]) * 0.3)

    return df, activity_mapping, user_history

# ----------------------------------------------------
# ✅ /predict: Recommend Optimal Workouts
# ----------------------------------------------------
import os
import requests
import numpy as np
import pandas as pd
from flask import jsonify, request
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ✅ Correct path to the trained model
MODEL_PATH = "/Users/ashwaryeyadav/rl_NEWGAME/api/trained_workout_model.zip"
API_URL = "http://127.0.0.1:5000/mergedData"

class WorkoutEnv(gym.Env):
    def __init__(self, df, activity_mapping, user_history, sequence_length=5):  
        super(WorkoutEnv, self).__init__()

        self.df = df
        self.activity_mapping = activity_mapping
        self.user_history = user_history
        self.current_step = 0
        self.sequence_length = sequence_length

        self.num_activities = len(activity_mapping)
        self.action_space = spaces.MultiDiscrete([self.num_activities] * sequence_length)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6,), dtype=np.float32
        )

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        return np.array([
            row["bpm_daily"],
            row["steps_daily"],
            row["stress_daily"],
            row["sleep_respiration"],
            row["time_of_day"],
            row["fatigue_score"]
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self._next_observation()
        return obs, {}

    def step(self, actions):
        reward = 0
        done = False

        for action in actions:
            chosen_activity = list(self.activity_mapping.keys())[action]

            if chosen_activity in self.user_history:
                past_effects = self.user_history[chosen_activity]

                bpm_values = [val for val in past_effects["bpm_effect"] if not np.isnan(val)]
                stress_values = [val for val in past_effects["stress_effect"] if not np.isnan(val)]
                sleep_values = [val for val in past_effects["sleep_effect"] if not np.isnan(val)]

                avg_bpm_effect = np.nanmean(bpm_values) if bpm_values else 0.5
                avg_stress_effect = np.nanmean(stress_values) if stress_values else 0.5
                avg_sleep_effect = np.nanmean(sleep_values) if sleep_values else 0.5

                if avg_bpm_effect < 0.4:
                    reward += 2  
                if avg_stress_effect < 0.3:
                    reward += 2  
                if avg_sleep_effect > 0.7:
                    reward += 3  

            reward -= self.df.loc[self.current_step, "fatigue_score"] * 4  

            self.current_step += 1
            if self.current_step >= len(self.df) - 1:
                done = True
                break

        return self._next_observation(), reward, done, False, {}

# ✅ Now define predict AFTER WorkoutEnv
@app.route("/predict", methods=['GET'])
def predict():
    """Fetches data, preprocesses it, and predicts an optimal workout plan using RL."""
    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({"error": "Trained model not found at specified path"}), 500

        response = requests.get(API_URL)
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch mergedData"}), 500

        merged_data = response.json()
        if not merged_data:
            return jsonify({"error": "No data available"}), 400

        df, activity_mapping, user_history = preprocess_data(merged_data)

        model = PPO.load(MODEL_PATH)

        env = DummyVecEnv([lambda: WorkoutEnv(df, activity_mapping, user_history, sequence_length=5)])
        obs = env.reset()

        actions, _ = model.predict(obs, deterministic=True)
        recommended_sequence = [list(activity_mapping.keys())[action] for action in actions[0]]
        predicted_fatigue = np.mean(df["fatigue_score"])

        return jsonify({
            "recommended_workout": recommended_sequence,
            "predicted_fatigue_score": predicted_fatigue
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ----------------------------------------------------
# ✅ Start the Flask App
# ----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

