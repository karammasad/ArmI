import requests
import json
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ----------------------------------------
# ✅ Step 1: Fetch Data from API
# ----------------------------------------

API_URL = "http://127.0.0.1:5000/mergedData"

def fetch_data():
    response = requests.get(API_URL)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching data: {response.status_code}")

# ----------------------------------------
# ✅ Step 2: Preprocess Data
# ----------------------------------------
def preprocess_data(mergedData):
    data_list = []
    activity_mapping = {}  # Maps activity names to numbers
    user_history = {}  # Tracks past workout effects

    for entry in mergedData:
        date = entry["date"]
        
        # ✅ Ensure numeric conversion and set NaN for invalid values
        bpm_daily = pd.to_numeric(entry["daily"].get("bpm_daily", np.nan), errors="coerce")
        steps_daily = pd.to_numeric(entry["daily"].get("steps_daily", np.nan), errors="coerce")
        stress_daily = pd.to_numeric(entry["daily"].get("stress_daily", np.nan), errors="coerce")
        sleep_respiration = pd.to_numeric(entry["sleep"].get("respiration_sleep", np.nan), errors="coerce")

        for activity in entry["activities"]:
            activity_name = activity["activity_name"]
            bpm_activity = pd.to_numeric(activity["bpm_activity"], errors="coerce")
            timestamp = activity["timestamp"]
            time_of_day = pd.to_datetime(timestamp).hour

            # ✅ Assign integer values to activity names
            if activity_name not in activity_mapping:
                activity_mapping[activity_name] = len(activity_mapping)

            activity_encoded = activity_mapping[activity_name]

            # ✅ Track past workouts' effects on health
            if activity_name not in user_history:
                user_history[activity_name] = {
                    "bpm_effect": [],
                    "stress_effect": [],
                    "sleep_effect": [],
                }

            # ✅ Store only valid numerical values (ignore NaN values)
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

    # ✅ Handle NaN values before normalization
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # ✅ Normalize numeric values
    df["bpm_daily"] /= 200
    df["steps_daily"] /= 40000
    df["stress_daily"] /= 100
    df["sleep_respiration"] /= 100
    df["time_of_day"] /= 24

    # ✅ Compute Fatigue Score
    df = compute_fatigue(df)

    return df, activity_mapping, user_history

# ----------------------------------------
# ✅ Step 3: Define RL Environment
# ----------------------------------------
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
        """Resets the environment and initializes the seed."""
        super().reset(seed=seed)  # ✅ Ensures compatibility with SB3
        self.current_step = 0
        obs = self._next_observation()
        return obs, {}  # ✅ Explicitly return (obs, info)

    def step(self, actions):
        reward = 0
        done = False

        for action in actions:
            chosen_activity = list(self.activity_mapping.keys())[action]

            # ✅ Dynamic Rewards Based on Past Data
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

# ----------------------------------------
# ✅ Step 4: Compute Fatigue Score
# ----------------------------------------
def compute_fatigue(df):
    fatigue = (
        (df["bpm_daily"] * 0.4) + 
        ((1 - df["steps_daily"]) * 0.3) +  
        ((1 - df["sleep_respiration"]) * 0.3)  
    )

    df["fatigue_score"] = (fatigue - fatigue.min()) / (fatigue.max() - fatigue.min())
    return df

# ----------------------------------------
# ✅ Step 5: Train RL Model
# ----------------------------------------

def train_model(df, activity_mapping, user_history, sequence_length=5, model_path="trained_workout_model.zip"):
    env = DummyVecEnv([lambda: WorkoutEnv(df, activity_mapping, user_history, sequence_length)])
    model = PPO("MlpPolicy", env, learning_rate=0.0001, verbose=1)
    
    print("Training model...")
    model.learn(total_timesteps=200000)

    # ✅ Save model
    model.save(model_path)
    print(f"Model saved successfully to {model_path}")

    return model


# # ----------------------------------------
# # ✅ Step 6: Predict Next Workout
# # ----------------------------------------
def predict_best_workout_sequence(model, df, activity_mapping, user_history, sequence_length=7):
    env = DummyVecEnv([lambda: WorkoutEnv(df, activity_mapping, user_history, sequence_length)])
    
    obs = env.reset()  # ✅ Fix: No unpacking needed with DummyVecEnv

    actions, _ = model.predict(obs, deterministic=True)

    recommended_sequence = [list(activity_mapping.keys())[action] for action in actions[0]]

    predicted_fatigue = np.mean(df["fatigue_score"])

    return recommended_sequence, predicted_fatigue
# # ----------------------------------------
# ✅ Step 6: Load Model and Predict Next Workout
# ----------------------------------------
# def load_and_predict(model_path="trained_workout_model.zip", df=None, activity_mapping=None, user_history=None, sequence_length=7):
#     env = DummyVecEnv([lambda: WorkoutEnv(df, activity_mapping, user_history, sequence_length)])

#     # ✅ Load saved model
#     model = PPO.load(model_path)
#     print(f"Model loaded from {model_path}")

#     obs = env.reset()

#     actions, _ = model.predict(obs, deterministic=True)
#     recommended_sequence = [list(activity_mapping.keys())[action] for action in actions[0]]
#     predicted_fatigue = np.mean(df["fatigue_score"])

#     return recommended_sequence, predicted_fatigue


# # ----------------------------------------
# # ✅ Step 7: Main Execution
# # ----------------------------------------
if __name__ == "__main__":
    print("Fetching data...")
    mergedData = fetch_data()
    
    df, activity_mapping, user_history = preprocess_data(mergedData)

    print("Training RL Model...")
    model = train_model(df, activity_mapping, user_history, sequence_length=7)

    print("Predicting Optimal Workout Sequence...")
    best_sequence, predicted_fatigue = predict_best_workout_sequence(model, df, activity_mapping, user_history, sequence_length=5)

    print(f"Recommended Workout Plan: {best_sequence}")
    print(f"Predicted Fatigue Score After Workouts: {predicted_fatigue:.2f}")
    
# if __name__ == "__main__":
#     print("Fetching data...")
#     mergedData = fetch_data()
    
#     df, activity_mapping, user_history = preprocess_data(mergedData)

#     # Train and save model
#     model_path = "trained_workout_model.zip"
#     model = train_model(df, activity_mapping, user_history, sequence_length=7, model_path=model_path)

#     # Load model and predict
#     print("Predicting Optimal Workout Sequence...")
#     best_sequence, predicted_fatigue = load_and_predict(model_path, df, activity_mapping, user_history, sequence_length=5)

#     print(f"Recommended Workout Plan: {best_sequence}")
#     print(f"Predicted Fatigue Score After Workouts: {predicted_fatigue:.2f}")

