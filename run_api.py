# from fastapi import FastAPI
# import pandas as pd
# import numpy as np
# from stable_baselines3 import PPO
# from gymnasium import spaces
# from stable_baselines3.common.vec_env import DummyVecEnv

# # Import your functions
# from main import fetch_data, preprocess_data, WorkoutEnv

# app = FastAPI()

# # Load or train model
# MODEL_PATH = "api/trained_workout_model.zip"

# print("🚀 Training new model...")
# mergedData = fetch_data()
# df, activity_mapping, user_history = preprocess_data(mergedData)

# # Train model
# env = DummyVecEnv([lambda: WorkoutEnv(df, activity_mapping, user_history, sequence_length=7)])
# model = PPO("MlpPolicy", env, learning_rate=0.0001, verbose=1)
# model.learn(total_timesteps=50000)
# model.save(MODEL_PATH)
# print(f"✅ Model trained and saved at {MODEL_PATH}")

# @app.get("/")
# def home():
#     return {"message": "RL Workout API is running!"}

# @app.get("/predict")
# def predict_workout():
#     """API endpoint for FlutterFlow to get recommended workouts"""
#     env = DummyVecEnv([lambda: WorkoutEnv(df, activity_mapping, user_history, sequence_length=5)])
#     obs = env.reset()
#     actions, _ = model.predict(obs, deterministic=True)

#     recommended_sequence = [list(activity_mapping.keys())[action] for action in actions[0]]
#     predicted_fatigue = np.mean(df["fatigue_score"])

#     return {
#         "recommended_workout": recommended_sequence,
#         "predicted_fatigue": round(predicted_fatigue, 2)
#     }

from fastapi import FastAPI
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
import uvicorn

# Import your functions
from main import fetch_data, preprocess_data, WorkoutEnv


app = FastAPI()

# Load or train model
MODEL_PATH = "api/trained_workout_model.zip"

print("🚀 Training new model...")
mergedData = fetch_data()
df, activity_mapping, user_history = preprocess_data(mergedData)

# Train model
env = DummyVecEnv([lambda: WorkoutEnv(df, activity_mapping, user_history, sequence_length=7)])
model = PPO("MlpPolicy", env, learning_rate=0.0001, verbose=1)
model.learn(total_timesteps=200000)
model.save(MODEL_PATH)
print(f"✅ Model trained and saved at {MODEL_PATH}")

@app.get("/")
def home():
    return {"message": "RL Workout API is running!"}

@app.get("/predict")
def predict_workout():
    """API endpoint for FlutterFlow to get recommended workouts"""
    env = DummyVecEnv([lambda: WorkoutEnv(df, activity_mapping, user_history, sequence_length=5)])
    obs = env.reset()
    actions, _ = model.predict(obs, deterministic=True)

    recommended_sequence = [list(activity_mapping.keys())[action] for action in actions[0]]
    predicted_fatigue = np.mean(df["fatigue_score"])

    return {
        "recommended_workout": recommended_sequence,
        "predicted_fatigue": round(predicted_fatigue, 2)
    }

# # ✅ Ensure FastAPI Runs on Port 8080 for Google Cloud Run
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080)
 
