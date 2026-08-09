import pickle
import numpy as np

with open("demonstrations.pkl", "rb") as f:
    demos = pickle.load(f)

steering_values = np.array([d["action"][0] for d in demos])
print("Total transitions:", len(steering_values))
print("Mean steering:", steering_values.mean())
print("Fraction steering left (<0):", (steering_values < 0).mean())
print("Fraction steering right (>0):", (steering_values > 0).mean())
print("Fraction near-zero (|steer|<0.05):", (np.abs(steering_values) < 0.05).mean())

step_indices = []
current_step = 0
for i, d in enumerate(demos):
    if i == 0 or (demos[i-1]["terminated"] or demos[i-1]["truncated"]):
        current_step = 0
    else:
        current_step += 1
    if abs(d["action"][0]) > 0.1:
        step_indices.append(current_step)

step_indices = np.array(step_indices)
print("Number of 'meaningful turn' transitions:", len(step_indices))
print("Mean step-since-reset at these moments:", step_indices.mean())
print("Std dev:", step_indices.std())
print("Median:", np.median(step_indices))