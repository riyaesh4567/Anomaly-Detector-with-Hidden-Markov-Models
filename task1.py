import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM


np.random.seed(42)

n_train_samples = 2000
train_data = np.sin(np.linspace(0, 50, n_train_samples)) + np.random.normal(0, 0.2, (n_train_samples, 1))

n_test_samples = 1000
test_data = np.sin(np.linspace(0, 25, n_test_samples)) + np.random.normal(0, 0.2, (n_test_samples, 1))


test_data[300:350] += 4.5  
test_data[700:730] -= 4.0  

n_states = 3
model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=42)
model.fit(train_data)


window_size = 50
train_scores = []

for i in range(len(train_data) - window_size + 1):
    window = train_data[i : i + window_size]
    score = model.score(window)
    train_scores.append(score)

train_scores = np.array(train_scores)
threshold = np.percentile(train_scores, 1.0)
print(f"Anomaly Threshold (Log-Likelihood): {threshold:.2f}")


test_scores = []
anomaly_flags = np.zeros(len(test_data), dtype=bool)

for i in range(len(test_data) - window_size + 1):
    window = test_data[i : i + window_size]
    score = model.score(window)
    test_scores.append(score)
    
    if score < threshold:
        anomaly_flags[i + window_size - 1] = True

test_scores = np.array(test_scores)


padded_scores = np.full(len(test_data), np.nan)
padded_scores[window_size - 1:] = test_scores


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)


ax1.plot(test_data, label='Sensor Reading', color='royalblue', alpha=0.7)
anomaly_indices = np.where(anomaly_flags)[0]
ax1.scatter(anomaly_indices, test_data[anomaly_indices], color='crimson', label='Detected Anomaly', zorder=5)
ax1.set_title('Real-Time Sensor Anomaly Detection using HMM')
ax1.set_ylabel('Sensor Value')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.5)


ax2.plot(padded_scores, label='Rolling Window Log-Likelihood', color='forestgreen')
ax2.axhline(threshold, color='crimson', linestyle='--', label=f'Threshold ({threshold:.2f})')
ax2.set_title('HMM Sequence Log-Likelihood per Window')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Log-Likelihood')
ax2.legend(loc='lower right')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()