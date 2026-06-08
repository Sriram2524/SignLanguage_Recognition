import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data_dict = pickle.load(open('./data.pickle', 'rb'))

data_raw   = data_dict['data']
labels_raw = data_dict['labels']

# Keep only samples with exactly 42 features
EXPECTED = 42
data_clean   = [d for d in data_raw                        if len(d) == EXPECTED]
labels_clean = [l for d, l in zip(data_raw, labels_raw)   if len(d) == EXPECTED]

removed = len(data_raw) - len(data_clean)
if removed:
    print(f'Warning: removed {removed} corrupt samples.')

data   = np.array(data_clean,   dtype=np.float32)
labels = np.array(labels_clean)

print(f'Training on {len(data)} samples across {len(set(labels))} classes.')

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(f'\nOverall accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print('\nPer-class report:')
print(classification_report(y_test, y_pred))

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print('Model saved to model.p')