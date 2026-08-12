import pickle
with open('encodings.pkl', 'rb') as f:
    encodeListKnown, classNames = pickle.load(f)

print("Stored Users in encodings.pkl:")
for i, name in enumerate(classNames):
    print(f"{i + 1}. {name}")