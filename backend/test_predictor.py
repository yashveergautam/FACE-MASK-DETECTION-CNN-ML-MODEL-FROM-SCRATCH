from backend.predictor import FaceMaskPredictor

print("Creating Predictor...")

predictor = FaceMaskPredictor()

print("Predictor Loaded!")

result = predictor.predict(
    "dataset/with_mask/with_mask_1.jpg"   # use an image that exists
)

print(result)