from .metadata_extractor import MetadataExtractor

from deepface import DeepFace


class FaceCharacteristicsExtractor(MetadataExtractor):
    def extract(self, image):
        result = DeepFace.analyze(
            img_path=image,
            actions=["age", "gender", "emotion", "race"],
            enforce_detection=False,
        )

        if isinstance(result, list) and result:
            result = result[0]

        metadata = {}
        if result:
            gender_scores = result.get("gender", {})
            gender = (
                max(gender_scores, key=gender_scores.get) if gender_scores else None
            )

            metadata = {
                "age": result.get("age"),
                "gender": gender,
                "race": result.get("dominant_race"),
                "emotion": result.get("dominant_emotion"),
            }
        return metadata
