from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from hey_jude.config import Settings
from hey_jude.models import DetectedEntity
from hey_jude.services.recognizers import build_recognizers

_engine: AnalyzerEngine | None = None


def _get_engine() -> AnalyzerEngine:
    global _engine
    if _engine is None:
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
            "ner_model_configuration": {
                "model_to_presidio_entity_mapping": {
                    "ORG": "ORGANIZATION",
                    "PERSON": "PERSON",
                    "GPE": "LOCATION",
                    "LOC": "LOCATION",
                    "DATE": "DATE_TIME",
                    "TIME": "DATE_TIME",
                },
                "labels_to_ignore": ["O"],
            },
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        _engine = AnalyzerEngine(nlp_engine=nlp_engine)
    return _engine


async def detect_entities(text: str, settings: Settings) -> list[DetectedEntity]:
    engine = _get_engine()
    ad_hoc_recognizers = build_recognizers(settings.custom_recognizer_specs)
    results = engine.analyze(
        text=text,
        entities=settings.presidio_entities,
        language="en",
        ad_hoc_recognizers=ad_hoc_recognizers or None,
    )
    entities = []
    for result in results:
        entities.append(
            DetectedEntity(
                text=text[result.start:result.end],
                entity_type=result.entity_type,
                start=result.start,
                end=result.end,
                score=result.score,
            )
        )
    return entities
