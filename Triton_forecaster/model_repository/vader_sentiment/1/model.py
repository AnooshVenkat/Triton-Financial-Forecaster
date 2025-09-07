import json
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        nltk.download('vader_lexicon')
        self.analyzer = SentimentIntensityAnalyzer()
        print("VADER sentiment analyzer initialized.")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT_INPUT")
            texts = [t.decode('utf-8') for t in input_tensor.as_numpy()]
            scores = [self.analyzer.polarity_scores(text)['compound'] for text in texts]
            output_tensor = pb_utils.Tensor(
                "SENTIMENT_SCORE",
                np.array(scores, dtype=np.float32)
            )
            responses.append(pb_utils.InferenceResponse([output_tensor]))
        return responses