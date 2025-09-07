import json
import numpy as np
import triton_python_backend_utils as pb_utils
import xgboost as xgb

class TritonPythonModel:
    def initialize(self, args):
        self.model = xgb.Booster()
        self.model.load_model(f"{args['model_repository']}/1/model.json")
        print("XGBoost model initialized.")

    def execute(self, requests):
        responses = []
        feature_names = [
            'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'OBV', 'SMA_50',
            'Volume', 'sentiment_score'
        ]

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input__0")
            input_data = input_tensor.as_numpy()

            dmatrix = xgb.DMatrix(input_data, feature_names=feature_names)
            predictions = self.model.predict(dmatrix)

            output_tensor = pb_utils.Tensor(
                "output__0",
                predictions.astype(np.float32).reshape([-1, 1])
            )
            responses.append(pb_utils.InferenceResponse([output_tensor]))
        return responses