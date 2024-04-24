from .parsing_api import onnx_inference
from ..libs.utils import install_package

class HumanParsing:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None

    def __call__(self, input_image, mask_components):
        if self.session is None:
            install_package('onnxruntime')
            import onnxruntime as ort

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # session_options.add_session_config_entry('gpu_id', str(gpu_id))
            self.session = ort.InferenceSession(self.model_path, sess_options=session_options,
                                                providers=['CPUExecutionProvider'])

        parsed_image, mask = onnx_inference(self.session, input_image, mask_components)
        return parsed_image, mask

