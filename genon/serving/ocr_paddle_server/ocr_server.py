import argparse
import io
import multiprocessing as mp
from concurrent import futures

import grpc
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

import ocr_pb2
import ocr_pb2_grpc

parser = argparse.ArgumentParser(description="")
parser.add_argument("-c", "--server_count", type=int, default=4, help="Number of gRPC servers to start.")
args = parser.parse_args()

import paddle

print("[init] compiled_with_cuda:", paddle.is_compiled_with_cuda())
print("[init] device:", paddle.get_device())
try:
    print("[init] cuda_devices:", paddle.device.cuda.device_count())
except Exception as e:
    print("[init] cuda_count_err:", e)


class OCRServiceServicer(ocr_pb2_grpc.OCRServiceServicer):
    def __init__(self, port):
        self.port = port

        # PaddleOCR 모델 초기화
        self.ocr = PaddleOCR(
            lang="korean",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_dir="/models/paddleocr_model/PP-OCRv5_server_det",
            text_recognition_model_dir="/models/paddleocr_model/korean_PP-OCRv5_mobile_rec",
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="korean_PP-OCRv5_mobile_rec"
        )

    def PerformOCR(self, request, context):
        # 현재 서버의 포트 정보 출력
        # print(f"Received request on port {self.port}")

        # 바이너리 데이터를 메모리에서 이미지로 읽기
        image_data = request.image_data
        image = Image.open(io.BytesIO(image_data))  # 바이너리 데이터를 이미지로 변환

        # RGBA 이미지를 RGB로 변환
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # PIL 이미지에서 numpy 배열로 변환
        image = np.array(image)

        # OCR 처리
        result = self.ocr.predict(image)

        ocr_results = [ocr_pb2.OCRResult(text=text, confidence=score, box=box) for (text, score, box) in
                       zip(result[0]["rec_texts"], result[0]["rec_scores"], result[0]["rec_boxes"])]

        return ocr_pb2.OCRResponse(results=ocr_results)


def serve(port: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    ocr_pb2_grpc.add_OCRServiceServicer_to_server(OCRServiceServicer(port), server)
    server.add_insecure_port(f'[::]:{port}')
    print(f"Server started at [::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # 멀티프로세싱 시작 방법 설정

    ports = [50051 + i for i in range(args.server_count)]  # 포트 번호 설정
    procs = []
    for port in ports:
        proc = mp.Process(target=serve, args=(port,), daemon=False)
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
