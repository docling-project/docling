import grpc
import ocr_pb2
import ocr_pb2_grpc

def run():
    # 바이너리로 이미지를 읽기
    with open("ocr_sample.png", "rb") as f:
        image_data = f.read()  # 이미지 파일을 바이너리 형식으로 읽음

    # gRPC 서버와 연결
    channel = grpc.insecure_channel('localhost:50051')
    stub = ocr_pb2_grpc.OCRServiceStub(channel)

    # OCR 요청: 이미지 데이터를 바이너리로 전송
    response = stub.PerformOCR(ocr_pb2.OCRRequest(image_data=image_data))

    # 결과 출력
    print("OCR Results:")
    for result in response.results:
        print(f"Text: {result.text}, Confidence: {result.confidence}, Box: {result.box}")

if __name__ == '__main__':
    run()
