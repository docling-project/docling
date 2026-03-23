import os
import json
from pathlib import Path

# 1. 필수 모듈 임포트
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
# 파일 위치가 docling/backend/genos_hwp_backend.py 라고 가정합니다.
from docling.backend.genos_hwp_backend import GenosHwpDocumentBackend

def test_hwp_backend():
    # --- [설정 구간] ---
    # 테스트할 HWP 파일 경로 설정
    current_dir = Path(__file__).parent 
    hwp_path = current_dir.parent / "samples" / "01_[핵심이슈]가계별 금리익스포저를 감안한 금리상승의 소비 영향 점검_24.2.19_공개용.hwp"
    #hwp_path = current_dir.parent / "samples" / "03_2023.05.04_BOK조사연구_(국제국) 외환국제금융동향(2302).hwpx"
    #hwp_path = current_dir.parent / "samples" / "04_I.+4.+부록3_국내외+주요+경제지표_공개용_물가동향팀.hwp"  

    # (선택 사항) 경로가 잘 잡혔는지 출력해서 확인
    print(f"📍 파일 경로 확인: {hwp_path.resolve()}")

    # 결과물을 저장할 폴더
    output_dir = Path("./test_results")
    output_dir.mkdir(exist_ok=True)
    
    # SDK 환경 확인 (convtext 파일이 실행 위치에 있어야 함)
    if not Path("./convtext").exists():
        print("❌ 경고: 'convtext' 바이너리가 현재 디렉토리에 없습니다.")
        print("백엔드 내부에서 subprocess 실행 시 경로 에러가 발생할 수 있습니다.")
    # ------------------

    print(f"🚀 테스트 시작: {hwp_path.name}")

    try:
        # 2. InputDocument 생성
        # backend 인자에 우리가 만든 클래스를 직접 전달합니다.
        in_doc = InputDocument(
            path_or_stream=hwp_path,
            format=InputFormat.HWP,
            backend=GenosHwpDocumentBackend,
        )

        # 3. 백엔드 직접 인스턴스화 및 변환 실행
        backend = GenosHwpDocumentBackend(in_doc=in_doc, path_or_stream=hwp_path)
        
        print("⏳ SDK 변환 및 DoclingDocument 매핑 중...")
        doc = backend.convert()
        print("✅ 변환 완료!")

        # 4. 결과 검증 및 저장
        # (1) Markdown 출력 확인
        md_text = doc.export_to_markdown()
        md_path = output_dir / f"{hwp_path.stem}.md"
        md_path.write_text(md_text, encoding="utf-8")
        print(f"📄 Markdown 저장 완료: {md_path}")

        # (2) JSON 구조 확인 (구조가 깨지지 않았는지 확인용)
        json_dict = doc.export_to_dict()
        json_path = output_dir / f"{hwp_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=2)
        print(f"📂 JSON 데이터 저장 완료: {json_path}")

        # (3) 간단한 통계 출력
        item_counts = {}
        for item, _ in doc.iterate_items():
            label = item.label
            item_counts[label] = item_counts.get(label, 0) + 1
        
        print("\n--- [변환 결과 요약] ---")
        for label, count in item_counts.items():
            print(f"- {label}: {count}개")
        print("----------------------")

    except Exception as e:
        print(f"🔥 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 가비지 컬렉션 전 명시적 리소스 해제 테스트
        if 'backend' in locals():
            backend.unload()

if __name__ == "__main__":
    test_hwp_backend()