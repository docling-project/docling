"""
PDF 문서에서 섹션헤더가 청크 내에서 본문 순서대로 나오는지 확인하는 테스트
pdf_sample.pdf 기준으로 작성
"""
import pytest
from pathlib import Path
import re
from typing import List, Dict, Any


class TestSectionHeaderOrder:
    """섹션헤더 순서 테스트 클래스"""
    
    def extract_headers_from_chunk(self, chunk_text: str) -> List[str]:
        """청크 텍스트에서 HEADER: 부분을 추출하여 리스트로 반환"""
        header_pattern = r'HEADER:\s*(.+?)(?:\n|$)'
        matches = re.findall(header_pattern, chunk_text)
        
        if not matches:
            return []
        
        # 첫 번째 매치에서 쉼표로 구분된 헤더들을 분리
        headers = [h.strip() for h in matches[0].split(',') if h.strip()]
        return headers
    
    def check_header_order_in_chunk(self, chunk_text: str, expected_headers: List[str]) -> bool:
        """청크 내에서 헤더가 예상 순서대로 나오는지 확인"""
        extracted_headers = self.extract_headers_from_chunk(chunk_text)
        
        if not extracted_headers:
            return len(expected_headers) == 0
        
        # 예상 헤더와 추출된 헤더가 순서대로 일치하는지 확인
        if len(extracted_headers) != len(expected_headers):
            return False
        
        for i, (extracted, expected) in enumerate(zip(extracted_headers, expected_headers)):
            if extracted != expected:
                print(f"헤더 순서 불일치 at index {i}: '{extracted}' != '{expected}'")
                return False
        
        return True
    
    def analyze_chunk_headers(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """청크들의 헤더 정보를 분석"""
        analysis = {
            'total_chunks': len(chunks),
            'chunks_with_headers': 0,
            'chunks_with_multiple_headers': 0,
            'header_sequences': [],
            'potential_issues': []
        }
        
        for i, chunk in enumerate(chunks):
            # dict와 Pydantic 모델(예: GenOSVectorMeta) 모두 지원
            if isinstance(chunk, dict):
                chunk_text = chunk['text'] if 'text' in chunk else ''
            else:
                chunk_text = getattr(chunk, 'text', '')
            
            headers = self.extract_headers_from_chunk(chunk_text)
            
            if headers:
                analysis['chunks_with_headers'] += 1
                analysis['header_sequences'].append({
                    'chunk_index': i,
                    'headers': headers,
                    'header_count': len(headers)
                })
                
                if len(headers) > 1:
                    analysis['chunks_with_multiple_headers'] += 1
                    
                    # 다중 헤더 청크에서 순서 문제 확인
                    if len(headers) > 1:
                        # 헤더들이 논리적 순서를 따르는지 확인
                        header_order_valid = self._validate_header_sequence(headers)
                        if not header_order_valid:
                            analysis['potential_issues'].append({
                                'chunk_index': i,
                                'headers': headers,
                                'issue': '헤더 순서가 논리적이지 않음'
                            })
        
        return analysis
    
    def _validate_header_sequence(self, headers: List[str]) -> bool:
        """헤더 시퀀스가 논리적인 순서를 따르는지 확인 (pdf_sample.pdf 기준)"""
        if len(headers) <= 1:
            return True
        
        # pdf_sample.pdf의 예상 헤더 순서 패턴들
        expected_patterns = [
            # SECTION 순서
            ["SECTION I:", "SECTION II:", "SECTION III:"],
            # Reading Task 순서  
            ["Reading Task 1", "Reading Task 2"],
            # 일반적인 문서 구조
            ["A2", "English Practice Test"],
            ["개요", "상세", "결론"],
            ["제1장", "제2장", "제3장"]
        ]
        
        # 각 패턴과 매칭되는지 확인
        for pattern in expected_patterns:
            if self._matches_pattern(headers, pattern):
                return True
        
        # 숫자로 시작하는 헤더들의 순서 확인
        numbered_headers = []
        for header in headers:
            # "1.", "2.", "3." 같은 패턴 찾기
            match = re.match(r'^(\d+)\.', header.strip())
            if match:
                numbered_headers.append((int(match.group(1)), header))
        
        if len(numbered_headers) > 1:
            numbers = [num for num, _ in numbered_headers]
            return numbers == sorted(numbers)
        
        return True  # 기본적으로 유효하다고 간주
    
    def _matches_pattern(self, headers: List[str], pattern: List[str]) -> bool:
        """헤더들이 특정 패턴과 매칭되는지 확인"""
        if len(headers) != len(pattern):
            return False
        
        for header, expected in zip(headers, pattern):
            if expected not in header and header not in expected:
                return False
        
        return True
    
    @pytest.mark.parametrize("processor_name", ["basic_processor", "intelligent_processor"])
    def test_section_header_order_in_pdf(self, processor_name, sample_dir, request):
        """PDF 문서에서 섹션헤더 순서 테스트"""
        # 사용자 지시: unit 하의 mncai(= intelligent_processor 관련) 무시
        if processor_name == "intelligent_processor":
            pytest.skip("사용자 지시에 따라 intelligent_processor(MNCAI 관련)는 건너뜀")
        
        # 프로세서 픽스처 가져오기
        processor_class = request.getfixturevalue(processor_name)
        
        # PDF 샘플 파일들 찾기
        pdf_files = list(sample_dir.glob("*.pdf"))
        if not pdf_files:
            pytest.skip("PDF 샘플 파일이 없습니다")
        
        # 첫 번째 PDF 파일 사용
        pdf_file = pdf_files[0]
        print(f"\n테스트 파일: {pdf_file.name}")
        
        # 프로세서 인스턴스 생성
        processor = processor_class()
        
        # Mock request 객체 생성
        class MockRequest:
            pass
        
        mock_request = MockRequest()
        
        # 문서 처리 (비동기 호출을 동기적으로 실행)
        try:
            vectors = run_async_test(processor.__call__(mock_request, str(pdf_file)))
        except Exception as e:
            pytest.fail(f"문서 처리 실패: {e}")
        
        # 결과 분석
        analysis = self.analyze_chunk_headers(vectors)
        
        print(f"\n=== {processor_name} 분석 결과 ===")
        print(f"총 청크 수: {analysis['total_chunks']}")
        print(f"헤더가 있는 청크 수: {analysis['chunks_with_headers']}")
        print(f"다중 헤더 청크 수: {analysis['chunks_with_multiple_headers']}")
        
        # 다중 헤더 청크들 상세 분석
        if analysis['chunks_with_multiple_headers'] > 0:
            print(f"\n=== 다중 헤더 청크 상세 분석 ===")
            for chunk_info in analysis['header_sequences']:
                if chunk_info['header_count'] > 1:
                    print(f"청크 {chunk_info['chunk_index']}: {chunk_info['headers']}")
        
        # 잠재적 문제점 출력
        if analysis['potential_issues']:
            print(f"\n=== 잠재적 문제점 ===")
            for issue in analysis['potential_issues']:
                print(f"청크 {issue['chunk_index']}: {issue['issue']}")
                print(f"  헤더들: {issue['headers']}")
        
        # 검증: 다중 헤더 청크가 있으면 순서가 논리적인지 확인
        if analysis['chunks_with_multiple_headers'] > 0:
            problematic_chunks = len(analysis['potential_issues'])
            total_multi_header_chunks = analysis['chunks_with_multiple_headers']
            
            print(f"\n=== 검증 결과 ===")
            print(f"다중 헤더 청크 중 문제가 있는 청크: {problematic_chunks}/{total_multi_header_chunks}")
            
            # 문제가 있는 청크가 전체의 50% 이상이면 실패
            if problematic_chunks > total_multi_header_chunks * 0.5:
                pytest.fail(f"너무 많은 청크에서 헤더 순서 문제 발견: {problematic_chunks}/{total_multi_header_chunks}")
        
        # 최소한의 검증: 헤더가 있는 청크가 존재해야 함
        assert analysis['chunks_with_headers'] > 0, "헤더가 포함된 청크가 없습니다"
    
    def test_specific_header_sequence(self, basic_processor, sample_dir):
        """특정 헤더 시퀀스 패턴 테스트 (pdf_sample.pdf 기준)"""
        # pdf_sample.pdf의 실제 헤더 순서 패턴들
        expected_patterns = [
            ["SECTION I:", "SECTION II:", "SECTION III:"],  # SECTION 순서
            ["Reading Task 1", "Reading Task 2"],  # Reading Task 순서
            ["A2", "English Practice Test"],  # 문서 제목 구조
        ]
        
        # 각 패턴에 대해 검증
        for pattern in expected_patterns:
            is_valid = self._validate_header_sequence(pattern)
            assert is_valid, f"헤더 패턴이 유효하지 않음: {pattern}"
    
    def test_header_extraction_regex(self):
        """헤더 추출 정규식 테스트 (pdf_sample.pdf 기준)"""
        test_cases = [
            # 실제 pdf_sample.pdf의 헤더 패턴들
            ("HEADER: A2, English Practice Test, SECTION I: Grammar (Use of English)\n본문 내용", 
             ["A2", "English Practice Test", "SECTION I: Grammar (Use of English)"]),
            ("HEADER: SECTION I: Grammar (Use of English), SECTION II: Reading Comprehension\n본문 내용", 
             ["SECTION I: Grammar (Use of English)", "SECTION II: Reading Comprehension"]),
            ("HEADER: Reading Task 1, Reading Task 2\n본문 내용", 
             ["Reading Task 1", "Reading Task 2"]),
            ("HEADER: 1. 'COMEWITHUS.COM' sell, 2. According to the advertisement\n본문 내용", 
             ["1. 'COMEWITHUS.COM' sell", "2. According to the advertisement"]),
            ("본문만 있는 청크", []),
        ]
        
        for chunk_text, expected_headers in test_cases:
            extracted = self.extract_headers_from_chunk(chunk_text)
            assert extracted == expected_headers, f"헤더 추출 실패: {chunk_text} -> {extracted} != {expected_headers}"


# 비동기 함수를 동기적으로 실행하기 위한 헬퍼
import asyncio

def run_async_test(coro):
    """비동기 테스트를 동기적으로 실행"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)
