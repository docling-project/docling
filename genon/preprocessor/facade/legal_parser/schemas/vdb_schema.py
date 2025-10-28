from typing import Optional

from pydantic import BaseModel, Field, field_validator

from commons.utils import normalize_to_nfc

#  From https://github.com/mindsandcompany/GenOS/blob/develop/admin-api/src/schema/data/vdb_schema.py

class VDBUploadFile(BaseModel):
    """
    VDB에 업로드할 파일을 나타내는 스키마입니다.
    /document/upload/token API 호출 시 Request Body에 사용됩니다.

    속성:
        file_name (str): 파일 이름.
        file_content (bytes): 파일 내용 (바이너리 데이터).
    """
    file_name: str
    file_content: bytes

class DocumentFile(BaseModel):
    """
    문서 파일을 나타내는 스키마입니다.
    
    속성:
        name (str): 사용자 선택 위치의 상대 경로로 보내야 하는 파일 경로.
        path (str): 파일의 전체 경로.
    """
    name: str = Field(..., description="파일 경로. 사용자 선택한 위치의 상대 경로로 보내주세요")
    path: str
    
    @field_validator("name", mode='before')
    def validate_params(cls, value):
        """
        파일 이름을 NFC 형식으로 정규화합니다.

        Args:
            value (str): 파일 이름.

        Returns:
            str: 정규화된 파일 이름.
        """
        return normalize_to_nfc(value)

class VDBRegisterRequest(BaseModel):
    """
    벡터 문서를 등록하는 요청을 나타내는 스키마입니다.
    
    속성:
        vdb_id (int): 벡터 데이터베이스의 ID.
        description (Optional[str]): 벡터 문서에 대한 설명 (선택 사항).
        serving_id (int): 벡터 문서의 서빙 ID.
        serving_rev_id (int): 서빙 문서의 리비전 ID.
        preprocessor_id (int): 전처리기 ID.
        batch_size (int): 처리할 배치 크기.
        params (str): 처리 파라미터 (예: 청크 크기 및 겹침 크기).
        files (List[DocumentFile]): 등록할 문서 파일 목록.
    """
    vdb_id: str
    description: Optional[str] = None
    serving_id: int
    serving_rev_id: int
    preprocessor_id: int
    batch_size: int
    params: str         # ex) "{\"chunk_size\":1000,\"chunk_overlap\":100}",
    files: list[DocumentFile]  # 하나씩 올리기를 권장

class DataModel(BaseModel):
    """
    업로드된 파일 데이터를 포함하는 스키마입니다.

    속성:
        files (list[DocumentFile]): 업로드된 문서 파일 목록.
    """
    files: list[DocumentFile]

class VDBUploadResponse(BaseModel):
    """
    VDB 업로드 응답을 나타내는 스키마입니다.

    속성:
        code (int): 응답 코드.
        errMsg (str): 에러 메시지.
        data (Optional[DataModel]): 업로드된 데이터 정보.
    """
    code: int
    errMsg: str
    data: Optional[DataModel]

class RegisterData(BaseModel):
    """
    벡터 등록 결과 데이터를 나타내는 스키마입니다.

    속성:
        doc_ids (list[int]): 등록된 문서 ID 목록.
        upsert_ids (list[int]): 업서트 ID 목록.
    """
    doc_ids: list[int]
    upsert_ids: list[int]

class VDBRegisterResponse(BaseModel):
    """
    VDB 벡터 등록 응답을 나타내는 스키마입니다.

    속성:
        code (int): 응답 코드.
        errMsg (str): 에러 메시지.
        data (RegisterData): 등록된 데이터 정보.
    """
    code: int
    errMsg: str
    data: RegisterData

class LawVectorMapping(BaseModel):
    """
    법령 데이터와 벡터 데이터 간의 매핑 정보를 나타내는 스키마입니다.

    속성:
        law_id (str): 법령 ID.
        law_num (str): 법령 번호.
        law_type (str): 법령 유형 ("law"/"admrule").
        file_name (str): 파일 이름.
        doc_id (int): 문서 ID.
        vdb_id (int): VDB ID.
        upsert_id (int): 업서트 ID.
    """
    law_id: str
    law_num: str
    law_type: str  # "law" or "admrule"
    file_name: str
    doc_id: int
    vdb_id: int
    upsert_id: int

class LawFileInfo(BaseModel):
    """
    법령 파일 정보를 나타내는 스키마입니다.

    속성:
        law_type (str): 법령 유형 ("law" 또는 "admrule").
        law_id (str): 법령 ID.
        law_num (str): 법령 번호.
        filename (str): 파일 이름.
    """
    law_type: str  # "law" or "admrule"
    law_id: str
    law_num: str
    filename: str  # 파일명을 포함하여 추후 참조

class VDBResponse(BaseModel):
    """
    VDB 벡터화 작업의 결과를 나타내는 스키마입니다.

    속성:
        total_count (int): 총 처리된 파일 수.
        success_count (int): 성공적으로 처리된 파일 수.
        fail_count (int): 실패한 파일 수.
        fail_filenames (set[str]): 실패한 파일 이름 목록.
        mappings (list[LawVectorMapping]): 법령과 벡터 데이터 간의 매핑 목록.
    """
    total_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    fail_filenames: set[str] = set()
    mappings: list[LawVectorMapping] = []

    def increment_total(self):
        """총 처리된 파일 수를 증가시킵니다."""
        self.total_count += 1

    def increment_success(self):
        """성공적으로 처리된 파일 수를 증가시킵니다."""
        self.success_count += 1

    def increment_fail(self, filename: str):
        """
        실패한 파일 수를 증가시키고, 실패한 파일 이름을 추가합니다.

        Args:
            filename (str): 실패한 파일 이름.
        """
        self.fail_count += 1
        self.fail_filenames.add(filename)