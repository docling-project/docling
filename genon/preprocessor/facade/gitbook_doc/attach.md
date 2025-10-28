---
description: >-
  Genos 는 크게 적재용(내부), 적재용(외부), 적재용(규정), 첨부용 4가지 유형의 전처리기 (document parser)를
  지원합니다. 여기서는 첨부용 Intelligent Doc Parser - text 추출형 전처리기의 코드 원형에 대해서 설명합니다.
icon: forward
---

# 첨부용 문서 전처리기

<figure><img src="../../../../.gitbook/assets/preprocess_code.png" alt=""><figcaption><p>전처리기 상세에서 아래 코드를 확인하실 수 있습니다.</p></figcaption></figure>

***

여기서는 Genos 첨부용 문서 파서의 전처리 파이프라인 내 주요 구성 요소에 대한 코드 중심의 설명을 제공합니다. 코드 조각과 함께 각 부분의 기능을 이해함으로써, 특정 요구 사항 및 문서 유형에 맞게 문서 처리 프로세스를 보다 효과적으로 조정할 수 있습니다.

### 🔧 공통 구성요소

#### `GenOSVectorMetaBuilder` 및 `GenOSVectorMeta`

`GenOSVectorMetaBuilder`는 각 청크에 대한 상세 메타데이터 객체인 `GenOSVectorMeta`를 단계적으로 생성하는 역할을 합니다.

**`GenOSVectorMeta` (Pydantic 모델)**

먼저, 최종적으로 생성될 메타데이터의 구조를 정의하는 Pydantic 모델입니다.

Python

```python
class GenOSVectorMeta(BaseModel):
    class Config:
        extra = 'allow' # Pydantic v2에서는 extra='allow' 대신 model_config 사용 가능

    text: str = None
    n_char: int = None
    n_word: int = None
    n_line: int = None
    e_page: int = None
    i_page: int = None
    i_chunk_on_page: int = None
    n_chunk_of_page: int = None
    i_chunk_on_doc: int = None
    n_chunk_of_doc: int = None
    n_page: int = None
    reg_date: str = None
    chunk_bboxes: str = None
    media_files: str = None
```

**설명:**

* `BaseModel`을 상속받아 Pydantic 모델로 정의됩니다. 이는 데이터 유효성 검사 및 직렬화/역직렬화를 용이하게 합니다.
* `Config.extra = 'allow'`: 모델에 정의되지 않은 추가 필드가 입력 데이터에 존재하더라도 오류를 발생시키지 않고 허용합니다. (Pydantic V2에서는 `model_config = ConfigDict(extra='allow')` 형태로 사용)
* 각 필드는 청크의 메타데이터 항목을 나타냅니다.
  * `text`: 청크의 텍스트 내용.
  * `n_char`, `n_word`, `n_line`: 문자 수, 단어 수, 줄 수.
  * `e_page`, `i_page`, `i_chunk_on_page`, `n_chunk_of_page`: 페이지 내에서의 청크 위치 정보.
  * `i_chunk_on_doc`, `n_chunk_of_doc`: 문서 전체에서의 청크 위치 정보.
  * `n_page`: 문서의 총 페이지 수.
  * `reg_date`: 처리 등록 시간.
  * `bboxes`: 페이지 내 해당 청크의 경계 상자 (JSON 문자열 형태).
  * `chunk_bboxes`: 청크를 구성하는 각 `DocItem`의 상세 경계 상자 정보 리스트.
  * `media_files`: 청크 내 포함된 미디어 파일(이미지) 정보 리스트.

**사용자 정의 포인트:**

* 고객사에서 필요한 추가적인 메타데이터 항목이 있다면, 이 `GenOSVectorMeta` 모델에 새로운 필드를 추가로 정의할 수 있습니다.
* 필드 타입을 보다 엄격하게 정의하거나 (예: `Optional[str]`), 기본값을 설정하거나, 유효성 검사 로직을 추가할 수 있습니다.

***

**`GenOSVectorMetaBuilder` 클래스 및 주요 메서드**

`GenOSVectorMeta` 객체를 생성하는 빌더 클래스입니다.

Python

```python
class GenOSVectorMetaBuilder:
    def __init__(self):
        """빌더 초기화"""
        self.text: Optional[str] = None
        self.n_char: Optional[int] = None
        # ... (다른 필드들도 초기화) ...

    def set_text(self, text: str) -> "GenOSVectorMetaBuilder":
        """텍스트와 관련된 데이터를 설정"""
        self.text = text
        self.n_char = len(text)
        self.n_word = len(text.split())
        self.n_line = len(text.splitlines())
        return self

    def set_page_info(
            self, i_page: int, i_chunk_on_page: int, n_chunk_of_page: int
    ) -> "GenOSVectorMetaBuilder":
        """페이지 정보 설정"""
        self.i_page = i_page
        self.i_chunk_on_page = i_chunk_on_page
        self.n_chunk_of_page = n_chunk_of_page
        return self

    def set_chunk_index(self, i_chunk_on_doc: int) -> "GenOSVectorMetaBuilder":
        """문서 전체의 청크 인덱스 설정"""
        self.i_chunk_on_doc = i_chunk_on_doc
        return self

    def set_global_metadata(self, **global_metadata) -> "GenOSVectorMetaBuilder":
        """글로벌 메타데이터 병합"""
        for key, value in global_metadata.items():
            if hasattr(self, key): # 빌더 내에 해당 속성이 정의되어 있는지 확인
                setattr(self, key, value)
        return self

    def set_chunk_bboxes(self, doc_items: list, document: DoclingDocument) -> "GenOSVectorMetaBuilder":
        chunk_bboxes = []
        for item in doc_items:
            for prov in item.prov:
                label = item.self_ref
                type_ = item.label
                size = document.pages.get(prov.page_no).size
                page_no = prov.page_no
                bbox = prov.bbox
                bbox_data = {
                    'l': bbox.l / size.width,
                    't': bbox.t / size.height,
                    'r': bbox.r / size.width,
                    'b': bbox.b / size.height,
                    'coord_origin': bbox.coord_origin.value
                }
                chunk_bboxes.append({
                    'page': page_no,
                    'bbox': bbox_data,
                    'type': type_,
                    'ref': label
                })
        self.e_page = max([bbox['page'] for bbox in chunk_bboxes]) if chunk_bboxes else None
        self.chunk_bboxes = json.dumps(chunk_bboxes)
        return self

    def set_media_files(self, doc_items: list) -> "GenOSVectorMetaBuilder":
        temp_list = []
        if not doc_items:
            self.media_files = ""
            return self
        for item in doc_items:
            if isinstance(item, PictureItem):
                path = str(item.image.uri)
                name = path.rsplit("/", 1)[-1]
                temp_list.append({'name': name, 'type': 'image', 'ref': item.self_ref})
        self.media_files = json.dumps(temp_list)
        return self

    def build(self) -> GenOSVectorMeta:
        """설정된 데이터를 사용해 최종적으로 GenOSVectorMeta 객체 생성"""
        return GenOSVectorMeta(
            text=self.text,
            n_char=self.n_char,
            # ... (모든 필드를 GenOSVectorMeta 생성자에 전달) ...
        )

```

**설명:**

* **`__init__`**: 빌더 내부의 모든 속성들을 초기화합니다. 이 속성들은 `GenOSVectorMeta`의 필드들과 대부분 일치합니다.
* **`set_text`**: 청크의 텍스트를 설정하고, 문자 수, 단어 수, 줄 수를 계산하여 내부 속성에 저장합니다.
* **`set_page_info`**: 페이지 번호, 페이지 내 청크 인덱스, 페이지 내 총 청크 수를 설정합니다.
* **`set_chunk_index`**: 문서 전체에서의 청크 인덱스를 설정합니다.
* **`set_global_metadata`**: `DocumentProcessor.compose_vectors`에서 전달받은 `global_metadata` 딕셔너리의 값들을 빌더의 해당 속성에 할당합니다. 빌더 내에 `global_metadata`의 키와 동일한 이름의 속성이 있어야 값이 할당됩니다.
* **`set_chunk_bboxes`**: 청크를 구성하는 모든 `DocItem`들의 상세한 경계 상자 정보를 추출하여 리스트로 저장합니다. 각 항목은 페이지 번호, 정규화된 좌표(0\~1 값), `DocItem`의 타입 및 참조 ID를 포함합니다. 정규화된 좌표는 페이지 크기에 상대적인 위치를 나타내므로, 다양한 크기의 페이지에서도 일관되게 위치를 표현할 수 있습니다.
* **`set_media_files`**: 청크 내에 `PictureItem`(이미지)이 포함되어 있다면, 해당 이미지의 파일 이름, 타입("image"), 참조 ID를 추출하여 리스트로 저장합니다.
* **`build`**: 지금까지 `set_...` 메서드들을 통해 빌더 내부에 축적된 모든 속성값들을 사용하여 최종적으로 `GenOSVectorMeta` Pydantic 모델 객체를 생성하고 반환합니다.

**사용자 정의 포인트:**

* `GenOSVectorMeta` 모델에 새로운 필드를 추가했다면, 이 빌더에도 해당 필드를 위한 내부 속성과 `set_...` 메서드를 추가해야 합니다.
* `build` 메서드에서 `GenOSVectorMeta` 객체를 생성할 때 새로 추가된 필드도 인자로 전달하도록 수정해야 합니다.
* 특정 필드값을 설정하기 전에 추가적인 가공 로직(예: 날짜 형식 변환, 특정 코드값 매핑 등)이 필요하다면 해당 `set_...` 메서드 내부에 구현할 수 있습니다.

***

### 공통 문서 로더

#### `HwpLoader`

hwp 파일을 로드하고 변환하는 데 사용됩니다. HWP 파일을 XHTML로 변환한 후 PDF로 저장하고, 최종적으로 PyMuPDFLoader를 사용하여 문서 내용을 로드합니다.

Python

```python
class HwpLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.output_dir = os.path.join('/tmp', str(uuid.uuid4()))
        os.makedirs(self.output_dir, exist_ok=True)

    def load(self):
        try:
            subprocess.run(['hwp5html', self.file_path, '--output', self.output_dir], check=True, timeout=600)
            converted_file_path = os.path.join(self.output_dir, 'index.xhtml')
            pdf_save_path = self.file_path.replace('.hwp', '.pdf')
            HTML(converted_file_path).write_pdf(pdf_save_path)
            loader = PyMuPDFLoader(pdf_save_path)
            return loader.load()
        except Exception as e:
            print(f"Failed to convert {self.file_path} to XHTML")
            raise e
        finally:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
```

**설명:**

* `HwpLoader`: HWP 파일을 로드하고 변환하는 데 사용되는 클래스입니다. HWP 파일을 XHTML로 변환한 후 PDF로 저장하고, 최종적으로 PyMuPDFLoader를 사용하여 문서 내용을 로드합니다.

***

#### `TextLoader`

텍스트 파일을 로드하고 변환하는 데 사용됩니다. 다양한 인코딩을 지원하며, PDF로 변환할 수 있는 기능을 포함하고 있습니다.

Python

```python
class TextLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.output_dir = os.path.join('/tmp', str(uuid.uuid4()))
        os.makedirs(self.output_dir, exist_ok=True)

    def load(self):
        try:
            # 1) 샘플로 인코딩 추정(150바이트)
            with open(self.file_path, 'rb') as f:
                sample = f.read(150)
            enc = chardet.detect(sample).get('encoding') or ''
            encodings = [enc] if enc and enc.lower() not in ('ascii','unknown') else []
            encodings += ['utf-8', 'cp949', 'euc-kr', 'iso-8859-1', 'latin-1']
            # 2) 전체 파일 바이트/텍스트 확보
            with open(self.file_path, 'rb') as f:
                raw = f.read()

            content = None
            for e in encodings:
                try:
                    content = raw.decode(e)  # 전체 파일로 디코딩
                    break
                except UnicodeDecodeError:
                    continue
            if content is None:
                content = raw.decode('utf-8', errors='replace')

            # 4) PDF 변환 유지
            html = f"<html><meta charset='utf-8'><body><pre>{content}</pre></body></html>"
            html_path = os.path.join(self.output_dir, 'temp.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
            pdf_path = (self.file_path
                        .replace('.txt', '.pdf')
                        .replace('.json', '.pdf'))
            if HTML:
                HTML(html_path).write_pdf(pdf_path)
                loader = PyMuPDFLoader(pdf_path)
                return loader.load()
            # PDF가 불가하면 Document 직접 반환 (원형 스키마 유지)
            return [Document(page_content=content, metadata={'source': self.file_path, 'page': 0})]

        except Exception:
            # 실패 시에도 스키마는 그대로 유지해 반환
            for e in ['utf-8', 'cp949', 'euc-kr', 'iso-8859-1']:
                try:
                    with open(self.file_path, 'r', encoding=e) as f:
                        content = f.read()
                    return [Document(page_content=content, metadata={'source': self.file_path, 'page': 0})]
                except UnicodeDecodeError:
                    continue
            with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            return [Document(page_content=content, metadata={'source': self.file_path, 'page': 0})]
        finally:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
```

**설명:**

* `TextLoader`: 텍스트 파일을 로드하고 변환하는 데 사용되는 클래스입니다. 다양한 인코딩을 지원하며, PDF로 변환할 수 있는 기능을 포함하고 있습니다.

***

#### `TabularLoader`

표 형식의 데이터를 로드하고 변환하는 데 사용됩니다. CSV, Excel 등 다양한 형식의 표 데이터를 지원합니다.

Python

```python
class TabularLoader:
    def __init__(self, file_path: str, ext: str):
        packages = ['openpyxl', 'chardet']
        install_packages(packages)

        self.file_path = file_path
        if ext == ".csv":
            self.data_dict = self.load_csv_documents(file_path)
        elif ext == ".xlsx":
            self.data_dict = self.load_xlsx_documents(file_path)
        else:
            print(f"[!] Inadequate extension for TabularLoader: {ext}")
            return

    def check_sql_dtypes(self, df):
        df = df.convert_dtypes()
        res = []
        for col in df.columns:
            # col_name = col.strip().replace(' ', '_')
            dtype = str(df.dtypes[col]).lower()

            if 'int' in dtype:
                if '64' in dtype:
                    sql_dtype = 'BIGINT'
                else:
                    sql_dtype = 'INT'
            elif 'float' in dtype:
                sql_dtype = 'FLOAT'
            elif 'bool' in dtype:
                sql_dtype = 'BOOLEAN'
            elif 'date' in dtype:
                sql_dtype = 'DATE'
                df[col] = df[col].astype(str)
            elif 'datetime' in dtype:
                sql_dtype = 'DATETIME'
                df[col] = df[col].astype(str)
            # else:
            #     max_len = df[col].str.len().max().item() + 10
            #     sql_dtype = f'VARCHAR({max_len})'
            else:
                lens = df[col].astype(str).str.len()
                max_len_val = lens.max()
                max_len = int(0 if pd.isna(max_len_val) else max_len_val) + 10
                sql_dtype = f'VARCHAR({max_len})'

            res.append([col, sql_dtype])

        return df, res

    def process_data_rows(self, data: dict):
        """Arg: data (keys: 'sheet_name', 'page_column', 'page_column_type', 'documents')"""

        rows = []
        for doc in data["documents"]:
            row = {}
            if 'int' in data["page_column_type"]:
                row[data["page_column"]] = int(doc.page_content)
            elif 'float' in data["page_column_type"]:
                row[data["page_column"]] = float(doc.page_content)
            elif 'bool' in data["page_column_type"]:
                if doc.page_content.lower() == 'true':
                    row[data["page_column"]] = True
                elif doc.page_content.lower() == 'false':
                    row[data["page_column"]] = False
                else:
                    raise ValueError(f"Invalid boolean string: {doc.page_content}")
            else:
                row[data["page_column"]] = doc.page_content

            row.update(doc.metadata)
            rows.append(row)

        processed_data = {"sheet_name": data["sheet_name"], "data_rows": rows, "data_types": data["dtypes"]}
        return processed_data

    def load_csv_documents(self, file_path: str, **kwargs: dict):
        import chardet

        with open(file_path, "rb") as f:
            raw_file = f.read(10000)
        enc_type = chardet.detect(raw_file)['encoding']
        df = pd.read_csv(file_path, encoding=enc_type, index_col=False)
        df = df.fillna('null') # csv 파일에서도 xlsx 파일과 동일하게 null로 채움
        df, dtypes_str = self.check_sql_dtypes(df)

        for i in range(len(df.columns)):
            try:
                col = df.columns[0]
                # col_type = str(type(col))
                col_type = str(df[col].dtype)
                df = df.astype({col: 'str'})
                break
            except:
                raise ValueError(
                    f"Any columns cannot be converted into the string type so that can't load LangChain Documents: {dtypes_str}")

        loader = DataFrameLoader(df, page_content_column=col)
        documents = loader.load()

        data = {
            "sheet_name": "table_1",
            "page_column": col,
            "page_column_type": col_type,
            "documents": documents,
            "dtypes": dtypes_str
        }
        data = self.process_data_rows(data)  # including only one sheet as it's a csv file
        data_dict = {"data": [data]}
        return data_dict

    def load_xlsx_documents(self, file_path: str, **kwargs: dict):
        dfs = pd.read_excel(file_path, sheet_name=None)
        sheets = []
        for sheet_name, df in dfs.items():
            df = df.fillna('null')
            df, dtypes_str = self.check_sql_dtypes(df)

            for i in range(len(df.columns)):
                try:
                    col = df.columns[0]
                    col_type = str(type(col))
                    df = df.astype({col: 'str'})
                    break
                except:
                    raise ValueError(
                        f"Any columns cannot be converted into string type so that can't load LangChain Documents: {dtypes_str}")

            loader = DataFrameLoader(df, page_content_column=col)
            documents = loader.load()

            sheet = {
                "sheet_name": sheet_name,
                "page_column": col,
                "page_column_type": col_type,
                "documents": documents,
                "dtypes": dtypes_str
            }
            sheets.append(sheet)

        data_dict = {"data": []}
        for sheet in sheets:
            data = self.process_data_rows(sheet)
            data_dict["data"].append(data)

        return data_dict

    def return_vectormeta_format(self):
        if not self.data_dict:
            return None

        text = "[DA] " + str(self.data_dict)  # Add a token to indicate this string is for data analysis
        vectors = [GenOSVectorMeta.model_validate({
            'text': text,
            'n_chars': 1,
            'n_words': 1,
            'n_lines': 1,
            'i_page': 1,
            'e_page': 1,
            'n_page': 1,
            'i_chunk_on_page': 1,
            'n_chunk_of_page': 1,
            'i_chunk_on_doc': 1,
            'reg_date': datetime.now().isoformat(timespec='seconds') + 'Z',
            'chunk_bboxes': ".",
            'media_files': "."
        })]
        return vectors
```

**설명:**

* `check_sql_dtypes`: 데이터프레임의 각 열에 대한 SQL 데이터 유형을 확인하고, 필요한 경우 형 변환을 수행합니다.
* `process_data_rows`: 데이터 행을 처리하고, 필요한 경우 변환을 수행합니다.
* `load_csv_documents`: CSV 문서를 로드하고, 필요한 전처리를 수행합니다.
* `load_xlsx_documents`: Excel 문서를 로드하고, 필요한 전처리를 수행합니다.
* `return_vectormeta_format`: 현재 데이터의 메타 정보를 벡터 형식으로 변환하여 반환합니다.

***

#### `AudioLoader`

`AudioLoader` 클래스는 오디오 파일을 로드하고, 청크로 분할하며, 각 청크에 대한 텍스트를 전사하는 기능을 제공합니다.

Python

```python
class AudioLoader:
    def __init__(self,
                 file_path: str,
                 req_url: str,
                 req_data: dict,
                 chunk_sec: int = 29,
                 tmp_path: str = '.',
                 ):
        self.file_path = file_path
        self.tmp_path = tmp_path
        self.chunk_sec = chunk_sec
        self.req_url = req_url
        self.req_data = req_data

    def split_file_as_chunks(self) -> list:
        audio = pydub.AudioSegment.from_file(self.file_path)
        chunk_len = self.chunk_sec * 1000
        n_chunks = math.ceil(len(audio) / chunk_len)

        for i in range(n_chunks):
            start_ms = i * chunk_len
            overlap_start_ms = start_ms - 300 if start_ms > 0 else start_ms
            end_ms = start_ms + chunk_len
            audio_chunk = audio[overlap_start_ms:end_ms]
            audio_chunk.export(os.path.join(self.tmp_path, "tmp_{}.wav".format(str(i))), format="wav")
        tmp_files = glob(os.path.join(self.tmp_path, "*.wav"))
        return tmp_files

    def transcribe_audio(self, file_path_lst: list):
        transcribed_text_chunks = []

        def _send_request(filepath: str):
            """Send a request to 'whisper' model served"""
            files = {
                'file': (filepath, open(filepath, 'rb'), 'audio/mp3'),
            }

            response = requests.post(self.req_url, data=self.req_data, files=files)
            text = response.json().get('text', ', ')
            transcribed_text_chunks.append({
                'file_name': os.path.basename(filepath),
                'text': text
            })

        # Send parallel requests
        threads = [threading.Thread(target=_send_request, args=(f,)) for f in file_path_lst]
        for t in threads: t.start()
        for t in threads: t.join()

        # Merge transcribed text snippets in order
        transcribed_text_chunks.sort(key=lambda x: x['file_name'])
        transcribed_text = "[AUDIO]" + ' '.join([t['text'] for t in transcribed_text_chunks])
        return transcribed_text

    def return_vectormeta_format(self):
        audio_chunks = self.split_file_as_chunks()
        transcribed_text = self.transcribe_audio(audio_chunks)
        res = [GenOSVectorMeta.model_validate({
            'text': transcribed_text,
            'n_chars': 1,
            'n_words': 1,
            'n_lines': 1,
            'i_page': 1,
            'e_page': 1,
            'n_page': 1,
            'i_chunk_on_page': 1,
            'n_chunk_of_page': 1,
            'i_chunk_on_doc': 1,
            'reg_date': datetime.now().isoformat(timespec='seconds') + 'Z',
            'chunk_bboxes': ".",
            'media_files': "."
        })]
        return res
```

**설명:**

* `AudioLoader`: 오디오 파일을 로드하고, 청크로 분할하며, 각 청크에 대한 텍스트를 전사하는 기능을 제공합니다.
  * `split_file_as_chunks`: 오디오 파일을 지정된 길이의 청크로 분할합니다.
  * `transcribe_audio`: 분할된 오디오 청크를 전사하여 텍스트로 변환합니다.
  * `return_vectormeta_format`: 전사된 텍스트를 벡터 메타 형식으로 변환하여 반환합니다.

***

#### `HWPX`

`HWPX` 클래스는 HWPX 형식의 문서를 로드하고, 필요한 전처리를 수행하는 기능을 제공합니다. `HierarchicalChunker`, `HybridChunker`, `HwpxProcessor` 등의 구성 요소를 사용하여 문서를 청크로 분할하고, 각 청크에 대한 메타데이터를 생성합니다.

##### ``HierarchicalChunker` 및 `HybridChunker`

`HierarchicalChunker`는 문서를 계층적으로 청크로 나누는 역할을 하며, `HybridChunker`는 토큰 제한을 고려하여 섹션별 청크를 분할하고 병합하는 고급 청커입니다.

Python

```python
class HierarchicalChunker(BaseChunker):
    merge_list_items: bool = True

    def chunk(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        # 모든 아이템과 헤더 정보 수집
        heading_by_level: dict[LevelNumber, str] = {}
        list_items: list[TextItem] = []
        for item, level in dl_doc.iterate_items():
            captions = None
            if isinstance(item, DocItem):
                # first handle any merging needed
                if self.merge_list_items:
                    if isinstance(
                            item, ListItem
                    ) or (  # TODO remove when all captured as ListItem:
                            isinstance(item, TextItem)
                            and item.label == DocItemLabel.LIST_ITEM
                    ):
                        list_items.append(item)
                        continue
                # ... 헤더 처리 로직

                c = DocChunk(
                    text=text,
                    meta=DocMeta(
                        doc_items=[item],
                        headings=[heading_by_level[k] for k in sorted(heading_by_level)] or None,
                        captions=captions,
                        origin=dl_doc.origin,
                    ),
                )
                yield c

        if self.merge_list_items and list_items:  # need to yield
            yield DocChunk(
                text=self.delim.join([i.text for i in list_items]),
                meta=DocMeta(
                    doc_items=list_items,
                    headings=[heading_by_level[k] for k in sorted(heading_by_level)] or None,
                    origin=dl_doc.origin,
                ),
            )

class HybridChunker(BaseChunker):

    def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        r"""Chunk the provided document.
        Args:
            dl_doc (DLDocument): document to chunk
        Yields:
            Iterator[Chunk]: iterator over extracted chunks
        """
        res: Iterable[DocChunk]
        res = self._inner_chunker.chunk(dl_doc=dl_doc, **kwargs)  # type: ignore
        res = [x for c in res for x in self._split_by_doc_items(c)]
        res = [x for c in res for x in self._split_using_plain_text(c)]

        if self.merge_peers:
            res = self._merge_chunks_with_matching_metadata(res)
        return iter(res)
```

***

##### `HwpxProcessor`

`HwpxProcessor` 클래스는 HWPX 형식의 문서를 처리하는 데 필요한 다양한 기능을 제공합니다. 이 클래스는 문서의 청크를 생성하고, 메타데이터를 구성하며, 최종적으로 벡터 형태로 변환하는 작업을 수행합니다.

Python

```python
class HwpxProcessor:
    def __init__(self):
        self.page_chunk_counts = defaultdict(int)
        self.pipeline_options = PipelineOptions()
        self.pipeline_options.save_images = False
        self.converter = DocumentConverter(
            format_options={
                InputFormat.XML_HWPX: HwpxFormatOption(
                    pipeline_options=self.pipeline_options
                )
            }
        )

    def get_paths(self, file_path: str):
        output_path, output_file = os.path.split(file_path)
        filename, _ = os.path.splitext(output_file)
        artifacts_dir = Path(f"{output_path}/{filename}")
        if artifacts_dir.is_absolute():
            reference_path = None
        else:
            reference_path = artifacts_dir.parent
        return artifacts_dir, reference_path

    def get_media_files(self, doc_items: list):
        temp_list = []
        for item in doc_items:
            if isinstance(item, PictureItem):
                path = str(item.image.uri)
                name = path.rsplit("/", 1)[-1]
                temp_list.append({'path': path, 'name': name})
        return temp_list

    def safe_join(self, iterable):
        if not isinstance(iterable, (list, tuple, set)):
            return ''
        return ''.join(map(str, iterable)) + '\n'

    def load_documents(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        save_images = kwargs.get('save_images', False)

        if self.pipeline_options.save_images != save_images:
            self.pipeline_options.save_images = save_images
            # self._create_converters()

        conv_result: ConversionResult = self.converter.convert(file_path, raises_on_error=True)
        return conv_result.document

    def split_documents(self, documents: DoclingDocument, **kwargs: dict) -> List[DocChunk]:
        chunker = HybridChunker(max_tokens=int(1e30), merge_peers=True)
        chunks: List[DocChunk] = list(chunker.chunk(dl_doc=documents, **kwargs))
        for chunk in chunks:
            self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
        return chunks

    async def compose_vectors(self, document: DoclingDocument, chunks: List[DocChunk], file_path: str, request: Request,
                              **kwargs: dict) -> list[dict]:
        global_metadata = dict(
            n_chunk_of_doc=len(chunks),
            n_page=document.num_pages(),
            reg_date=datetime.now().isoformat(timespec='seconds') + 'Z',
        )

        current_page = None
        chunk_index_on_page = 0
        vectors = []
        upload_tasks = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk_page = chunk.meta.doc_items[0].prov[0].page_no
            content = self.safe_join(chunk.meta.headings) + chunk.text

            if chunk_page != current_page:
                current_page = chunk_page
                chunk_index_on_page = 0

            vector = (GenOSVectorMetaBuilder()
                      .set_text(content)
                      .set_page_info(chunk_page, chunk_index_on_page, self.page_chunk_counts[chunk_page])
                      .set_chunk_index(chunk_idx)
                      .set_global_metadata(**global_metadata)
                      .set_chunk_bboxes(chunk.meta.doc_items, document)
                      .set_media_files(chunk.meta.doc_items)
                      ).build()
            vectors.append(vector)

            chunk_index_on_page += 1
            file_list = self.get_media_files(chunk.meta.doc_items)
            upload_tasks.append(asyncio.create_task(
                upload_files(file_list, request=request)
            ))

        if upload_tasks:
            await asyncio.gather(*upload_tasks)
        return vectors

    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        document: DoclingDocument = self.load_documents(file_path, **kwargs)
        artifacts_dir, reference_path = self.get_paths(file_path)
        document = document._with_pictures_refs(image_dir=artifacts_dir, reference_path=reference_path)

        chunks: list[DocChunk] = self.split_documents(document, **kwargs)

        vectors = []
        if len(chunks) >= 1:
            vectors: list[dict] = await self.compose_vectors(document, chunks, file_path, request, **kwargs)
        else:
            raise GenosServiceException(1, f"chunk length is 0")
        return vectors
```

**설명:**

* `get_paths`: 주어진 파일 경로에 대한 아티팩트 디렉토리와 참조 경로를 반환합니다.
* `get_media_files`: 주어진 문서에서 미디어 파일의 경로를 추출합니다.
* `safe_join`: 주어진 iterable의 요소를 안전하게 연결하여 문자열로 반환합니다.
* `load_documents`: 주어진 파일 경로에서 문서를 로드합니다.
* `split_documents`: 로드된 문서를 의미 있는 작은 단위(청크)로 분할합니다.
  * `chunker: HybridChunker = HybridChunker()`: 문서를 청크로 나누기 위해 `HybridChunker` 인스턴스를 생성합니다. `HybridChunker`는 내부적으로 계층적 구조(Hierarchical)와 의미론적 분할(Semantic, 주석 처리된 `semchunk` 의존성 부분에서 유추)을 결합하여 문서를 분할합니다.&#x20;
    * `max_tokens`는 각 청크의 최대 토큰 수를 제한합니다.
    * `merge_peers`는 인접한 청크들을 병합할지 여부를 결정합니다.
  * `chunks: List[DocChunk] = list(chunker.chunk(dl_doc=documents, **kwargs))`: `chunker`의 `chunk` 메서드를 호출하여 `DoclingDocument`를 `DocChunk` 객체들의 리스트로 변환합니다. `**kwargs`는 청킹 과정에 필요한 추가 옵션을 전달하는 데 사용될 수 있습니다 .
  * `for chunk in chunks: self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1`: 각 청크가 어떤 페이지에서 왔는지 파악하여 `self.page_chunk_counts` 딕셔너리에 페이지별 청크 수를 기록합니다. 이는 추후 메타데이터 생성 시 활용됩니다. (`chunk.meta.doc_items[0].prov[0].page_no`는 청크를 구성하는 첫번째 문서 아이템의 첫번째 출처 정보에서 페이지 번호를 가져옵니다.)
* `compose_vectors`: 분할된 청크들에 대해 메타데이터를 생성하고 최종적인 벡터(딕셔너리 형태) 리스트를 구성합니다. 이 부분이 고객사별 요구사항을 반영하는 부분이며 매우 중요합니다.
* `__call__`: 문서 처리를 수행하는 메서드입니다.
  * `document: DoclingDocument = self.load_documents(file_path, **kwargs)`: `load_documents` 메서드를 호출하여 입력된 `file_path`의 문서를 로드합니다. `**kwargs`는 사용자가 입력 UI 를 통해서 지정하거나, 혹은 수집기가 수집단에서 지정한 정보를 전달합니다.
  * `artifacts_dir, reference_path = self.get_paths(file_path)`: 문서의 아티팩트 디렉토리와 참조 경로를 가져옵니다.
  * `document = document._with_pictures_refs(image_dir=artifacts_dir, reference_path=reference_path)`: `DoclingDocument` 객체 내의 그림(PictureItem)들이 실제 파일로 저장될 위치(`image_dir`)와 참조 경로(`reference_path`)를 설정하여, 그림 객체가 실제 파일 경로를 참조하도록 업데이트합니다. `PdfPipelineOptions`에서 `generate_picture_images = True`로 설정된 경우, `docling` 라이브러리가 내부적으로 이 경로에 이미지들을 저장하고, 이 메서드를 통해 문서 객체 내의 참조를 업데이트합니다.
  * `chunks: List[DocChunk] = self.split_documents(document, **kwargs)`: 업데이트된 `document` 객체를 `split_documents` 메서드에 전달하여 청크 리스트를 얻습니다.
    * text가 있는 item이 없을 때 document에 임의의 text item 추가합니다.
  * `vectors = [] ...`:
   * 만약 생성된 청크가 1개 이상이면 (`len(chunks) >= 1`), `compose_vectors` 메서드를 호출하여 최종 메타데이터 벡터 리스트를 생성합니다.
   * 청크가 하나도 없으면 `GenosServiceException`을 발생시켜 오류 상황임을 알립니다.
  * `return vectors`: 생성된 벡터 리스트를 반환합니다.

***

### 📂 공통 전처리 흐름

#### `DocumentProcessor`

`DocumentProcessor` 클래스는 Genos 의 전처리기가 호출되는 관문입니다. 내부 구성을 보면, 문서를 로드, 변환, 분할하고 각 부분에 대한 메타데이터를 구성하는 핵심 요소입니다.

**`__init__` (초기화)**

`DocumentProcessor` 인스턴스가 생성될 때 호출되는 초기화 메서드입니다. 여기서 문서 처리 파이프라인의 주요 설정들이 정의됩니다.

Python

```python
class DocumentProcessor:

    def __init__(self):
        self.page_chunk_counts = defaultdict(int)
        self.hwpx_processor = HwpxProcessor()
```

**설명:**

* `self.page_chunk_counts = defaultdict(int)`: 페이지별로 생성된 청크의 수를 저장하기 위한 딕셔너리입니다.
* `self.hwpx_processor = HwpxProcessor()`: HWPX 문서 처리를 위한 프로세서입니다.

***

**`get_loader`**

문서 변환에 사용할 변환기를 설정하는 함수입니다.

Python

```python
    def get_loader(self, file_path: str):
        ext = os.path.splitext(file_path)[-1].lower()
        real_type = self.get_real_file_type(file_path)

        # 확장자와 실제 파일 타입이 다를 때만 real_type 사용
        if ext != real_type and real_type == 'pdf':
            return PyMuPDFLoader(file_path)
        elif ext != real_type and real_type in ['txt', 'json', 'md']:
            return TextLoader(file_path)
        # 원래 확장자 기반 로직
        elif ext == '.pdf':
            return PyMuPDFLoader(file_path)
        elif ext in ['.doc', '.docx']:
            return UnstructuredWordDocumentLoader(file_path)
        elif ext in ['.ppt', '.pptx']:
            return UnstructuredPowerPointLoader(file_path)
        elif ext in ['.jpg', '.jpeg', '.png']:
            return UnstructuredImageLoader(file_path)
        elif ext in ['.txt', '.json', '.md']:
            return TextLoader(file_path)
        elif ext == '.hwp':
            return HwpLoader(file_path)
        elif ext == '.md':
            return UnstructuredMarkdownLoader(file_path)
        else:
            return UnstructuredFileLoader(file_path)
```

**설명:**

* 각 확장자에 맞는 로더를 반환합니다.

***

**`convert_to_pdf`, `convert_md_to_pdf`**

문서를 PDF로 변환하는 메서드입니다.

Python

```python
    def convert_to_pdf(self, file_path: str):
    out_path = ""
    try:
        subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', out_path, file_path],
                       check=True)
        pdf_path = os.path.basename(file_path).replace(file_path.split('.')[-1], 'pdf')
        return pdf_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting PPT to PDF: {e}")
        return False


def convert_md_to_pdf(self, md_path):
    """Markdown 파일을 PDF로 변환"""
    install_packages(['chardet'])
    import chardet

    pdf_path = md_path.replace('.md', '.pdf')
    with open(md_path, 'rb') as f:
        raw_file = f.read(100)
    enc_type = chardet.detect(raw_file)['encoding']
    with open(md_path, 'r', encoding=enc_type) as f:
        md_content = f.read()

    html_content = markdown(md_content)
    HTML(string=html_content).write_pdf(pdf_path)
    return pdf_path
```

**설명:**

* `convert_to_pdf`:
  * 주어진 파일 경로에 있는 문서를 PDF로 변환합니다.
  * LibreOffice의 명령줄 도구를 사용하여 변환을 수행합니다.
* `convert_md_to_pdf`:
  * Markdown 파일을 PDF로 변환하는 메서드입니다.

**`load_documents`**

문서 파일을 로드하여 문서 객체 리스트를 생성하는 메서드입니다.

Python

```python
    def load_documents(self, file_path: str, **kwargs: dict) -> list[Document]:
        loader = self.get_loader(file_path)
        documents = loader.load()
        return documents
```

**설명:**

* `loader = self.get_loader(file_path)`: 파일 경로에 맞는 로더를 가져옵니다.
* `documents = loader.load()`: 로더를 사용하여 문서 파일을 로드합니다.
* `return documents`: 로드된 문서 객체 리스트를 반환합니다.

***

**`split_documents`**

로드된 문서를 의미 있는 작은 단위(청크)로 분할합니다.

Python

```python
    def split_documents(self, documents, **kwargs: dict) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(**kwargs)
        chunks = text_splitter.split_documents(documents)
        chunks = [chunk for chunk in chunks if chunk.page_content]
        if not chunks:
            raise Exception('Empty document')

        for chunk in chunks:
            page = chunk.metadata.get('page', 0)
            self.page_chunk_counts[page] += 1
        return chunks
```

**설명:**

* `text_splitter = RecursiveCharacterTextSplitter(**kwargs)`: 문서를 의미 있는 작은 단위로 분할하기 위해 `RecursiveCharacterTextSplitter` 인스턴스를 생성합니다. 이 클래스는 재귀적으로 문서를 분할하여 각 청크의 내용을 최대한 유지합니다.
* `chunks = text_splitter.split_documents(documents)`: `text_splitter`의 `split_documents` 메서드를 호출하여 문서를 청크로 분할합니다.
* `chunks = [chunk for chunk in chunks if chunk.page_content]`: 페이지 콘텐츠가 있는 청크만 필터링합니다.
* `if not chunks: raise Exception('Empty document')`: 청크가 비어있으면 예외를 발생시킵니다.
* `for chunk in chunks: page = chunk.metadata.get('page', 0); self.page_chunk_counts[page] += 1`: 각 청크의 페이지 정보를 기반으로 페이지별 청크 수를 업데이트합니다.
* `return chunks`: 최종 청크 리스트를 반환합니다.

***

**`compose_vectors`**

분할된 청크들에 대해 메타데이터를 생성하고 최종적인 벡터(딕셔너리 형태) 리스트를 구성합니다. 이 부분이 고객사별 요구사항을 반영하는 부분이며 매우 중요합니다.

Python

```python
    def compose_vectors(self, file_path: str, chunks: list[Document], **kwargs: dict) -> list[dict]:
        ext = os.path.splitext(file_path)[-1].lower()
        real_type = self.get_real_file_type(file_path)

        # 확장자와 실제 파일 타입이 다를 때만 real_type 사용
        if ext != real_type and real_type == 'pdf':
            pdf_path = file_path
        elif ext != real_type and real_type in ['txt', 'json', 'md']:
            # pdf_path = None  # PDF 변환 없이 직접 처리
            pdf_path = file_path.replace('.hwp', '.pdf').replace('.txt', '.pdf').replace('.json', '.pdf')
        # 원래 확장자 기반 로직
        elif file_path.endswith('.md'):
            pdf_path = self.convert_md_to_pdf(file_path)
        elif file_path.endswith('.ppt'):
            pdf_path = self.convert_to_pdf(file_path)
            if not pdf_path:
                return False
        else:
            pdf_path = file_path.replace('.hwp', '.pdf').replace('.txt', '.pdf').replace('.json', '.pdf')

        doc = fitz.open(pdf_path) if (pdf_path and os.path.exists(pdf_path)) else None

        if file_path.endswith('.ppt'):
            if os.path.exists(pdf_path):
                subprocess.run(["rm", pdf_path], check=True)

        global_metadata = dict(
            n_chunk_of_doc=len(chunks),
            n_page=max([chunk.metadata.get('page', 0) for chunk in chunks]),
            reg_date=datetime.now().isoformat(timespec='seconds') + 'Z'
        )
        current_page = None
        chunk_index_on_page = 0

        vectors = []
        for chunk_idx, chunk in enumerate(chunks):
            page = chunk.metadata.get('page', 0)
            text = chunk.page_content

            if page != current_page:
                current_page = page
                chunk_index_on_page = 0

            if doc:
                fitz_page = doc.load_page(page)
                global_metadata['chunk_bboxes'] = json.dumps(merge_overlapping_bboxes([{
                    'page': page + 1,
                    'type': 'text',
                    'bbox': {
                        'l': rect[0] / fitz_page.rect.width,
                        't': rect[1] / fitz_page.rect.height,
                        'r': rect[2] / fitz_page.rect.width,
                        'b': rect[3] / fitz_page.rect.height,
                    }
                } for rect in fitz_page.search_for(text)], x_tolerance=1 / fitz_page.rect.width,
                    y_tolerance=1 / fitz_page.rect.height))

            vectors.append(GenOSVectorMeta.model_validate({
                'text': text,
                'n_char': len(text),
                'n_word': len(text.split()),
                'n_line': len(text.splitlines()),
                'i_page': page,
                'e_page': page,
                'i_chunk_on_page': chunk_index_on_page,
                'n_chunk_of_page': self.page_chunk_counts[page],
                'i_chunk_on_doc': chunk_idx,
                **global_metadata
            }))
            chunk_index_on_page += 1

        return vectors
```

**설명:**

**한국은행 기록물의 메타데이타 Mapping 예**

* **`global_metadata`**: 문서 전체에 공통적으로 적용될 메타데이터를 딕셔너리로 구성합니다.
  * `n_chunk_of_doc=len(chunks)`: 문서 내 총 청크 수.
  * `n_page=max([chunk.metadata.get('page', 0) for chunk in chunks])`: 문서의 총 페이지 수.
  * `reg_date`: 현재 시간을 ISO 형식의 문자열로 등록일로 사용합니다.
* 루프 (`for chunk_idx, chunk in enumerate(chunks):`): 각 청크를 순회하며 메타데이터를 생성합니다.
  * `page = chunk.metadata.get('page', 0)`: 현재 청크의 시작 페이지 번호를 가져옵니다.
  * `text = chunk.page_content`: 청크의 실제 텍스트 내용을 가져옵니다.
  * `global_metadata['chunk_bboxes'] = json.dumps(merge_overlapping_bboxes(...)`: 청크의 경계 상자 정보를 병합하여 JSON 문자열로 저장합니다.
  * `vectors.append(...)`: 생성된 `GenOSVectorMeta` Pydantic 모델 객체를 `vectors` 리스트에 추가합니다.
* 페이지 변경 감지 로직: `current_page`와 `chunk_index_on_page`를 사용하여 페이지가 바뀔 때마다 페이지 내 청크 인덱스를 0으로 초기화합니다.

***

**`__call__`**

`DocumentProcessor` 인스턴스를 GenOS 에서 호출할때의 진입점으로, 함수처럼 호출했을 때 실행되는 메인 로직입니다. 문서 처리의 전체 흐름을 담당합니다.

Python

```python
    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        ext = os.path.splitext(file_path)[-1].lower()
        if ext in ('.wav', '.mp3', '.m4a'):
            # Generate a temporal path saving audio chunks: the audio file is supposed to be splited to several chunks due to limitted length by the model
            tmp_path = "./tmp_audios_{}".format(os.path.basename(file_path).split('.')[0])
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)

            # Use 'Whisper' model served in-house
            # [!] Modify the request parameters to change a STT model to be used
            loader = AudioLoader(
                file_path=file_path,
                req_url="http://192.168.74.164:30100/v1/audio/transcriptions",
                req_data={
                    'model': 'model',
                    'language': 'ko',
                    'response_format': 'json',
                    'temperature': '0',
                    'stream': 'false',
                    'timestamp_granularities[]': 'word'
                },
                chunk_sec=29,  # length(sec) of a chunk from the uploaded audio
                tmp_path=tmp_path
            )
            vectors = loader.return_vectormeta_format()
            await assert_cancelled(request)

            # Remove the temporal chunks
            try:
                subprocess.run(['rm', '-r', tmp_path], check=True)
            except:
                pass
            await assert_cancelled(request)
            return vectors

        elif ext in ('.csv', '.xlsx'):
            loader = TabularLoader(file_path, ext)
            vectors = loader.return_vectormeta_format()
            await assert_cancelled(request)
            return vectors

        elif ext == '.hwp':
            documents: list[Document] = self.load_documents(file_path, **kwargs)
            await assert_cancelled(request)
            chunks: list[Document] = self.split_documents(documents, **kwargs)
            await assert_cancelled(request)
            vectors: list[dict] = self.compose_vectors(file_path, chunks, **kwargs)
            return vectors

        elif ext in ('.hwpx'):
            return await self.hwpx_processor(request, file_path, **kwargs)

        else:
            documents: list[Document] = self.load_documents(file_path, **kwargs)
            await assert_cancelled(request)

            chunks: list[Document] = self.split_documents(documents, **kwargs)
            await assert_cancelled(request)

            vectors: list[dict] = self.compose_vectors(file_path, chunks, **kwargs)
            return vectors
```

**설명:**

* `ext = os.path.splitext(file_path)[-1].lower()`: 파일 경로에서 확장자를 추출하고 소문자로 변환합니다. 확장자에 따라서 적합한 처리 로직을 선택하기 위함입니다.
* `if ext in ('.wav', '.mp3', '.m4a'):`: 오디오 파일인 경우, 오디오 전용 처리 로직을 수행합니다.
  * `tmp_path = "./tmp_audios_{}".format(os.path.basename(file_path).split('.')[0])`: 오디오 파일을 임시로 저장할 디렉토리 경로를 생성합니다. 이 디렉토리는 오디오 청크를 저장하는 데 사용됩니다.
  * `if not os.path.exists(tmp_path): os.makedirs(tmp_path)`: 임시 디렉토리가 존재하지 않으면 생성합니다.
  * `loader = AudioLoader(...)`: `AudioLoader` 인스턴스를 생성하여 오디오 파일을 처리합니다. 여기서 `req_url`과 `req_data`는 음성 인식 모델에 대한 요청 정보를 포함합니다.
  * `vectors = loader.return_vectormeta_format()`: 오디오 파일을 처리하고 벡터 메타데이터 형식으로 변환된 결과를 얻습니다.
  * `await assert_cancelled(request)`: 요청이 취소되었는지 확인합니다. 취소된 경우 예외를 발생시킵니다.
  * `subprocess.run(['rm', '-r', tmp_path], check=True)`: 임시 오디오 청크 디렉토리를 삭제하여 정리합니다.
  * `return vectors`: 생성된 벡터 메타데이터를 반환합니다.
* `elif ext in ('.csv', '.xlsx'):`: CSV 또는 XLSX 파일인 경우, 표 형식 전용 처리 로직을 수행합니다.
  * `loader = TabularLoader(file_path, ext)`: `TabularLoader` 인스턴스를 생성하여 표 형식 파일을 처리합니다.
  * `vectors = loader.return_vectormeta_format()`: 표 형식 파일을 처리하고 벡터 메타데이터 형식으로 변환된 결과를 얻습니다.
  * `await assert_cancelled(request)`: 요청이 취소되었는지 확인합니다. 취소된 경우 예외를 발생시킵니다.
  * `return vectors`: 생성된 벡터 메타데이터를 반환합니다.
* `elif ext == '.hwp': ...`: HWP 파일인 경우, HWP 전용 처리 로직을 수행합니다.
  * `documents: list[Document] = self.load_documents(file_path, **kwargs)`: HWP 파일을 로드하여 문서 객체 리스트를 생성합니다.
  * `await assert_cancelled(request)`: 요청이 취소되었는지 확인합니다. 취소된 경우 예외를 발생시킵니다.
  * `chunks: list[Document] = self.split_documents(documents, **kwargs)`: 문서 객체를 청크 단위로 분할합니다.
  * `await assert_cancelled(request)`: 요청이 취소되었는지 확인합니다. 취소된 경우 예외를 발생시킵니다.
  * `vectors: list[dict] = self.compose_vectors(file_path, chunks, **kwargs)`: 청크를 기반으로 벡터 메타데이터를 생성합니다.
  * `return vectors`: 생성된 벡터 메타데이터를 반환합니다.
* `elif ext == '.hwpx': ...`: HWPX 파일인 경우, HWPX 전용 처리 로직을 수행합니다.
  * `return await self.hwpx_processor(request, file_path, **kwargs)`: `HwpxProcessor` 인스턴스를 호출하여 HWPX 파일을 처리하고 벡터 메타데이터를 반환합니다.
* `else: ...`: 그 외의 파일 형식인 경우, 일반적인 문서 처리 로직을 수행합니다.
  * `documents: list[Document] = self.load_documents(file_path, **kwargs)`: 문서 파일을 로드하여 문서 객체 리스트를 생성합니다.
  * `await assert_cancelled(request)`: 요청이 취소되었는지 확인합니다. 취소된 경우 예외를 발생시킵니다.
  * `chunks: list[Document] = self.split_documents(documents, **kwargs)`: 문서 객체를 청크 단위로 분할합니다.
  * `await assert_cancelled(request)`: 요청이 취소되었는지 확인합니다. 취소된 경우 예외를 발생시킵니다.
  * `vectors: list[dict] = self.compose_vectors(file_path, chunks, **kwargs)`: 청크를 기반으로 벡터 메타데이터를 생성합니다.
  * `return vectors`: 생성된 벡터 메타데이터를 반환합니다.

**사용자 정의 포인트:**

* `**kwargs` 활용: `__call__` 메서드에 전달되는 `**kwargs`는 내부적으로 `load_documents`, `split_documents`, `compose_vectors`로 전파될 수 있으므로, 문서 처리 전 과정에 걸쳐 동적인 설정을 주입하는 통로로 활용될 수 있습니다. 예를 들어, API 요청으로부터 특정 파라미터를 받아 `kwargs`로 전달하고, 이 값을 기반으로 `PdfPipelineOptions`의 일부를 변경하거나 `compose_vectors`에서 특정 메타데이터를 추가/제외하는 등의 로직을 구현할 수 있습니다.

### ✨ 사용자 정의 포인트

* **메타 필드 확장**: `GenOSVectorMeta`와 `Builder` 클래스에 필드 추가
* **페이지 처리 로직 수정**: `set_page_info` 파라미터 조정
* **청크 분할 커스터마이징**:  `HybridChunker` 파라미터 기준값 수정
* **문자열 변환**: `NaN`, `None` 등 값은 전처리 단계에서 빈 문자열 처리

### ✅ 유지보수 팁

* Pydantic `extra='allow'` 설정으로 필드 변경이 유연하게 허용됨
* Builder 패턴을 사용하여 필드 설정 오류를 방지하고 유지보수를 단순화

***

이와 같이 코드 조각과 함께 설명을 보면서 `DocumentProcessor`와 `GenOSVectorMetaBuilder`의 작동 방식과 사용자 정의 지점을 파악하시면, 요구사항에 맞게 전처리 파이프라인을 효과적으로 수정하고 확장하실 수 있을 것입니다.
