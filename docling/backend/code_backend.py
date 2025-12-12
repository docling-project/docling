from typing import Union, Set
from pathlib import Path  
from io import BytesIO  
  
from docling_core.types.doc import DoclingDocument, DocumentOrigin  
from docling.backend.abstract_backend import DeclarativeDocumentBackend  
from docling.datamodel.base_models import InputFormat  
  
class CodeFileBackend(DeclarativeDocumentBackend):  
      
    LANGUAGE_MAPPINGS = {  
        '.py': 'python',  
        '.js': 'javascript',   
        '.java': 'java'
    }  
      
    def __init__(self, in_doc, path_or_stream: Union[BytesIO, Path]):    
        super().__init__(in_doc, path_or_stream)    
        self.path_or_stream = path_or_stream
        self.valid = True    
        
        try:    
            if isinstance(self.path_or_stream, BytesIO):
                self.source_code = self.path_or_stream.getvalue().decode("utf-8")
            if isinstance(self.path_or_stream, Path):
                with open(self.path_or_stream, encoding="utf-8") as f:
                    self.source_code = f.read() 
                
            self.language = self._detect_language()  
          
        except Exception as e:
            raise RuntimeError(
                f"Could not initialize code backend for file with hash {self.document_hash}."
            ) from e

    def _detect_language(self) -> str:  
        """Detect programming language from file extension."""  
        file_ext = self.file.suffix.lower()  
        return self.LANGUAGE_MAPPINGS.get(file_ext, 'text')  
      
    @classmethod  
    def supported_formats(cls) -> Set[InputFormat]:  
        return {  
            InputFormat.CODE_PYTHON,  
            InputFormat.CODE_JAVASCRIPT,   
            InputFormat.CODE_JAVA
        }  
      
    @classmethod  
    def supports_pagination(cls) -> bool:  
        return False  
      
    def is_valid(self) -> bool:  
        return self.valid  
      
    def convert(self) -> DoclingDocument:  
        mime_type = f"text/x-{self.language}-source" if self.language in ['java'] else f"text/x-{self.language}"  

        origin = DocumentOrigin(  
            filename=self.file.name or f"file{self.file.suffix}",  
            mimetype=mime_type,  
            binary_hash=self.document_hash,  
        )  
          
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)  
        
        if self.is_valid():  
            doc.add_code(  
                text=self.source_code,  
                code_language=self.language  
            )  
          
        return doc