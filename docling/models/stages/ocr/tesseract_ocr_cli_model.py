import csv
import io
import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from subprocess import DEVNULL, PIPE, Popen
from typing import List, Optional, Tuple, Type

import pandas as pd
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import TextCell

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    OcrOptions,
    TesseractCliOcrOptions,
)
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.ocr_utils import (
    map_tesseract_script,
    parse_tesseract_orientation,
    tesseract_box_to_bounding_rectangle,
)
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)

# Regex for valid Tesseract language identifiers (e.g. "eng", "script/Latin", "eng+deu")
_VALID_LANG_RE = re.compile(r"^[a-zA-Z0-9_/+-]+$")


class TesseractOcrCliModel(BaseOcrModel):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: TesseractCliOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: TesseractCliOcrOptions

        self.scale = 3  # multiplier for 72 dpi == 216 dpi.

        self._name: Optional[str] = None
        self._version: Optional[str] = None
        self._tesseract_languages: Optional[List[str]] = None
        self._script_prefix: Optional[str] = None
        self._is_auto: bool = "auto" in self.options.lang

        # Pre-validate and store sanitized subprocess arguments at construction time
        # so that all subsequent subprocess calls use only these already-validated values.
        self._safe_tesseract_cmd: str = self._sanitize_cmd(self.options.tesseract_cmd)
        self._safe_tessdata_path: Optional[str] = (
            self._sanitize_path(self.options.path)
            if self.options.path is not None
            else None
        )
        if self.options.lang:
            for _lang_token in self.options.lang:
                if _lang_token != "auto":
                    self._sanitize_lang(_lang_token)

        if self.enabled:
            try:
                self._get_name_and_version()
                self._set_languages_and_prefix()

            except Exception as exc:
                raise RuntimeError(
                    f"Tesseract is not available, aborting: {exc} "
                    "Install tesseract on your system and the tesseract binary is discoverable. "
                    "The actual command for Tesseract can be specified in `pipeline_options.ocr_options.tesseract_cmd='tesseract'`. "
                    "Alternatively, Docling has support for other OCR engines. See the documentation."
                )

    @staticmethod
    def _sanitize_lang(lang: str) -> str:
        """Validate and sanitize a Tesseract language identifier to prevent argument injection.

        Valid identifiers (e.g. ``eng``, ``script/Latin``, ``eng+deu``) contain only
        alphanumeric characters, underscores, hyphens, forward slashes, and plus signs.
        """
        if not _VALID_LANG_RE.match(lang):
            raise ValueError(
                f"Invalid Tesseract language identifier: {lang!r}. "
                "Language identifiers must only contain alphanumeric characters, "
                "underscores, hyphens, forward slashes, and plus signs."
            )
        return lang

    @staticmethod
    def _sanitize_path(path: str) -> str:
        """Validate and sanitize a Tesseract data directory path to prevent argument injection.

        Rejects paths containing null bytes and resolves the path to an absolute form.
        """
        if "\x00" in path:
            raise ValueError("Invalid Tesseract data path: contains null byte.")
        return str(Path(path).resolve())

    @staticmethod
    def _sanitize_cmd(cmd: str) -> str:
        """Validate and sanitize the Tesseract executable name/path to prevent injection.

        Rejects values containing null bytes or shell metacharacters.
        """
        if "\x00" in cmd:
            raise ValueError("Invalid Tesseract command: contains null byte.")
        _UNSAFE_CMD_CHARS = set("|;&`$(){}[]<>!\\\"'")
        if any(c in _UNSAFE_CMD_CHARS for c in cmd):
            raise ValueError(
                f"Invalid Tesseract command: {cmd!r} contains unsafe characters."
            )
        return cmd

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Validate and sanitize a filename passed to the Tesseract CLI.

        Rejects paths containing null bytes and resolves to an absolute path.
        """
        if "\x00" in filename:
            raise ValueError("Invalid filename: contains null byte.")
        return str(Path(filename).resolve())

    def _get_name_and_version(self) -> Tuple[str, str]:
        if self._name is not None and self._version is not None:
            return self._name, self._version  # type: ignore

        cmd = [self._safe_tesseract_cmd, "--version"]

        proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=False)
        stdout, stderr = proc.communicate()

        proc.wait()

        # HACK: Windows versions of Tesseract output the version to stdout, Linux versions
        # to stderr, so check both.
        version_line = (
            (stdout.decode("utf8").strip() or stderr.decode("utf8").strip())
            .split("\n")[0]
            .strip()
        )

        # If everything else fails...
        if not version_line:
            version_line = "tesseract XXX"

        name, version = version_line.split(" ")

        self._name = name
        self._version = version

        return name, version

    def _run_tesseract(self, ifilename: str, osd: Optional[pd.DataFrame]):
        r"""
        Run tesseract CLI
        """
        cmd = [self._safe_tesseract_cmd]
        if self._is_auto and osd is not None:
            lang = self._parse_language(osd)
            if lang is not None:
                cmd.append("-l")
                cmd.append(self._sanitize_lang(lang))
        elif self.options.lang is not None and len(self.options.lang) > 0:
            cmd.append("-l")
            cmd.append("+".join(self._sanitize_lang(l) for l in self.options.lang))

        if self._safe_tessdata_path is not None:
            cmd.append("--tessdata-dir")
            cmd.append(self._safe_tessdata_path)

        # Add PSM option if specified in the configuration; cast to int to
        # reject any non-numeric value and prevent argument injection.
        if self.options.psm is not None:
            cmd.extend(["--psm", str(int(self.options.psm))])

        cmd.append(self._sanitize_filename(ifilename))
        cmd.extend(["stdout", "tsv"])
        _log.info("command: {}".format(" ".join(cmd)))

        output = subprocess.run(
            cmd, stdout=PIPE, stderr=DEVNULL, stdin=DEVNULL, check=True, shell=False
        )

        # _log.info(output)

        # Decode the byte string to a regular string
        decoded_data = output.stdout.decode("utf-8")
        # _log.info(decoded_data)

        # Read the TSV file generated by Tesseract
        df_result = pd.read_csv(
            io.StringIO(decoded_data), quoting=csv.QUOTE_NONE, sep="\t"
        )

        # Display the dataframe (optional)
        # _log.info("df: ", df.head())

        # Filter rows that contain actual text (ignore header or empty rows)
        df_filtered = df_result[
            df_result["text"].notna() & (df_result["text"].apply(str).str.strip() != "")
        ]

        return df_filtered

    def _perform_osd(self, ifilename: str) -> pd.DataFrame:
        r"""
        Run tesseract in PSM 0 mode to detect the language
        """

        cmd = [self._safe_tesseract_cmd, "--psm", "0", "-l", "osd",
               self._sanitize_filename(ifilename), "stdout"]
        _log.info("command: {}".format(" ".join(cmd)))
        output = subprocess.run(
            cmd, capture_output=True, stdin=DEVNULL, check=True, shell=False
        )
        decoded_data = output.stdout.decode("utf-8")
        df_detected = pd.read_csv(
            io.StringIO(decoded_data), sep=":", header=None, names=["key", "value"]
        )
        return df_detected

    def _parse_language(self, df_osd: pd.DataFrame) -> Optional[str]:
        assert self._tesseract_languages is not None
        scripts = df_osd.loc[df_osd["key"] == "Script"].value.tolist()
        if len(scripts) == 0:
            _log.warning("Tesseract cannot detect the script of the page")
            return None

        script = map_tesseract_script(scripts[0].strip())
        lang = f"{self._script_prefix}{script}"

        # Check if the detected language has been installed
        if lang not in self._tesseract_languages:
            msg = f"Tesseract detected the script '{script}' and language '{lang}'."
            msg += " However this language is not installed in your system and will be ignored."
            _log.warning(msg)
            return None

        _log.debug(
            f"Using tesseract model for the detected script '{script}' and language '{lang}'"
        )
        return lang

    def _set_languages_and_prefix(self):
        r"""
        Read and set the languages installed in tesseract and decide the script prefix
        """
        # Get all languages
        cmd = [self._safe_tesseract_cmd, "--list-langs"]
        _log.info("command: {}".format(" ".join(cmd)))
        output = subprocess.run(
            cmd, stdout=PIPE, stderr=DEVNULL, stdin=DEVNULL, check=True, shell=False
        )
        decoded_data = output.stdout.decode("utf-8")
        df_list = pd.read_csv(io.StringIO(decoded_data), header=None)
        self._tesseract_languages = df_list[0].tolist()[1:]

        # Decide the script prefix
        if any(lang.startswith("script/") for lang in self._tesseract_languages):
            script_prefix = "script/"
        else:
            script_prefix = ""

        self._script_prefix = script_prefix

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page_i, page in enumerate(page_batch):
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "ocr"):
                    ocr_rects = self.get_ocr_rects(page)

                    all_ocr_cells = []
                    for ocr_rect_i, ocr_rect in enumerate(ocr_rects):
                        # Skip zero area boxes
                        if ocr_rect.area() == 0:
                            continue
                        high_res_image = page._backend.get_page_image(
                            scale=self.scale, cropbox=ocr_rect
                        )
                        try:
                            with tempfile.NamedTemporaryFile(
                                suffix=".png", mode="w+b", delete=False
                            ) as image_file:
                                fname = image_file.name
                                high_res_image.save(image_file)
                            doc_orientation = 0
                            df_osd: Optional[pd.DataFrame] = None
                            try:
                                df_osd = self._perform_osd(fname)
                                doc_orientation = _parse_orientation(df_osd)
                            except subprocess.CalledProcessError as exc:
                                _log.error(
                                    "OSD failed (doc %s, page: %s, "
                                    "OCR rectangle: %s, processed image file %s):\n %s",
                                    conv_res.input.file,
                                    page_i,
                                    ocr_rect_i,
                                    image_file,
                                    exc.stderr,
                                )
                                # Skipping if OSD fail when in auto mode, otherwise proceed
                                # to OCR in the hope OCR will succeed while OSD failed
                                if self._is_auto:
                                    continue
                            if doc_orientation != 0:
                                high_res_image = high_res_image.rotate(
                                    -doc_orientation, expand=True
                                )
                                high_res_image.save(fname)
                            try:
                                df_result = self._run_tesseract(fname, df_osd)
                            except subprocess.CalledProcessError as exc:
                                _log.error(
                                    "tesseract OCR failed (doc %s, page: %s, "
                                    "OCR rectangle: %s, processed image file %s):\n %s",
                                    conv_res.input.file,
                                    page_i,
                                    ocr_rect_i,
                                    image_file,
                                    exc.stderr,
                                )
                                continue
                        finally:
                            if os.path.exists(fname):
                                os.remove(fname)

                        # _log.info(df_result)

                        # Print relevant columns (bounding box and text)
                        for ix, row in df_result.iterrows():
                            text = row["text"]
                            conf = row["conf"]

                            left, top = float(row["left"]), float(row["top"])
                            right = left + float(row["width"])
                            bottom = top + row["height"]
                            bbox = BoundingBox(
                                l=left,
                                t=top,
                                r=right,
                                b=bottom,
                                coord_origin=CoordOrigin.TOPLEFT,
                            )
                            rect = tesseract_box_to_bounding_rectangle(
                                bbox,
                                original_offset=ocr_rect,
                                scale=self.scale,
                                orientation=doc_orientation,
                                im_size=high_res_image.size,
                            )
                            cell = TextCell(
                                index=ix,
                                text=str(text),
                                orig=str(text),
                                from_ocr=True,
                                confidence=conf / 100.0,
                                rect=rect,
                            )
                            all_ocr_cells.append(cell)

                    # Post-process the cells
                    self.post_process_cells(all_ocr_cells, page)

                # DEBUG code:
                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return TesseractCliOcrOptions


def _parse_orientation(df_osd: pd.DataFrame) -> int:
    # For strictly optimal performance with invariant dataframe format:
    mask = df_osd["key"].to_numpy() == "Orientation in degrees"
    orientation_val = df_osd["value"].to_numpy()[mask][0]
    orientation = parse_tesseract_orientation(orientation_val.strip())
    return orientation
