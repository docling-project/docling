## [v2.70.0](https://github.com/docling-project/docling/releases/tag/v2.70.0) - 2026-01-23

### Feature

* Drop support for Python 3.9 ([#2905](https://github.com/docling-project/docling/issues/2905)) ([`7f38658`](https://github.com/docling-project/docling/commit/7f386587ed9a28a839a928f3815d5ce1f3e05f8b))

### Fix

* **md:** Handle pipe symbols that are not table markers ([#2904](https://github.com/docling-project/docling/issues/2904)) ([`86eaef5`](https://github.com/docling-project/docling/commit/86eaef5b4544d638099657d38f18966ddd3e73f2))
* Remove direct vllm dependency ([#2910](https://github.com/docling-project/docling/issues/2910)) ([`7a1952a`](https://github.com/docling-project/docling/commit/7a1952ae3dbdaf781700b35d226b7ec9f73ea399))
* PPTX parsing: bullet points not grouped correctly under subheadings (#2663) ([#2855](https://github.com/docling-project/docling/issues/2855)) ([`999dbb2`](https://github.com/docling-project/docling/commit/999dbb2765ab47e00ee1d3d710c4e337f79452bc))

### Documentation

* Add comprehensive docstrings to PdfPipelineOptions ([#2827](https://github.com/docling-project/docling/issues/2827)) ([`ab91786`](https://github.com/docling-project/docling/commit/ab91786f3b5dc531c7ec8c3973fac58e8288b3f0))

## [v2.69.1](https://github.com/docling-project/docling/releases/tag/v2.69.1) - 2026-01-21

### Fix

* Off-by-one error for page indexing in vlm_pipeline ([#2902](https://github.com/docling-project/docling/issues/2902)) ([`08f49e2`](https://github.com/docling-project/docling/commit/08f49e2abc74bfbc6be3433f64698c2b4ac7ddce))

## [v2.69.0](https://github.com/docling-project/docling/releases/tag/v2.69.0) - 2026-01-20

### Feature

* New picture classifier v2.0 ([#2889](https://github.com/docling-project/docling/issues/2889)) ([`43badc3`](https://github.com/docling-project/docling/commit/43badc3838ccfc98fd28d9d66ffe0811585f90fd))
* Add classification filters for picture description ([#2836](https://github.com/docling-project/docling/issues/2836)) ([`ac16a26`](https://github.com/docling-project/docling/commit/ac16a26a047ccf5edd88775197ca43d146d00528))

### Fix

* Torch compatibility for xpu ([#2894](https://github.com/docling-project/docling/issues/2894)) ([`00273f6`](https://github.com/docling-project/docling/commit/00273f6574ad2f72f8e8b727833f1e742ab59265))
* Standardize page_no to 1-based indexing ([#2847](https://github.com/docling-project/docling/issues/2847)) ([`1b4d82d`](https://github.com/docling-project/docling/commit/1b4d82de2d6a7998656ff0a4224a55ad4ed24fbf))
* Usage of direct logging ([#2884](https://github.com/docling-project/docling/issues/2884)) ([`2fe9def`](https://github.com/docling-project/docling/commit/2fe9def2828735dbeecb606361302521bfa36e9c))
* Relax pypdfium2 version constraint, support 5.x ([#2873](https://github.com/docling-project/docling/issues/2873)) ([`daf2bc6`](https://github.com/docling-project/docling/commit/daf2bc6d55ab9fd3c818e5319d4820361cc28e47))

### Documentation

* Correct broken link to supported formats ([#2878](https://github.com/docling-project/docling/issues/2878)) ([`16e88d5`](https://github.com/docling-project/docling/commit/16e88d50fa9e50351b1847897ff10aa4a3324872))

## [v2.68.0](https://github.com/docling-project/docling/releases/tag/v2.68.0) - 2026-01-13

### Feature

* Support for DeepSeek-OCR in VLM pipeline ([#2798](https://github.com/docling-project/docling/issues/2798)) ([`19af03f`](https://github.com/docling-project/docling/commit/19af03f539b40d88eedd132644ed085b572664d7))

### Fix

* **logging:** Include page numbers in preprocess error messages ([#2858](https://github.com/docling-project/docling/issues/2858)) ([`89bea24`](https://github.com/docling-project/docling/commit/89bea245392b840a0c25c5fc35c931477a34d881))
* **docx:** Handle grouped pictures ([#2861](https://github.com/docling-project/docling/issues/2861)) ([`5c1f8f0`](https://github.com/docling-project/docling/commit/5c1f8f0171c92d33e65f6407f08d07353fe06ebf))

### Documentation

* Fix Colab badge links and Weaviate typo in docs examples ([#2871](https://github.com/docling-project/docling/issues/2871)) ([`72851cc`](https://github.com/docling-project/docling/commit/72851cce66c95ab66278d8e366a80abe331d1b7c))
* **example:** Fix update sample image path to be relative ([#2864](https://github.com/docling-project/docling/issues/2864)) ([`211c759`](https://github.com/docling-project/docling/commit/211c759fd043e81c26d2dee4418e7bf99d128673))
* Add Semantica integration ([#2860](https://github.com/docling-project/docling/issues/2860)) ([`bf80e32`](https://github.com/docling-project/docling/commit/bf80e329d1adae99327dd0ef9ccade0d4690305a))

## [v2.67.0](https://github.com/docling-project/docling/releases/tag/v2.67.0) - 2026-01-09

### Feature

* Enrichment annotations in the new meta format ([#2859](https://github.com/docling-project/docling/issues/2859)) ([`aab3ff5`](https://github.com/docling-project/docling/commit/aab3ff5d82fc54864657c0c2ff8e0aa21461f23f))
* Add XPU device support for Intel GPUs ([#2809](https://github.com/docling-project/docling/issues/2809)) ([`2b83fdd`](https://github.com/docling-project/docling/commit/2b83fdd0deeec0f1ad016cc78ea42d3144a86cad))
* Add option to report timings details ([#2772](https://github.com/docling-project/docling/issues/2772)) ([`cbc6537`](https://github.com/docling-project/docling/commit/cbc6537ee856349cef9b1ed453310e269667534c))

### Fix

* Lock new deps and update python 3.14 warnings ([#2844](https://github.com/docling-project/docling/issues/2844)) ([`d9295df`](https://github.com/docling-project/docling/commit/d9295df30202e251c9e282a4d0ce61653c7268b6))
* Correct type hint for table_structure_options usage ([#2823](https://github.com/docling-project/docling/issues/2823)) ([`a0530a2`](https://github.com/docling-project/docling/commit/a0530a271e5411bae96cfcb8a6ae23b3431a7462))
* Transformers models lazy-loaded ([#2826](https://github.com/docling-project/docling/issues/2826)) ([`3ef4525`](https://github.com/docling-project/docling/commit/3ef45258b7de6f5efe13b1f5ba392d72e2854349))
* Font download by passing font_path to RapidOcr ([#2822](https://github.com/docling-project/docling/issues/2822)) ([`ffafe58`](https://github.com/docling-project/docling/commit/ffafe58ad25e0ebdca44a41f8df9f691c9a393bf))
* **cli:** Add Layout and Table models to --show-external-plugins ([#2832](https://github.com/docling-project/docling/issues/2832)) ([`ed57089`](https://github.com/docling-project/docling/commit/ed5708907fcd29e56b114ce7062c4c2693c67b6b))

## [v2.66.0](https://github.com/docling-project/docling/releases/tag/v2.66.0) - 2025-12-24

### Feature

* Add preset for using granite-docling via vllm and other apis ([#2792](https://github.com/docling-project/docling/issues/2792)) ([`241d19e`](https://github.com/docling-project/docling/commit/241d19ed6f1b6d4327df250497ff8d8dd2686b5d))

### Fix

* **docx:** Handle tables with merged cells causing IndexError ([#2813](https://github.com/docling-project/docling/issues/2813)) ([`faff935`](https://github.com/docling-project/docling/commit/faff935b0e9f7a6f450b3bbc0329a05ac1b00ff2))
* **markdown:** Allow text before headers also in mixed markdown and html ([#2801](https://github.com/docling-project/docling/issues/2801)) ([`595115d`](https://github.com/docling-project/docling/commit/595115d89259687548a96ea2f120bba572d69091))

### Documentation

* **RTX:** Guidelines for best performance on RTX GPUs ([#2765](https://github.com/docling-project/docling/issues/2765)) ([`be085c0`](https://github.com/docling-project/docling/commit/be085c0e39dd5c51572b883d0f795c5a7abefd5d))
* Add docstrings to DocumentConverter #2748 ([#2782](https://github.com/docling-project/docling/issues/2782)) ([`cc5e3ce`](https://github.com/docling-project/docling/commit/cc5e3cee740fd41e27c0cc6b4a3db71308539986))
* **style:** Fix link visibility in dark mode ([#2804](https://github.com/docling-project/docling/issues/2804)) ([`150fe90`](https://github.com/docling-project/docling/commit/150fe90728b1719b6a59e2ed058ec026d573d04b))

## [v2.65.0](https://github.com/docling-project/docling/releases/tag/v2.65.0) - 2025-12-15

### Feature

* Add YAML output format to CLI ([#2768](https://github.com/docling-project/docling/issues/2768)) ([`da7678a`](https://github.com/docling-project/docling/commit/da7678a754b62df5cf0a9a1efe98c288bda20bd7))

### Fix

* **rapidocr:** Use correct parameter name for rec_keys_path ([#2762](https://github.com/docling-project/docling/issues/2762)) ([`1d78418`](https://github.com/docling-project/docling/commit/1d78418cefb5b90691481fa92c35e8b4909b6de5))
* **docx:** Handle missing value in paragraph style name ([#2761](https://github.com/docling-project/docling/issues/2761)) ([`a97d950`](https://github.com/docling-project/docling/commit/a97d950d74335fbc14d124edda67fa0fa272f11d))

### Documentation

* Add Pydantic field documentation for PipelineOptions ([#2771](https://github.com/docling-project/docling/issues/2771)) ([`7c24b01`](https://github.com/docling-project/docling/commit/7c24b014f6cee6978f1bd1c98cc0478aed96b4a1))
* **gpu:** Add benchmarks of standard pipeline with OCR ([#2764](https://github.com/docling-project/docling/issues/2764)) ([`d03439c`](https://github.com/docling-project/docling/commit/d03439ccc54bab43bf4b5c79bb21a1491caeb728))

## [v2.64.1](https://github.com/docling-project/docling/releases/tag/v2.64.1) - 2025-12-09

### Fix

* Clear word/char cells when force_full_page_ocr is used ([#2738](https://github.com/docling-project/docling/issues/2738)) ([`1df0560`](https://github.com/docling-project/docling/commit/1df0560ec2cafcd95f2240e6188385e1ec117110))
* Add missing font download in the rapidocr artifacts ([#2735](https://github.com/docling-project/docling/issues/2735)) ([`edbabfc`](https://github.com/docling-project/docling/commit/edbabfcac2fd53345b1a0677e81f206285d58bae))
* Ensure proper image_scale for generated page images in VLM pipelines ([#2728](https://github.com/docling-project/docling/issues/2728)) ([`609069d`](https://github.com/docling-project/docling/commit/609069d12c128f30fbf5132f017c9b62aed6a762))
* **html:** Tackle paragraphs with block-level elements ([#2720](https://github.com/docling-project/docling/issues/2720)) ([`d007ba0`](https://github.com/docling-project/docling/commit/d007ba0e6f257fcc4e96aad388512455011b7952))
* **html:** Prevent hierarchy reset in rich table cells ([#2716](https://github.com/docling-project/docling/issues/2716)) ([`aebe25c`](https://github.com/docling-project/docling/commit/aebe25cf0027efc35c50de36751820c9a62e30ff))
* **docx:** Parse integrals as n-ary objects without chr element ([#2712](https://github.com/docling-project/docling/issues/2712)) ([`c97715f`](https://github.com/docling-project/docling/commit/c97715f5fd07556306026a1291f3bf1749dcd020))

## [v2.64.0](https://github.com/docling-project/docling/releases/tag/v2.64.0) - 2025-12-02

### Feature

* **experimental:** Add experimental TableCropsLayoutModel ([#2669](https://github.com/docling-project/docling/issues/2669)) ([`1344362`](https://github.com/docling-project/docling/commit/134436245a1ebdadbfd8ba3c870f0f3c866f39a7))
* Factory and plugin-capability for Layout and Table models ([#2637](https://github.com/docling-project/docling/issues/2637)) ([`ad97e52`](https://github.com/docling-project/docling/commit/ad97e5285126388847ba9a219ac73f006c759f09))

### Fix

* InputFormat.IMAGE must have correct pipeline ([#2707](https://github.com/docling-project/docling/issues/2707)) ([`6ef4ffd`](https://github.com/docling-project/docling/commit/6ef4ffd643285a27afccdbc6bda6843c7f8c8d79))
* Do not consider singleton cells in xlsx as TableItems but rather TextItems ([#2589](https://github.com/docling-project/docling/issues/2589)) ([`54cd6d7`](https://github.com/docling-project/docling/commit/54cd6d7406ee9431d45e8bcd1b0ae53389cfb56c))
* **docx:** Missing list items after numbered header ([#2665](https://github.com/docling-project/docling/issues/2665)) ([`e580554`](https://github.com/docling-project/docling/commit/e58055465cb3635ff17ee03e0a74126c03c55f49))

### Documentation

* Example on how to apply external OCR as post processing ([#2517](https://github.com/docling-project/docling/issues/2517)) ([`fa21128`](https://github.com/docling-project/docling/commit/fa211281385072b61576ec58dd48025f8e433dcd))
* More GPU results and improvements in the example docs ([#2674](https://github.com/docling-project/docling/issues/2674)) ([`b75c646`](https://github.com/docling-project/docling/commit/b75c6461f46d1a012d2b2b06b2b4da870786fda3))
* Fix typo on jobkit page ([#2671](https://github.com/docling-project/docling/issues/2671)) ([`146b4f0`](https://github.com/docling-project/docling/commit/146b4f05351042a446394c16895c73c2f26cad7b))

## [v2.63.0](https://github.com/docling-project/docling/releases/tag/v2.63.0) - 2025-11-20

### Feature

* Add save and load for conversion result ([#2648](https://github.com/docling-project/docling/issues/2648)) ([`b559813`](https://github.com/docling-project/docling/commit/b559813b9becf7950bc539c1334e55ef17bed2ad))

### Fix

* Respect document_timeout in new threaded StandardPdfPipeline ([#2653](https://github.com/docling-project/docling/issues/2653)) ([`2087c6b`](https://github.com/docling-project/docling/commit/2087c6bf9f65f279dd2ff0631768996aecd640fe))
* In DocumentConverter.convert_string() make nullable name parameter optional ([#2660](https://github.com/docling-project/docling/issues/2660)) ([`6fb9a5f`](https://github.com/docling-project/docling/commit/6fb9a5f98aa33f0e37d720487e71467cd9aae846))
* Enable GPU for RapidOCR when available ([#2659](https://github.com/docling-project/docling/issues/2659)) ([`463a3fd`](https://github.com/docling-project/docling/commit/463a3fd4743708ccebe32fa1ddb462a6899277b9))
* Remove py3.14 requirement for default rapidocr ([#2639](https://github.com/docling-project/docling/issues/2639)) ([`da4c2e9`](https://github.com/docling-project/docling/commit/da4c2e9dbe5ea363d29af188e24885f1ef3efba9))

### Documentation

* Add Hector as compatible AI agent platform integration ([#2662](https://github.com/docling-project/docling/issues/2662)) ([`ce5a099`](https://github.com/docling-project/docling/commit/ce5a099dfde50b5416f7a9676914c938740e0ada))
* Added documentation to use SuryaOCR via plugin docling-surya ([#2533](https://github.com/docling-project/docling/issues/2533)) ([`b216ad8`](https://github.com/docling-project/docling/commit/b216ad848dfe6d8cdfdbb90d10000773d8563633))
* Fix broken homepage links ([#2651](https://github.com/docling-project/docling/issues/2651)) ([`03e7c7d`](https://github.com/docling-project/docling/commit/03e7c7d924f4973c679ef988bc861741e02d804a))
* **examples:** Processing parquet file of images ([#2641](https://github.com/docling-project/docling/issues/2641)) ([`8af228f`](https://github.com/docling-project/docling/commit/8af228f1e2e5f12815f19beb979e7b1253c15115))
* Move Installation and Quickstart (Usage) under Getting started ([#2644](https://github.com/docling-project/docling/issues/2644)) ([`d549445`](https://github.com/docling-project/docling/commit/d549445e78fc8839c510a52e6c72e2e711ba153e))
* Add redirection from getting started page ([#2640](https://github.com/docling-project/docling/issues/2640)) ([`ac9fc58`](https://github.com/docling-project/docling/commit/ac9fc585bbb515c3e00b042eeb979800ae548660))
* **examples:** Remove deprecation warnings with export_to_dataframe ([#2638](https://github.com/docling-project/docling/issues/2638)) ([`f552862`](https://github.com/docling-project/docling/commit/f5528623a77d9a3b0d821035f66baefd49da8dd9))

## [v2.62.0](https://github.com/docling-project/docling/releases/tag/v2.62.0) - 2025-11-17

### Feature

* Add the Image backend ([#2627](https://github.com/docling-project/docling/issues/2627)) ([`3495b73`](https://github.com/docling-project/docling/commit/3495b73de875c2438108b4362dbac770b6d322ca))
* **experimental:** Layout + VLM model with layout prompt ([#2244](https://github.com/docling-project/docling/issues/2244)) ([`4852d8b`](https://github.com/docling-project/docling/commit/4852d8b4f2938434f1d6250984fa18ec5428055f))

### Fix

* Correct the model-repo name ([#2624](https://github.com/docling-project/docling/issues/2624)) ([`14b436d`](https://github.com/docling-project/docling/commit/14b436d59075c6e78fb0aac9d2fc9742954920ea))
* **docx:** Parse page headers and footers ([#2599](https://github.com/docling-project/docling/issues/2599)) ([`054c4a6`](https://github.com/docling-project/docling/commit/054c4a634d91c5480df473b289b9403498065334))

### Documentation

* Combine Home and Getting Started pages ([#2600](https://github.com/docling-project/docling/issues/2600)) ([`ae30373`](https://github.com/docling-project/docling/commit/ae30373ee7fd5e91e24511eb06d6b24a7bc79e07))

## [v2.61.2](https://github.com/docling-project/docling/releases/tag/v2.61.2) - 2025-11-10

### Fix

* Default to EasyOCR in Python 3.14 ([#2605](https://github.com/docling-project/docling/issues/2605)) ([`5c27567`](https://github.com/docling-project/docling/commit/5c27567c4160b6ec43857855c8d5cd3a58c031c5))

## [v2.61.1](https://github.com/docling-project/docling/releases/tag/v2.61.1) - 2025-11-06

### Fix

* **docx:** Slow table parsing ([#2553](https://github.com/docling-project/docling/issues/2553)) ([`ef623ff`](https://github.com/docling-project/docling/commit/ef623ffceefe40aa237e163b564310ed81296bcf))
* **html:** Slow table parsing ([#2582](https://github.com/docling-project/docling/issues/2582)) ([`0ba8d5d`](https://github.com/docling-project/docling/commit/0ba8d5d9e325390626268744f289458e91689b4b))

### Documentation

* Make navigation menus collapse and expand ([#2573](https://github.com/docling-project/docling/issues/2573)) ([`8da3d28`](https://github.com/docling-project/docling/commit/8da3d287edd6a65877c147882430422e4373230b))

## [v2.61.0](https://github.com/docling-project/docling/releases/tag/v2.61.0) - 2025-11-06

### Feature

* **vlm:** Track generated tokens and stop reasons for VLM models ([#2543](https://github.com/docling-project/docling/issues/2543)) ([`6a04e27`](https://github.com/docling-project/docling/commit/6a04e273528691eb22a5708f1270d4c5fa8f5b7c))

### Fix

* Temporarily pin NuExtract to working revision ([#2588](https://github.com/docling-project/docling/issues/2588)) ([`fa92574`](https://github.com/docling-project/docling/commit/fa925741b6dc00c7bd2806c62cb75cb539649c9f))
* **ocr:** Use PSM integer values directly instead of constructor ([#2578](https://github.com/docling-project/docling/issues/2578)) ([`1a5146a`](https://github.com/docling-project/docling/commit/1a5146abc9dbd4ed9d08da186695fbbcb07814a2))

## [v2.60.1](https://github.com/docling-project/docling/releases/tag/v2.60.1) - 2025-11-04

### Fix

* Extract response from api_image_request in picture description ([#2571](https://github.com/docling-project/docling/issues/2571)) ([`8360aa5`](https://github.com/docling-project/docling/commit/8360aa54492bc5b5e07fcd07b0b85284910f1a14))

## [v2.60.0](https://github.com/docling-project/docling/releases/tag/v2.60.0) - 2025-10-31

### Feature

* Use threading in the standard pipeline and move old behavior to legacy ([#2452](https://github.com/docling-project/docling/issues/2452)) ([`268d027`](https://github.com/docling-project/docling/commit/268d027c8f2abae7339b4c7d33642c3135c56e7a))

### Fix

* **pdf:** Threadsafe for pypdfium2 backend ([#2527](https://github.com/docling-project/docling/issues/2527)) ([`a51275d`](https://github.com/docling-project/docling/commit/a51275d08037a30ebaa07e33b0c4e82623791259))

### Documentation

* Update link to Open WebUI docs ([#2549](https://github.com/docling-project/docling/issues/2549)) ([`01577e9`](https://github.com/docling-project/docling/commit/01577e92d15653a517c81226695821a7ea4c955e))
* Update installation options with extras and review FAQ ([#2548](https://github.com/docling-project/docling/issues/2548)) ([`cb10043`](https://github.com/docling-project/docling/commit/cb100437facdfc064c83396c3aebb185717bead6))
* Fix typos ([#2546](https://github.com/docling-project/docling/issues/2546)) ([`741c44f`](https://github.com/docling-project/docling/commit/741c44fa45250c7d0e5687547fc5d2453dc20e6b))

## [v2.59.0](https://github.com/docling-project/docling/releases/tag/v2.59.0) - 2025-10-30

### Feature

* **vlm:** Add num_tokens as attribtue for VlmPrediction ([#2489](https://github.com/docling-project/docling/issues/2489)) ([`b6c892b`](https://github.com/docling-project/docling/commit/b6c892b505bf29a12ce7e8d9b4e88e1253440ebc))
* Support for Python 3.14 ([#2530](https://github.com/docling-project/docling/issues/2530)) ([`cdffb47`](https://github.com/docling-project/docling/commit/cdffb47b9a12da23489e345ea633786914776f7d))

### Fix

* Xlsx cell parsing, now returning values instead of formulas ([#2520](https://github.com/docling-project/docling/issues/2520)) ([`d9c90eb`](https://github.com/docling-project/docling/commit/d9c90eb45e899b2f3b608d959cb664c016e63821))

### Documentation

* Add details and examples on optimal GPU setup ([#2531](https://github.com/docling-project/docling/issues/2531)) ([`97aa06b`](https://github.com/docling-project/docling/commit/97aa06bfbca6f9a0bfca81c4ed99bf83f36c9f9b))
* Update opensearch notebook and backend documentation ([#2519](https://github.com/docling-project/docling/issues/2519)) ([`9a6fdf9`](https://github.com/docling-project/docling/commit/9a6fdf936bd59421c9b133c065059c3a0783ca2b))

## [v2.58.0](https://github.com/docling-project/docling/releases/tag/v2.58.0) - 2025-10-22

### Feature

* **pdf:** Support for password-protected PDF documents ([#2499](https://github.com/docling-project/docling/issues/2499)) ([`bbe82a6`](https://github.com/docling-project/docling/commit/bbe82a68d08e5dc33191524bb636f06112edff87))
* **backend:** Add generic options support and HTML image handling modes ([#2011](https://github.com/docling-project/docling/issues/2011)) ([`a30e6a7`](https://github.com/docling-project/docling/commit/a30e6a76148079cc48fb179e4b9ca36371026b6f))
* **ASR:** MLX Whisper Support for Apple Silicon ([#2366](https://github.com/docling-project/docling/issues/2366)) ([`657ce8b`](https://github.com/docling-project/docling/commit/657ce8b01cecd4983075142a64c076abb35e80ae))

### Fix

* **markdown:** Set the correct discriminator in md backend options ([#2501](https://github.com/docling-project/docling/issues/2501)) ([`4227fcc`](https://github.com/docling-project/docling/commit/4227fcc3e1e123da1ffc1a8609aa93d9a756a67e))
* **xlsx:** Speed up by detecting the true last non-empty row/column ([#2404](https://github.com/docling-project/docling/issues/2404)) ([`b66624b`](https://github.com/docling-project/docling/commit/b66624bfff1b0db634925fe468a7fe1c2c78c077))

### Documentation

* Fix typo in mcp.md ([#2502](https://github.com/docling-project/docling/issues/2502)) ([`86556d8`](https://github.com/docling-project/docling/commit/86556d8367b1ed70ba59b8c6e96a09f91ad4798e))
* Discord badge with join link ([#2473](https://github.com/docling-project/docling/issues/2473)) ([`dd03b53`](https://github.com/docling-project/docling/commit/dd03b531178a4038291fbf5785a8f3624965de86))

### Performance

* Use docling-parse-v4 as default ([#2503](https://github.com/docling-project/docling/issues/2503)) ([`89820d0`](https://github.com/docling-project/docling/commit/89820d01b5272d4600bdd4ea7168f0d031fd52dc))

## [v2.57.0](https://github.com/docling-project/docling/releases/tag/v2.57.0) - 2025-10-15

### Feature

* **docx:** Process drawingml objects in docx ([#2453](https://github.com/docling-project/docling/issues/2453)) ([`1682993`](https://github.com/docling-project/docling/commit/16829939cf1f8d89974c51c1d7c5cdc2fe8045da))

### Fix

* Use proper page concatentation in VLM pipeline MD/HTML conversion ([#2458](https://github.com/docling-project/docling/issues/2458)) ([`cd7f7ba`](https://github.com/docling-project/docling/commit/cd7f7ba145c401fb6567ef1c7337c840100cded1))

### Documentation

* Example on PII obfuscation ([#2459](https://github.com/docling-project/docling/issues/2459)) ([`3e6da2c`](https://github.com/docling-project/docling/commit/3e6da2c62dee9a48b442245dbc1aa5e144c1446c))

## [v2.56.1](https://github.com/docling-project/docling/releases/tag/v2.56.1) - 2025-10-13

### Fix

* Avoid downloading easyocr models by default ([#2454](https://github.com/docling-project/docling/issues/2454)) ([`688a7df`](https://github.com/docling-project/docling/commit/688a7dfd38ba3e3aea64f7fe027815e910818785))

## [v2.56.0](https://github.com/docling-project/docling/releases/tag/v2.56.0) - 2025-10-13

### Feature

* AutoOCR model selecting the best OCR model available and deprecating the usage of EasyOCR ([#2391](https://github.com/docling-project/docling/issues/2391)) ([`f7244a4`](https://github.com/docling-project/docling/commit/f7244a433378327576e3554d41d80928ee38e2a7))
* Add Tesseract PSM options support ([#2411](https://github.com/docling-project/docling/issues/2411)) ([`f11f8c0`](https://github.com/docling-project/docling/commit/f11f8c0a8188f99179acd7e47a48b908b1ea64d0))

### Fix

* **asr:** Implement robust status check in AsrPipeline ([#2442](https://github.com/docling-project/docling/issues/2442)) ([`db985bb`](https://github.com/docling-project/docling/commit/db985bb159173b56977db1228222592dcdbcb297))
* Deal with chartsheets in workbooks ([#2433](https://github.com/docling-project/docling/issues/2433)) ([`cce18b2`](https://github.com/docling-project/docling/commit/cce18b2ff7f14ab0643364055a1ec8a77479a211))
* Skip temporary docx files ([#2413](https://github.com/docling-project/docling/issues/2413)) ([`ee55013`](https://github.com/docling-project/docling/commit/ee5501320e003f45b7f4b9fdaaae69d481ec9ccc))
* AsrPipeline to handle absolute paths and BytesIO streams correctly ([#2407](https://github.com/docling-project/docling/issues/2407)) ([`b5f7fef`](https://github.com/docling-project/docling/commit/b5f7fef29b4bbb51393a212041011ec5950aba28))
* Enrichment of documents without pages metadata (pptx and xlsx) ([#2401](https://github.com/docling-project/docling/issues/2401)) ([`0610d01`](https://github.com/docling-project/docling/commit/0610d01afae6e3a1c88c85894be52503d7c873e4))
* Proper heading support in rich tables for HTML backend ([#2394](https://github.com/docling-project/docling/issues/2394)) ([`9705f40`](https://github.com/docling-project/docling/commit/9705f4020c8e7f43cc5b0658c9641df24508502f))

### Documentation

* Remove deprecated call in custom_convert.py ([#2447](https://github.com/docling-project/docling/issues/2447)) ([`9020044`](https://github.com/docling-project/docling/commit/90200443bcd92ef52e1b1f5cc32a5ee9d85f74f2))
* Fixed a few typos ([#2441](https://github.com/docling-project/docling/issues/2441)) ([`2a0f563`](https://github.com/docling-project/docling/commit/2a0f56390a1426d15d8330d3f3f36aeec464693c))
* Add MongoDB + VoyageAI ([#2382](https://github.com/docling-project/docling/issues/2382)) ([`f2854b2`](https://github.com/docling-project/docling/commit/f2854b2e1d790c42b93f2925d42adb53d33fd01e))
* Add RAG example with MongoDB Atlas Vector Search and VoyageAI embeddings ([#2341](https://github.com/docling-project/docling/issues/2341)) ([`8a4b946`](https://github.com/docling-project/docling/commit/8a4b946a1acf89b831a65c99503290b3ea957ec7))

## [v2.55.1](https://github.com/docling-project/docling/releases/tag/v2.55.1) - 2025-10-03

### Fix

* **markdown:** Setext heading support ([#2359](https://github.com/docling-project/docling/issues/2359)) ([`ee73ffa`](https://github.com/docling-project/docling/commit/ee73ffae15b2bb60c42e333fca4684bd57eeff31))
* **docs:** Fixed the color scheme ([#2371](https://github.com/docling-project/docling/issues/2371)) ([`246de77`](https://github.com/docling-project/docling/commit/246de77d8ce53fbeb7a93a6412c461df82269685))
* Empty table handling ([#2365](https://github.com/docling-project/docling/issues/2365)) ([`ca2be7f`](https://github.com/docling-project/docling/commit/ca2be7ff3ad1223eaea9954543f8d82e4d64e53e))
* Add table raw content when no table structure model is used ([#1815](https://github.com/docling-project/docling/issues/1815)) ([`4f295ed`](https://github.com/docling-project/docling/commit/4f295ed0511e21d215129c5850a7d84e748c7a1d))

### Documentation

* Example using Hashicorp Vault PII transform ([#2373](https://github.com/docling-project/docling/issues/2373)) ([`a975a79`](https://github.com/docling-project/docling/commit/a975a790c997268bc5f613631943ca074844f290))
* Jobkit and connectors ([#2357](https://github.com/docling-project/docling/issues/2357)) ([`e6c3b05`](https://github.com/docling-project/docling/commit/e6c3b05e63e7b4ef0a3e52205dedde4accbea1f1))

## [v2.55.0](https://github.com/docling-project/docling/releases/tag/v2.55.0) - 2025-09-30

### Feature

* Repetition-based StoppingCriteria for GraniteDocling ([#2323](https://github.com/docling-project/docling/issues/2323)) ([`1e9dc43`](https://github.com/docling-project/docling/commit/1e9dc43b722aeffa4574ae2a87bae1eb180c1201))
* Rich tables support for HTML backend ([#2324](https://github.com/docling-project/docling/issues/2324)) ([`c803abe`](https://github.com/docling-project/docling/commit/c803abed9ae98489184791a70bf49cac0c83ab89))

### Fix

* Pin wider range of typer ([#2309](https://github.com/docling-project/docling/issues/2309)) ([`68ae7cc`](https://github.com/docling-project/docling/commit/68ae7ccf3cf945fc18093d27a5344d55f77a510b))
* Update Transformers & VLLM inference code, CLI and VLM specs ([#2322](https://github.com/docling-project/docling/issues/2322)) ([`654c70f`](https://github.com/docling-project/docling/commit/654c70f99096636006e67fefe4a0809ef58fd236))
* Support escaped characters in markdown backend ([#2304](https://github.com/docling-project/docling/issues/2304)) ([`9d67bb9`](https://github.com/docling-project/docling/commit/9d67bb9ed61e8c3072816b13823b908817b45518))

### Documentation

* **styling:** Update color scheme ([#2154](https://github.com/docling-project/docling/issues/2154)) ([`325877a`](https://github.com/docling-project/docling/commit/325877aee977efae0a87998aeb82088b5ca4e13d))
* **vlm:** Update SmolDocling to GraniteDocling references ([#2315](https://github.com/docling-project/docling/issues/2315)) ([`a873200`](https://github.com/docling-project/docling/commit/a873200c9d6656de86833baa314f1e0faa8af50c))

## [v2.54.0](https://github.com/docling-project/docling/releases/tag/v2.54.0) - 2025-09-22

### Feature

* Rich tables for MSWord backend ([#2291](https://github.com/docling-project/docling/issues/2291)) ([`e2482a2`](https://github.com/docling-project/docling/commit/e2482a2ada52b2b8a41c4402b27e125adbe4385f))
* Add a backend parser for WebVTT files ([#2288](https://github.com/docling-project/docling/issues/2288)) ([`46efaae`](https://github.com/docling-project/docling/commit/46efaaefee17a6b83e02a050f9f3c8a51afbbd53))

### Fix

* Correct y-axis scaling in draw_table_cells ([#2287](https://github.com/docling-project/docling/issues/2287)) ([`b5628f1`](https://github.com/docling-project/docling/commit/b5628f12273297d9db1393f4b734cfa337caa8c9))

### Documentation

* Update API VLM example with granite-docling ([#2294](https://github.com/docling-project/docling/issues/2294)) ([`8b7e83a`](https://github.com/docling-project/docling/commit/8b7e83a8c7b9e333c31d5ae0b96213e3c70c6bf3))
* Fix examples rendering ([#2281](https://github.com/docling-project/docling/issues/2281)) ([`8322c2e`](https://github.com/docling-project/docling/commit/8322c2ea9b4fbb1625bcbf1ec1b3dea6c1cd3ed0))

## [v2.53.0](https://github.com/docling-project/docling/releases/tag/v2.53.0) - 2025-09-17

### Feature

* Add granite-docling model ([#2272](https://github.com/docling-project/docling/issues/2272)) ([`17afb66`](https://github.com/docling-project/docling/commit/17afb664d005168b5a6f12a2df4432076a9329bb))
* **RapidOcr:** Support generic extra arguments for RapidOcr ([#2266](https://github.com/docling-project/docling/issues/2266)) ([`0e95171`](https://github.com/docling-project/docling/commit/0e95171dd64733ba52f2f0906642be24f6237977))

### Fix

* Handle empty result from RapidOCR to avoid crash ([#2264](https://github.com/docling-project/docling/issues/2264)) ([`609d902`](https://github.com/docling-project/docling/commit/609d902eef157ae68e33faa26d73533ef7a4a749))

### Documentation

* Describe examples ([#2262](https://github.com/docling-project/docling/issues/2262)) ([`ff351fd`](https://github.com/docling-project/docling/commit/ff351fd40c4b635133401e77dea89bec8cd0ca33))

## [v2.52.0](https://github.com/docling-project/docling/releases/tag/v2.52.0) - 2025-09-11

### Feature

* Enrichment steps on all convert pipelines (incl docx, html, etc) ([#2251](https://github.com/docling-project/docling/issues/2251)) ([`2c91234`](https://github.com/docling-project/docling/commit/2c9123419f541feda8cc98c53aeb37288fabcaee))

### Fix

* Add missing features in ThreadedStandardPdfPipeline ([#2252](https://github.com/docling-project/docling/issues/2252)) ([`0700af2`](https://github.com/docling-project/docling/commit/0700af212cce8d90dbe0477dcb06d69370649e97))
* Address deprecation warnings of dependencies ([#2237](https://github.com/docling-project/docling/issues/2237)) ([`c696549`](https://github.com/docling-project/docling/commit/c6965495a22703d0e35105b5daafcaaf8a8063d6))

### Documentation

* Add an example of RAG with OpenSearch ([#2238](https://github.com/docling-project/docling/issues/2238)) ([`f8cc545`](https://github.com/docling-project/docling/commit/f8cc545bab07e5fdd79bcff7042e9279e18926c6))
* Add instructions for using Docling with MCP to README ([#2219](https://github.com/docling-project/docling/issues/2219)) ([`e5cd702`](https://github.com/docling-project/docling/commit/e5cd7020bd281aea63519db9a5332dd2dcca54b4))
* Document VLM support requirement in extraction example ([#2231](https://github.com/docling-project/docling/issues/2231)) ([`55f5f37`](https://github.com/docling-project/docling/commit/55f5f3752f33f5b495cb2af5e6a3aee5d157fad8))

## [v2.51.0](https://github.com/docling-project/docling/releases/tag/v2.51.0) - 2025-09-05

### Feature

* Updating default parameters to get better performance with docling-parse ([#2208](https://github.com/docling-project/docling/issues/2208)) ([`b49d1ad`](https://github.com/docling-project/docling/commit/b49d1ad4f1af6eeadc3f8d0e35123dc52c6e228e))
* Updated the backend for new docling-parse ([#2187](https://github.com/docling-project/docling/issues/2187)) ([`b3d7542`](https://github.com/docling-project/docling/commit/b3d754206172d08d6d01f29f132dcb66383f955b))

### Documentation

* Add information extraction example ([#2199](https://github.com/docling-project/docling/issues/2199)) ([`a9f41b0`](https://github.com/docling-project/docling/commit/a9f41b088eae6f1ffe34d567057f80180f445a05))

## [v2.50.0](https://github.com/docling-project/docling/releases/tag/v2.50.0) - 2025-09-03

### Feature

* Heron layout model as new default ([#1971](https://github.com/docling-project/docling/issues/1971)) ([`e38aa0f`](https://github.com/docling-project/docling/commit/e38aa0f7f2e8a7881c0f97131bf776556778f9a2))

### Fix

* **html:** Access to variable not yet declared ([#2171](https://github.com/docling-project/docling/issues/2171)) ([`293e81b`](https://github.com/docling-project/docling/commit/293e81bf9d341edd1d35f3b66faf726b82ad4885))

## [v2.49.0](https://github.com/docling-project/docling/releases/tag/v2.49.0) - 2025-09-01

### Feature

* [Beta] Extraction with schema ([#2138](https://github.com/docling-project/docling/issues/2138)) ([`9f4bc5b`](https://github.com/docling-project/docling/commit/9f4bc5b2f19d700208b0b233c88fbe960758bdbd))
* **msexcel:** Set ContentLayer.INVISIBLE for invisible sheet ([#1876](https://github.com/docling-project/docling/issues/1876)) ([`a283ccf`](https://github.com/docling-project/docling/commit/a283ccff25a25ebbe6e9b2decfaaad6f300597db))

### Fix

* **pypdfium2:** Fix OCR bounding box misalignment caused by mismatched rotation metadata ([#2039](https://github.com/docling-project/docling/issues/2039)) ([`4d94e38`](https://github.com/docling-project/docling/commit/4d94e382239b19fc06b13317c80b4772a94f5269))
* Translation example ([#2166](https://github.com/docling-project/docling/issues/2166)) ([`9f0286b`](https://github.com/docling-project/docling/commit/9f0286bcac75870a2cf5a119813791aa46538d55))
* Extend offline mode for rapidocr fonts ([#2155](https://github.com/docling-project/docling/issues/2155)) ([`9904d14`](https://github.com/docling-project/docling/commit/9904d14e6a9bdc641f796382140967d11f315ea7))

### Documentation

* Enrich landing pages ([#2165](https://github.com/docling-project/docling/issues/2165)) ([`96cab6b`](https://github.com/docling-project/docling/commit/96cab6b536e532980b5130faf22be39f977a4d4b))

## [v2.48.0](https://github.com/docling-project/docling/releases/tag/v2.48.0) - 2025-08-26

### Feature

* Upgrade to RapidOCR 3.x ([#2088](https://github.com/docling-project/docling/issues/2088)) ([`3f60a0f`](https://github.com/docling-project/docling/commit/3f60a0fa7814aa36f396131e53b0ca6982b7638e))

### Fix

* **html:** Preserve code blocks in list items ([#2131](https://github.com/docling-project/docling/issues/2131)) ([`fa3327e`](https://github.com/docling-project/docling/commit/fa3327e1a6f7f4de6d5fd1f83588d6b5cac324ca))

## [v2.47.1](https://github.com/docling-project/docling/releases/tag/v2.47.1) - 2025-08-23

### Fix

* Vllm extra only for linux x86_64 ([#2126](https://github.com/docling-project/docling/issues/2126)) ([`488f6cd`](https://github.com/docling-project/docling/commit/488f6cdd2da1e6065e56843413b56b41cb208b2c))

## [v2.47.0](https://github.com/docling-project/docling/releases/tag/v2.47.0) - 2025-08-22

### Feature

* **CLI:** Option to download arbitrary HuggingFace model ([#2123](https://github.com/docling-project/docling/issues/2123)) ([`cdf079d`](https://github.com/docling-project/docling/commit/cdf079dd06955358d5bb74d6d3be9dee7b30c082))
* Batching support for VLMs in transformers backend, add initial VLLM backend ([#2094](https://github.com/docling-project/docling/issues/2094)) ([`3c660c0`](https://github.com/docling-project/docling/commit/3c660c051180720c6ae2321f1243a9a3b9696ddc))
* **html:** Support formatting tags in HTML texts ([#2111](https://github.com/docling-project/docling/issues/2111)) ([`94fcc46`](https://github.com/docling-project/docling/commit/94fcc46aa9098fe70450046bd97f8df7f93bce33))

### Fix

* Improve numbered list detection for msword docs ([#2100](https://github.com/docling-project/docling/issues/2100)) ([`3f03709`](https://github.com/docling-project/docling/commit/3f03709885a7280d6ca389816002ea606c3778d6))

### Documentation

* DPK pipeline example using docling library ([#2112](https://github.com/docling-project/docling/issues/2112)) ([`e76298c`](https://github.com/docling-project/docling/commit/e76298c40d9a860fe5c8e2d5922397eed4a71763))
* Add Getting Started page ([#2113](https://github.com/docling-project/docling/issues/2113)) ([`8996d61`](https://github.com/docling-project/docling/commit/8996d612aa47c5d500ddefcab4e1d42a87f19033))

## [v2.46.0](https://github.com/docling-project/docling/releases/tag/v2.46.0) - 2025-08-20

### Feature

* New code formula model ([#2042](https://github.com/docling-project/docling/issues/2042)) ([`d2494da`](https://github.com/docling-project/docling/commit/d2494da8b84c37731dd25b2f82d6559539ab4cb2))

### Fix

* **HTML:** Parse footer tag as a group in furniture content layer ([#2106](https://github.com/docling-project/docling/issues/2106)) ([`c5f2e2f`](https://github.com/docling-project/docling/commit/c5f2e2fdd68365f1dfa18d462f1e8051cd1089da))

### Performance

* Clean up resources with docling-parse v4, no parsed_page output by default ([#2105](https://github.com/docling-project/docling/issues/2105)) ([`5f57ff2`](https://github.com/docling-project/docling/commit/5f57ff2a45eaf1f2feb63080e9590ad6b9d4ea1e))
* Speed up function `_parse_orientation` ([#1934](https://github.com/docling-project/docling/issues/1934)) ([`8820b55`](https://github.com/docling-project/docling/commit/8820b5558ba18306cfa7a2def3f04cbd480fef94))

## [v2.45.0](https://github.com/docling-project/docling/releases/tag/v2.45.0) - 2025-08-18

### Feature

* Add backend for METS with Google Books profile ([#1989](https://github.com/docling-project/docling/issues/1989)) ([`31087f3`](https://github.com/docling-project/docling/commit/31087f3fcc2edb6e7c01cc444cf1bd93689999a6))
* **html:** Support in-line anchor tags in HTML texts ([#1659](https://github.com/docling-project/docling/issues/1659)) ([`9687297`](https://github.com/docling-project/docling/commit/968729726224279e688d5bdb0fccd2ed1a739aaa))
* **vlm:** Ability to preprocess VLM response ([#1907](https://github.com/docling-project/docling/issues/1907)) ([`5f050f9`](https://github.com/docling-project/docling/commit/5f050f94e19aeb9ea4e4457c20a55d31d1211567))

### Documentation

* Add docling Quarkus integration ([#2083](https://github.com/docling-project/docling/issues/2083)) ([`76c1fbd`](https://github.com/docling-project/docling/commit/76c1fbd6e8255d63cf483fd3d3465f13fd34a4ef))

## [v2.44.0](https://github.com/docling-project/docling/releases/tag/v2.44.0) - 2025-08-12

### Feature

* Add convert_string to document-converter ([#2069](https://github.com/docling-project/docling/issues/2069)) ([`b09033c`](https://github.com/docling-project/docling/commit/b09033cb73e3d99bb8b299675a539b4f10e41cb1))

### Fix

* **html:** Parse rawspan and colspan when they include non numerical values ([#2048](https://github.com/docling-project/docling/issues/2048)) ([`ed56f2d`](https://github.com/docling-project/docling/commit/ed56f2de5d58d8257df09beadc03bfddd521511a))
* Support new mlx-vlm module ([#2001](https://github.com/docling-project/docling/issues/2001)) ([`0130e3a`](https://github.com/docling-project/docling/commit/0130e3ae96bf7679e28ce1ec939ac750c71c6756))
* Extend error reporting when verbose logging is enabled ([#2017](https://github.com/docling-project/docling/issues/2017)) ([`2eb760d`](https://github.com/docling-project/docling/commit/2eb760d060911102be6f58fc1322b648e11099ab))
* **HTML:** Replace non-standard Unicode characters ([#2006](https://github.com/docling-project/docling/issues/2006)) ([`86f7012`](https://github.com/docling-project/docling/commit/86f70128aae9a4c4187c69757e77ebf90c8cf0c0))

### Documentation

* Add Langflow integration ([#2068](https://github.com/docling-project/docling/issues/2068)) ([`e2cca93`](https://github.com/docling-project/docling/commit/e2cca931be24f05de843e074a91a0a0fa53a1393))
* Add Arconia integration ([#2061](https://github.com/docling-project/docling/issues/2061)) ([`bfda6d3`](https://github.com/docling-project/docling/commit/bfda6d34d87965056fd0f7f4adcca6613f89ea3a))

## [v2.43.0](https://github.com/docling-project/docling/releases/tag/v2.43.0) - 2025-07-28

### Feature

* Threaded PDF pipeline ([#1951](https://github.com/docling-project/docling/issues/1951)) ([`aed772a`](https://github.com/docling-project/docling/commit/aed772ab332a9c4d2a95875c249f8646268b016d))

### Fix

* **markdown:** Ensure correct parsing of nested lists ([#1995](https://github.com/docling-project/docling/issues/1995)) ([`aec29a7`](https://github.com/docling-project/docling/commit/aec29a73152ef09028ebf09d9527e15cda8d1437))
* **HTML:** Remove an unnecessary print command ([#1988](https://github.com/docling-project/docling/issues/1988)) ([`945721a`](https://github.com/docling-project/docling/commit/945721a15dde80bbf836df8d87f278cf9a1baa51))

## [v2.42.2](https://github.com/docling-project/docling/releases/tag/v2.42.2) - 2025-07-24

### Fix

* **HTML:** Concatenation of child strings in table cells and list items ([#1981](https://github.com/docling-project/docling/issues/1981)) ([`5132f06`](https://github.com/docling-project/docling/commit/5132f061a8125332ba10a4a30e0dd4973637a11b))
* **docx:** Adding plain latex equations to table cells ([#1986](https://github.com/docling-project/docling/issues/1986)) ([`0b83609`](https://github.com/docling-project/docling/commit/0b836095319ebf2133c4a3a77602718034915e55))
* Preserve PARTIAL_SUCCESS status when document timeout hits ([#1975](https://github.com/docling-project/docling/issues/1975)) ([`98e2fcf`](https://github.com/docling-project/docling/commit/98e2fcff63660c158bafb9a1b5584c1439d7a533))
* Multi-page image support (tiff) ([#1928](https://github.com/docling-project/docling/issues/1928)) ([`8d50a59`](https://github.com/docling-project/docling/commit/8d50a59d4887caac1c214add8037ed0b5250f68c))

### Documentation

* Add chat with dosu ([#1984](https://github.com/docling-project/docling/issues/1984)) ([`7b5f860`](https://github.com/docling-project/docling/commit/7b5f86098d07b734f2b6aa8c88ae7cafa265246a))

## [v2.42.1](https://github.com/docling-project/docling/releases/tag/v2.42.1) - 2025-07-22

### Fix

* Keep formula clusters also when empty ([#1970](https://github.com/docling-project/docling/issues/1970)) ([`67441ca`](https://github.com/docling-project/docling/commit/67441ca4188d532c79df788c461e7f6f7d2f8170))

### Documentation

* Enrich existing DoclingDocument ([#1969](https://github.com/docling-project/docling/issues/1969)) ([`90a7cc4`](https://github.com/docling-project/docling/commit/90a7cc4bdda7272cd87d6f4ab3c0b7966f6e9c73))
* Add documentation for confidence scores ([#1912](https://github.com/docling-project/docling/issues/1912)) ([`5d98bce`](https://github.com/docling-project/docling/commit/5d98bcea1bd03aff426f903211c931620ff8fcc1))

## [v2.42.0](https://github.com/docling-project/docling/releases/tag/v2.42.0) - 2025-07-18

### Feature

* Add option to control empty clusters in layout postprocessing ([#1940](https://github.com/docling-project/docling/issues/1940)) ([`a436be7`](https://github.com/docling-project/docling/commit/a436be73676101cc9461a17ae7a9ae72316a5096))

### Fix

* Safe pipeline init, use device_map in transformers models ([#1917](https://github.com/docling-project/docling/issues/1917)) ([`cca05c4`](https://github.com/docling-project/docling/commit/cca05c45eaec154ae8470f9eb3577852d17773cd))
* Fix HTML table parser and JATS backend bugs ([#1948](https://github.com/docling-project/docling/issues/1948)) ([`e1e3053`](https://github.com/docling-project/docling/commit/e1e305369552b82d3f09f0c113ea8b54d5c90658))
* KeyError: 'fPr' when processing latex fractions in DOCX files ([#1926](https://github.com/docling-project/docling/issues/1926)) ([`95e7096`](https://github.com/docling-project/docling/commit/95e70962f1d7cf1f339a88fde9c907111e194726))
* Change granite vision model URL from preview to stable version ([#1925](https://github.com/docling-project/docling/issues/1925)) ([`c5fb353`](https://github.com/docling-project/docling/commit/c5fb353f109dfe79b51c201ebb1ff33fceeae34a))

### Documentation

* Fix typos ([#1943](https://github.com/docling-project/docling/issues/1943)) ([`d6d2dbe`](https://github.com/docling-project/docling/commit/d6d2dbe2f99bd965c1bc8eec3d332d0acf731189))

## [v2.41.0](https://github.com/docling-project/docling/releases/tag/v2.41.0) - 2025-07-10

### Feature

* Layout model specification and multiple choices ([#1910](https://github.com/docling-project/docling/issues/1910)) ([`2b8616d`](https://github.com/docling-project/docling/commit/2b8616d6d5e2b4ca1de587cd7a0746d3fe8e227b))
* Enable precision control in float serialization ([#1914](https://github.com/docling-project/docling/issues/1914)) ([`ec588df`](https://github.com/docling-project/docling/commit/ec588df97148818a6bee2512d5d81972b723a554))
* Add image-text-to-text models in transformers ([#1772](https://github.com/docling-project/docling/issues/1772)) ([`a07ba86`](https://github.com/docling-project/docling/commit/a07ba863c4c3dacfecaca159faa5653097662755))
* **vlm:** Dynamic prompts ([#1808](https://github.com/docling-project/docling/issues/1808)) ([`b8813ee`](https://github.com/docling-project/docling/commit/b8813eea806a33f3bcc4f865d7e6ceba8b2fffa5))

### Fix

* **ocr-utils:** Unit test and fix the `rotate_bounding_box` function ([#1897](https://github.com/docling-project/docling/issues/1897)) ([`931eb55`](https://github.com/docling-project/docling/commit/931eb55b8820765eb872961f295be0676852c73e))
* Docs are missing osd packages for tesseract on RHEL ([#1905](https://github.com/docling-project/docling/issues/1905)) ([`e25873d`](https://github.com/docling-project/docling/commit/e25873d55766761741ad5781efd18bc3bfea5e3d))
* Use only backend for picture classifier ([#1904](https://github.com/docling-project/docling/issues/1904)) ([`edd4356`](https://github.com/docling-project/docling/commit/edd4356aac25b62c30cae6d2e8c69095f63bd442))
* Typo in asr options ([#1902](https://github.com/docling-project/docling/issues/1902)) ([`dd8fde7`](https://github.com/docling-project/docling/commit/dd8fde7f19ecd9695d6bc6cf94896a2cf87a0e7c))

## [v2.40.0](https://github.com/docling-project/docling/releases/tag/v2.40.0) - 2025-07-04

### Feature

* Introduce LayoutOptions to control layout postprocessing behaviour ([#1870](https://github.com/docling-project/docling/issues/1870)) ([`ec6cf6f`](https://github.com/docling-project/docling/commit/ec6cf6f7e8050db30c14f0625d6d5c6bbfeb6aeb))
* Integrate ListItemMarkerProcessor into document assembly ([#1825](https://github.com/docling-project/docling/issues/1825)) ([`56a0e10`](https://github.com/docling-project/docling/commit/56a0e104f76c5ba30ac0fcd247be61f911b560c1))

### Fix

* Secure torch model inits with global locks ([#1884](https://github.com/docling-project/docling/issues/1884)) ([`598c9c5`](https://github.com/docling-project/docling/commit/598c9c53d401de6aac89b7c51bccd57160dace1e))
* Ensure that TesseractOcrModel does not crash in case OSD is not installed ([#1866](https://github.com/docling-project/docling/issues/1866)) ([`ae39a94`](https://github.com/docling-project/docling/commit/ae39a9411a09b2165ac745af358dea644f868e26))

### Performance

* **msexcel:** _find_table_bounds use iter_rows/iter_cols instead of Worksheet.cell ([#1875](https://github.com/docling-project/docling/issues/1875)) ([`13865c0`](https://github.com/docling-project/docling/commit/13865c06f5c564b9e57f3dbb60d26e60c75258b6))
* Move expensive imports closer to usage ([#1863](https://github.com/docling-project/docling/issues/1863)) ([`3089cf2`](https://github.com/docling-project/docling/commit/3089cf2d26918eed4007398a528f53971c19f839))

## [v2.39.0](https://github.com/docling-project/docling/releases/tag/v2.39.0) - 2025-06-27

### Feature

* Leverage new list modeling, capture default markers ([#1856](https://github.com/docling-project/docling/issues/1856)) ([`0533da1`](https://github.com/docling-project/docling/commit/0533da1923598e4a2d6392283f6de0f9c7002b01))

### Fix

* **markdown:** Make parsing of rich table cells valid ([#1821](https://github.com/docling-project/docling/issues/1821)) ([`e79e4f0`](https://github.com/docling-project/docling/commit/e79e4f0ab6c5b8276316e423b14c9821165049f2))

## [v2.38.1](https://github.com/docling-project/docling/releases/tag/v2.38.1) - 2025-06-25

### Fix

* Updated granite vision model version for picture description ([#1852](https://github.com/docling-project/docling/issues/1852)) ([`d337825`](https://github.com/docling-project/docling/commit/d337825b8ef9ab3ec00c1496c340041e406bd271))
* **markdown:** Fix single-formatted headings & list items ([#1820](https://github.com/docling-project/docling/issues/1820)) ([`7c5614a`](https://github.com/docling-project/docling/commit/7c5614a37a316950c9a1d123e4fd94e0e831aca0))
* Fix response type of ollama ([#1850](https://github.com/docling-project/docling/issues/1850)) ([`41e8cae`](https://github.com/docling-project/docling/commit/41e8cae26b625b95ffab021fb4dc337249e8caad))
* Handle missing runs to avoid out of range exception ([#1844](https://github.com/docling-project/docling/issues/1844)) ([`4002de1`](https://github.com/docling-project/docling/commit/4002de1f9220a6568ed87ba726254cde3ab1168a))

## [v2.38.0](https://github.com/docling-project/docling/releases/tag/v2.38.0) - 2025-06-23

### Feature

* Support audio input ([#1763](https://github.com/docling-project/docling/issues/1763)) ([`1557e7c`](https://github.com/docling-project/docling/commit/1557e7ce3e036fb51eb118296f5cbff3b6dfbfa7))
* **markdown:** Add formatting & improve inline support ([#1804](https://github.com/docling-project/docling/issues/1804)) ([`861abcd`](https://github.com/docling-project/docling/commit/861abcdcb0d406342b9566f81203b87cf32b7ad0))
* Maximum image size for Vlm models ([#1802](https://github.com/docling-project/docling/issues/1802)) ([`215b540`](https://github.com/docling-project/docling/commit/215b540f6c078a72464310ef22975ebb6cde4f0a))

### Fix

* **docx:** Ensure list items have a list parent ([#1827](https://github.com/docling-project/docling/issues/1827)) ([`d26dac6`](https://github.com/docling-project/docling/commit/d26dac61a86b0af5b16686f78956ba047bcbddba))
* **msword_backend:** Identify text in the same line after an image #1425 ([#1610](https://github.com/docling-project/docling/issues/1610)) ([`1350a8d`](https://github.com/docling-project/docling/commit/1350a8d3e5ea3c4b4d506757758880c8f78efd8c))
* Ensure uninitialized pages are removed before assembling document ([#1812](https://github.com/docling-project/docling/issues/1812)) ([`dd7f64f`](https://github.com/docling-project/docling/commit/dd7f64ff28226cd9964fc4d8ba807b2c8a6358ef))
* Formula conversion with page_range param set ([#1791](https://github.com/docling-project/docling/issues/1791)) ([`dbab30e`](https://github.com/docling-project/docling/commit/dbab30e92cc1d130ce7f9335ab9c46aa7a30930d))

### Documentation

* Update readme and add ASR example ([#1836](https://github.com/docling-project/docling/issues/1836)) ([`f3ae302`](https://github.com/docling-project/docling/commit/f3ae3029b8a6d6f0109383fbc82ebf9da3942afd))
* Support running examples from root or subfolder ([#1816](https://github.com/docling-project/docling/issues/1816)) ([`64ac043`](https://github.com/docling-project/docling/commit/64ac043786efdece0c61827051a5b41dddf6c5d7))

## [v2.37.0](https://github.com/docling-project/docling/releases/tag/v2.37.0) - 2025-06-16

### Feature

* Make Page.parsed_page the only source of truth for text cells, add OCR cells to it ([#1745](https://github.com/docling-project/docling/issues/1745)) ([`7d3302c`](https://github.com/docling-project/docling/commit/7d3302cb48dd91cd29673d7c4eaf7326736d0685))
* Support xlsm files ([#1520](https://github.com/docling-project/docling/issues/1520)) ([`df14022`](https://github.com/docling-project/docling/commit/df140227c3b8bcad0c68bf3d129930cccd96a07e))

### Fix

* Pptx line break and space handling ([#1664](https://github.com/docling-project/docling/issues/1664)) ([`f28d23c`](https://github.com/docling-project/docling/commit/f28d23cf03d059619d1d3482594596ab7c87d197))
* **asciidoc:** Set default size when missing in image directive ([#1769](https://github.com/docling-project/docling/issues/1769)) ([`b886e4d`](https://github.com/docling-project/docling/commit/b886e4df312447d39f58cf6e3c45b0f863940321))
* Handle NoneType error in MsPowerpointDocumentBackend ([#1747](https://github.com/docling-project/docling/issues/1747)) ([`7a275c7`](https://github.com/docling-project/docling/commit/7a275c763731d9c96b7cf32f2e27b8dc8bebacd7))
* Prov for merged-elems ([#1728](https://github.com/docling-project/docling/issues/1728)) ([`6613b9e`](https://github.com/docling-project/docling/commit/6613b9e98bc8b89791dc0334de8970ff243aba82))
* **tesseract:** Initialize df_osd to avoid uninitialized variable error ([#1718](https://github.com/docling-project/docling/issues/1718)) ([`e979750`](https://github.com/docling-project/docling/commit/e979750ce93b2fae89dbb60ff06333f80c1c2908))
* Allow custom torch_dtype in vlm models ([#1735](https://github.com/docling-project/docling/issues/1735)) ([`f7f3113`](https://github.com/docling-project/docling/commit/f7f31137f10999fefdb70da7e5ef56536f650400))
* Improve extraction from textboxes in Word docs ([#1701](https://github.com/docling-project/docling/issues/1701)) ([`9dbcb3d`](https://github.com/docling-project/docling/commit/9dbcb3d7d4f27d1c935c8681c57ed59524452d53))
* Add WEBP to the list of image file extensions ([#1711](https://github.com/docling-project/docling/issues/1711)) ([`a2b83fe`](https://github.com/docling-project/docling/commit/a2b83fe4aea66c273a83bf17177e87d45d3f18d1))

### Documentation

* Update vlm models api examples with LM Studio ([#1759](https://github.com/docling-project/docling/issues/1759)) ([`0432a31`](https://github.com/docling-project/docling/commit/0432a31b2f7c9fe944c3a1d4b608ef938b4f2299))
* Add open webui ([#1734](https://github.com/docling-project/docling/issues/1734)) ([`49b10e7`](https://github.com/docling-project/docling/commit/49b10e74191d4d580c9305ac08d9898a79346d7d))

## [v2.36.1](https://github.com/docling-project/docling/releases/tag/v2.36.1) - 2025-06-04

### Fix

* Remove typer and click constraints ([#1707](https://github.com/docling-project/docling/issues/1707)) ([`8846f1a`](https://github.com/docling-project/docling/commit/8846f1a393923a6badcca3a78a664a4dd31eae0d))

### Documentation

* Flash-attn usage and install ([#1706](https://github.com/docling-project/docling/issues/1706)) ([`be42b03`](https://github.com/docling-project/docling/commit/be42b03f9b366bed33e95c1033b90c63f300b914))

## [v2.36.0](https://github.com/docling-project/docling/releases/tag/v2.36.0) - 2025-06-03

### Feature

* Simplify dependencies, switch to uv ([#1700](https://github.com/docling-project/docling/issues/1700)) ([`cdd4018`](https://github.com/docling-project/docling/commit/cdd401847a35f16d69944eb6dddf57e4e0b65020))
* New vlm-models support ([#1570](https://github.com/docling-project/docling/issues/1570)) ([`cfdf4ce`](https://github.com/docling-project/docling/commit/cfdf4cea25e681fc557df310b8bf34f3dd892e15))

## [v2.35.0](https://github.com/docling-project/docling/releases/tag/v2.35.0) - 2025-06-02

### Feature

* Add visualization of bbox on page with html export. ([#1663](https://github.com/docling-project/docling/issues/1663)) ([`b356b33`](https://github.com/docling-project/docling/commit/b356b33059bdeeaf1584d9d189cbf1c4832e367c))

### Fix

* Guess HTML content starting with script tag ([#1673](https://github.com/docling-project/docling/issues/1673)) ([`984cb13`](https://github.com/docling-project/docling/commit/984cb137f6a8ae2f3a63623add6c474d97ef8739))
* UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd0 in position 0: invalid continuation byte ([#1665](https://github.com/docling-project/docling/issues/1665)) ([`51d3450`](https://github.com/docling-project/docling/commit/51d34509156e2dbec9e697276681d59f9ca7e020))

### Documentation

* Fix typo in index.md ([#1676](https://github.com/docling-project/docling/issues/1676)) ([`11ca4f7`](https://github.com/docling-project/docling/commit/11ca4f7a7bd8068bee472510dd71f1cd58f86f17))

## [v2.34.0](https://github.com/docling-project/docling/releases/tag/v2.34.0) - 2025-05-22

### Feature

* **ocr:** Auto-detect rotated pages in Tesseract ([#1167](https://github.com/docling-project/docling/issues/1167)) ([`45265bf`](https://github.com/docling-project/docling/commit/45265bf8b1a6d6ad5367bb3f17fb3fa9d4366a05))
* Establish confidence estimation for document and pages ([#1313](https://github.com/docling-project/docling/issues/1313)) ([`9087524`](https://github.com/docling-project/docling/commit/90875247e5813da1de17f3cd4475937e8bd45571))

### Fix

* Fix ZeroDivisionError for cell_bbox.area() ([#1636](https://github.com/docling-project/docling/issues/1636)) ([`c2f595d`](https://github.com/docling-project/docling/commit/c2f595d2830ca2e28e68c5da606e89541264f156))
* **integration:** Update the Apify Actor integration ([#1619](https://github.com/docling-project/docling/issues/1619)) ([`14d4f5b`](https://github.com/docling-project/docling/commit/14d4f5b109fa65d777ab147b3ce9b5174d020a5d))

## [v2.33.0](https://github.com/docling-project/docling/releases/tag/v2.33.0) - 2025-05-20

### Feature

* Add textbox content extraction in msword_backend ([#1538](https://github.com/docling-project/docling/issues/1538)) ([`12a0e64`](https://github.com/docling-project/docling/commit/12a0e648929ce75da73617904792a50f5145fe4a))

### Fix

* Fix issue with detecting docx files, and files with upper case extensions ([#1609](https://github.com/docling-project/docling/issues/1609)) ([`f4d9d41`](https://github.com/docling-project/docling/commit/f4d9d4111b0a6eb87fc1c05a56618fc430d1e7a2))
* Load_from_doctags static usage ([#1617](https://github.com/docling-project/docling/issues/1617)) ([`0e00a26`](https://github.com/docling-project/docling/commit/0e00a263fa0c45f6cf2ae0bd94f9387c28e51ed0))
* Incorrect force_backend_text behaviour for VLM DocTag pipelines ([#1371](https://github.com/docling-project/docling/issues/1371)) ([`f2e9c07`](https://github.com/docling-project/docling/commit/f2e9c0784c842612641171754ce51362e298088d))
* **pypdfium:** Resolve overlapping text when merging bounding boxes ([#1549](https://github.com/docling-project/docling/issues/1549)) ([`98b5eeb`](https://github.com/docling-project/docling/commit/98b5eeb8440d34ac84f58271c8b8eea88881260a))

## [v2.32.0](https://github.com/docling-project/docling/releases/tag/v2.32.0) - 2025-05-14

### Feature

* Improve parallelization for remote services API calls ([#1548](https://github.com/docling-project/docling/issues/1548)) ([`3a04f2a`](https://github.com/docling-project/docling/commit/3a04f2a367e32913f91faa2325f928b85112e632))
* Support image/webp file type ([#1415](https://github.com/docling-project/docling/issues/1415)) ([`12dab0a`](https://github.com/docling-project/docling/commit/12dab0a1e8d181d99e4711ffdbbc33d158234fb4))

### Fix

* **ocr:** Orig field in TesseractOcrCliModel as str ([#1553](https://github.com/docling-project/docling/issues/1553)) ([`9f8b479`](https://github.com/docling-project/docling/commit/9f8b479f17bbfaf79c3c897980ad15742ec86568))
* **settings:** Fix nested settings load via environment variables ([#1551](https://github.com/docling-project/docling/issues/1551)) ([`2efb7a7`](https://github.com/docling-project/docling/commit/2efb7a7c06a8e51516cc9b93e5dbcdea69f562fa))

### Documentation

* Add advanced chunking & serialization example ([#1589](https://github.com/docling-project/docling/issues/1589)) ([`9f28abf`](https://github.com/docling-project/docling/commit/9f28abf0610560645b40352dfdfc3525fa86c28d))

## [v2.31.2](https://github.com/docling-project/docling/releases/tag/v2.31.2) - 2025-05-13

### Fix

* AsciiDoc header identification (#1562) ([#1563](https://github.com/docling-project/docling/issues/1563)) ([`4046d0b`](https://github.com/docling-project/docling/commit/4046d0b2f38254679de5fc78aaf2fe630d6bb61c))
* Restrict click version and update lock file ([#1582](https://github.com/docling-project/docling/issues/1582)) ([`8baa85a`](https://github.com/docling-project/docling/commit/8baa85a49d3a456d198c52aac8e0b4ac70c92e72))

## [v2.31.1](https://github.com/docling-project/docling/releases/tag/v2.31.1) - 2025-05-12

### Fix

* Add smoldocling in download utils ([#1577](https://github.com/docling-project/docling/issues/1577)) ([`127e386`](https://github.com/docling-project/docling/commit/127e38646fd7f23fcda0e392e756fe27f123bd78))
* **HTML:** Handle row spans in header rows ([#1536](https://github.com/docling-project/docling/issues/1536)) ([`776e7ec`](https://github.com/docling-project/docling/commit/776e7ecf9ac93d62c66b03f33e5c8560e81b6fb3))
* Mime error in document streams ([#1523](https://github.com/docling-project/docling/issues/1523)) ([`f1658ed`](https://github.com/docling-project/docling/commit/f1658edbad5c7205bb457322d2c89f7f4d8a4659))
* Usage of hashlib for FIPS ([#1512](https://github.com/docling-project/docling/issues/1512)) ([`7c70573`](https://github.com/docling-project/docling/commit/7c705739f9db1cfc6c0a502fd5ba8b2093376d7f))
* Guard against attribute errors in TesseractOcrModel __del__ ([#1494](https://github.com/docling-project/docling/issues/1494)) ([`4ab7e9d`](https://github.com/docling-project/docling/commit/4ab7e9ddfb9d8fd0abc483efb70e701447a602c5))
* Enable cuda_use_flash_attention2 for PictureDescriptionVlmModel ([#1496](https://github.com/docling-project/docling/issues/1496)) ([`cc45396`](https://github.com/docling-project/docling/commit/cc453961a9196c79f6428305b9007402e448f300))
* Updated the time-recorder label for reading order ([#1490](https://github.com/docling-project/docling/issues/1490)) ([`976e92e`](https://github.com/docling-project/docling/commit/976e92e289a414b6b70c3e3ca37a60c85fa12535))
* Incorrect scaling of TableModel bboxes when do_cell_matching is False ([#1459](https://github.com/docling-project/docling/issues/1459)) ([`94d66a0`](https://github.com/docling-project/docling/commit/94d66a076559c4e48017bd619508cfeef104079b))

### Documentation

* Update links in data_prep_kit ([#1559](https://github.com/docling-project/docling/issues/1559)) ([`844babb`](https://github.com/docling-project/docling/commit/844babb39034b39d9c4edcc3f145684991cda174))
* Add serialization docs, update chunking docs ([#1556](https://github.com/docling-project/docling/issues/1556)) ([`3220a59`](https://github.com/docling-project/docling/commit/3220a592e720174940a3b958555f90352d7320d8))
* Update supported formats guide ([#1463](https://github.com/docling-project/docling/issues/1463)) ([`3afbe6c`](https://github.com/docling-project/docling/commit/3afbe6c9695d52cf6ed8b48b2f403df7d53342e5))

## [v2.31.0](https://github.com/docling-project/docling/releases/tag/v2.31.0) - 2025-04-25

### Feature

* Add tutorial using Milvus and Docling for RAG pipeline ([#1449](https://github.com/docling-project/docling/issues/1449)) ([`a2fbbba`](https://github.com/docling-project/docling/commit/a2fbbba9f7f889a1f84f8642cf5c75feb57e8668))

### Fix

* **html:** Handle address, details, and summary tags ([#1436](https://github.com/docling-project/docling/issues/1436)) ([`ed20124`](https://github.com/docling-project/docling/commit/ed20124544a1b10f068b11bbdf12e1bfc7567195))
* Treat overflowing -v flags as DEBUG ([#1419](https://github.com/docling-project/docling/issues/1419)) ([`8012a3e`](https://github.com/docling-project/docling/commit/8012a3e4d6b9ce4cae28210d525d87175da2f5c2))
* **codecov:** Fix codecov argument and yaml file ([#1399](https://github.com/docling-project/docling/issues/1399)) ([`fa7fc9e`](https://github.com/docling-project/docling/commit/fa7fc9e63d45f44af57dd6ad7636a2a16f04b8c4))

### Documentation

* Fix wrong output format in example code ([#1427](https://github.com/docling-project/docling/issues/1427)) ([`c2470ed`](https://github.com/docling-project/docling/commit/c2470ed216eaf3aae0ad16306de19682fa55b99b))
* Add OpenSSF Best Practices badge ([#1430](https://github.com/docling-project/docling/issues/1430)) ([`64918a8`](https://github.com/docling-project/docling/commit/64918a81ac315ea0108f1411a1537dd12117e49c))
* Typo fixes in docling_document.md ([#1400](https://github.com/docling-project/docling/issues/1400)) ([`995b3b0`](https://github.com/docling-project/docling/commit/995b3b0ab1c4e566eaba2ea31af3db21eb12a7ae))
* Updated the [Usage] link in architecture.md ([#1416](https://github.com/docling-project/docling/issues/1416)) ([`88948b0`](https://github.com/docling-project/docling/commit/88948b0bbaba2ecbaa71f703d2cc94055a3e6b3e))
* **ocr:** Add docs entry for OnnxTR OCR plugin ([#1382](https://github.com/docling-project/docling/issues/1382)) ([`a7dd59c`](https://github.com/docling-project/docling/commit/a7dd59c5cb3e7f1eba76c7e2e20be79d8fa5b367))
* **security:** More statements about secure development ([#1381](https://github.com/docling-project/docling/issues/1381)) ([`293c28c`](https://github.com/docling-project/docling/commit/293c28ca7c4a44dcd56595ed2fe0372fe1b531b2))
* Add testing in the docs ([#1379](https://github.com/docling-project/docling/issues/1379)) ([`01fbfd5`](https://github.com/docling-project/docling/commit/01fbfd565204258acb2986dcaefad3a328626c66))
* Add Notes for Installing in Intel macOS ([#1377](https://github.com/docling-project/docling/issues/1377)) ([`a026b4e`](https://github.com/docling-project/docling/commit/a026b4e84bcc8e11ceaa6d9a46c7c741000aff44))

## [v2.30.0](https://github.com/docling-project/docling/releases/tag/v2.30.0) - 2025-04-14

### Feature

* **cli:** Add option for html with split-page mode ([#1355](https://github.com/docling-project/docling/issues/1355)) ([`c0ba88e`](https://github.com/docling-project/docling/commit/c0ba88edf1d9dcef91979be9b674660b34c2d46a))
* **xlsx:** Create a page for each worksheet in XLSX backend ([#1332](https://github.com/docling-project/docling/issues/1332)) ([`eef2bde`](https://github.com/docling-project/docling/commit/eef2bdea77fa32061e798f538bf2cd95f8d72165))
* OllamaVlmModel for Granite Vision 3.2 ([#1337](https://github.com/docling-project/docling/issues/1337)) ([`c605edd`](https://github.com/docling-project/docling/commit/c605edd8e91d988f6dca2bdfc67c54d6396fe903))

### Fix

* **deps:** Widen typer upper bound ([#1375](https://github.com/docling-project/docling/issues/1375)) ([`7e40ad3`](https://github.com/docling-project/docling/commit/7e40ad3261147ba70204f7006ccaf9741025e58a))
* Auto-recognize .xlsx, .docx and .pptx files ([#1340](https://github.com/docling-project/docling/issues/1340)) ([`0de70e7`](https://github.com/docling-project/docling/commit/0de70e799100878b2aa48dfd49858c426f3f1b10))
* **docx:** Declare image_data variable when handling pictures ([#1359](https://github.com/docling-project/docling/issues/1359)) ([`415b877`](https://github.com/docling-project/docling/commit/415b877984fd89884e97b4740bc553e800055e0e))
* Implement PictureDescriptionApiOptions.bitmap_area_threshold ([#1248](https://github.com/docling-project/docling/issues/1248)) ([`2503999`](https://github.com/docling-project/docling/commit/250399948de69fe01cd789e328194f38a03598a7))
* Properly address page in pipeline _assemble_document when page_range is provided ([#1334](https://github.com/docling-project/docling/issues/1334)) ([`6b696b5`](https://github.com/docling-project/docling/commit/6b696b504a03ba49f05237d0e1b23fcced1a538a))

## [v2.29.0](https://github.com/docling-project/docling/releases/tag/v2.29.0) - 2025-04-10

### Feature

* Handle <code> tags as code blocks ([#1320](https://github.com/docling-project/docling/issues/1320)) ([`0499cd1`](https://github.com/docling-project/docling/commit/0499cd1c1e93f74260754476a8423059915f59c2))
* **docx:** Add text formatting and hyperlink support ([#630](https://github.com/docling-project/docling/issues/630)) ([`bfcab3d`](https://github.com/docling-project/docling/commit/bfcab3d6778e6f622bb4a6b241bdb4bab22ba378))

### Fix

* **docx:** Adding new latex symbols, simplifying how equations are added to text ([#1295](https://github.com/docling-project/docling/issues/1295)) ([`14e9c0c`](https://github.com/docling-project/docling/commit/14e9c0ce9a7559fac96ba5ed82befa12a7f53bfa))
* **pptx:** Check if picture shape has an image attached ([#1316](https://github.com/docling-project/docling/issues/1316)) ([`dc3bf9c`](https://github.com/docling-project/docling/commit/dc3bf9ceacb7048a97ceb8b7aa80bfccc8a05ca5))
* **docx:** Improve text parsing ([#1268](https://github.com/docling-project/docling/issues/1268)) ([`d2d6874`](https://github.com/docling-project/docling/commit/d2d68747f9c31be897f3e63c160c835086d37014))
* Tesseract OCR CLI can't process images composed with numbers only ([#1201](https://github.com/docling-project/docling/issues/1201)) ([`b3d111a`](https://github.com/docling-project/docling/commit/b3d111a3cdb90b653ddaaa356f9299e9cd39b340))

### Documentation

* Add plugins docs ([#1319](https://github.com/docling-project/docling/issues/1319)) ([`2e99e5a`](https://github.com/docling-project/docling/commit/2e99e5a54fafd901d8f26b56b25bb006c0e8e8b0))
* Add visual grounding example ([#1270](https://github.com/docling-project/docling/issues/1270)) ([`71148eb`](https://github.com/docling-project/docling/commit/71148eb381747a6b899c84b72946ba9bde665a40))

## [v2.28.4](https://github.com/docling-project/docling/releases/tag/v2.28.4) - 2025-03-29

### Fix

* Fixes tables when using OCR ([#1261](https://github.com/docling-project/docling/issues/1261)) ([`7afad7e`](https://github.com/docling-project/docling/commit/7afad7e52da642b258edd67f8f4815ea430f05e1))

## [v2.28.3](https://github.com/docling-project/docling/releases/tag/v2.28.3) - 2025-03-28

### Fix

* Word-level pdf cells for tables ([#1238](https://github.com/docling-project/docling/issues/1238)) ([`8bd71e8`](https://github.com/docling-project/docling/commit/8bd71e8e331de3a176110341554e026c9e0ecf6c))

## [v2.28.2](https://github.com/docling-project/docling/releases/tag/v2.28.2) - 2025-03-26

### Fix

* Improve HTML layer detection, various MD fixes ([#1241](https://github.com/docling-project/docling/issues/1241)) ([`9210812`](https://github.com/docling-project/docling/commit/9210812bfaad1fb138194464f0d563788f63f4c2))
* **html:** Fix HTML parsed heading level ([#1244](https://github.com/docling-project/docling/issues/1244)) ([`85c4df8`](https://github.com/docling-project/docling/commit/85c4df887b4dfc566f38ce25c6cf2824ff092b8e))

## [v2.28.1](https://github.com/docling-project/docling/releases/tag/v2.28.1) - 2025-03-25

### Fix

* **converter:** Cache same pipeline class with different options ([#1152](https://github.com/docling-project/docling/issues/1152)) ([`825b226`](https://github.com/docling-project/docling/commit/825b226fab0acabf2920f1af33c048675e8b0139))
* **debug:** Missing translation of bbox to to_bounding_box ([#1220](https://github.com/docling-project/docling/issues/1220)) ([`6df8827`](https://github.com/docling-project/docling/commit/6df882723112f9ddc22a5ace4048e1d5acb30737))
* **docx:** Identifying numbered headers ([#1231](https://github.com/docling-project/docling/issues/1231)) ([`f739d0e`](https://github.com/docling-project/docling/commit/f739d0e4c5a29046587e3c348eff7fdd30262d44))

### Documentation

* **examples:** Batch conversion doc `raises_on_error` ([#1147](https://github.com/docling-project/docling/issues/1147)) ([`0974ba4`](https://github.com/docling-project/docling/commit/0974ba4e1ca165f9916d17861491f482e33909ca))

## [v2.28.0](https://github.com/docling-project/docling/releases/tag/v2.28.0) - 2025-03-19

### Feature

* **SmolDocling:** Support MLX acceleration in VLM pipeline ([#1199](https://github.com/docling-project/docling/issues/1199)) ([`1c26769`](https://github.com/docling-project/docling/commit/1c26769785bcd17c0b8b621c5182ad81134d3915))
* Add PPTX notes slides ([#474](https://github.com/docling-project/docling/issues/474)) ([`b454aa1`](https://github.com/docling-project/docling/commit/b454aa1551b891644ce4028ed2d7ec8f82c167ab))
* Updated vlm pipeline (with latest changes from docling-core) ([#1158](https://github.com/docling-project/docling/issues/1158)) ([`2f72167`](https://github.com/docling-project/docling/commit/2f72167ff6421424dea4d93018b0d43af16ec153))

### Fix

* Determine correct page size in DoclingParseV4Backend ([#1196](https://github.com/docling-project/docling/issues/1196)) ([`f5adfb9`](https://github.com/docling-project/docling/commit/f5adfb9724aae1207f23e21d74033f331e6e1ffb))
* **msword:** Fixing function return in equations handling ([#1194](https://github.com/docling-project/docling/issues/1194)) ([`0b707d0`](https://github.com/docling-project/docling/commit/0b707d0882f5be42505871799387d0b1882bffbf))

### Documentation

* Linux Foundation AI & Data ([#1183](https://github.com/docling-project/docling/issues/1183)) ([`1d680b0`](https://github.com/docling-project/docling/commit/1d680b0a321d95fc6bd65b7bb4d5e15005a0250a))
* Move apify to docs ([#1182](https://github.com/docling-project/docling/issues/1182)) ([`54a78c3`](https://github.com/docling-project/docling/commit/54a78c307de833b93f9b84cf1f8ed6dace8573cb))

## [v2.27.0](https://github.com/docling-project/docling/releases/tag/v2.27.0) - 2025-03-18

### Feature

* Add factory for ocr engines via plugins ([#1010](https://github.com/docling-project/docling/issues/1010)) ([`6eaae3c`](https://github.com/docling-project/docling/commit/6eaae3cba034599020dc06ebdad3bc3ff0b5a8eb))
* Add DoclingParseV4 backend, using high-level docling-parse API ([#905](https://github.com/docling-project/docling/issues/905)) ([`3960b19`](https://github.com/docling-project/docling/commit/3960b199d63d0e9d660aeb0cbced02b38bb0b593))
* **actor:** Docling Actor on Apify infrastructure ([#875](https://github.com/docling-project/docling/issues/875)) ([`772487f`](https://github.com/docling-project/docling/commit/772487f9c91ad2ee53c591c314c72443f9cbfd23))
* Equations to latex in MSWord backend (with inline groups) ([#1114](https://github.com/docling-project/docling/issues/1114)) ([`6eb718f`](https://github.com/docling-project/docling/commit/6eb718f8493038d1b4b6ae836df5a24aa13cd17e))

### Fix

* **html:** Handle nested empty lists ([#1154](https://github.com/docling-project/docling/issues/1154)) ([`f94da44`](https://github.com/docling-project/docling/commit/f94da44ec5c7a8c92b9dd60e4df5dc945ed6d1ea))
* Use first table row as col headers ([#1156](https://github.com/docling-project/docling/issues/1156)) ([`0945973`](https://github.com/docling-project/docling/commit/0945973b79d67b74281aba5102ee985ac1de74ea))
* Pass tests, update docling-core to 2.22.0 ([#1150](https://github.com/docling-project/docling/issues/1150)) ([`aa92a57`](https://github.com/docling-project/docling/commit/aa92a57fa9e7228e894efb9050a0cdb9f287ebfd))

### Documentation

* Fix spelling of picture in usage ([#1165](https://github.com/docling-project/docling/issues/1165)) ([`7e01798`](https://github.com/docling-project/docling/commit/7e01798417c424c05685e0ff5f6f89f70dc3bfcd))

## [v2.26.0](https://github.com/docling-project/docling/releases/tag/v2.26.0) - 2025-03-11

### Feature

* Use new TableFormer model weights and default to accurate model version ([#1100](https://github.com/docling-project/docling/issues/1100)) ([`eb97357`](https://github.com/docling-project/docling/commit/eb97357b0560b59c14a8be3fb52d6a1362ad0a1d))

### Fix

* **CLI:** Fix help message for abort options ([#1130](https://github.com/docling-project/docling/issues/1130)) ([`4d64c4c`](https://github.com/docling-project/docling/commit/4d64c4c0b67b23f53d4ea21fb754455840fe4556))

### Documentation

* Add description of DOCLING_ARTIFACTS_PATH env var ([#1124](https://github.com/docling-project/docling/issues/1124)) ([`e1c49ad`](https://github.com/docling-project/docling/commit/e1c49ad72710ff76c1b0574bb4d2bdab93077902))

### Performance

* New revision code formula model and document picture classifier ([#1140](https://github.com/docling-project/docling/issues/1140)) ([`5e30381`](https://github.com/docling-project/docling/commit/5e30381c0dd3b4f9e3b2d8af3863ed51fa51194a))

## [v2.25.2](https://github.com/docling-project/docling/releases/tag/v2.25.2) - 2025-03-05

### Fix

* Proper handling of orphan IDs in layout postprocessing ([#1118](https://github.com/docling-project/docling/issues/1118)) ([`c56ab3a`](https://github.com/docling-project/docling/commit/c56ab3a66b79e0d1b6f4b22880aabb7ee909d9d7))

### Documentation

* Enrichment models ([#1097](https://github.com/docling-project/docling/issues/1097)) ([`357d41c`](https://github.com/docling-project/docling/commit/357d41cc47bcf69684643d193ed38f9baaf0d2f1))

## [v2.25.1](https://github.com/docling-project/docling/releases/tag/v2.25.1) - 2025-03-03

### Fix

* Enable locks for threadsafe pdfium ([#1052](https://github.com/docling-project/docling/issues/1052)) ([`8dc0562`](https://github.com/docling-project/docling/commit/8dc0562542299cf972d14eeeb4393e50b589c8ad))
* **html:** Use 'start' attribute when parsing ordered lists from HTML docs ([#1062](https://github.com/docling-project/docling/issues/1062)) ([`de7b963`](https://github.com/docling-project/docling/commit/de7b963b09a34916f0a8d99649269aeb37db1408))

### Documentation

* Improve docs on token limit warning triggered by HybridChunker ([#1077](https://github.com/docling-project/docling/issues/1077)) ([`db3ceef`](https://github.com/docling-project/docling/commit/db3ceefd4ae6251a97e333bcb03051698b3fa71a))

## [v2.25.0](https://github.com/docling-project/docling/releases/tag/v2.25.0) - 2025-02-26

### Feature

* [Experimental] Introduce VLM pipeline using HF AutoModelForVision2Seq, featuring SmolDocling model ([#1054](https://github.com/docling-project/docling/issues/1054)) ([`3c9fe76`](https://github.com/docling-project/docling/commit/3c9fe76b706b7714b25d49cb09050c42e3b8c849))
* **cli:** Add option for downloading all models, refine help messages ([#1061](https://github.com/docling-project/docling/issues/1061)) ([`ab683e4`](https://github.com/docling-project/docling/commit/ab683e4fb6df4973d2efda04f00c269a2dc95f5b))

### Fix

* Vlm using artifacts path ([#1057](https://github.com/docling-project/docling/issues/1057)) ([`e197225`](https://github.com/docling-project/docling/commit/e1972257399151503d60b4806976c8b9b6911aa8))
* **html:** Parse text in div elements as TextItem ([#1041](https://github.com/docling-project/docling/issues/1041)) ([`1b0ead6`](https://github.com/docling-project/docling/commit/1b0ead69078030a0e4d25b51450ef2aa4a2e79fc))

### Documentation

* Extend chunking docs, add FAQ on token limit ([#1053](https://github.com/docling-project/docling/issues/1053)) ([`c84b973`](https://github.com/docling-project/docling/commit/c84b973959a254db22ac9a7dc8810628e4808a2d))

## [v2.24.0](https://github.com/docling-project/docling/releases/tag/v2.24.0) - 2025-02-20

### Feature

* Implement new reading-order model ([#916](https://github.com/docling-project/docling/issues/916)) ([`c93e369`](https://github.com/docling-project/docling/commit/c93e36988f1e1e461477223143c2c1fb2162d11f))

## [v2.23.1](https://github.com/docling-project/docling/releases/tag/v2.23.1) - 2025-02-20

### Fix

* Runtime error when Pandas Series is not always of string type ([#1024](https://github.com/docling-project/docling/issues/1024)) ([`6796f0a`](https://github.com/docling-project/docling/commit/6796f0a13263281cd48712b3c71579bfd81bb0d1))

### Documentation

* Revamp picture description example ([#1015](https://github.com/docling-project/docling/issues/1015)) ([`27c0400`](https://github.com/docling-project/docling/commit/27c04007bc1be7a6f6c90aaf04ea9f4ff8eb1f3d))

## [v2.23.0](https://github.com/docling-project/docling/releases/tag/v2.23.0) - 2025-02-17

### Feature

* Support cuda:n GPU device allocation ([#694](https://github.com/docling-project/docling/issues/694)) ([`77eb77b`](https://github.com/docling-project/docling/commit/77eb77bdc2c07b632a1d171826d1855a5218399e))
* **xml-jats:** Parse XML JATS documents ([#967](https://github.com/docling-project/docling/issues/967)) ([`428b656`](https://github.com/docling-project/docling/commit/428b656793cb75d108c69f20c254be7c198cee5c))

### Fix

* Revise DocTags, fix iterate_items to output content_layer in items ([#965](https://github.com/docling-project/docling/issues/965)) ([`6e75f0b`](https://github.com/docling-project/docling/commit/6e75f0b5d3ee42738a80049d4cf2fa6d34e8ab97))

## [v2.22.0](https://github.com/docling-project/docling/releases/tag/v2.22.0) - 2025-02-14

### Feature

* Add support for CSV input with new backend to transform CSV files to DoclingDocument ([#945](https://github.com/docling-project/docling/issues/945)) ([`00d9405`](https://github.com/docling-project/docling/commit/00d9405b0ac519d321ae54e8150f5facbaabbe14))
* Introduce the enable_remote_services option to allow remote connections while processing ([#941](https://github.com/docling-project/docling/issues/941)) ([`2716c7d`](https://github.com/docling-project/docling/commit/2716c7d4ffb836664178178d3f8d01b7f9112595))
* Allow artifacts_path to be defined as ENV ([#940](https://github.com/docling-project/docling/issues/940)) ([`5101e25`](https://github.com/docling-project/docling/commit/5101e2519e7a5bb727531b1412b1131a7cfbda52))

### Fix

* Update Pillow constraints ([#958](https://github.com/docling-project/docling/issues/958)) ([`af19c03`](https://github.com/docling-project/docling/commit/af19c03f6e5e0b24e12d6a3baac6c46a4c8b10d1))
* Fix the initialization of the TesseractOcrModel ([#935](https://github.com/docling-project/docling/issues/935)) ([`c47ae70`](https://github.com/docling-project/docling/commit/c47ae700ece2ea4efee17f82e4667c1ce9a0ed2a))

### Documentation

* Update example Dockerfile with download CLI ([#929](https://github.com/docling-project/docling/issues/929)) ([`7493d5b`](https://github.com/docling-project/docling/commit/7493d5b01f8be60294afeffdfb54a62bb74bcc92))
* Examples for picture descriptions ([#951](https://github.com/docling-project/docling/issues/951)) ([`2d66e99`](https://github.com/docling-project/docling/commit/2d66e99b69f39a282109c366fae3679f41c6e081))

## [v2.21.0](https://github.com/docling-project/docling/releases/tag/v2.21.0) - 2025-02-10

### Feature

* Add content_layer property to items to address body, furniture and other roles ([#735](https://github.com/docling-project/docling/issues/735)) ([`cf78d5b`](https://github.com/docling-project/docling/commit/cf78d5b7b9f12728270e673857fd299efc01a7db))

## [v2.20.0](https://github.com/docling-project/docling/releases/tag/v2.20.0) - 2025-02-07

### Feature

* Describe pictures using vision models ([#259](https://github.com/docling-project/docling/issues/259)) ([`4cc6e3e`](https://github.com/docling-project/docling/commit/4cc6e3ea5e858b367136acc729b723ea0552d22a))

### Fix

* Remove unused httpx ([#919](https://github.com/docling-project/docling/issues/919)) ([`c18f47c`](https://github.com/docling-project/docling/commit/c18f47c5c032c49bf3175aecd2236df37c0e9ae1))

## [v2.19.0](https://github.com/docling-project/docling/releases/tag/v2.19.0) - 2025-02-07

### Feature

* New artifacts path and CLI utility ([#876](https://github.com/docling-project/docling/issues/876)) ([`ed74fe2`](https://github.com/docling-project/docling/commit/ed74fe2ec0a702834f0deacfdb5717c8c587dab1))

### Fix

* **markdown:** Handle nested lists ([#910](https://github.com/docling-project/docling/issues/910)) ([`90b766e`](https://github.com/docling-project/docling/commit/90b766e2ae1695a759191df37c272efc09be5ee3))
* Test cases for RTL programmatic PDFs and fixes for the formula model ([#903](https://github.com/docling-project/docling/issues/903)) ([`9114ada`](https://github.com/docling-project/docling/commit/9114ada7bc4dd45ce0046de2f9d00a80ccb25c79))
* **msword_backend:** Handle conversion error in label parsing ([#896](https://github.com/docling-project/docling/issues/896)) ([`722a6eb`](https://github.com/docling-project/docling/commit/722a6eb7b994a0261312a356df80b2fced121812))
* Enrichment models batch size and expose picture classifier ([#878](https://github.com/docling-project/docling/issues/878)) ([`5ad6de0`](https://github.com/docling-project/docling/commit/5ad6de05600315617b574bd12af553e00b4d316e))

### Documentation

* Introduce example with custom models for RapidOCR ([#874](https://github.com/docling-project/docling/issues/874)) ([`6d3fea0`](https://github.com/docling-project/docling/commit/6d3fea019635bd6ca94bd36c3928b28c245d638d))

## [v2.18.0](https://github.com/docling-project/docling/releases/tag/v2.18.0) - 2025-02-03

### Feature

* Expose equation exports ([#869](https://github.com/docling-project/docling/issues/869)) ([`6a76b49`](https://github.com/docling-project/docling/commit/6a76b49a4756fd00503d0baec5db8d23be8207e8))
* Add option to define page range ([#852](https://github.com/docling-project/docling/issues/852)) ([`70d68b6`](https://github.com/docling-project/docling/commit/70d68b6164c6c7029b39dd65c5a278278768c381))
* **docx:** Support of SDTs in docx backend ([#853](https://github.com/docling-project/docling/issues/853)) ([`d727b04`](https://github.com/docling-project/docling/commit/d727b04ad080df0b3811902059e0fe0539f7037e))
* Python 3.13 support ([#841](https://github.com/docling-project/docling/issues/841)) ([`4df085a`](https://github.com/docling-project/docling/commit/4df085aa6c6f5cc043f4f7a9f0c1b4af43f95e8f))

### Fix

* **markdown:** Fix parsing if doc ending with table ([#873](https://github.com/docling-project/docling/issues/873)) ([`5ac2887`](https://github.com/docling-project/docling/commit/5ac2887e4ad52ed6e7147e3af1e3ee5eb0006a70))
* **markdown:** Add support for HTML content ([#855](https://github.com/docling-project/docling/issues/855)) ([`94751a7`](https://github.com/docling-project/docling/commit/94751a78f4f61b78f64952190717440ec6d84c62))
* **docx:** Merged table cells not properly converted ([#857](https://github.com/docling-project/docling/issues/857)) ([`0cd81a8`](https://github.com/docling-project/docling/commit/0cd81a81226c0d4aa4f20e4e58c3b33e4fe50ce0))
* Processing of placeholder shapes in pptx that have text but no bbox ([#868](https://github.com/docling-project/docling/issues/868)) ([`eff16b6`](https://github.com/docling-project/docling/commit/eff16b62ccdb0eb764eeacee550563898784dd6a))
* KeyError in tableformer prediction ([#854](https://github.com/docling-project/docling/issues/854)) ([`b1cf796`](https://github.com/docling-project/docling/commit/b1cf796730901222ad0882ff44efa0ef43a743ee))
* Fixed docx import with headers that are also lists ([#842](https://github.com/docling-project/docling/issues/842)) ([`2c037ae`](https://github.com/docling-project/docling/commit/2c037ae62e123967eddf065ccb2abbaf78cdcab3))
* Use new add_code in html backend and add more typing hints ([#850](https://github.com/docling-project/docling/issues/850)) ([`2a1f8af`](https://github.com/docling-project/docling/commit/2a1f8afe7e8d9d508aebcfd3998ee1625c938933))
* **markdown:** Fix empty block handling ([#843](https://github.com/docling-project/docling/issues/843)) ([`bccb022`](https://github.com/docling-project/docling/commit/bccb022fc82d4d0ef2ed2d8bea5f5d8e6400c1d9))
* Fix for the crash when encountering WMF images in pptx and docx ([#837](https://github.com/docling-project/docling/issues/837)) ([`fea0a99`](https://github.com/docling-project/docling/commit/fea0a99a95d97e72687f48f8174d31102655483e))

### Documentation

* Updated the readme with upcoming features ([#831](https://github.com/docling-project/docling/issues/831)) ([`d7c0828`](https://github.com/docling-project/docling/commit/d7c082894e3ef85881665d20167198adcbc1becd))
* Add example for inspection of picture content ([#624](https://github.com/docling-project/docling/issues/624)) ([`f9144f2`](https://github.com/docling-project/docling/commit/f9144f2bb6b322244c9d37683dca1e537ec6d781))

## [v2.17.0](https://github.com/docling-project/docling/releases/tag/v2.17.0) - 2025-01-28

### Feature

* **CLI:** Expose code and formula models in the CLI ([#820](https://github.com/docling-project/docling/issues/820)) ([`6882e6c`](https://github.com/docling-project/docling/commit/6882e6c38df30e4d4a1b83e01b13900ca7ea001f))
* Add platform info to CLI version printout ([#816](https://github.com/docling-project/docling/issues/816)) ([`95b293a`](https://github.com/docling-project/docling/commit/95b293a72356f94c7076e3649be970c8a51121a3))
* **ocr:** Expose `rec_keys_path` in RapidOcrOptions to support custom dictionaries ([#786](https://github.com/docling-project/docling/issues/786)) ([`5332755`](https://github.com/docling-project/docling/commit/53327552e83ced079ae50d8067ba7a8ce80cd9ad))
* Introduce automatic language detection in TesseractOcrCliModel ([#800](https://github.com/docling-project/docling/issues/800)) ([`3be2fb5`](https://github.com/docling-project/docling/commit/3be2fb581fe5a2ebd5cec9c86bb22eb1dec6fd0f))

### Fix

* Fix single newline handling in MD backend ([#824](https://github.com/docling-project/docling/issues/824)) ([`5aed9f8`](https://github.com/docling-project/docling/commit/5aed9f8aeba1624ba1a721e2ed3ba4aceaa7a482))
* Use file extension if filetype fails with PDF ([#827](https://github.com/docling-project/docling/issues/827)) ([`adf6353`](https://github.com/docling-project/docling/commit/adf635348365f82daa64e3f879076a7baf71edc0))
* Parse html with omitted body tag ([#818](https://github.com/docling-project/docling/issues/818)) ([`a112d7a`](https://github.com/docling-project/docling/commit/a112d7a03512e8a00842a100416426254d6ecfc0))

### Documentation

* Document Docling JSON parsing ([#819](https://github.com/docling-project/docling/issues/819)) ([`6875913`](https://github.com/docling-project/docling/commit/6875913e34abacb8d71b5d31543adbf7b5bd5e92))
* Add SSL verification error mitigation ([#821](https://github.com/docling-project/docling/issues/821)) ([`5139b48`](https://github.com/docling-project/docling/commit/5139b48e4e62bb061d956c132958ec2e6d88e40a))
* **backend XML:** Do not delete temp file in notebook ([#817](https://github.com/docling-project/docling/issues/817)) ([`4d41db3`](https://github.com/docling-project/docling/commit/4d41db3f7abb86c8c65386bf94e7eb0bf22bb82b))
* Typo ([#814](https://github.com/docling-project/docling/issues/814)) ([`8a4ec77`](https://github.com/docling-project/docling/commit/8a4ec77576b8a9fd60d0047939665d00cf93b4dd))
* Added markdown headings to enable TOC in github pages ([#808](https://github.com/docling-project/docling/issues/808)) ([`b885b2f`](https://github.com/docling-project/docling/commit/b885b2fa3c2519c399ed4b9a3dd4c2f6f62235d1))
* Description of supported formats and backends ([#788](https://github.com/docling-project/docling/issues/788)) ([`c2ae1cc`](https://github.com/docling-project/docling/commit/c2ae1cc4cab0f9e693c7ca460fe8afa5b515ee94))

## [v2.16.0](https://github.com/docling-project/docling/releases/tag/v2.16.0) - 2025-01-24

### Feature

* New document picture classifier ([#805](https://github.com/docling-project/docling/issues/805)) ([`16a218d`](https://github.com/docling-project/docling/commit/16a218d871c48fd9cc636b77f7b597dc40cbeeec))
* Add Docling JSON ingestion ([#783](https://github.com/docling-project/docling/issues/783)) ([`88a0e66`](https://github.com/docling-project/docling/commit/88a0e66adc19238f57a942b0504926cdaeacd8cc))
* Code and equation model for PDF and code blocks in markdown ([#752](https://github.com/docling-project/docling/issues/752)) ([`3213b24`](https://github.com/docling-project/docling/commit/3213b247ad6870ff984271f09f7720be68d9479b))
* Add "auto" language for TesseractOcr ([#759](https://github.com/docling-project/docling/issues/759)) ([`8543c22`](https://github.com/docling-project/docling/commit/8543c22687fee40459d393bf4adcfc059712de02))

### Fix

* Added extraction of byte-images in excel ([#804](https://github.com/docling-project/docling/issues/804)) ([`a458e29`](https://github.com/docling-project/docling/commit/a458e298ca64da2c6df29d953e95645525817bed))
* Update docling-parse-v2 backend version with new parsing fixes ([#769](https://github.com/docling-project/docling/issues/769)) ([`670a08b`](https://github.com/docling-project/docling/commit/670a08bdedda847ff3b6942bcaa1a2adef79afe2))

### Documentation

* Fix minor typos ([#801](https://github.com/docling-project/docling/issues/801)) ([`c58f75d`](https://github.com/docling-project/docling/commit/c58f75d0f75040e32820cc2915ec00755211c02f))
* Add Azure RAG example ([#675](https://github.com/docling-project/docling/issues/675)) ([`9020a93`](https://github.com/docling-project/docling/commit/9020a934be35b0798c972eb77a22fb62ce654ca5))
* Fix links between docs pages ([#697](https://github.com/docling-project/docling/issues/697)) ([`c49b352`](https://github.com/docling-project/docling/commit/c49b3526fb7b72e8007f785b1fcfdf58c2457756))
* Fix correct Accelerator pipeline options in docs/examples/custom_convert.py ([#733](https://github.com/docling-project/docling/issues/733)) ([`7686083`](https://github.com/docling-project/docling/commit/768608351d40376c3504546f52e967195536b3d5))
* Example to translate documents ([#739](https://github.com/docling-project/docling/issues/739)) ([`f7e1cbf`](https://github.com/docling-project/docling/commit/f7e1cbf629ae5f3e279296e72f656b7a453ab7a3))

## [v2.15.1](https://github.com/docling-project/docling/releases/tag/v2.15.1) - 2025-01-10

### Fix

* Improve OCR results, stricten criteria before dropping bitmap areas ([#719](https://github.com/docling-project/docling/issues/719)) ([`5a060f2`](https://github.com/docling-project/docling/commit/5a060f237d1decd0ff9db9e73478978419315778))
* Allow earlier requests versions ([#716](https://github.com/docling-project/docling/issues/716)) ([`e64b5a2`](https://github.com/docling-project/docling/commit/e64b5a2f628acc340a6d94ee6f1ada2aa267cecc))

### Documentation

* Add pointers to LangChain-side docs ([#718](https://github.com/docling-project/docling/issues/718)) ([`9a6b5c8`](https://github.com/docling-project/docling/commit/9a6b5c8c8debc81e0ddcbe91df6afbbeb29e97e6))
* Add LangChain docs ([#717](https://github.com/docling-project/docling/issues/717)) ([`4fa8028`](https://github.com/docling-project/docling/commit/4fa8028bd8120d7557e1d45ba31e200e130af698))

## [v2.15.0](https://github.com/docling-project/docling/releases/tag/v2.15.0) - 2025-01-08

### Feature

* Added http header support for document converter and cli ([#642](https://github.com/docling-project/docling/issues/642)) ([`0ee849e`](https://github.com/docling-project/docling/commit/0ee849e8bc8cf24d1c5597af3fe20a7fa19a29e0))

### Fix

* Correct scaling of debug visualizations, tune OCR ([#700](https://github.com/docling-project/docling/issues/700)) ([`5cb4cf6`](https://github.com/docling-project/docling/commit/5cb4cf6f19f91e6c87141e93400c4b54b93aa5d7))
* Let BeautifulSoup detect the HTML encoding ([#695](https://github.com/docling-project/docling/issues/695)) ([`42856fd`](https://github.com/docling-project/docling/commit/42856fdf79559188ec4617bc5d3a007286f114d2))
* **mspowerpoint:** Handle invalid images in PowerPoint slides ([#650](https://github.com/docling-project/docling/issues/650)) ([`d49650c`](https://github.com/docling-project/docling/commit/d49650c54ffa60bc6d6106970e104071689bc7b0))

### Documentation

* Specify docstring types ([#702](https://github.com/docling-project/docling/issues/702)) ([`ead396a`](https://github.com/docling-project/docling/commit/ead396ab407f6bbd43176abd6ed2bed7ed8c7c43))
* Add link to rag with granite ([#698](https://github.com/docling-project/docling/issues/698)) ([`6701f34`](https://github.com/docling-project/docling/commit/6701f34c855992c52918b210c65a2edb1c827c01))
* Add integrations, revamp docs ([#693](https://github.com/docling-project/docling/issues/693)) ([`2d24fae`](https://github.com/docling-project/docling/commit/2d24faecd96bfa656b2b8c80f25cdf251a50526a))
* Add OpenContracts as an integration ([#679](https://github.com/docling-project/docling/issues/679)) ([`569038d`](https://github.com/docling-project/docling/commit/569038df4205703f87517ea58da7902d143e7699))
* Add Weaviate RAG recipe notebook ([#451](https://github.com/docling-project/docling/issues/451)) ([`2b591f9`](https://github.com/docling-project/docling/commit/2b591f98726ed0d883236dd0550201b95203eebb))
* Document Haystack & Vectara support ([#628](https://github.com/docling-project/docling/issues/628)) ([`fc645ea`](https://github.com/docling-project/docling/commit/fc645ea531ddc67959640b428007851d641c923e))

## [v2.14.0](https://github.com/docling-project/docling/releases/tag/v2.14.0) - 2024-12-18

### Feature

* Create a backend to transform PubMed XML files to DoclingDocument ([#557](https://github.com/docling-project/docling/issues/557)) ([`fd03480`](https://github.com/docling-project/docling/commit/fd034802b65a0e567531b8ecc9a283aaf030e050))

## [v2.13.0](https://github.com/docling-project/docling/releases/tag/v2.13.0) - 2024-12-17

### Feature

* Updated Layout processing with forms and key-value areas ([#530](https://github.com/docling-project/docling/issues/530)) ([`60dc852`](https://github.com/docling-project/docling/commit/60dc852f16dc1adbb5e9284c81a146043a301ec1))
* Create a backend to parse USPTO patents into DoclingDocument ([#606](https://github.com/docling-project/docling/issues/606)) ([`4e08750`](https://github.com/docling-project/docling/commit/4e087504cc4b04210574e69f616badcddfa1f8e5))
* Add Easyocr parameter recog_network ([#613](https://github.com/docling-project/docling/issues/613)) ([`3b53bd3`](https://github.com/docling-project/docling/commit/3b53bd38c8efcc5ba54421fbfa90d047f1a61f82))

### Documentation

* Add Haystack RAG example ([#615](https://github.com/docling-project/docling/issues/615)) ([`3e599c7`](https://github.com/docling-project/docling/commit/3e599c7bbeef211dc346e9bc1d3a249113fcc4e4))
* Fix the path to the run_with_accelerator.py example ([#608](https://github.com/docling-project/docling/issues/608)) ([`3bb3bf5`](https://github.com/docling-project/docling/commit/3bb3bf57150c9705a055982e6fb0cc8d1408f161))

## [v2.12.0](https://github.com/docling-project/docling/releases/tag/v2.12.0) - 2024-12-13

### Feature

* Introduce support for GPU Accelerators ([#593](https://github.com/docling-project/docling/issues/593)) ([`19fad92`](https://github.com/docling-project/docling/commit/19fad9261cb61f732a0426393866c8c1a9efbf4f))

## [v2.11.0](https://github.com/docling-project/docling/releases/tag/v2.11.0) - 2024-12-12

### Feature

* Add timeout limit to document parsing job. DS4SD#270 ([#552](https://github.com/docling-project/docling/issues/552)) ([`3da166e`](https://github.com/docling-project/docling/commit/3da166eafa3c119de961510341cb92397652c222))

### Fix

* Do not import python modules from deepsearch-glm ([#569](https://github.com/docling-project/docling/issues/569)) ([`aee9c0b`](https://github.com/docling-project/docling/commit/aee9c0b324a07190ad03ad3a6266e76c465d4cdf))
* Handle no result from RapidOcr reader ([#558](https://github.com/docling-project/docling/issues/558)) ([`f45499c`](https://github.com/docling-project/docling/commit/f45499ce9349fe55538dfb36d74c395e9193d9b1))
* Make enum serializable with human-readable value ([#555](https://github.com/docling-project/docling/issues/555)) ([`a7df337`](https://github.com/docling-project/docling/commit/a7df337654fa5fa7633af8740fb5e4cc4a06f250))

### Documentation

* Update chunking usage docs, minor reorg ([#550](https://github.com/docling-project/docling/issues/550)) ([`d0c9e8e`](https://github.com/docling-project/docling/commit/d0c9e8e508d7edef5e733be6cdea2cea0a9a0695))

## [v2.10.0](https://github.com/docling-project/docling/releases/tag/v2.10.0) - 2024-12-09

### Feature

* Docling-parse v2 as default PDF backend ([#549](https://github.com/docling-project/docling/issues/549)) ([`aca57f0`](https://github.com/docling-project/docling/commit/aca57f0527dddcc027dc1ee840e2e492ab997170))

### Fix

* Call into docling-core for legacy document transform ([#551](https://github.com/docling-project/docling/issues/551)) ([`7972d47`](https://github.com/docling-project/docling/commit/7972d47f88604f02d6a32527116c4d78eb1005e2))
* Introduce Image format options in CLI. Silence the tqdm downloading messages. ([#544](https://github.com/docling-project/docling/issues/544)) ([`78f61a8`](https://github.com/docling-project/docling/commit/78f61a8522d3a19ecc1d605e8441fb543ca0fa96))

## [v2.9.0](https://github.com/docling-project/docling/releases/tag/v2.9.0) - 2024-12-09

### Feature

* Expose new hybrid chunker, update docs ([#384](https://github.com/docling-project/docling/issues/384)) ([`c8ecdd9`](https://github.com/docling-project/docling/commit/c8ecdd987e80227db3850ea729ecb36d2b609040))
* **MS Word backend:** Make detection of headers and other styles localization agnostic ([#534](https://github.com/docling-project/docling/issues/534)) ([`3e073df`](https://github.com/docling-project/docling/commit/3e073dfbebbc65f995d4df946c1650699a26782c))

### Fix

* Correcting DefaultText ID for MS Word backend ([#537](https://github.com/docling-project/docling/issues/537)) ([`eb7ffcd`](https://github.com/docling-project/docling/commit/eb7ffcdd1cda1caa8ec8ba2fc313ff1e7d9acd4f))
* Add `py.typed` marker file ([#531](https://github.com/docling-project/docling/issues/531)) ([`9102fe1`](https://github.com/docling-project/docling/commit/9102fe1adcd43432e5fb3f35af704b7442c5d633))
* Enable HTML export in CLI and add options for image mode ([#513](https://github.com/docling-project/docling/issues/513)) ([`0d11e30`](https://github.com/docling-project/docling/commit/0d11e30dd813020c0189de849cd7b2e285d08694))
* Missing text in docx (t tag) when embedded in a table ([#528](https://github.com/docling-project/docling/issues/528)) ([`b730b2d`](https://github.com/docling-project/docling/commit/b730b2d7a04a8773a00ed88889d28b0c476ba052))
* Restore pydantic version pin after fixes ([#512](https://github.com/docling-project/docling/issues/512)) ([`c830b92`](https://github.com/docling-project/docling/commit/c830b92b2e043ea63d216f65b3f9d88d2a8c33f7))
* Folder input in cli ([#511](https://github.com/docling-project/docling/issues/511)) ([`8ada0bc`](https://github.com/docling-project/docling/commit/8ada0bccc744df94f755adf71cf8b163e6304375))

### Documentation

* Document new integrations ([#532](https://github.com/docling-project/docling/issues/532)) ([`e780333`](https://github.com/docling-project/docling/commit/e7803334409a343a59c536c529a03d6f5cdbfe15))

## [v2.8.3](https://github.com/docling-project/docling/releases/tag/v2.8.3) - 2024-12-03

### Fix

* Improve handling of disallowed formats ([#429](https://github.com/docling-project/docling/issues/429)) ([`34c7c79`](https://github.com/docling-project/docling/commit/34c7c798580476a86ce8abec30b1115fbb36fdd8))

## [v2.8.2](https://github.com/docling-project/docling/releases/tag/v2.8.2) - 2024-12-03

### Fix

* ParserError EOF inside string (#470) ([#472](https://github.com/docling-project/docling/issues/472)) ([`c90c41c`](https://github.com/docling-project/docling/commit/c90c41c391de4366db554d7a71ce9a35467c981e))
* PermissionError when using tesseract_ocr_cli_model ([#496](https://github.com/docling-project/docling/issues/496)) ([`d3f84b2`](https://github.com/docling-project/docling/commit/d3f84b2457125feacd0c21d6513e7ae69a308ea5))

### Documentation

* Add styling for faq ([#502](https://github.com/docling-project/docling/issues/502)) ([`5ba3807`](https://github.com/docling-project/docling/commit/5ba3807f315a01b1a4e8df9bab40e34a4238205a))
* Typo in faq ([#484](https://github.com/docling-project/docling/issues/484)) ([`33cff98`](https://github.com/docling-project/docling/commit/33cff98d360c02a382a66850c696a0cf511659ac))
* Add automatic api reference ([#475](https://github.com/docling-project/docling/issues/475)) ([`d487210`](https://github.com/docling-project/docling/commit/d4872103b8f24e38b37a8cd3ac414d3e02e7d6e8))
* Introduce faq section ([#468](https://github.com/docling-project/docling/issues/468)) ([`8ccb3c6`](https://github.com/docling-project/docling/commit/8ccb3c6db69318789af7deec26cfa2a3fd71302e))

### Performance

* Prevent temp file leftovers, reuse core type ([#487](https://github.com/docling-project/docling/issues/487)) ([`051789d`](https://github.com/docling-project/docling/commit/051789d01706d3823dd6307eca4dc5faacd1b7ce))

## [v2.8.1](https://github.com/docling-project/docling/releases/tag/v2.8.1) - 2024-11-29

### Fix

* **cli:** Expose debug options ([#467](https://github.com/docling-project/docling/issues/467)) ([`dd8de46`](https://github.com/docling-project/docling/commit/dd8de462676993b81926610fd573d51d3272cbaf))
* Remove unused deps ([#466](https://github.com/docling-project/docling/issues/466)) ([`af63818`](https://github.com/docling-project/docling/commit/af63818df5636c4cbe77c0a01e6dcc0d47c4bfdb))

### Documentation

* Extend integration docs & README ([#456](https://github.com/docling-project/docling/issues/456)) ([`84c46fd`](https://github.com/docling-project/docling/commit/84c46fdeb344502edf9647c610c4828ab0ffb9dd))

## [v2.8.0](https://github.com/docling-project/docling/releases/tag/v2.8.0) - 2024-11-27

### Feature

* **ocr:** Added support for RapidOCR engine ([#415](https://github.com/docling-project/docling/issues/415)) ([`85b2999`](https://github.com/docling-project/docling/commit/85b29990be6468516b6dbe49f880d9f1f4c11c5a))

### Fix

* Use correct image index in word backend ([#442](https://github.com/docling-project/docling/issues/442)) ([`767563b`](https://github.com/docling-project/docling/commit/767563bf8b331304892285c0789bba481acaa1b5))
* Update tests and examples for docling-core 2.5.1 ([#449](https://github.com/docling-project/docling/issues/449)) ([`29807a2`](https://github.com/docling-project/docling/commit/29807a2d687896c67ada934c6a626401f5930e50))

## [v2.7.1](https://github.com/docling-project/docling/releases/tag/v2.7.1) - 2024-11-26

### Fix

* Fixes for wordx ([#432](https://github.com/docling-project/docling/issues/432)) ([`d0a1180`](https://github.com/docling-project/docling/commit/d0a118047804765b1b8532e72e08272e678c0c93))
* Force pydantic < 2.10.0 ([#407](https://github.com/docling-project/docling/issues/407)) ([`d7072b4`](https://github.com/docling-project/docling/commit/d7072b4b56227756eb2c7abd3a6e7387eeefe7c1))

### Documentation

* Add DocETL, Kotaemon, spaCy integrations; minor docs improvements ([#408](https://github.com/docling-project/docling/issues/408)) ([`7a45b92`](https://github.com/docling-project/docling/commit/7a45b92078b3a9fdd8f0650002eddc03e9d780af))

## [v2.7.0](https://github.com/docling-project/docling/releases/tag/v2.7.0) - 2024-11-20

### Feature

* Add support for `ocrmac` OCR engine on macOS ([#276](https://github.com/docling-project/docling/issues/276)) ([`6efa96c`](https://github.com/docling-project/docling/commit/6efa96c983fc509b2c7b35a4a25a714284f2f782))

### Fix

* Python3.9 support ([#396](https://github.com/docling-project/docling/issues/396)) ([`7b013ab`](https://github.com/docling-project/docling/commit/7b013abcf31ba49e2141dfd408bc8c23e8d87d91))
* Propagate document limits to converter ([#388](https://github.com/docling-project/docling/issues/388)) ([`32ebf55`](https://github.com/docling-project/docling/commit/32ebf55e3338dd22f9a23c55595f15835794d961))

## [v2.6.0](https://github.com/docling-project/docling/releases/tag/v2.6.0) - 2024-11-19

### Feature

* Added support for exporting DocItem to an image when page image is available ([#379](https://github.com/docling-project/docling/issues/379)) ([`3f91e7d`](https://github.com/docling-project/docling/commit/3f91e7d3f166901c139ab036c4d9dad5fa560aa9))
* Expose ocr-lang in CLI ([#375](https://github.com/docling-project/docling/issues/375)) ([`ed785ea`](https://github.com/docling-project/docling/commit/ed785ea122d8d736c2031a38fce81dc5c19e244c))
* Added excel backend ([#334](https://github.com/docling-project/docling/issues/334)) ([`926dfd2`](https://github.com/docling-project/docling/commit/926dfd29d51c52628fe9fe8acb0ad0121c88e58a))
* Extracting picture data for raster images found in PPTX ([#349](https://github.com/docling-project/docling/issues/349)) ([`7a97d71`](https://github.com/docling-project/docling/commit/7a97d7119f69a83042477d4272e8ef93a2252cc8))

### Fix

* Fixing images in the input Word files ([#330](https://github.com/docling-project/docling/issues/330)) ([`8533039`](https://github.com/docling-project/docling/commit/8533039b0c0eff131b524da765f15c3279b554c5))
* Reduce logging by keeping option for more verbose ([#323](https://github.com/docling-project/docling/issues/323)) ([`8b437ad`](https://github.com/docling-project/docling/commit/8b437adcde4acc1d309c81c707c264bcca05d394))

### Documentation

* Fixed typo in v2 example v2 ([#378](https://github.com/docling-project/docling/issues/378)) ([`911c3bd`](https://github.com/docling-project/docling/commit/911c3bda27c4108167b89fa70ec8204c854c583b))
* Add automatic generation of CLI reference ([#325](https://github.com/docling-project/docling/issues/325)) ([`ca8524e`](https://github.com/docling-project/docling/commit/ca8524ecaea93cca0c808c8e7dee29fda0c1977e))
* Add architecture outline ([#341](https://github.com/docling-project/docling/issues/341)) ([`25fd149`](https://github.com/docling-project/docling/commit/25fd149c3839343f8bd42ae993e35f80acda2a52))
* Fix parameter in usage.md ([#332](https://github.com/docling-project/docling/issues/332)) ([`835e077`](https://github.com/docling-project/docling/commit/835e077b021d0a3615247906dd82ecfa19f3cf98))

## [v2.5.2](https://github.com/docling-project/docling/releases/tag/v2.5.2) - 2024-11-13

### Fix

* Skip glm model downloads ([#322](https://github.com/docling-project/docling/issues/322)) ([`c9341bf`](https://github.com/docling-project/docling/commit/c9341bf22e08920284cbc14821c190eaf6abf8a6))

## [v2.5.1](https://github.com/docling-project/docling/releases/tag/v2.5.1) - 2024-11-12

### Fix

* Handling of single-cell tables in DOCX backend ([#314](https://github.com/docling-project/docling/issues/314)) ([`fb8ba86`](https://github.com/docling-project/docling/commit/fb8ba861e28eda0079daa44fb1ea3ed17745f1d2))

### Documentation

* Hybrid RAG with Qdrant ([#312](https://github.com/docling-project/docling/issues/312)) ([`7f5d35e`](https://github.com/docling-project/docling/commit/7f5d35ea3c225ce1ce7328825842f98755c0104f))
* Add Data Prep Kit integration ([#316](https://github.com/docling-project/docling/issues/316)) ([`93fc1be`](https://github.com/docling-project/docling/commit/93fc1be61abfe0669daf26c0984a51ec8675bf62))

## [v2.5.0](https://github.com/docling-project/docling/releases/tag/v2.5.0) - 2024-11-12

### Feature

* **OCR:** Introduce the OcrOptions.force_full_page_ocr parameter that forces a full page OCR scanning ([#290](https://github.com/docling-project/docling/issues/290)) ([`c6b3763`](https://github.com/docling-project/docling/commit/c6b3763ecb6ef862840a30978ee177b907f86505))

### Fix

* Configure env prefix for docling settings ([#315](https://github.com/docling-project/docling/issues/315)) ([`5d4a10b`](https://github.com/docling-project/docling/commit/5d4a10b121317fa481208dacbee47032b08ff928))
* Added handling of grouped elements in pptx backend ([#307](https://github.com/docling-project/docling/issues/307)) ([`81c8243`](https://github.com/docling-project/docling/commit/81c8243a8bf177feed8f87ea283b5bb6836350cb))
* Allow mps usage for easyocr ([#286](https://github.com/docling-project/docling/issues/286)) ([`97f214e`](https://github.com/docling-project/docling/commit/97f214efddcf66f0734a95c17c08936f6111d113))

### Documentation

* Add navigation indices ([#305](https://github.com/docling-project/docling/issues/305)) ([`1239ade`](https://github.com/docling-project/docling/commit/1239ade2750349d13d4e865d88449b232bbad944))

## [v2.4.2](https://github.com/docling-project/docling/releases/tag/v2.4.2) - 2024-11-08

### Fix

* **EasyOcrModel:** Support the use_gpu pipeline parameter in EasyOcrModel. Initialize easyocr ([#282](https://github.com/docling-project/docling/issues/282)) ([`0eb065e`](https://github.com/docling-project/docling/commit/0eb065e9b6e4619d4c412ed98bc7408915ca3f95))

## [v2.4.1](https://github.com/docling-project/docling/releases/tag/v2.4.1) - 2024-11-08

### Fix

* **tesserocr:** Raise Exception if tesserocr has not loaded any languages ([#279](https://github.com/docling-project/docling/issues/279)) ([`704d792`](https://github.com/docling-project/docling/commit/704d792a7997c4ca34f9f9045ed4ae02b4f5df47))
* Dockerfile example copy command ([#234](https://github.com/docling-project/docling/issues/234)) ([`90836db`](https://github.com/docling-project/docling/commit/90836db90accf4a66c9c20544c98452840e3a308))

### Documentation

* Update badges & credits ([#248](https://github.com/docling-project/docling/issues/248)) ([`a84ec27`](https://github.com/docling-project/docling/commit/a84ec276b0997c4ba9b32e18e911a966124dc3bc))
* Add coming-soon section ([#235](https://github.com/docling-project/docling/issues/235)) ([`5ce02c5`](https://github.com/docling-project/docling/commit/5ce02c5c598a2efa615ad15f0ead8d752d3ad2ea))
* Add artifacts-path param to CLI ([#233](https://github.com/docling-project/docling/issues/233)) ([`d5e65ae`](https://github.com/docling-project/docling/commit/d5e65aedac23d6849c805a0e88dd06f2a285eb18))

## [v2.4.0](https://github.com/docling-project/docling/releases/tag/v2.4.0) - 2024-11-04

### Feature

* Pdf backend, table mode as options and artifacts path ([#203](https://github.com/docling-project/docling/issues/203)) ([`40ad987`](https://github.com/docling-project/docling/commit/40ad98730356218359d6fa9b3deb5bc094d6c699))

### Documentation

* Add explicit artifacts path example ([#224](https://github.com/docling-project/docling/issues/224)) ([`eeee3b4`](https://github.com/docling-project/docling/commit/eeee3b4371cb8207a8e7a887acba3fc5afc6de4d))
* Update custom convert and dockerfile ([#226](https://github.com/docling-project/docling/issues/226)) ([`5f5fea9`](https://github.com/docling-project/docling/commit/5f5fea90a963f73a92b551dfefb353fa3e9657d7))
* Correct spelling of 'individual' ([#219](https://github.com/docling-project/docling/issues/219)) ([`41acaa9`](https://github.com/docling-project/docling/commit/41acaa9e2ef4cff8d781f79fb5ae1b31762fa644))
* Update LlamaIndex docs ([#196](https://github.com/docling-project/docling/issues/196)) ([`244ca69`](https://github.com/docling-project/docling/commit/244ca69cfd8a17b449a0a6baaf062b0b5b798bb1))

## [v2.3.1](https://github.com/docling-project/docling/releases/tag/v2.3.1) - 2024-10-30

### Fix

* Simplify torch dependencies and update pinned docling deps ([#190](https://github.com/docling-project/docling/issues/190)) ([`eb679cc`](https://github.com/docling-project/docling/commit/eb679ccbb484fc3ef50dcf00be54ccd488d4a34d))
* Allow to explicitly initialize the pipeline ([#189](https://github.com/docling-project/docling/issues/189)) ([`904d24d`](https://github.com/docling-project/docling/commit/904d24d6005d113c50bde0ad398fdaafbbfb3027))

## [v2.3.0](https://github.com/docling-project/docling/releases/tag/v2.3.0) - 2024-10-30

### Feature

* Add pipeline timings and toggle visualization, establish debug settings ([#183](https://github.com/docling-project/docling/issues/183)) ([`2a2c65b`](https://github.com/docling-project/docling/commit/2a2c65bf4f89a715c27310eaa9cd9db635e0f673))

### Fix

* Fix duplicate title and heading + add e2e tests for html and docx ([#186](https://github.com/docling-project/docling/issues/186)) ([`f542460`](https://github.com/docling-project/docling/commit/f542460af3c7676e5f8dee3b6ce729b139560cd6))

## [v2.2.1](https://github.com/docling-project/docling/releases/tag/v2.2.1) - 2024-10-28

### Fix

* Fix header levels for DOCX & HTML ([#184](https://github.com/docling-project/docling/issues/184)) ([`b9f5c74`](https://github.com/docling-project/docling/commit/b9f5c74a7d13827c2b7887ddbf0b4eb43edd0846))
* Handling of long sequence of unescaped underscore chars in markdown ([#173](https://github.com/docling-project/docling/issues/173)) ([`94d0729`](https://github.com/docling-project/docling/commit/94d0729c500b0be8ac4a1cd3025b42048f6e8d5a))
* HTML backend, fixes for Lists and nested texts ([#180](https://github.com/docling-project/docling/issues/180)) ([`7d19418`](https://github.com/docling-project/docling/commit/7d19418b779408c345473af684de6b7f60872b6e))
* MD Backend, fixes to properly handle trailing inline text and emphasis in headers ([#178](https://github.com/docling-project/docling/issues/178)) ([`88c1673`](https://github.com/docling-project/docling/commit/88c16730571afdd3bfb8894f64d41b5e99bc5a5b))

### Documentation

* Update LlamaIndex docs for Docling v2 ([#182](https://github.com/docling-project/docling/issues/182)) ([`2cece27`](https://github.com/docling-project/docling/commit/2cece27208c4bce715d20000b845794dfb97843d))
* Fix batch convert ([#177](https://github.com/docling-project/docling/issues/177)) ([`189d3c2`](https://github.com/docling-project/docling/commit/189d3c2d44ec389856f48696eaa78ac9f02f8cde))
* Add export with embedded images ([#175](https://github.com/docling-project/docling/issues/175)) ([`8d356aa`](https://github.com/docling-project/docling/commit/8d356aa24715433d458eff4f5f0937ff5cb9cc69))

## [v2.2.0](https://github.com/docling-project/docling/releases/tag/v2.2.0) - 2024-10-23

### Feature

* Update to docling-parse v2 without history ([#170](https://github.com/docling-project/docling/issues/170)) ([`4116819`](https://github.com/docling-project/docling/commit/4116819b515a0611e8e5bf2bb0e1e39f1096b7bf))
* Support AsciiDoc and Markdown input format ([#168](https://github.com/docling-project/docling/issues/168)) ([`3023f18`](https://github.com/docling-project/docling/commit/3023f18ba0462a224f75ea40953b5605abef6427))

### Fix

* Set valid=false for invalid backends ([#171](https://github.com/docling-project/docling/issues/171)) ([`3496b48`](https://github.com/docling-project/docling/commit/3496b4838fd52cb0d74eadf78b27c19f633871b1))

## [v2.1.0](https://github.com/docling-project/docling/releases/tag/v2.1.0) - 2024-10-18

### Feature

* Add coverage_threshold to skip OCR for small images ([#161](https://github.com/docling-project/docling/issues/161)) ([`b346faf`](https://github.com/docling-project/docling/commit/b346faf622190c4895dffdc1ee2365b3f7808cbc))

### Fix

* Fix legacy doc ref ([#162](https://github.com/docling-project/docling/issues/162)) ([`63bef59`](https://github.com/docling-project/docling/commit/63bef59d9ed6cfd937aefd36a4ef38a54a10dac5))

### Documentation

* Typo fix ([#155](https://github.com/docling-project/docling/issues/155)) ([`f799e77`](https://github.com/docling-project/docling/commit/f799e777c1248559eb2f84bc334e392cd3c98e49))
* Add graphical band in readme ([#154](https://github.com/docling-project/docling/issues/154)) ([`034a411`](https://github.com/docling-project/docling/commit/034a4110573df3ac88fd623970958f02309dd6da))
* Add use docling ([#150](https://github.com/docling-project/docling/issues/150)) ([`61c092f`](https://github.com/docling-project/docling/commit/61c092f445ccde8ed5d7c0f2fa91a3d19a1f7a1e))

## [v2.0.0](https://github.com/docling-project/docling/releases/tag/v2.0.0) - 2024-10-16

### Feature

* Docling v2 ([#117](https://github.com/docling-project/docling/issues/117)) ([`7d3be0e`](https://github.com/docling-project/docling/commit/7d3be0edebb420f5840499aa04e4ab928d33cda2))

### Breaking

* Docling v2 ([#117](https://github.com/docling-project/docling/issues/117)) ([`7d3be0e`](https://github.com/docling-project/docling/commit/7d3be0edebb420f5840499aa04e4ab928d33cda2))

### Documentation

* Introduce docs site ([#141](https://github.com/docling-project/docling/issues/141)) ([`d504432`](https://github.com/docling-project/docling/commit/d504432c1ee250ea417e8239ff5c16c5bec5a2c7))

## [v1.20.0](https://github.com/docling-project/docling/releases/tag/v1.20.0) - 2024-10-11

### Feature

* New experimental docling-parse v2 backend ([#131](https://github.com/docling-project/docling/issues/131)) ([`5e4944f`](https://github.com/docling-project/docling/commit/5e4944f15f0ac1faf3e6a532c3e3ab4da56517a3))

## [v1.19.1](https://github.com/docling-project/docling/releases/tag/v1.19.1) - 2024-10-11

### Fix

* Remove stderr from tesseract cli and introduce fuzziness in the text validation of OCR tests ([#138](https://github.com/docling-project/docling/issues/138)) ([`dae2a3b`](https://github.com/docling-project/docling/commit/dae2a3b66732e1e135b00cce24226c7d9f2eb2e4))

### Documentation

* Simplify LlamaIndex example using Docling extension ([#135](https://github.com/docling-project/docling/issues/135)) ([`5f1bd9e`](https://github.com/docling-project/docling/commit/5f1bd9e9c8a19c667d1d587a557c3c36df494762))

## [v1.19.0](https://github.com/docling-project/docling/releases/tag/v1.19.0) - 2024-10-08

### Feature

* Add options for choosing OCR engines ([#118](https://github.com/docling-project/docling/issues/118)) ([`f96ea86`](https://github.com/docling-project/docling/commit/f96ea86a00fd1aafaa57025e46b5288b43958725))

## [v1.18.0](https://github.com/docling-project/docling/releases/tag/v1.18.0) - 2024-10-03

### Feature

* New torch-based docling models ([#120](https://github.com/docling-project/docling/issues/120)) ([`2422f70`](https://github.com/docling-project/docling/commit/2422f706a1b02a679bcbaaba097fef2f69aba0f4))

## [v1.17.0](https://github.com/docling-project/docling/releases/tag/v1.17.0) - 2024-10-03

### Feature

* Windows support ([#122](https://github.com/docling-project/docling/issues/122)) ([`d44c62d`](https://github.com/docling-project/docling/commit/d44c62d7ce6990bbc78bf53315dd76d35d1f6c2e))

## [v1.16.1](https://github.com/docling-project/docling/releases/tag/v1.16.1) - 2024-09-27

### Fix

* Allow usage of opencv 4.6.x ([#110](https://github.com/docling-project/docling/issues/110)) ([`34bd887`](https://github.com/docling-project/docling/commit/34bd887a7f9c11b2b051bdb4707dfdc5f43e6ad5))

### Documentation

* Document chunking ([#111](https://github.com/docling-project/docling/issues/111)) ([`c05b692`](https://github.com/docling-project/docling/commit/c05b692d69b6dae1ac5f518e84b17f32e7d94372))

## [v1.16.0](https://github.com/docling-project/docling/releases/tag/v1.16.0) - 2024-09-27

### Feature

* Support tableformer model choice ([#90](https://github.com/docling-project/docling/issues/90)) ([`d6df76f`](https://github.com/docling-project/docling/commit/d6df76f90b249bf48a509b63fa18f570be39482e))

## [v1.15.0](https://github.com/docling-project/docling/releases/tag/v1.15.0) - 2024-09-24

### Feature

* Add figure in markdown ([#98](https://github.com/docling-project/docling/issues/98)) ([`6a03c20`](https://github.com/docling-project/docling/commit/6a03c208ecc9176b0be413594114ce8a3f213371))

## [v1.14.0](https://github.com/docling-project/docling/releases/tag/v1.14.0) - 2024-09-24

### Feature

* Add URL support to CLI ([#99](https://github.com/docling-project/docling/issues/99)) ([`3c46e42`](https://github.com/docling-project/docling/commit/3c46e4266cf1ad8d3a99aa33636d84d34222b4fe))

### Fix

* Fix OCR setting for pypdfium, minor refactor ([#102](https://github.com/docling-project/docling/issues/102)) ([`d96b96c`](https://github.com/docling-project/docling/commit/d96b96c8481a8ae68545a34aaf9b8d5a6637a6be))

### Documentation

* Document CLI, minor README revamp ([#100](https://github.com/docling-project/docling/issues/100)) ([`f8f2303`](https://github.com/docling-project/docling/commit/f8f2303348c4bbcb7903ff172746a69607e53271))

## [v1.13.1](https://github.com/docling-project/docling/releases/tag/v1.13.1) - 2024-09-23

### Fix

* Updated the render_as_doctags with the new arguments from docling-core ([#93](https://github.com/docling-project/docling/issues/93)) ([`4794ce4`](https://github.com/docling-project/docling/commit/4794ce460a542a730fd5a72a7be7f94a07ed5d12))

## [v1.13.0](https://github.com/docling-project/docling/releases/tag/v1.13.0) - 2024-09-18

### Feature

* Add table exports ([#86](https://github.com/docling-project/docling/issues/86)) ([`f19bd43`](https://github.com/docling-project/docling/commit/f19bd437984f77067d33d591e25c5d5c92d7e0a9))

### Fix

* Bumped the glm version and adjusted the tests ([#83](https://github.com/docling-project/docling/issues/83)) ([`442443a`](https://github.com/docling-project/docling/commit/442443a102d91b19a7eb38b316dada89c86ea8a8))

### Documentation

* Updated Docling logo.png with transparent background ([#88](https://github.com/docling-project/docling/issues/88)) ([`0da7519`](https://github.com/docling-project/docling/commit/0da75198967c9cffd42be3f3acd6ade2341fc1f5))

## [v1.12.2](https://github.com/docling-project/docling/releases/tag/v1.12.2) - 2024-09-17

### Fix

* **tests:** Adjust the test data to match the new version of LayoutPredictor ([#82](https://github.com/docling-project/docling/issues/82)) ([`fa9699f`](https://github.com/docling-project/docling/commit/fa9699fa3cd2d367382d7b952d0365983a870848))

## [v1.12.1](https://github.com/docling-project/docling/releases/tag/v1.12.1) - 2024-09-16

### Fix

* CLI compatibility with python 3.10 and 3.11 ([#79](https://github.com/docling-project/docling/issues/79)) ([`2870fdc`](https://github.com/docling-project/docling/commit/2870fdc857d02efeb8f1de7852e9577dd3eb2f51))

## [v1.12.0](https://github.com/docling-project/docling/releases/tag/v1.12.0) - 2024-09-13

### Feature

* Add docling cli ([#75](https://github.com/docling-project/docling/issues/75)) ([`9899078`](https://github.com/docling-project/docling/commit/98990784dfa6009b72ee2e1508948b22b00245ec))

### Documentation

* Showcase RAG with LlamaIndex and LangChain ([#71](https://github.com/docling-project/docling/issues/71)) ([`53569a1`](https://github.com/docling-project/docling/commit/53569a10238a821dfbbfcef9d2376d179e62a1db))

## [v1.11.0](https://github.com/docling-project/docling/releases/tag/v1.11.0) - 2024-09-10

### Feature

* Adding txt and doctags output ([#68](https://github.com/docling-project/docling/issues/68)) ([`bdfdfbf`](https://github.com/docling-project/docling/commit/bdfdfbf092fdaca43ddef28f763ef04456b82890))

## [v1.10.0](https://github.com/docling-project/docling/releases/tag/v1.10.0) - 2024-09-10

### Feature

* Linux arm64 support and reducing dependencies ([#69](https://github.com/docling-project/docling/issues/69)) ([`27a7a15`](https://github.com/docling-project/docling/commit/27a7a152e1123df7a22c44bb1adab5acce8f5390))

## [v1.9.0](https://github.com/docling-project/docling/releases/tag/v1.9.0) - 2024-09-03

### Feature

* Export document pages as multimodal output ([#54](https://github.com/docling-project/docling/issues/54)) ([`1de2e4f`](https://github.com/docling-project/docling/commit/1de2e4f924f562139c2a1e6314364845f9256575))

### Documentation

* Update MAINTAINERS.md ([#59](https://github.com/docling-project/docling/issues/59)) ([`69e5d95`](https://github.com/docling-project/docling/commit/69e5d951a389a9d36134629cfa2a0496c3bf095a))
* Mention quackling on README ([#58](https://github.com/docling-project/docling/issues/58)) ([`85b7348`](https://github.com/docling-project/docling/commit/85b7348846c87b28981f23c4855e49857c5bb782))

## [v1.8.5](https://github.com/docling-project/docling/releases/tag/v1.8.5) - 2024-08-30

### Fix

* Add unit tests ([#51](https://github.com/docling-project/docling/issues/51)) ([`48f4d1b`](https://github.com/docling-project/docling/commit/48f4d1ba5288b54d96740a1132b0d7977bef01cf))

## [v1.8.4](https://github.com/docling-project/docling/releases/tag/v1.8.4) - 2024-08-30

### Fix

* Propagate row_section in tables ([#57](https://github.com/docling-project/docling/issues/57)) ([`de85e46`](https://github.com/docling-project/docling/commit/de85e46ced293bdef7957f72fff301fec178cc94))

### Documentation

* Add instructions for cpu-only installation ([#56](https://github.com/docling-project/docling/issues/56)) ([`a8a60d5`](https://github.com/docling-project/docling/commit/a8a60d52b17fc25e71a421d4f89240bc7f02e154))

## [v1.8.3](https://github.com/docling-project/docling/releases/tag/v1.8.3) - 2024-08-28

### Fix

* Table cells overlap and model warnings ([#53](https://github.com/docling-project/docling/issues/53)) ([`f49ee82`](https://github.com/docling-project/docling/commit/f49ee825c3b95ffd5de29242aec764b074c773f7))

## [v1.8.2](https://github.com/docling-project/docling/releases/tag/v1.8.2) - 2024-08-27

### Fix

* Refine conversion result ([#52](https://github.com/docling-project/docling/issues/52)) ([`e46a66a`](https://github.com/docling-project/docling/commit/e46a66a17606a26f351b798ecf4fdeae71465f9c))

### Documentation

* Update interface in README ([#50](https://github.com/docling-project/docling/issues/50)) ([`fe817b1`](https://github.com/docling-project/docling/commit/fe817b11d730c55d48b6a60fc4e6f173da51a66b))

## [v1.8.1](https://github.com/docling-project/docling/releases/tag/v1.8.1) - 2024-08-26

### Fix

* Align output formats ([#49](https://github.com/docling-project/docling/issues/49)) ([`8cc147b`](https://github.com/docling-project/docling/commit/8cc147bc56753144915709a48b08830d0c3ad44e))

## [v1.8.0](https://github.com/docling-project/docling/releases/tag/v1.8.0) - 2024-08-23

### Feature

* Page-level error reporting from PDF backend, introduce PARTIAL_SUCCESS status ([#47](https://github.com/docling-project/docling/issues/47)) ([`a294b7e`](https://github.com/docling-project/docling/commit/a294b7e64a4d66ebb9fd328c084e5f74647805ee))

## [v1.7.1](https://github.com/docling-project/docling/releases/tag/v1.7.1) - 2024-08-23

### Fix

* Better raise exception when a page fails to parse ([#46](https://github.com/docling-project/docling/issues/46)) ([`8808463`](https://github.com/docling-project/docling/commit/8808463cecd7ff3a92bd99d2e3d65fd248672c9e))
* Upgrade docling-parse to 1.1.1, safety checks for failed parse on pages ([#45](https://github.com/docling-project/docling/issues/45)) ([`7e84533`](https://github.com/docling-project/docling/commit/7e845332992ab37386daee087573773051bfd065))

## [v1.7.0](https://github.com/docling-project/docling/releases/tag/v1.7.0) - 2024-08-22

### Feature

* Upgrade docling-parse PDF backend and interface to use page-by-page parsing ([#44](https://github.com/docling-project/docling/issues/44)) ([`a8c6b29`](https://github.com/docling-project/docling/commit/a8c6b29a67ca303d6eec3fabb6b5e75ad5a7676d))

## [v1.6.3](https://github.com/docling-project/docling/releases/tag/v1.6.3) - 2024-08-22

### Fix

* Usage of bytesio with docling-parse ([#43](https://github.com/docling-project/docling/issues/43)) ([`fac5745`](https://github.com/docling-project/docling/commit/fac5745dc846281bfae64bc631658bb2a2c90982))

## [v1.6.2](https://github.com/docling-project/docling/releases/tag/v1.6.2) - 2024-08-22

### Fix

* Remove [ocr] extra to fix wheel install ([#42](https://github.com/docling-project/docling/issues/42)) ([`6995268`](https://github.com/docling-project/docling/commit/69952682edd014a3f252e9c87edffa7c34f1033b))

## [v1.6.1](https://github.com/docling-project/docling/releases/tag/v1.6.1) - 2024-08-21

### Fix

* Add scipy as dependency ([#40](https://github.com/docling-project/docling/issues/40)) ([`f19871a`](https://github.com/docling-project/docling/commit/f19871a5a164b5369da10f7753d7c7da7fde35cc))

## [v1.6.0](https://github.com/docling-project/docling/releases/tag/v1.6.0) - 2024-08-20

### Feature

* Add adaptive OCR, factor out treatment of OCR areas and cell filtering ([#38](https://github.com/docling-project/docling/issues/38)) ([`e94d317`](https://github.com/docling-project/docling/commit/e94d317c022d2b916332d43cdc2aa90fd4738df9))

## [v1.5.0](https://github.com/docling-project/docling/releases/tag/v1.5.0) - 2024-08-20

### Feature

* Allow computing page images on-demand with scale and cache them ([#36](https://github.com/docling-project/docling/issues/36)) ([`78347bf`](https://github.com/docling-project/docling/commit/78347bf679c393378eab0bd383929fced88afeae))

### Documentation

* Add technical paper ref ([#37](https://github.com/docling-project/docling/issues/37)) ([`a13114b`](https://github.com/docling-project/docling/commit/a13114bafdcf4b62eb97df32cbfaf5695596b77c))

## [v1.4.0](https://github.com/docling-project/docling/releases/tag/v1.4.0) - 2024-08-14

### Feature

* Update parser with bytesio interface and set as new default backend ([#32](https://github.com/docling-project/docling/issues/32)) ([`90dd676`](https://github.com/docling-project/docling/commit/90dd676422f87584395a8949fa842fc9a6bdbd19))

### Fix

* Allow newer torch versions ([#34](https://github.com/docling-project/docling/issues/34)) ([`349b0e9`](https://github.com/docling-project/docling/commit/349b0e914f7194ee778571a7189b7eaff6f5966a))

## [v1.3.0](https://github.com/docling-project/docling/releases/tag/v1.3.0) - 2024-08-12

### Feature

* Output page images and extracted bbox ([#31](https://github.com/docling-project/docling/issues/31)) ([`63d80ed`](https://github.com/docling-project/docling/commit/63d80edca2fa4e64a07d8b00172d563d81ecb781))

## [v1.2.1](https://github.com/docling-project/docling/releases/tag/v1.2.1) - 2024-08-07

### Fix

* Update (vuln) deps ([#29](https://github.com/docling-project/docling/issues/29)) ([`79ef8d2`](https://github.com/docling-project/docling/commit/79ef8d2f2f6732f94c6777877ac9d0a45915ac84))
* Type of path_or_stream in PdfDocumentBackend ([#28](https://github.com/docling-project/docling/issues/28)) ([`794b20a`](https://github.com/docling-project/docling/commit/794b20a50ad089b39d4a4a84dcd826935b2b83ed))

### Documentation

* Improve examples ([#27](https://github.com/docling-project/docling/issues/27)) ([`9550db8`](https://github.com/docling-project/docling/commit/9550db8e64c4d638a429be33c10f10f18871f795))

## [v1.2.0](https://github.com/docling-project/docling/releases/tag/v1.2.0) - 2024-08-07

### Feature

* Introducing docling_backend ([#26](https://github.com/docling-project/docling/issues/26)) ([`b8f5e38`](https://github.com/docling-project/docling/commit/b8f5e38a8c8b3fd734fa119cae216a3da0b363f7))

## [v1.1.2](https://github.com/docling-project/docling/releases/tag/v1.1.2) - 2024-07-31

### Fix

* Set page number using 1-based indexing ([#22](https://github.com/docling-project/docling/issues/22)) ([`d2d9543`](https://github.com/docling-project/docling/commit/d2d9543415d37c54add917803b96d9959dc4d2cc))

## [v1.1.1](https://github.com/docling-project/docling/releases/tag/v1.1.1) - 2024-07-30

### Fix

* Correct text extraction for table cells ([#21](https://github.com/docling-project/docling/issues/21)) ([`f4bf3d2`](https://github.com/docling-project/docling/commit/f4bf3d25b955b71729833a18aa3a5b643fecfa75))

## [v1.1.0](https://github.com/docling-project/docling/releases/tag/v1.1.0) - 2024-07-26

### Feature

* Add simplified single-doc conversion ([#20](https://github.com/docling-project/docling/issues/20)) ([`d603137`](https://github.com/docling-project/docling/commit/d60313738340c20f9af64dfe51e28b7670ff64ef))

## [v1.0.2](https://github.com/docling-project/docling/releases/tag/v1.0.2) - 2024-07-24

### Fix

* Add easyocr to main deps for valid extra ([#19](https://github.com/docling-project/docling/issues/19)) ([`54b3dda`](https://github.com/docling-project/docling/commit/54b3dda141fc09e8c17ba4cb301d0c4394b680d8))

## [v1.0.1](https://github.com/docling-project/docling/releases/tag/v1.0.1) - 2024-07-24

### Fix

* Expose ocr as extra ([#18](https://github.com/docling-project/docling/issues/18)) ([`b0725e0`](https://github.com/docling-project/docling/commit/b0725e0aa693058b4962efa69730777dbe1d5bec))

## [v1.0.0](https://github.com/docling-project/docling/releases/tag/v1.0.0) - 2024-07-18

### Feature

* V1.0.0 release ([#16](https://github.com/docling-project/docling/issues/16)) ([`71c3a9c`](https://github.com/docling-project/docling/commit/71c3a9c8cde5b3a8884430eddcb33a9fbd7bf354))

### Breaking

* v1.0.0 release ([#16](https://github.com/docling-project/docling/issues/16)) ([`71c3a9c`](https://github.com/docling-project/docling/commit/71c3a9c8cde5b3a8884430eddcb33a9fbd7bf354))

## [v0.4.0](https://github.com/docling-project/docling/releases/tag/v0.4.0) - 2024-07-17

### Feature

* Optimize table extraction quality, add configuration options ([#11](https://github.com/docling-project/docling/issues/11)) ([`e9526bb`](https://github.com/docling-project/docling/commit/e9526bb11e21dc85c787af5c38e6f77eaca05f69))

## [v0.3.1](https://github.com/docling-project/docling/releases/tag/v0.3.1) - 2024-07-17

### Fix

* Missing type for default values ([#12](https://github.com/docling-project/docling/issues/12)) ([`d1d1724`](https://github.com/docling-project/docling/commit/d1d1724537d6a1f37591cdea44052207caae2ee2))

### Documentation

* Reflect supported Python versions, add badges ([#10](https://github.com/docling-project/docling/issues/10)) ([`2baa35c`](https://github.com/docling-project/docling/commit/2baa35c548dd6d15dba449eb1dc707f8f08c0a2a))

## [v0.3.0](https://github.com/docling-project/docling/releases/tag/v0.3.0) - 2024-07-17

### Feature

* Enable python 3.12 support by updating glm ([#8](https://github.com/docling-project/docling/issues/8)) ([`fb72688`](https://github.com/docling-project/docling/commit/fb72688ff7413083c864fe62d2dbfc420c1e5268))

### Documentation

* Add setup with pypi to Readme ([#7](https://github.com/docling-project/docling/issues/7)) ([`2803222`](https://github.com/docling-project/docling/commit/2803222ee1708481c779d435dbf1c031929d3cf6))

## [v0.2.0](https://github.com/docling-project/docling/releases/tag/v0.2.0) - 2024-07-16

### Feature

* Build with ci ([#6](https://github.com/docling-project/docling/issues/6)) ([`b1479cf`](https://github.com/docling-project/docling/commit/b1479cf4ecf8a586703b31c7cf6917b3293c6a85))
