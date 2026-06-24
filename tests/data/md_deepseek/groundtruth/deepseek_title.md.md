# DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis

Birgit Pfitzmann IBM Research Rueschlikon, Switzerland bpf@zurich.ibm.com

Christoph Auer IBM Research Rueschlikon, Switzerland cau@zurich.ibm.com

Michele Dolfi IBM Research Rueschlikon, Switzerland dol@zurich.ibm.com

Ahmed S. Nassar IBM Research Rueschlikon, Switzerland ahn@zurich.ibm.com

Peter Staar IBM Research Rueschlikon, Switzerland taa@zurich.ibm.com

## ABSTRACT

Accurate document layout analysis is a key requirement for high- quality PDF document conversion. With the recent availability of public, large ground- truth datasets such as PubLayNet and DocBank, deep- learning models have proven to be very effective at layout detection and segmentation. While these datasets are of adequate size to train such models, they severely lack in layout variability since they are sourced from scientific article repositories such as PubMed and arXiv only. Consequently, the accuracy of the layout segmentation drops significantly when these models are applied on more challenging and diverse layouts. In this paper, we present DocLayNet, a new, publicly available, document- layout annotation dataset in COCO format. It contains 80863 manually annotated pages from diverse data sources to represent a wide variability in layouts. For each PDF page, the layout annotations provide labelled bounding- boxes with a choice of 11 distinct classes. DocLayNet also provides a subset of double- and triple- annotated pages to determine the inter- annotator agreement. In multiple experiments, we provide baseline accuracy scores (in mAP) for a set of popular object detection models. We also demonstrate that these models fall approximately \(10\%\) behind the inter- annotator agreement. Furthermore, we provide evidence that DocLayNet is of sufficient size. Lastly, we compare models trained on PubLayNet, DocBank and DocLayNet, showing that layout predictions of the DocLayNet- trained models are more robust and thus the preferred choice for general- purpose document- layout analysis.

## CCS CONCEPTS

- Information systems \(\rightarrow\) Document structure; \(\cdot\) Applied computing \(\rightarrow\) Document analysis; \(\cdot\) Computing methodologies \(\rightarrow\) Machine learning; Computer vision; Object detection;

&lt;center&gt;Figure 1: Four examples of complex page layouts across different document categories &lt;/center&gt;

<!-- image -->

## KEYWORDS

PDF document conversion, layout segmentation, object- detection, data set, Machine Learning

## ACM Reference Format:

Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S. Nassar, and Peter Staar. 2022. DocLayNet: A Large Human- Annotated Dataset for Document- Layout Analysis. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22), August 14- 18, 2022, Washington, DC, USA. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3534678.3539043