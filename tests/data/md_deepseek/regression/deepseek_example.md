order to compute the TED score. Inference timing results for all experiments were obtained from the same machine on a single core with AMD EPYC 7763 CPU @2.45 GHz.  

<|ref|>sub_title<|/ref|><|det|>[[217, 209, 520, 225]]<|/det|>
### 5.1 Hyper Parameter Optimization  

<|ref|>text<|/ref|><|det|>[[217, 230, 785, 321]]<|/det|>
We have chosen the PubTabNet data set to perform HPO, since it includes a highly diverse set of tables. Also we report TED scores separately for simple and complex tables (tables with cell spans). Results are presented in Table. It is evident that with OTSL, our model achieves the same TED score and slightly better mAP scores in comparison to HTML. However OTSL yields a \(2x\) speed up in the inference runtime over HTML.  

<|ref|>table<|/ref|><|det|>[[225, 421, 777, 595]]<|/det|>
<|ref|>table_caption<|/ref|><|det|>[[217, 342, 785, 413]]<|/det|>
Table 1. HPO performed in OTSL and HTML representation on the same transformer-based TableFormer 9 architecture, trained only on PubTabNet [22]. Effects of reducing the # of layers in encoder and decoder stages of the model show that smaller models trained on OTSL perform better, especially in recognizing complex table structures, and maintain a much higher mAP score than the HTML counterpart.   

<table><tr><th rowspan="2"># enc-layers</th><th rowspan="2"># dec-layers</th><th rowspan="2">Language</th><th colspan="3">TEDs</th><th rowspan="2">mAP (0.75)</th><th rowspan="2">Inference time (secs)</th></tr><tr><th>simple</th><th>complex</th><th>all</th></tr><tr><td rowspan="2">6</td><td rowspan="2">6</td><td>OTSL</td><td>0.965</td><td>0.934</td><td>0.955</td><td>0.88</td><td>2.73</td></tr><tr><td>HTML</td><td>0.969</td><td>0.927</td><td>0.955</td><td>0.857</td><td>5.39</td></tr><tr><td rowspan="2">4</td><td rowspan="2">4</td><td>OTSL</td><td>0.938</td><td>0.904</td><td>0.927</td><td>0.853</td><td>1.97</td></tr><tr><td>HTML</td><td>0.952</td><td>0.909</td><td>0.938</td><td>0.843</td><td>3.77</td></tr><tr><td rowspan="2">2</td><td rowspan="2">4</td><td>OTSL</td><td>0.923</td><td>0.897</td><td>0.915</td><td>0.859</td><td>1.91</td></tr><tr><td>HTML</td><td>0.945</td><td>0.901</td><td>0.931</td><td>0.834</td><td>3.81</td></tr><tr><td rowspan="2">4</td><td rowspan="2">2</td><td>OTSL</td><td>0.952</td><td>0.92</td><td>0.942</td><td>0.857</td><td>1.22</td></tr><tr><td>HTML</td><td>0.944</td><td>0.903</td><td>0.931</td><td>0.824</td><td>2</td></tr></table>

<|ref|>sub_title<|/ref|><|det|>[[217, 636, 432, 652]]<|/det|>
### 5.2 Quantitative Results  

<|ref|>text<|/ref|><|det|>[[217, 656, 785, 777]]<|/det|>
We picked the model parameter configuration that produced the best prediction quality (enc=6, dec=6, heads=8) with PubTabNet alone, then independently trained and evaluated it on three publicly available data sets: PubTabNet (395k samples), FinTabNet (113k samples) and PubTables- 1M (about 1M samples). Performance results are presented in Table. 2 It is clearly evident that the model trained on OTSL outperforms HTML across the board, keeping high TEDs and mAP scores even on difficult financial tables (FinTabNet) that contain sparse and large tables.  

<|ref|>text<|/ref|><|det|>[[217, 778, 785, 838]]<|/det|>
Additionally, the results show that OTSL has an advantage over HTML when applied on a bigger data set like PubTables- 1M and achieves significantly improved scores. Finally, OTSL achieves faster inference due to fewer decoding steps which is a result of the reduced sequence representation.
