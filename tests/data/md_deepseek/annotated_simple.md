text[[217, 146, 785, 191]]
order to compute the TED score. Inference timing results for all experiments were obtained from the same machine on a single core with AMD EPYC 7763 CPU @2.45 GHz.  

sub_title[[217, 209, 520, 225]]
### 5.1 Hyper Parameter Optimization  

text[[217, 230, 785, 321]]
We have chosen the PubTabNet data set to perform HPO, since it includes a highly diverse set of tables. Also we report TED scores separately for simple and complex tables (tables with cell spans). Results are presented in Table. It is evident that with OTSL, our model achieves the same TED score and slightly better mAP scores in comparison to HTML. However OTSL yields a \(2x\) speed up in the inference runtime over HTML.  

sub_title[[217, 636, 432, 652]]
### 5.2 Quantitative Results  

text[[217, 657, 785, 778]]
We picked the model parameter configuration that produced the best prediction quality (enc=6, dec=6, heads=8) with PubTabNet alone, then independently trained and evaluated it on three publicly available data sets: PubTabNet (395k samples), FinTabNet (113k samples) and PubTables- 1M (about 1M samples). Performance results are presented in Table. It is clearly evident that the model trained on OTSL outperforms HTML across the board, keeping high TEDs and mAP scores even on difficult financial tables (FinTabNet) that contain sparse and large tables.  

text[[217, 779, 785, 839]]
Additionally, the results show that OTSL has an advantage over HTML when applied on a bigger data set like PubTables- 1M and achieves significantly improved scores. Finally, OTSL achieves faster inference due to fewer decoding steps which is a result of the reduced sequence representation.
