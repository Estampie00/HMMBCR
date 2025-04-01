# HMMBCR
This is a project that uses raw sensory data collected by wearable sensors and mobile devices for Behavioral Context Recognition, which can been as an extension of Human Activity Recognition. <br>
![总体框图8](https://github.com/user-attachments/assets/cadd2adb-70e4-4fe0-a7c6-30964ca082a9) <br>


Code structure:<br>
-model.HMMBCR.py<br>

--module.feature_extraction.py (multi-modal feature extraction)<br>

  --module.MULT.py (multi-modal fusion module, i.e. cross-modal fusion encoder)<br>
    ----component.position_embeddings.py<br>
    ----component.mult_encoder.py<br>
    
  --module.Heterogeous_decoder (Heterogeneous Modality-to-Label Dependence Module)<br>
    ----component.modality_label_fusion.py<br>
    
  --module.HGAT (Heterogeneous Label-to-Label Dependence Module)<br>
    ----component.HGAT_layers.py (discretely, we perform the dual-level attention here)<br>


Requirements on this work:<br>
python 3.8.13<br>
torch 1.8.1<br>
numpy 1.18.1<br>
typing 4.8.0<br>
The datasets are available at:<br>
Extrasensory dataset: http://extrasensory.ucsd.edu/ <br>
ETRI Lifelog dataset: https://nanum.etri.re.kr/share/schung/ETRILifelogDataset2020?lang=En_us <br>
Article link: To be continued
