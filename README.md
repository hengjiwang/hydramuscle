Code for paper _From neuron to muscle to movement: a complete biomechanical model of *Hydra* contractile behaviors_

- **hydramuscle/model/** : biophysical model for simulating calcium dynamics
- **hydramuscle/notebooks/** : jupyter notebooks for saving data and generating figures
- **hydramuscle/test/** : codes for running the biophysical model under different conditions
- **hydramuscle/midline/** : code for extracting midline
- **hydramuscle/biomech/** : biomechanical model for transforming stress to movements
- **hydramuscle/utils/** : useful wheels
- **hydramuscle/dataset/** : relevant data of the paper
  - **2020-09-28-23-41-19-017767.h5** : simulated calcium dynamics data (for Figure 8CDEF)
  - **DyWat_Ecto_80X_Snap12d_label.tif** : hydra photo for Figure 3B
  - **coords_017767_rlx60_100-350.txt** : simulated length of the model (Figure 8G, 100s-350s)
  - **coords_017767_rlx60_350-600.txt** : simulated length of the model (Figure 8G, 350s-600s)
  - **dual_channel_cycles.mov** : dual-channel imaging video for Figure 7
  - **lengths_Pre_Bisect_40x_4fps_ex3.csv** : measured length of a real hydra (Figure 8G)
  - **neural_gcamp_video.avi** : neural GCaMP imaging video for extracting stimulation times (Figure 8A)
