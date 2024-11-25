# D3MBR: Dual-Level Diffusion Denoiser with Preference Guidance for Multi-Behavior Recommendation

## Environments

The codes of D3MBR are implemented and tested under the following development environment:

python = 3.8.20

torch = 2.1.0

numpy = 1.24.4

scipy = 1.10.1



## Datasets

We utilized three datasets : <i>Beibei, Tmall, </i>and <i>IJCAI</i>. The <i>purchase</i> behavior is taken as the target behavior for all datasets. We filter out users whose interactions are fewer than 5 under the purchase behavior and adopt the widely used leave-one-out strategy for evaluation. For each user, the last interacted item under the purchase behavior constitutes the test set, the penultimate item is used to construct the validation set, and the remaining positive items are used for training.



## Reproductivity

The hyperparameters corresponding to different downstream MBR models and datasets are as follows:

### MBGCN

- beibeild32_nl2_ep50_lr0.001_dlr0.001_d[200,600]_ad[300]_st3_ns0.005_nm0.005_nx0.01_ss0_rwTrue_al0.1_be0.2_rsTrue_kr0.8_ktr0.8'
- IJCAI_15ld32_nl2_ep50_lr0.0001_dlr0.0005_d[200,600]_ad[300]_st2_ns0.005_nm0.005_nx0.01_ss2_rwTrue_al0.1_be0.2_rsTrue_kr0.8_ktr0.8'
- Tmallld32_nl2_ep50_lr0.001_dlr0.0001_d[200,600]_ad[300]_st2_ns0.005_nm0.005_nx0.01_ss2_rwTrue_al0.1_be0.2_rsTrue_kr0.8_ktr0.8'

### CRGCN

- 'beibeild32_nl2_ep50_lr0.001_dlr0.001_d[200,600]_ad[300]_st2_ns0.005_nm0.005_nx0.01_ss2_rwTrue_al0.1_be0.2_rsTrue_kr0.8_ktr0.8'

- 'IJCAI_15ld32_nl2_ep50_lr0.0001_dlr0.0001_d[200,600]_ad[300]_st2_ns0.005_nm0.005_nx0.01_ss2_rwTrue_al0.1_be0.2_rsTrue_kr0.8_ktr0.8'

- 'Tmallld32_nl2_ep50_lr0.0001_dlr0.0001_d[200,600]_ad[300]_st2_ns0.005_nm0.005_nx0.01_ss2_rwTrue_al0.1_be0.2_rsTrue_kr0.8_ktr0.8'

  

### BIPN

- 'beibeild32_nl2_ep50_lr0.0001_dlr0.0001_d[200,600]_ad[300]_st2_ns0.005_nm0.005_nx0.01_ss2_rwTrue_al0.1_be0.2_rsTrue_kr0.8_ktr0.8

- 'IJCAI_15ld32_nl2_ep50_lr0.0001_dlr0.0001_d[200,600]_ad[300]_st2_ns0.005_nm0.005_nx0.01_ss2_rwTrue_al0.1_be0.2_rsTrue_kr0.8_ktr0.9'
- 'Tmallld32_nl2_ep50_lr0.0001_dlr0.0005_d[200,600]_ad[300]_st2_ns0.005_nm0.005_nx0.01_ss2_rwTrue_al0.1_be0.2_rsTrue_kr0.8_ktr0.8'



