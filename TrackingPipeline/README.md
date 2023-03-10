# LitMAE: Masked Auto-Emphasizer for Nighttime Aerial Tracking

# Keep excellent tracking in the nighttime!

## Tracking Test 

To evaluate the ability of LitMAE to lighten up semantics for aerial trackers, you need to meet the enviroment requirements of base trackers, as well as download their snapshots to corresponding folders at first. Details can be found in their repos. Currently supporting trackers including [LPAT](https://github.com/vision4robotics/LPAT), [HiFT](https://github.com/vision4robotics/HiFT), [SiamAPN](https://github.com/vision4robotics/SiamAPN), [SiamAPN++](https://github.com/vision4robotics/SiamAPN), and [SiamRPN++](https://github.com/STVIR/pysot).

Take the test of LitMAE_SiamAPN++ as an example:

```
python test.py                      \
  --dataset UAVDark135                            \ # dataset_name
  --datasetpath ./test_dataset                    \ # dataset_path
  
  --config ./experiments/SiamAPN++/config.yaml      \ # tracker_config
  --snapshot ./experiments/SiamAPN++/model.pth      \ # tracker_model
  --trackername LitMAE_SiamAPN++                    \ # tracker_name

  --e_weights ./experiments/LitMAE/model.pth         \ # enhancer_model
  --enhancername LitMAE                              \ # enhancer_name

```

## Evaluation 

Please put those results into `results` directory as `results/<dataset_name>/<tracker_name>`.

```
python tools/eval.py                              \
  --dataset UAVDark135                            \ # dataset_name
  --datasetpath path/of/your/dataset              \ # dataset_path
  --tracker_path ./results                        \ # result_path
  --tracker_prefix 'LitMAE_SiamAPN'                # tracker_name
```

## Contact

If you have any questions, please contact me.

Weiyu Peng

Email: 2032698@tongji.edu.cn .

## Acknowledgements
- The SOTA trackers' tracking pipelines are implemented based on [LPAT](https://github.com/vision4robotics/LPAT), [HiFT](https://github.com/vision4robotics/HiFT), [SiamAPN](https://github.com/vision4robotics/SiamAPN), [SiamAPN++](https://github.com/vision4robotics/SiamAPN), and [SiamRPN++](https://github.com/STVIR/pysot). The integeration of tracking pipelines is based on [SNOT](https://github.com/vision4robotics/SiameseTracking4UAV).

- We would also like to express our sincere thanks to the contributors.
