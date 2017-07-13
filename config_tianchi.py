config = {'datapath':'/mnt/media/wentao/tianchi/', # use val_subset02, val_subset03, val_subset04
          'preprocess_result_path':'./prep_result_tianchi/',
          'bbox_result_path':'./bbox_result_tianchi',
          
          'detector_model':'net_detector',
         'detector_param':'./model/detector.ckpt',
         'n_gpu':4,
         'n_worker_preprocessing':15,
         'use_exsiting_preprocessing':False,
         'skip_preprocessing':False,
         'skip_detect':False}
