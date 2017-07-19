config = {'datapath':'/mnt/media/wentao/tianchi/test/', 
          'preprocess_result_path':'/mnt/media/wentao/tianchi/preprocess_test/',
          'bbox_result_path':'/mnt/media/wentao/tianchi/bbox_test/',
          
          'detector_model':'net_detector',
         'detector_param':'./model/detector.ckpt',
         'n_gpu':4,
         'n_worker_preprocessing':10,
         'use_exsiting_preprocessing':False,
         'skip_preprocessing':False,
         'skip_detect':False}
