config = {'train_data_path':'/mnt/media/wentao/tianchi/train/',
          'val_data_path':'/mnt/media/wentao/tianchi/val/', 
          'test_data_path':'/mnt/media/wentao/tianchi/val/', 
          
          'train_preprocess_result_path':'/mnt/media/wentao/tianchi/preprocessing/train/',  
          'val_preprocess_result_path':'/mnt/media/wentao/tianchi/preprocessing/val/',  
          'test_preprocess_result_path':'/mnt/media/wentao/tianchi/preprocessing/val/',
          
          'train_bbox_path':'/mnt/media/wentao/tianchi/bbox/train/',  
          'val_bbox_path':'/mnt/media/wentao/tianchi/bbox/val/',  
          'test_bbox_path':'/mnt/media/wentao/tianchi/bbox/val/',  
          
          'train_annos_path':'/mnt/media/wentao/tianchi/csv/train/annotations.csv',
          'val_annos_path':'/mnt/media/wentao/tianchi/csv/val/annotations.csv',
          'test_annos_path':'/mnt/media/wentao/tianchi/csv/val/annotations.csv',

          'black_list':['LKDS-00192', 'LKDS-00319', 'LKDS-00238', 'LKDS-00926', 'LKDS-00504',
                        'LKDS-00648', 'LKDS-00829', 'LKDS-00931', 'LKDS-00359', 'LKDS-00379', 
                        'LKDS-00541', 'LKDS-00353', 'LKDS-00598', 'LKDS-00684', 'LKDS-00065'],
          
          'preprocessing_backend':'python'
         } # LKDS-00648 - end is found by the labelmapping function
# 'LKDS-00192','LKDS-00319','LKDS-00238','LKDS-00926', 'LKDS-00504' is found from preprocessing
# 'LKDS-00504',