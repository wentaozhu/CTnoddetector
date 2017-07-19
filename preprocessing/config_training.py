config = {'stage1_data_path':'/mnt/media/wentao/tianchi/databowl/stage1/',
          'luna_raw':'/mnt/media/wentao/tianchi/luna16/',
          'luna_segment':'/mnt/media/wentao/tianchi/luna16/seg-lungs-LUNA16/',
          
          'luna_data':'/mnt/media/wentao/tianchi/luna16/allset/',
          'preprocess_result_path':'/mnt/media/wentao/tianchi/databowl/preprocess/',       
          
          'luna_abbr':'./detector/labels/shorter.csv',
          'luna_label':'./detector/labels/lunaqualified.csv',
          'stage1_annos_path':['./detector/labels/label_job5.csv',
                './detector/labels/label_job4_2.csv',
                './detector/labels/label_job4_1.csv',
                './detector/labels/label_job0.csv',
                './detector/labels/label_qualified.csv'],
          'bbox_path':'../detector/results/res18/bbox/',
          'preprocessing_backend':'python'
         }
