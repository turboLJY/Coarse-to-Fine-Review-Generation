CUDA_LAUNCH_BLOCKING=1 
python main.py --train electronic --save_dir ./data/electronic
#python main.py --test ./data/beer/model/2_512_48/32_review_model.tar --tp_load /home/junyi_li/Project/TopicTrans/topic/data/beer/model/2_512_1024/26_topic_model.tar --pt_load /home/junyi_li/Project/TopicTrans/pattern/data/beer/model/2_512_64/64_pattern_model.tar   --corpus beer
