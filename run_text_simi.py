# coding=utf-8
from main import main


if __name__ == "__main__":

    model_name = "BertOrigin"
    data_dir = "/search/hadoop02/suanfa/kinghuangdd/text_similarity/train_test"
    output_dir = ".text_simi_output/"
    cache_dir = ".text_simi_cache"
    log_dir = ".text_simi_log/"

    model_times = "model_5/"   # 第几次保存的模型，主要是用来获取最佳结果

    bert_vocab_file = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-chinese-vocab.txt"
    bert_model_dir = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-chinese"

    from Processors.text_simi_Processor import text_simi_Processor

    if model_name == "BertOrigin":
        from BertOrigin import args

    elif model_name == "BertCNN":
        from BertCNN import args

    elif model_name == "BertATT":
        from BertATT import args

    elif model_name == "BertRCNN":
        from BertRCNN import args
        
    main(args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir),
             model_times, text_simi_Processor)
