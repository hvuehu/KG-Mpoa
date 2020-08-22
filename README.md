1.使用命令运行文件：
python main_vua.py --w2v_embedding_dim 300 --elmo_embedding_dim 1024 --attention_query_size 300 --run_type train --pretrained_w2v_model_path static_data/semeval/semeval_embedding.txt --elmo_path static_data/semeval/semeval_elmo.hdf5  --train_path static_data/semeval/train_triples.txt --dev_path static_data/semeval/dev_triples.txt --test_path static_data/semeval/test_triples.txt --vocab_path static_data/semeval/vocab.txt --summary_result_path results --input_dim 300 --output_result_path result/semeval/none/mpoa/rand --attention_layer mpoa --pretrain_model_type none --query_matrix_path static_data/emo_vector_en.json --language_type en --m_a_type rand --triples_vocab_path static_data/semeval/bert_new/350 --epochs_num 200 --triples_embedding_dim 350 --batch_size 32 --dropout 0.3 --learning_rate 0.01 --momentum 0.95 --concat_mode concat
2.参数说明：
--vocab_path 词表
--run_type 运行类型,可选[train/test]
--output_result_path 模型保存路径
--summary_result_path 测试集结果保存路径
--attention_layer 注意力机制类型,可选[att/m_a/m_pre_orl_a/m_pre_orl_pun_a/m_pol_untrain_a/mpa/mpoa/none]
--pretrain_model_type 文本词向量预训练方式,可选[w2v/elmo/random]
--query_matrix_path 查询向量路径
--language_type 数据集类型,可选[中/英]
--concat_mode 文本与知识结合方式,可选[concat/graph_attention]
