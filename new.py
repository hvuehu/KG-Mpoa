## 拼接
import torch.nn as nn
import torch.nn.functional as F
from util import sort_batch_by_length
from util import last_dim_softmax
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from util import modified_Linear
from util import output_concat_triples
import torch

class RNNSequenceClassifier(nn.Module):
    def __init__(self, args, embedding, embeddings_entity, embeddings_relation, query_embedding):
        # Always call the superclass (nn.Module) constructor first
        super(RNNSequenceClassifier, self).__init__()
        self.args = args
        # self.rnn = nn.LSTM(input_size=args.input_dim+args.triples_embedding_dim, hidden_size=args.hidden_size,
        #                    num_layers=args.layers_num, dropout=args.dropout, batch_first=True, bidirectional=args.is_bidir)
        self.rnn = nn.LSTM(input_size=args.elmo_embedding_dim+args.triples_embedding_dim, hidden_size=args.hidden_size,
                           num_layers=args.layers_num, dropout=args.dropout, batch_first=False, bidirectional=args.is_bidir)
        # nn.LSTM()隐层输出维数为hidden_size
        if args.is_bidir == True:
            is_bidir_number = 2
        else:
            is_bidir_number = 1

        self.embedding = embedding
        self.embeddings_entity = embeddings_entity
        self.embeddings_relation = embeddings_relation
        self.seq_length = args.seq_length
        self.triples_number = args.triples_number
        self.batch_size = args.batch_size
        self.triples_embedding_dim = args.triples_embedding_dim
        self.input_dim = args.elmo_embedding_dim
        # self.input_dim = args.input_dim
        # self.linear = modified_Linear(args.w2v_embedding_dim + args.triples_embedding_dim*2,
        #         args.w2v_embedding_dim, False) # 词向量维度+三元组实体维度，词向量维度，可按需修改
        # nn.Linear()定义的是第二个维度的大小，第一个维度的大小跟输入数据的大小一致
        # self.linear = modified_Linear(args.elmo_embedding_dim + args.triples_embedding_dim,
        #                               args.elmo_embedding_dim, False)
        # self.linear = nn.Linear(args.elmo_embedding_dim + args.triples_embedding_dim*2,
        #                         args.elmo_embedding_dim, False)
        # self.linear = nn.Linear(args.w2v_embedding_dim + args.triples_embedding_dim*2,
        #                        args.w2v_embedding_dim, False)
        self.entity_transformed = nn.Linear(args.triples_embedding_dim * 2, args.triples_embedding_dim, False)
        self.relation_transformed = nn.Linear(args.triples_embedding_dim, args.triples_embedding_dim, False)
        # self.change_average = nn.Linear(args.triples_embedding_dim * 2, args.input_dim+args.triples_embedding_dim * 2)
        self.change_average = nn.Linear(args.triples_embedding_dim * 2, args.elmo_embedding_dim+args.triples_embedding_dim * 2)
        if args.attention_layer == 'att':
            self.attention_weights = nn.Linear(args.hidden_size * is_bidir_number, 1)
            self.output_projection = nn.Linear(args.hidden_size * is_bidir_number, args.num_classes)
        else:
            self.query_embedding = query_embedding
            self.proquery_weights_mp = nn.Linear(args.hidden_size * is_bidir_number, args.attention_query_size)
            self.multi_output_projection = nn.Linear(args.hidden_size * is_bidir_number* args.num_classes, args.num_classes)
        self.dropout_on_input_to_LSTM = nn.Dropout(args.dropout, inplace=False)
        self.dropout_on_input_to_linear_layer = nn.Dropout(args.dropout, inplace=False)

    def forward(self, inputs, triples, lengths, elmo_embedding, id2_ids_batch, add_triples = False):
        # print("begin...")
        # 0. input层和预训练层
        if self.args.pretrain_model_type == 'elmo': # 使用elmo训练好的词向量
            elmo_inputs = torch.Tensor().cuda()
            # torch.Tensor()生成的是FloatTensor的张量，如果没有提供参数，将会返回一个空的零维张量tensor([])；
            # torch.tensor()根据原始数据类型生成相应的torch.LongTensor，torch.FloatTensor，torch.DoubleTensor。
            for i in range(len(inputs)):
                elmo_input = torch.from_numpy(elmo_embedding[' '.join(map(str, inputs[i].cpu().numpy()))].value).type(torch.cuda.FloatTensor)
                # ' '.join(map())将id从数字形式转为字符串，如：‘5 8 9874 (中间为空格)5874’。
                # 猜想：elmo_embedding[]读取为一个字典，输根据入id查找字典中的词向量
                # torch.from_numpy(),将numpy中的ndarray转化成pytorch中的tensor
                try:
                    elmo_inputs = torch.cat((elmo_inputs, elmo_input.unsqueeze(dim=0)))
                    # unsqueeze()第一维度增加一维,torch.cat()默认在第一维进行拼接，如矩阵大小为3*4，拼接2*4，则得到5*4的矩阵
                except:
                    #print(elmo_inputs.shape, elmo_input.shape)
                    elmo_inputs = torch.cat((elmo_inputs, elmo_input.unsqueeze(dim=0)[:,:128,:]), dim=0)
            inputs = elmo_inputs
            #print(inputs)
        else:
            inputs = self.embedding(inputs) # 随机初始化或通过w2v

        t = torch.zeros(inputs.size(0), self.seq_length, self.input_dim + self.triples_embedding_dim).cuda()
        if self.args.combination_mode == "vocabulary_level" or self.args.combination_mode == "both_level":
            if add_triples == True:
                for i in range(len(inputs)):  # 输入的句子一句一句跟三元组结合
                    dict = {}
                    # print(inputs[i].size())
                    # print(id2_ids_batch[i].size())
                    # print(triples[i].size())
                    b = torch.full([self.seq_length, self.triples_number], -1, dtype=torch.long).cuda()
                    bb = torch.zeros(self.seq_length, self.triples_embedding_dim).cuda()
                    if (torch.equal(id2_ids_batch[i], b)):
                        t[i] = torch.cat((inputs[i], bb), dim=-1)
                    else:
                        for k in range(len(id2_ids_batch[i])):  # 每个词对应的三元组的拼接信息
                            # print(k)
                            a = 0
                            input = torch.Tensor().cuda()
                            c = torch.full([self.triples_number], -1, dtype=torch.long).cuda()
                            cc = torch.zeros(self.triples_embedding_dim).cuda()
                            if (torch.equal(id2_ids_batch[i][k], c)):
                                t[i][k] = torch.cat((inputs[i][k], cc), dim=-1)
                                # print(t[i][k].size())
                            # triple_entity = torch.LongTensor([triples[i][j][0:2]]).cuda()
                            else:
                                for j in range(len(id2_ids_batch[i][k])):  # 每个三元组的id2
                                    # print(id2_ids_batch[i][k][j].cpu().numpy())
                                    if id2_ids_batch[i][k][j].cpu().numpy() == 1:  # 看词与头实体还是尾实体匹配
                                        # print('》>》>》>》>》>。')
                                        inputs_triples = torch.cat(
                                            (inputs[i][k], self.embeddings_entity(triples[i][k][j][1])))
                                        # print(inputs_triples.size())
                                        # embeddings_entity[triples_ids_batch[i][j][1]]) 表示要匹配的实体在词表中的向量表示
                                        # inputs_triples_final = self.linear(inputs_triples.cuda())
                                        # print(inputs_triples_final)
                                        # print(inputs_triples_final.size())

                                    elif id2_ids_batch[i][k][j].cpu().numpy() == 2:
                                        # print('<《<《<《<《<《<《<。')
                                        # print(inputs[i][k].size())
                                        # print(self.embeddings_entity[triples[i][k][j][0]].size())
                                        # print(inputs[i][k].size())
                                        inputs_triples = torch.cat(
                                            (inputs[i][k], self.embeddings_entity(triples[i][k][j][0])))
                                        # print(inputs_triples.size())
                                        # inputs_triples_final = self.linear(inputs_triples.cuda())
                                        # print(inputs_triples_final.size())
                                    else:
                                        # print('。。。。。。。。。。。。。')
                                        continue

                                    if a == 0:
                                        # print(">》>》>》>》>》>》>》")
                                        a = a + 1
                                        input = torch.cat((inputs_triples, input))
                                        # input = torch.cat((inputs_triples, input))
                                    else:
                                        a = a + 1
                                        # print("《<《<《<《<《<《<《<")
                                        input = input + inputs_triples
                                        # input = input + inputs_triples

                            if a != 0:  # 计算平均
                                # print(a)
                                input = input / a
                                dict[k] = input
                            # print(dict)

                        for k in dict:
                            t[i][k] = dict[k]

            else:
                for i in range(len(inputs)):  # 每句话循环
                    ad = torch.zeros(self.seq_length, self.triples_embedding_dim).cuda()
                    t[i] = torch.cat((inputs[i], ad), dim=-1)

        # print(t.size())

        if self.args.combination_mode == "sentence_level":
            if add_triples == True:
                for i in range(len(inputs)): #　每句话循环
                    ad = torch.zeros(self.seq_length, self.triples_embedding_dim).cuda()
                    t[i]=torch.cat((inputs[i], ad),dim=-1)

                    # input_triples_middle = torch.zeros(600).cuda()
                    input_triples_middle = torch.zeros(self.triples_embedding_dim * 2).cuda() #　三元组维数一般是100，elmo是300
                    len_triples = 0
                    # print(triples[i])
                    # print(triples[i].size())
                    b = torch.full([self.seq_length, self.triples_number, 3], 0, dtype=torch.long).cuda()
                    # print(triples[i].size())
                    if (torch.equal(triples[i], b)):
                        continue
                    else:
                        for j in range(len(triples[i])): #　每个词对应的所有的三元组
                            # 将三元组头、尾实体向量拼接
                            # print(triples[i][j])
                            # print(triples[i][j].size())
                            c = torch.full([self.triples_number, 3], 0, dtype=torch.long).cuda()
                            if  (torch.equal(triples[i][j], c)):
                                continue
                            else:
                                for k in range(len(triples[i][j])): #　每个三元组
                                    if triples[i][j][k][0].cpu().numpy() != 0 and triples[i][j][k][1].cpu().numpy() != 0:
                                        #print("?？?？?？?？?？?？?？?")
                                        len_triples = len_triples + 1
                                        embed_triple = self.embeddings_entity(triples[i][j][k][0:2]).squeeze(0)
                                        #print(embed_triple)
                                        #print(embed_triple.size())
                                        input_triples = torch.cat((embed_triple[0].squeeze(0), embed_triple[1].squeeze(0))).cuda()
                                        #print("》>》>》>》>》>》>》>》>》>》>》")
                                        #print(input_triples.size())
                                        input_triples_middle = input_triples + input_triples_middle
                                        #print(input_triples_middle.size())
                                        #print(inputs[i][j].size())
                                        #print("?？?？?？?？?？?？")
                        if input_triples_middle.size() != t[i][j].size():
                            input_triples_middle1 = self.change_average(input_triples_middle)

                        if len_triples != 0:
                            average = input_triples_middle1 / len_triples  # 相加之后求平均
                            # print(average)
                            try:
                                t[i][lengths[i]] = average
                                # print("?？?？?？?？?？?？?")
                            except:
                                pass

                            if lengths[i] == self.args.seq_length:
                                pass
                            else:
                                lengths[i] = lengths[i] + 1
                        else:
                            pass


            else:
                for i in range(len(inputs)):  # 每句话循环
                    ad = torch.zeros((self.triples_embedding_dim)).cuda()
                    for j in range(len(inputs[i])):
                        t[i][j] = torch.cat((inputs[i][j], ad))


        if self.args.combination_mode == "both_level":
            if add_triples == True:
                for i in range(len(t)): #　每句话循环
                    input_triples_middle = torch.zeros(self.triples_embedding_dim * 2).cuda() #　三元组维数一般是100，elmo是300
                    # input_triples_middle = torch.zeros(600).cuda()
                    len_triples = 0
                    # print(triples[i])
                    # print(triples[i].size())
                    b = torch.full([self.seq_length, self.triples_number, 3], 0, dtype=torch.long).cuda()
                    # print(triples[i].size())
                    if (torch.equal(triples[i], b)):
                        continue
                    else:
                        for j in range(len(triples[i])): #　每个词对应的所有的三元组
                            # 将三元组头、尾实体向量拼接
                            # print(triples[i][j])
                            # print(triples[i][j].size())
                            c = torch.full([self.triples_number, 3], 0, dtype=torch.long).cuda()
                            if  (torch.equal(triples[i][j], c)):
                                continue
                            else:
                                for k in range(len(triples[i][j])): #　每个三元组
                                    if triples[i][j][k][0].cpu().numpy() != 0 and triples[i][j][k][1].cpu().numpy() != 0:
                                        #print("?？?？?？?？?？?？?？?")
                                        len_triples = len_triples + 1
                                        embed_triple = self.embeddings_entity(triples[i][j][k][0:2]).squeeze(0)
                                        #print(embed_triple)
                                        #print(embed_triple.size())
                                        input_triples = torch.cat((embed_triple[0].squeeze(0), embed_triple[1].squeeze(0))).cuda()
                                        #print("》>》>》>》>》>》>》>》>》>》>》")
                                        #print(input_triples.size())
                                        input_triples_middle = input_triples + input_triples_middle
                                        #print(input_triples_middle.size())
                                        #print(inputs[i][j].size())
                                        #print("?？?？?？?？?？?？")

                        if input_triples_middle.size() != t[i][j].size():
                            input_triples_middle1 = self.change_average(input_triples_middle)

                        if len_triples != 0:
                            average = input_triples_middle1 / len_triples  # 相加之后求平均
                            # print(average)
                            try:
                                t[i][lengths[i]] = average
                                # print("?？?？?？?？?？?？?")
                            except:
                                pass

                            if lengths[i] == self.args.seq_length:
                                pass
                            else:
                                lengths[i] = lengths[i] + 1
                        else:
                            pass



        # 1. input（双向LSTM）
        embedded_input = self.dropout_on_input_to_LSTM(t) # 输入进LSTM时随机失活
        #print("dddddddddddddddddddddddddddddd")
        #print(lengths)
        (sorted_input, sorted_lengths, input_unsort_indices, _) = sort_batch_by_length(embedded_input, lengths)
        # embedded_input是已经padding和截断后的句子。输入参数：已随机失活的inputs和句子长度，输出：
        # 按句子长度倒序排序的句子id表示、按句子长度倒序排序的句子长度、按句子长度倒序排序后再正序排序
        # 的句子（相当于无排序的句子）长度索引、按句子长度倒序排序的句子长度索引

        packed_input = pack_padded_sequence(sorted_input, sorted_lengths.data.tolist(), batch_first=True)
        # 将padding过的句子，去除pad,如tensor([[3., 7., 5., 7., 5.], 经过去掉pad之后成为
        #                                   [4., 9., 1., 4., 0.],
        #                                   [2., 8., 0., 0., 0.]])
        # data=tensor([3., 4., 2., 7., 9., 8., 5., 1., 7., 4., 5.]),
        # batch_sizes=tensor([3, 3, 2, 2, 1]（每一列有多少数据）
        packed_sorted_output, _ = self.rnn(packed_input)
        # 这里有2层lstm，output是最后一层lstm的每个词向量对应隐藏层的输出,其与层数无关，只与序列长度相关
        # hn,cn是所有层最后一个隐藏元和记忆元的输出
        # total_length = inputs.size(1)
        sorted_output, _ = pad_packed_sequence(packed_sorted_output, batch_first=True)
        # sorted_output, _ = pad_packed_sequence(packed_sorted_output, total_length=total_length, batch_first=True)
        # 输出变为原顺序，输出为一个batch的数据结果
        output = sorted_output[input_unsort_indices]
        #print(output.size())
        #print(output)
        #print(">》>》>》>》>》>》")

        # 2. use attention
        if self.args.attention_layer == 'att': # 普通注意力
            # 若attention_logits=0，则被mask,计算mask后的权重分布（注意力权重分布），与原输出相乘，得到
            # 整个batch中所有句子的表示
            attention_logits = self.attention_weights(output).squeeze(-1) # 线性层，得到一个值
            mask_attention_logits = (attention_logits != 0).type(
                torch.cuda.FloatTensor if inputs.is_cuda else torch.FloatTensor)
            # mask_attention_logits是Bool值 0/1,是否使用cuda
            softmax_attention_logits = last_dim_softmax(attention_logits, mask_attention_logits)
            # attention_logits=0的话mask掉，再进行归一化
            softmax_attention_logits0 = softmax_attention_logits.unsqueeze(dim=1) # 注意力权重
            input_encoding = torch.bmm(softmax_attention_logits0, output) # 得到最终表示
            # softmax_attention_logits, output 向量矩阵相乘，两矩阵维度必须为三维
            input_encoding0 = input_encoding.squeeze(dim=1) # 去掉一维
        else: # 多头注意力的各种形式
            input_encoding = torch.Tensor().cuda() # 空张量
            querys = self.query_embedding(torch.arange(0,self.args.num_classes,1).cuda()) # ？？？
            attention_weights = torch.Tensor(self.args.num_classes, len(output), len(output[0])).cuda()
            # 随机初始化args.num_classes, len(output), len(output[0])大小的张量矩阵
            for i in range(self.args.num_classes): # 每个类别下操作
                attention_logits = self.proquery_weights_mp(output) # 线性层
                attention_logits = torch.bmm(attention_logits, querys[i].unsqueeze(dim=1).repeat(len(output),1,1)).squeeze(dim=-1)
                # Q查询向量,K键，相乘（相当于缩放点积公式中的分子）（得分函数）。将查询向量重复len(output)次，第一维度重复
                mask_attention_logits = (attention_logits != 0).type(
                    torch.cuda.FloatTensor if inputs.is_cuda else torch.FloatTensor)
                softmax_attention_logits = last_dim_softmax(attention_logits, mask_attention_logits)
                # mask + 归一化
                input_encoding_part = torch.bmm(softmax_attention_logits.unsqueeze(dim=1), output)
                # 得到注意力权重分布的句子表示
                input_encoding = torch.cat((input_encoding,input_encoding_part.squeeze(dim=1)), dim=-1)
                #print(input_encoding.size())
                attention_weights[i] = softmax_attention_logits # 第i种注意力权重

        # 3. run linear layer
        if self.args.attention_layer == 'att':
            input_encodings = self.dropout_on_input_to_linear_layer(input_encoding0) # 输出随机失活
            unattized_output = self.output_projection(input_encodings) # 线性层，输出维度是类别数
            output_distribution = F.log_softmax(unattized_output, dim=-1) # 激活函数
            return output_distribution, softmax_attention_logits.squeeze(dim=1)
        else:
            input_encodings = self.dropout_on_input_to_linear_layer(input_encoding)
            unattized_output = self.multi_output_projection(input_encodings)
            # 线性层，输入维度和“att”中不一致，但输出维度均为类别数
            output_distribution = F.log_softmax(unattized_output, dim=-1)

            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-16) # cos相似度
            attention_loss = abs(cos(querys[0], querys[1])) + abs(cos(querys[1], querys[2])) \
                                                            + abs(cos(querys[0], querys[2]))
            # 计算两两查询向量间的相似度之和
            return output_distribution, attention_weights, attention_loss
            # output_distribution是预测分类的概率，attention_loss是两两查询向量间的相似度之和（两结果优化时使用）

