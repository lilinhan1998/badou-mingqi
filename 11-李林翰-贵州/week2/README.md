学员李林翰第一次作业 将只有一层的rnn改成有三层的结构 明白了nn.RNN的使用方法 
助教带来的提示：对于nn.RNN的输出有output和h，文本分类问题中，可以直接选择第1维进行分类（h），也可以选择第0维之后进行池化层进行维度缩减，再进行分类 
output,h=nn.RNN() 当batch_first=True或False时，对output的输出维度没有影响，对hidden输出的维度有影响 当batch_first=True时，hidden的输出为最后时间步的隐藏状态[num_layers,batch_size,num_hidden] 当batch_size=False时，hidden的输出为[num_layers,sentence_length(time_seq),num_hidden]
