
import torch
import torch.nn as nn

def argmax(vec):
    '''
    返回最大值的下标
    '''
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_boardcast = max_score.view(1,-1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec-max_score_boardcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, args):

        super(BiLSTM_CRF, self).__init__()

        self._vocab_size = args.vocab_size       # 词汇表大小
        self._embedding_dim = args.embedding_dim # 词向量维度
        self._hidden_dim = args.hidden_dim       # 隐藏层维度
        self._tagset_size = args.tagset_size     # 标签集大小
        self._tag_to_idx = args.tag_to_idx       # 标签到下标的映射
        self.START_TAG = args.START_TAG
        self.STOP_TAG = args.STOP_TAG

        # 词向量层
        self.embedding = nn.Embedding(self._vocab_size, self._embedding_dim)
        # LSTM层
        self.lstm = nn.LSTM(self._embedding_dim, self._hidden_dim//2, num_layers=1, bidirectional=True)
        # 线性层
        self.linear = nn.Linear(self._hidden_dim, self._tagset_size)

        # CRF
        # 变换矩阵（从 j 到 i ）
        self.transitions = nn.Parameter(torch.randn(self._tagset_size, self._tagset_size))
        # 初始化设置
        self.transitions.data[self._tag_to_idx[self.START_TAG], :] = -10000
        self.transitions.data[:, self._tag_to_idx[self.STOP_TAG]] = -10000

    def _init_hidden(self):
        return (torch.randn(2, 1, self._hidden_dim // 2),
                torch.randn(2, 1, self._hidden_dim // 2))

    def _get_lstm_feat(self, sentence):
        self.hidden = self._init_hidden()
        embed = self.embedding(sentence)
        lstm_out, self.hidden = self.lstm(embed, self.hidden)
        lstm_feat = self.linear(lstm_out.squeeze(1))
        return lstm_feat

    def _foward_alg(self, feats):
        '''
        前向算法计算得分
        '''
        # 初始化
        init_alphas = torch.full((1, self._tagset_size), -10000.0)
        init_alphas[0][self._tag_to_idx[self.START_TAG]] = 0
        # 包装为一个变量便于反向传播
        foward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self._tagset_size):
                # 发射得分
                emit_score = feat[next_tag].view(1,-1).expand(1, self._tagset_size)
                # 转移得分 
                trans_score = self.transitions[next_tag].view(1, -1)
                # 当前 tag得分 
                next_tag_var = foward_var + emit_score + trans_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            foward_var = torch.cat(alphas_t).view(1, -1)
        final_var = foward_var + self.transitions[self._tag_to_idx[self.STOP_TAG]]
        return log_sum_exp(final_var)
        
    def _sentence_score(self, feats, targets):
        '''
        根据已知序列计算得分
        '''
        score = torch.zeros(1)
        targets = torch.cat([torch.tensor([self._tag_to_idx[self.START_TAG]], dtype=torch.long), targets])

        for i, feat in enumerate(feats):
            score = score + self.transitions[targets[i+1], targets[i]] + feat[targets[i+1]]
        score = score + self.transitions[targets[self._tag_to_idx[self.STOP_TAG]], targets[-1]]
        return score

    def _viterbi_decode(self, feats):
        '''
        动态规划求解最优值以及对应路径
        '''
        backtrace = []

        init_vars = torch.full((1, self._tagset_size), -10000.0)
        init_vars[0][self._tag_to_idx[self.START_TAG]] = 0

        foward_var = init_vars

        for feat in feats:
            last = []
            max_val = []
            for next_tag in range(self._tagset_size):
                next_tag_var = foward_var + self.transitions[next_tag]
                max_idx = argmax(next_tag_var)
                last.append(max_idx)
                max_val.append(next_tag_var[0][max_idx].view(1))
            foward_var = (torch.cat(max_val) + feat).view(1, -1)
            backtrace.append(last)
        final_score = foward_var + self.transitions[self._tag_to_idx[self.STOP_TAG]]
        max_idx = argmax(final_score)
        max_val = final_score[0][max_idx].view(1)

        path = [max_idx]
        for last in reversed(backtrace):
            max_idx = last[max_idx]
            path.append(max_idx)
        
        start = path.pop()
        # 验证最终结点为start结点
        assert start == self._tag_to_idx[self.START_TAG]
        path.reverse()
        return max_val, path


    def neg_log_likelihood(self, sentence, targets):
        '''
        计算负对数似然损失函数
        '''
        feats = self._get_lstm_feat(sentence)
        foward_score = self._foward_alg(feats)
        gold_score = self._sentence_score(feats, targets)
        return foward_score - gold_score

    def forward(self, sentence):
        '''
        返回最优值和对应路径
        '''
        feats = self._get_lstm_feat(sentence)
        score, tags = self._viterbi_decode(feats)
        return score, tags

