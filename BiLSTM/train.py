import torch
import torch.optim as optim

def get_mrteics(predictions, targets, args):
    O_idx = args.tag_to_idx["O"]
    TP, FN, FP = 0, 0, 0
    for prediction, target in zip(predictions, targets):
        if prediction == target.item() and prediction != O_idx:
            TP += 1
        if prediction != target.item() and target.item() != O_idx:
            FN += 1
        if prediction != target.item() and prediction != O_idx:
            FP += 1
    return TP, FN, FP
def eval(valid_iter, model, args):
    TP = 0
    FN = 0
    FP = 0
    for batch in valid_iter:
        sentence = batch.text
        targets = batch.label.squeeze(1)
        score, sequence = model(sentence)
        _TP, _FN, _FP = get_mrteics(sequence, targets)
        TP += _TP
        FN += _FN
        FP += _FP
    precision = TP / (TP + FN)
    recall = TP / (TP +  FP)
    F1 = (precision * recall * 2) / (precision + recall)
    print("precision = {:.2f}%, recall = {:.2f}%, F1 = {:.2f}%".format(precision*100, recall*100, F1*100))

def train(train_iter, valid_iter, model, args):
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        print("Epoch : {}".format(epoch))
        for batch in train_iter:
            # 设置为 0梯度
            model.zero_grad()
            # 准备样本参数
            sentence = batch.text
            targets = batch.label.squeeze(1)
            # 计算loss损失
            loss = model.neg_log_likelihood(sentence, targets)
            # 回溯 loss 累积梯度
            loss.backward()
            optimizer.step()
        eval(valid_iter, model, args)
