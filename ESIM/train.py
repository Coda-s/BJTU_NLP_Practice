
import torch
import torch.optim as optim
import torch.nn.functional as F

def eval(iter, model, args):
    print("Evaluating...")
    model.eval()

    confusion_matrix = torch.zeros(2, 2)

    correct_num = 0
    total_num = 0

    for batch in iter:
        sentence1 = batch.sentence1.t()
        sentence2 = batch.sentence2.t()
        targets = batch.gold_label.squeeze(0)

        predictions = model(sentence1, sentence2)
        predictions = torch.max(predictions, 1)[1]

        correct_num += (predictions == targets).sum()
        total_num += len(targets)
    
    accuracy = correct_num / total_num * 100

    print("accuracy = {} / {} = {:.2f}".format(correct_num, total_num, accuracy))

        # for prediction, target in zip(predictions, targets):
        #     confusion_matrix[prediction, target] += 1
    
    # TP = confusion_matrix[0, 0]
    # TN = confusion_matrix[0, 1]
    # FP = confusion_matrix[1, 0]
    # FN = confusion_matrix[1, 1]

    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)

    # F1 = 2 * precision * recall / (precision + recall)

    # print("TP : {} | TN : {}\nFP : {} | FN : {}".format(TP, TN, FP, FN))
    # print("precision = {}\nrecall = {}\nF1 = {}".format(precision, recall, F1))

        

def train(train_iter, valid_iter, model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(1, args.epoches+1):
        print("Epoch : {}".format(epoch))
        # 训练
        model.train()
        for batch in train_iter:
            model.zero_grad()
            
            sentence1 = batch.sentence1.t()
            sentence2 = batch.sentence2.t()
            target = batch.gold_label.squeeze(0)
            
            
            logit = model(sentence1, sentence2)
            
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
        # 评估
        eval(valid_iter, model, args)
        torch.save(model, "./model/model_{}.pkl".format(epoch))
