import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os

INP_DATA_PATH = 'mat_comp'
TST_DATA_PATH = 'test_data.pt'
TR_DATA_PATH = 'training_data.pt'
PARAM_PATH = 'parameters.pt'
MODEL_PATH = 'model.pt'

BATCH_SIZE = 500
RANK = 50
EPOCHS = 10000000
LEARNING_RATE = 0.01
CROSS_VALIDATE = True
# rank = 20 => train loss = 1, test loss = 1.4 1000 epochs
# rank = 30 => 0.85 and 1.2 2200 epochs
# rank 30 with adam and svd/10: 0.9 and 1, 442 epochs
# batch 50
# TODO: see hints
# TODO: consider batch sizes
class MovieRecoModel(torch.nn.Module):
    def __init__(self, num_features, num_users, num_movies, M):
        super().__init__()
        # num_features is rank
        U, S, Vh = torch.linalg.svd(M, full_matrices = False)
        # TODO: remove this assertion and make code robust
        assert S.size(0) > num_features

        U = U[:, :num_features]
        S = S[:num_features] / 10
        Vh = Vh[:num_features, :]

        self.user_to_feature = nn.Parameter(nn.Parameter(U))
        self.movie_to_feature = nn.Parameter((torch.diag(S) @ Vh).t())

    def forward(self, user, movie):
        user_features = self.user_to_feature[user]
        movie_features = self.movie_to_feature[movie]
        return (user_features * movie_features).sum(1)


def process_data():
    if os.path.exists(TST_DATA_PATH) \
            and os.path.exists(TR_DATA_PATH) \
            and os.path.exists(PARAM_PATH):
        print("using previously processed data")
        training_data = torch.load(TR_DATA_PATH)
        test_data = torch.load(TST_DATA_PATH)
        parameters = torch.load(PARAM_PATH)
        return parameters[0].item(), parameters[1].item(), training_data, test_data

    with open(INP_DATA_PATH, "r") as file:
        print("Could not find processed data. Processing from scratch")
        n, m, k = map(int, file.readline().split())
        training_data = torch.zeros(k, 3, dtype=torch.float32)
        for index in range(k):
            i, j, M_ij = map(float, file.readline().split())
            # we want i, j to be 0 indexed instead of 1 indexed.
            training_data[index] = torch.tensor([i - 1, j - 1, M_ij])
        q = int(file.readline())
        test_data = torch.zeros(q, 2, dtype=torch.int64)
        for index in range(q):
            i, j = map(int, file.readline().split())
            test_data[index] = torch.tensor([i - 1, j - 1])
        # save data in files
        torch.save(training_data, TR_DATA_PATH)
        torch.save(test_data, TST_DATA_PATH)
        torch.save(torch.tensor([n, m, k]), PARAM_PATH)
        return n, m, training_data, test_data


def train():
    # TODO: check if path already exists
    writer = SummaryWriter('runs')
    num_users, num_movies, training_data, test_data = process_data()
    print("received input data")
    validation_data = None
    num_training_data = training_data.size(0)
    if CROSS_VALIDATE:
        print("cross validate was set to true, partitioning training data")
        shuffled_indices = torch.randperm(num_training_data)
        shuffled_train_data = training_data[shuffled_indices]
        num_training_data = int(num_training_data * 0.9)
        training_data = shuffled_train_data[:num_training_data]
        validation_data = shuffled_train_data[num_training_data:]
        num_validation_data = validation_data.size(0)
    M = torch.zeros(num_users, num_movies, dtype=torch.float32)
    for i, j, M_ij in training_data:
        M[int(i.item()), int(j.item())] = M_ij.item()
    rating_pred_model = MovieRecoModel(RANK, num_users, num_movies, M)
    loss_func = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(rating_pred_model.parameters(), lr=LEARNING_RATE)\
        #torch.optim.Adam(rating_pred_model.parameters(), lr=LEARNING_RATE) #torch.optim.SGD(rating_pred_model.parameters(), lr=LEARNING_RATE)
    #breakpoint()
    I = training_data[:, 0].to(int)
    J = training_data[:, 1].to(int)
    M_IJ = training_data[:, 2]
    try:
        for epoch in range(EPOCHS):
            print("epoch: ", epoch)
            shuffled_indices = torch.randperm(num_training_data)
            Is = I[shuffled_indices]
            Js = J[shuffled_indices]
            M_IJs = M_IJ[shuffled_indices]
            # TODO does not require grad
            avg_train_loss = 0

            updates_per_epoch = num_training_data // BATCH_SIZE
            for update in range(updates_per_epoch):
                #print(update, updates_per_epoch)
                Ib = Is[update*BATCH_SIZE:(update+1)*BATCH_SIZE]
                Jb = Js[update * BATCH_SIZE:(update + 1) * BATCH_SIZE]
                M_IJb = M_IJs[update * BATCH_SIZE:(update + 1) * BATCH_SIZE]
                optimizer.zero_grad()
                output = rating_pred_model(Ib, Jb)
                loss = loss_func(output, M_IJb)
                avg_train_loss += loss
                loss.backward()
                optimizer.step()
            avg_train_loss = avg_train_loss / updates_per_epoch
            print("train loss:", avg_train_loss)

            x = validation_data[:, 0].to(int)
            y = validation_data[:, 1].to(int)
            z = validation_data[:, 2]
            print("validation loss: ", loss_func(rating_pred_model(x, y), z))




            # if CROSS_VALIDATE:
            #         shuffled_indices = torch.randperm(num_validation_data)
            #         shuffled_validation_data = validation_data[shuffled_indices]
            #         validation_size = 100
            #         avg_test_loss = 0
            #         for i, j, M_ij in shuffled_validation_data[:validation_size]:
            #             avg_test_loss += \
            #                 loss_func(rating_pred_model(torch.LongTensor([i.item()]), torch.LongTensor([j.item()])),
            #                       torch.FloatTensor([M_ij.item()]))
            #         avg_test_loss = avg_test_loss / validation_size
            #         writer.add_scalar('training_loss', avg_train_loss, epoch)
            #         writer.add_scalar('test_loss', avg_test_loss, epoch)
            #         print("training_loss", epoch, update, avg_train_loss)
            #         print("test_loss", avg_test_loss)
    except KeyboardInterrupt:
        torch.save(rating_pred_model.state_dict(), MODEL_PATH)
    # TODO: save model
    # TODO: test data output


def main():
    train()


if __name__ == '__main__':
    main()
