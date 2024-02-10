import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os

INP_DATA_PATH = 'mat_comp'
TST_DATA_PATH = 'test_data.pt'
TR_DATA_PATH = 'training_data.pt'
PARAM_PATH = 'parameters.pt'

RANK = 20
EPOCHS = 10000000
LEARNING_RATE = 0.001
CROSS_VALIDATE = True


# TODO: see hints
# TODO: consider batch sizes
class MovieRecoModel(torch.nn.Module):
    def __init__(self, num_features, num_users, num_movies, M):
        super().__init__()
        # num_features is rank
        U, S, Vh = torch.linalg.svd(M)
        # TODO: remove this assertion and make code robust
        assert S.size(0) > num_features
        U = U[:, :num_features]
        S = S[:num_features]
        Vh = Vh[:num_features, :]

        self.user_to_feature = nn.Parameter(torch.randn(num_users, num_features))#nn.Parameter(U)
        self.movie_to_feature = nn.Parameter(torch.randn(num_movies, num_features))#(torch.diag(S) @ Vh).t())

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
    optimizer = torch.optim.SGD(rating_pred_model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        print("epoch: ", epoch)
        shuffled_indices = torch.randperm(num_training_data)
        shuffled_train_data = training_data[shuffled_indices]
        epoch_size = 10000
        avg_train_loss = 0
        for index in range(epoch_size):
            optimizer.zero_grad()
            i, j, M_ij = shuffled_train_data[index]
            output = rating_pred_model(torch.LongTensor([i.item()]), torch.LongTensor([j.item()]))
            loss = loss_func(output, torch.FloatTensor([M_ij.item()]))
            avg_train_loss += loss
            loss.backward()
            optimizer.step()
        avg_train_loss = avg_train_loss / epoch_size
        if CROSS_VALIDATE:
            shuffled_indices = torch.randperm(num_validation_data)
            shuffled_validation_data = validation_data[shuffled_indices]
            validation_size = 1000
            avg_test_loss = 0
            for i, j, M_ij in shuffled_validation_data[:validation_size]:
                avg_test_loss += \
                    loss_func(rating_pred_model(torch.LongTensor([i.item()]), torch.LongTensor([j.item()])),
                              torch.FloatTensor([M_ij.item()]))
            avg_test_loss = avg_test_loss / validation_size
        writer.add_scalar('training_loss', avg_train_loss, epoch)
        writer.add_scalar('test_loss', avg_test_loss, epoch)
        print("training_loss", avg_train_loss)
        print("test_loss", avg_test_loss)

    # TODO: save model
    # TODO: test data output


def main():
    train()


if __name__ == '__main__':
    main()
