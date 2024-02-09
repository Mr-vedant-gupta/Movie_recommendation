import torch
import torch.nn as nn

# TODO: see hints
# TODO: consider batch sizes
class MovieRecoModel(torch.nn.Module):
    def __init__(self, num_features, num_users, num_movies):
        # num_features is rank
        self.user_to_feature = nn.Parameter(torch.randn(num_users, num_features))
        self.feature_to_movie = nn.Parameter(torch.randn(num_features, num_movies))

    def forward(self, user, movie):
        user_features = self.user_to_feature[user]
        movie_features = self.feature_to_movie[movie]
        return torch.sum(user_features * movie_features)


def process_data(path):
    with open(path, "r") as file:
        n, m, k = map(int, file.readline().split())
        training_data = torch.zeros(k, 3, dtype=torch.float32)
        for index in range(k):
            i, j, M_ij = map(int, file.readline().split()))
            training_data[index] = torch.tensor([i, j, M_ij])
        q = int(file.readline())
        test_data = torch.zeros(q, 2, dtype=torch.int64)
        for index in range(q):
            i, j = map(int, file.readline().split()))
            test_data[index] = torch.tensor([i, j])
        # save data in files
        torch.save(training_data, 'training_data.pt')
        torch.save(test_data, 'test_data.pt')
        return n, m, training_data, test_data

def train(epochs, rank, input_file_path, learning_rate, cross_validate):
    #TODO: check if path already exists
    num_users, num_movies, training_data, test_data = process_data(input_file_path)
    validation_data = None
    rating_pred_model = MovieRecoModel(rank, num_users, num_movies)
    loss_func = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(rating_pred_model.parameters(), lr=learning_rate)
    num_training_data = training_data.size(0)
    if cross_validate:
        shuffled_indices = torch.randperm(num_training_data)
        shuffled_train_data = training_data(shuffled_indices)
        num_training_data = int(num_training_data * 0.9)
        training_data = shuffled_train_data[:num_training_data]
        validation_data = shuffled_train_data[num_training_data:]
        num_validation_data = validation_data.size(0)
    for epoch in range(epochs):
        shuffled_indices = torch.randperm(num_training_data)
        shuffled_train_data = training_data(shuffled_indices)
        for index in range(num_training_data):
            i, j, M_ij = shuffled_train_data[index]
            output = rating_pred_model(i, j)
            loss = loss_func(output, M_ij)
            loss.backward()
            optimizer.step()
        if cross_validate:
            print("Cross validation results after epoch: ", epoch)
            sum = 0
            for i, j, M_ij in validation_data:
                sum += (rating_pred_model(i, j) - M_ij)**2
            print("Averaged MSE:", sum / num_validation_data)
    #TODO: save model




def main():
    input_file_path = 'mat_comp'
    rank = 5
    epochs = 5
    learning_rate = 0.001
    cross_validate = True
    train(epochs, rank, input_file_path, learning_rate, cross_validate)















