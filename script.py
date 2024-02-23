import argparse
import csv
import torch
import torch.nn as nn

# File paths for input, validation, and test datasets, as well as model parameters and results
INP_USERS = 'inp_users.pt'
INP_MOVIES = 'inp_movies.pt'
INP_RATINGS = 'inp_ratings.pt'
VDN_USERS = 'vdn_users.pt'
VDN_MOVIES = 'vdn_movies.pt'
VDN_RATINGS = 'vdn_ratings.pt'
TST_USERS = 'test_users.pt'
TST_MOVIES = 'test_movies.pt'
PARAMETERS = 'parameters.pt'
INP_DATA_PATH = 'mat_comp'
RESULTS = 'results.csv'
PREDICTIONS = 'mat_comp_ans'

# Number of epochs for training
EPOCHS = 32

def process_data():
    """Read and process raw data from file, split into training and validation sets, and save to disk."""
    with open(INP_DATA_PATH, "r") as file:
        # Read the first line to get dimensions of the matrix
        n, m, k = map(int, file.readline().split())
        user_indexes = []
        movie_indexes = []
        ratings = []
        # Read user-movie rating data
        for index in range(k):
            i, j, M_ij = map(float, file.readline().split())
            i, j = int(i), int(j)
            # we want i, j to be 0 indexed instead of 1 indexed.
            user_indexes.append(i - 1)
            movie_indexes.append(j - 1)
            ratings.append(M_ij)
        # Read test data
        q = int(file.readline())
        test_user_indexes = []
        test_movie_indexes = []
        for index in range(q):
            i, j = map(int, file.readline().split())
            test_user_indexes.append(i - 1)
            test_movie_indexes.append(j - 1)
        user_indexes = torch.tensor(user_indexes)
        movie_indexes = torch.tensor(movie_indexes)
        ratings = torch.tensor(ratings)
        test_user_indexes = torch.tensor(test_user_indexes)
        test_movie_indexes = torch.tensor(test_movie_indexes)
        # shuffle data to create unbiased validation set
        num_training_data = int(k * 0.9)
        shuffled_indices = torch.randperm(k)
        shuffled_user_indexes = user_indexes[shuffled_indices]
        shuffled_movie_indexes = movie_indexes[shuffled_indices]
        shuffled_ratings = ratings[shuffled_indices]
        # save data in files
        torch.save(shuffled_user_indexes[:num_training_data], INP_USERS)
        torch.save(shuffled_movie_indexes[:num_training_data], INP_MOVIES)
        torch.save(shuffled_ratings[:num_training_data], INP_RATINGS)
        torch.save(shuffled_user_indexes[num_training_data:], VDN_USERS)
        torch.save(shuffled_movie_indexes[num_training_data:], VDN_MOVIES)
        torch.save(shuffled_ratings[num_training_data:], VDN_RATINGS)
        torch.save(test_user_indexes, TST_USERS)
        torch.save(test_movie_indexes, TST_MOVIES)
        # Save dimensions as parameters
        torch.save(torch.tensor([n, m, k]), PARAMETERS)


def load_data():
    """Load processed training, validation, and test datasets from disk."""
    tr_user_indexes = torch.load(INP_USERS)
    tr_movie_indexes = torch.load(INP_MOVIES)
    tr_ratings = torch.load(INP_RATINGS)
    vdn_user_indexes = torch.load(VDN_USERS)
    vdn_movie_indexes = torch.load(VDN_MOVIES)
    vdn_ratings = torch.load(VDN_RATINGS)
    tst_user_indexes = torch.load(TST_USERS)
    tst_movie_indexes = torch.load(TST_MOVIES)
    parameters = torch.load(PARAMETERS)
    return parameters, tr_user_indexes, tr_movie_indexes, tr_ratings, vdn_user_indexes, vdn_movie_indexes, \
           vdn_ratings, tst_user_indexes, tst_movie_indexes


class MovieRecoModel(torch.nn.Module):
    '''Low rank matrix completion model to predict user ratings'''
    def __init__(self, num_features, num_users, num_movies):
        super().__init__()
        # initialize using standard normal distribution.
        self.user_to_feature = nn.Parameter(torch.randn(num_users, num_features))
        self.movie_to_feature = nn.Parameter(torch.randn(num_movies, num_features))

    def forward(self, user, movie):
        # Compute the dot product of user and movie features to predict rating
        user_features = self.user_to_feature[user]
        movie_features = self.movie_to_feature[movie]
        return (user_features * movie_features).sum(1)


def run_train_loop(model, optimizer, loss_func, shuff_tr_user, shuff_tr_movie, shuff_tr_rating,
                   num_training_data, batch_size):
    """Execute the training loop for one epoch."""
    avg_train_loss = 0
    updates_per_epoch = num_training_data // batch_size
    for update in range(updates_per_epoch):
        # Slice the batch from shuffled dataset
        user_batch = shuff_tr_user[update * batch_size:(update + 1) * batch_size]
        movie_batch = shuff_tr_movie[update * batch_size:(update + 1) * batch_size]
        rating_batch = shuff_tr_rating[update * batch_size:(update + 1) * batch_size]
        # Zero the gradients before running the backward pass.
        optimizer.zero_grad()
        output = model(user_batch, movie_batch)
        loss = loss_func(output, rating_batch)
        avg_train_loss += loss
        # Backpropagation
        loss.backward()
        optimizer.step()
    avg_train_loss = avg_train_loss / updates_per_epoch
    return avg_train_loss


def run_validation(vdn_user_indexes, vdn_movie_indexes, vdn_ratings, loss_func, model):
    """Evaluate the model on the validation set."""
    return loss_func(model(vdn_user_indexes, vdn_movie_indexes), vdn_ratings)


def write_predictions(model, user_indexes, movie_indexes):
    """Generate predictions for the test dataset and write to file."""
    breakpoint()
    predictions = model(user_indexes, movie_indexes)
    with open(PREDICTIONS, "w") as file:
        for prediction in predictions:
            file.write(str(prediction.item()))
            file.write('\n')


def train(rank, batch_size, learning_rate, reg_factor):
    """Main training function."""
    print("rank: ", rank, "batch size: ", batch_size, "learning_rate: ", learning_rate, "reg factor: ", reg_factor)
    # Load data
    parameters, tr_user_indexes, tr_movie_indexes, tr_ratings, vdn_user_indexes, vdn_movie_indexes, \
    vdn_ratings, tst_user_indexes, tst_movie_indexes = load_data()
    num_training_data = tr_ratings.shape[0]
    # Initialize model, loss function, and optimizer
    rating_pred_model = MovieRecoModel(rank, parameters[0].item(), parameters[1].item())
    mse_func = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(rating_pred_model.parameters(), lr=learning_rate, weight_decay=reg_factor)
    # Training loop
    for epoch in range(EPOCHS):
        shuffled_indices = torch.randperm(num_training_data)
        shuff_tr_user = tr_user_indexes[shuffled_indices]
        shuff_tr_movie = tr_movie_indexes[shuffled_indices]
        shuff_tr_rating = tr_ratings[shuffled_indices]
        test_loss = run_train_loop(rating_pred_model, optimizer, mse_func, shuff_tr_user, shuff_tr_movie,
                                   shuff_tr_rating, num_training_data, batch_size)
        validation_loss = run_validation(vdn_user_indexes, vdn_movie_indexes, vdn_ratings, mse_func, rating_pred_model)
        print("EPOCH: ", epoch)
        print("training loss: ", test_loss)
        print("validation loss: ", validation_loss)
    # Save the trained model
    torch.save(rating_pred_model.state_dict(),
               "model:{}:{}:{}:{}.pt".format(rank, batch_size, learning_rate, reg_factor))
    write_predictions(rating_pred_model, tst_user_indexes, tst_movie_indexes)
    return test_loss, validation_loss


def main(mode):
    """Main function to process data or train the model based on the input mode."""
    if mode == "process_data":
        process_data()
    elif mode == "train":
        # Define training configurations
        ranks = [25]
        batch_sizes = [1000]
        learning_rates = [0.001]
        reg_factors = [0.00001]
        with open(RESULTS, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["batch_size", "learning_rate", "rank", "reg_factor", "test_loss", "validation_loss"])
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    for rank in ranks:
                        for reg_factor in reg_factors:
                            test_loss, validation_loss = train(rank, batch_size, learning_rate, reg_factor)
                            writer.writerow(
                                [batch_size, learning_rate, rank, reg_factor, test_loss.item(), validation_loss.item()])
    else:
        print("INVALID MODE")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        help='Must be process_data or train')
    args = parser.parse_args()
    main(args.mode)
