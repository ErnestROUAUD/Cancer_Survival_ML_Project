import pandas as pd
import torch, os
from torch import nn
from torch.optim import Adam


class ClinicalBranch(nn.Module):
    def __init__(self, n_feature_cli: int):
        super().__init__()
        self.lin = nn.Linear(n_feature_cli, 32)
        self.bn = nn.BatchNorm1d(32)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.lin(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class ExpressionBranch(nn.Module):
    def __init__(self, n_feature_exp: int):
        super().__init__()
        self.lin = nn.Linear(n_feature_exp, 64)
        self.bn = nn.BatchNorm1d(64)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.lin(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class DualBranchSurvival(nn.Module):
    def __init__(self, n_feature_cli: int, n_feature_exp: int):
        super().__init__()
        self.ClinicalBranch = ClinicalBranch(
            n_feature_cli
        )
        self.ExpressionBranch = ExpressionBranch(
            n_feature_exp
        )

        merged_input_dimension = 64 + 32

        self.merged = nn.Sequential(
            nn.Linear(merged_input_dimension, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1))

    def forward(self, clinical, expression):
        clinical_out = self.ClinicalBranch(clinical)
        expression_out = self.ExpressionBranch(expression)
        merged = torch.cat([clinical_out, expression_out], dim=1)
        final = self.merged(merged)
        return final
    
    
def Survival_Model(Clinical_train_df, Expression_train_df, Clinical_test_df, Expression_test_df, y_train, y_test, Sel_process = str, loss_fn = str, n_epochs=int, max_epochs=int, batch_size=int, patience = int):
    
    overall_best_loss = float('inf')
    overall_best_model = None
    overall_best_info = {}

    n_feature_cli=Clinical_train_df.shape[1]
    n_feature_exp=Expression_train_df.shape[1]

    training_df = pd.DataFrame()
    testing_df = pd.DataFrame()

    for run in range(5):

        model = DualBranchSurvival(n_feature_cli=n_feature_cli, n_feature_exp=n_feature_exp)
        optimizer = Adam(params=model.parameters(), lr=0.001, weight_decay=1e-4)

        training = {'train_loss': [], 'train_acc': []}
        testing = {'test_loss': [], 'test_acc': []}

        best_loss = float('inf')
        patience_counter = 0
        best_model_this_run = None
        best_epoch_this_run = None

        for epoch in range(min(n_epochs, max_epochs)):
            model.train()
            indice = torch.randperm(len(Clinical_train_df))
            X_clinical_train = Clinical_train_df[indice]
            X_expression_train = Expression_train_df[indice]
            y_train_shuffled = y_train[indice]

            # Step through every patient, including the final partial batch —
            # len(X_clinical_train) // batch_size previously floored down and
            # silently dropped the last ~35 patients from every epoch.
            for start in range(0, len(X_clinical_train), batch_size):
                cli_batch = X_clinical_train[start:start+batch_size]
                exp_batch = X_expression_train[start:start+batch_size]
                ybatch = y_train_shuffled[start:start+batch_size]

                y_pred = model(cli_batch, exp_batch)
                loss = loss_fn(y_pred, ybatch)
                acc = (torch.sigmoid(y_pred).round() == ybatch).float().mean()

                training['train_loss'].append(loss.item())
                training['train_acc'].append(acc.item())

                # backward step:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # evaluate the model on the test set each epoch
            model.eval()
            with torch.no_grad():
                y_pred = model(Clinical_test_df, Expression_test_df)
                loss = loss_fn(y_pred, y_test)
                testing['test_loss'].append(loss.item())
                acc = (torch.sigmoid(y_pred).round() == y_test).float().mean()
                testing['test_acc'].append(acc.item())

                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                    best_model_this_run = model.state_dict()
                    best_epoch_this_run = epoch
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("best model reached, early stopping")
                        break

        model.load_state_dict(best_model_this_run)

        if best_loss < overall_best_loss:
            overall_best_loss = best_loss
            overall_best_model = best_model_this_run
            overall_best_info = {
                'run': run,
                'epoch': best_epoch_this_run,
                'test_loss': best_loss.item(),
                'test_acc': acc.item()
            }

        train_col = pd.DataFrame({f'loss_{run}': training['train_loss']})
        train_col[f'acc_{run}'] = pd.Series(training['train_acc'])

        test_col = pd.DataFrame({f'loss_{run}': testing['test_loss']})
        test_col[f'acc_{run}'] = pd.Series(testing['test_acc'])

        training_df = pd.concat([training_df, train_col], axis=1)
        testing_df = pd.concat([testing_df, test_col], axis=1)

    os.makedirs(os.path.join(os.getenv("Output_path"), "Best_models_path"), exist_ok=True)
    torch.save({
        'model_state_dict': overall_best_model,
        'n_feature_cli': n_feature_cli,
        'n_feature_exp': n_feature_exp,
        'run': overall_best_info['run'],
        'epoch': overall_best_info['epoch'],
        'test_loss': overall_best_info['test_loss'],
        'test_acc': overall_best_info['test_acc']
    }, os.path.join(os.getenv("Best_models_path"), f'{Sel_process}_best_model.pt'))

    os.makedirs(os.path.join(os.getenv("Output_path"), "Training_folder_path"), exist_ok=True)
    training_df.to_csv(os.path.join(os.getenv("Training_folder_path"), f'{Sel_process}_training_df.csv'), header=True)
    os.makedirs(os.path.join(os.getenv("Output_path"), "Testing_folder_path"), exist_ok=True)
    testing_df.to_csv(os.path.join(os.getenv("Testing_folder_path"), f'{Sel_process}_testing_df.csv'), header=True)

    print(f'best model - run: {overall_best_info['run']}')
    print(f'best model - epoch: {overall_best_info['epoch']}')
    print(f'best model - test accuracy: {overall_best_info['test_acc']}')
    print(f'best model - test loss: {overall_best_info['test_loss']}')