import os
import json
import h5py
import argparse
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import set_seed
import numpy as np
import wandb


def load_hdf5_by_indices(file_path, indices, dataset_name='activations'):
    with h5py.File(file_path, "r") as f:
        dataset = f[dataset_name][:]

    data = torch.tensor(dataset, dtype=torch.float32)  # 转换为 torch.Tensor
    selected_data = data[indices]  # 选择指定的索引
    return selected_data


class ProbeDataset(Dataset):
    def __init__(self, data_path, activations_name, target_layer, target_head, tag='train'):
        super().__init__()
        if '.jsonl' in data_path:
            print(f"Loading data from {data_path}")
            self.data = [json.loads(line) for line in open(data_path)]

            print(f"Loading vectors from {os.path.join(os.path.dirname(data_path), activations_name)}")
            if 'activations_index' in self.data[0]:
                activations_indices = [instance['activations_index'] for instance in self.data]
                print(len(activations_indices), activations_indices[0])
                self.vectors = load_hdf5_by_indices(os.path.join(os.path.dirname(data_path), activations_name), activations_indices)
            else:
                self.vectors = torch.load(os.path.join(os.path.dirname(data_path), activations_name))
                if len(self.data) != self.vectors.shape[0]:
                    logging.info(f"Data length {len(self.data)} does not match vectors length {self.vectors.shape}")
                    assert 'projs_index' in self.data[0]
                    projs_indices = [instance['projs_index'] for instance in self.data]
                    self.vectors = self.vectors[projs_indices]
        else:
            raise NotImplementedError
        logging.info('Vectors shape: ' + str(self.vectors.shape))
        self.vectors = self.vectors.to(torch.float32)
        self.target_layer = target_layer
        self.target_head = target_head

        self.labels = [instance['label'] if 'label' in instance else instance['result'] for instance in self.data]

        self.tag = tag

    def get_d_model(self):
        return self.vectors.shape[-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.target_head != -1:
            return self.vectors[self.target_layer, self.target_head, idx], torch.tensor([self.labels[idx]], dtype=torch.float32)
        else:
            return self.vectors[idx, self.target_layer], torch.tensor([self.labels[idx]], dtype=torch.float32)
            # vectors: [bsz, num_layers, hidden_size]


class SingleLinearProber(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return F.sigmoid(self.hidden(x))


def compute_grouped_accuracy(ds, all_preds, all_targets):
    result_array = np.array([instance['result'] for instance in ds.data])  # 提取 result
    but_tag_array = np.array([instance['but_tag'] for instance in ds.data])  # 提取 but_tag
    all_preds = np.array(all_preds)  # 转换预测值
    all_targets = np.array(all_targets)  # 转换真实值

    accuracy_dict = {}
    for res_val in [True, False]:
        for but_val in [True, False]:
            # 筛选出 (result, but_tag) 组合的索引
            mask = (result_array == res_val) & (but_tag_array == but_val)

            # 计算当前组别的准确率
            if np.sum(mask) > 0:  # 确保该组合有数据
                acc = np.mean(all_preds[mask] == all_targets[mask])
            else:
                acc = None  # 该组别没有数据

            # 存入字典
            key = f"{res_val}_{but_val}"
            accuracy_dict[key] = acc

    return accuracy_dict


def compute_retraction_acc(ds, all_preds):
    result_array = np.array([instance['result'] for instance in ds.data])  # 提取 result
    but_tag_array = np.array([instance['but_tag'] for instance in ds.data])  # 提取 but_tag
    all_preds = np.array(all_preds)  # 转换预测值

    # 计算总的正确预测
    correct = ((all_preds == 0) & (but_tag_array == 1)) | ((all_preds == 1) & (but_tag_array == 0))
    total_acc = correct.mean()  # 总体准确率

    accuracy_dict = {}
    for res_val in [True, False]:
        for but_val in [True, False]:
            mask = (result_array == res_val) & (but_tag_array == but_val)

            if np.sum(mask) > 0:
                gt = 0 if but_val else 1
                acc = np.mean(all_preds[mask] == gt)
            else:
                acc = None

            key = f"{res_val}_{but_val}"
            accuracy_dict[key] = acc

    # 计算 all_preds = 1 的准确率
    mask_pred_1 = (all_preds == 1)
    num_correct_pred_1 = correct[mask_pred_1].sum()
    total_pred_1 = mask_pred_1.sum()
    acc_when_pred_1 = num_correct_pred_1 / total_pred_1 if total_pred_1 > 0 else 0.0

    # 计算 all_preds = 0 的准确率
    mask_pred_0 = (all_preds == 0)
    num_correct_pred_0 = correct[mask_pred_0].sum()
    total_pred_0 = mask_pred_0.sum()
    acc_when_pred_0 = num_correct_pred_0 / total_pred_0 if total_pred_0 > 0 else 0.0

    retraction_acc = {
        'total_acc': total_acc,
        'known_wrong_retraction_num': int(num_correct_pred_0),
        'known_wrong_num': int(total_pred_0),
        'known_wrong_retraction_acc': acc_when_pred_0,
        'known_right_retraction_num': int(num_correct_pred_1),
        'known_right_num': int(total_pred_1),
        'known_right_retraction_acc': acc_when_pred_1,
    }

    return retraction_acc, accuracy_dict


def evaluate(args, ds, loader, model, grouped_acc=False, retraction_acc=False, save_predictions=False, threshold=0.5):
    model.eval()
    with torch.no_grad():
        cum_loss = 0
        num_corr = 0
        all_preds, all_pred_scores, all_targets = [], [], []

        for _, (data, target) in enumerate(tqdm(loader)):
            data, target = data.to(args.accelerator), target.to(args.accelerator)

            # ➡ Forward pass
            prediction = model(data)
            loss = F.binary_cross_entropy(prediction, target.float())
            pred = prediction >= threshold
            cum_loss += loss.item()
            num_corr += torch.sum(pred == target)

            all_preds.extend(pred.squeeze().tolist())
            all_pred_scores.extend(prediction.squeeze().tolist())
            all_targets.extend(target.squeeze().tolist())

    # Calculate mean lss
    loss = cum_loss / len(loader)

    # Calculate accuracy
    acc = sum([1 for pred, target in zip(all_preds, all_targets) if pred == target]) / len(all_preds)

    # Calculate balanced accuracy
    pos_count = sum(all_targets)
    neg_count = len(all_preds) - pos_count

    pos_acc = sum([1 for pred, target in zip(all_preds, all_targets) if pred == target and target == 1]) / pos_count if pos_count > 0 else 0
    neg_acc = sum([1 for pred, target in zip(all_preds, all_targets) if pred == target and target == 0]) / neg_count if neg_count > 0 else 0
    balanced_acc = (pos_acc + neg_acc) / 2

    results = {
        "loss": loss,
        "acc": acc,
        "balanced_acc": balanced_acc,
        "pos_acc": pos_acc,
        "neg_acc": neg_acc
    }

    # 计算 grouped_acc
    if grouped_acc and 'but_tag' in ds.data[0]:
        results["grouped_acc"] = compute_grouped_accuracy(ds, all_preds, all_targets)

    # 计算 retraction_acc
    if retraction_acc and 'but_tag' in ds.data[0]:
        results["retraction_acc"], results['retraction_group_acc'] = compute_retraction_acc(ds, all_preds)

    if save_predictions:
        output_filename = os.path.join(args.save_dir, ds.tag, f"l{ds.target_layer}_predictions.jsonl")
        if not os.path.exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        # assert not os.path.exists(output_filename), f"File {output_filename} already exists. Please delete it first."
        with open(output_filename, 'w') as f:
            assert len(all_preds) == len(ds.data)
            for pred, pred_score, target, instance in zip(all_preds, all_pred_scores, all_targets, ds.data):
                if 'label' in instance:
                    assert target == instance['label']
                else:
                    assert target == instance['result']
                instance['probe_pred'] = pred
                instance['probe_pred_score'] = pred_score
                f.write(json.dumps(instance) + '\n')
        print(f"Saved predictions to {output_filename}")

    return results


def train(args):
    # Initialize a new wandb run
    wandb_prefix = args.wandb_prefix  # normally, train_data_dir/train_filename/model_name
    # hyperparameters
    if args.wandb_expnamesuffix is None:
        wandb_expname = f"prober_{wandb_prefix}_l{args.target_layer}_bs{args.batch_size}_lr{args.lr}_wd{args.weight_decay}_ep{args.num_epochs}"
    else:
        wandb_expname = f"prober_{wandb_prefix}_l{args.target_layer}_bs{args.batch_size}_lr{args.lr}_wd{args.weight_decay}_ep{args.num_epochs}_{args.wandb_expnamesuffix}"
    if args.target_head != -1:
        wandb_expname += f"_h{args.target_head}"

    wandb.init(project='probe', name=wandb_expname, config=args, mode='disabled' if args.disable_wandb else 'online')

    # Load data
    train_ds = ProbeDataset(args.train_filename, args.activations_name, args.target_layer, args.target_head)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
    logging.info(f"train set size: {len(train_ds)}")

    test_tag = args.test_tag[0]
    test_filename = args.test_filename[0]
    test_ds = ProbeDataset(test_filename, args.activations_name, args.target_layer, args.target_head, tag=test_tag)
    test_loader = DataLoader(test_ds, args.batch_size)
    logging.info(f"{args.test_tag[0]} test set size: {len(test_ds)}")

    d_model = train_ds.get_d_model()
    if args.singlelinear:
        model = SingleLinearProber(d_model, 1)
    else:
        raise NotImplementedError

    model.to(args.accelerator)
    results = {}

    if args.do_inference:
        logging.info(f"Doing inference on test set using model at {args.load_model_path}")
        model.load_state_dict(torch.load(args.load_model_path))
        for test_tag, test_filename in zip(args.test_tag[1:], args.test_filename[1:]):
            test_ds = ProbeDataset(test_filename, args.activations_name, args.target_layer, args.target_head, tag=test_tag)
            test_loader = DataLoader(test_ds, args.batch_size)

            result = evaluate(args, test_ds, test_loader, model, grouped_acc=args.compute_grouped_acc, retraction_acc=args.compute_retraction_acc, save_predictions=False, threshold=0.85)
            logging.info(f"Test Loss: {result['loss']}, Accuracy: {result['acc']}, Balanced acc: {result['balanced_acc']}, Positive acc: {result['pos_acc']}, Negative acc: {result['neg_acc']}, Grouped acc: True_True: {result['grouped_acc']['True_True']}, True_False: {result['grouped_acc']['True_False']}, False_True: {result['grouped_acc']['False_True']}, False_False: {result['grouped_acc']['False_False']}")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_test = {'loss': float("inf"), 'acc': 0, 'balanced_acc': 0, 'epoch': -1}
    best_model_path = os.path.join(args.save_dir, f"best-{wandb_expname}.pth")
    logging.info(f"Best model will be saved at {best_model_path}")
    for epoch in range(args.num_epochs):
        logging.info(f"Starting Epoch {epoch}")
        cumu_loss = 0
        num_corr = 0
        for _, (data, target) in enumerate(tqdm(train_loader)):
            model.train()
            data, target = data.to(args.accelerator), target.to(args.accelerator)
            optimizer.zero_grad()

            # ➡ Forward pass
            prediction = model(data)
            loss = F.binary_cross_entropy(prediction, target.float())
            pred = prediction >= 0.5
            cumu_loss += loss.item()
            num_corr += torch.sum(pred == target)

            # ⬅ Backward pass + weight update
            loss.backward()
            optimizer.step()
            wandb.log({"batch loss": loss.item()})

        result = evaluate(args, test_ds, test_loader, model)
        test_loss, test_acc, test_balanced_acc, test_pos_acc, test_neg_acc = result['loss'], result['acc'], result['balanced_acc'], result['pos_acc'], result['neg_acc']

        logging.info(f"Train Loss: {cumu_loss / len(train_loader)}, Train Accuracy: {num_corr / len(train_ds)}")
        logging.info(f"{test_tag} Loss: {test_loss}, Accuracy: {test_acc}, Balanced acc: {test_balanced_acc}, Positive acc: {test_pos_acc}, Negative acc: {test_neg_acc}")

        if best_test['balanced_acc'] < test_balanced_acc:
            best_test['loss'] = test_loss
            best_test['acc'] = test_acc
            best_test['balanced_acc'] = test_balanced_acc
            best_test['epoch'] = epoch
            torch.save(model.state_dict(), best_model_path)

        wandb.log({
            "Train_loss": cumu_loss / len(train_loader),
            "Train_acc": num_corr / len(train_ds),
            "Test_loss": test_loss,
            "Test_acc": test_acc,
            "Test_balanced_acc": test_balanced_acc,
            "Test_pos_acc": test_pos_acc,
            "Test_neg_acc": test_neg_acc,
            "epoch": epoch
        })

    model.load_state_dict(torch.load(best_model_path))

    assert len(args.test_tag) == len(args.test_filename)
    result = evaluate(args, test_ds, test_loader, model, grouped_acc=args.compute_grouped_acc, retraction_acc=args.compute_retraction_acc, save_predictions=True)
    results[test_tag] = result

    for test_tag, test_filename in zip(args.test_tag[1:], args.test_filename[1:]):
        logging.info(f"Evaluating on {test_tag} test set")

        test_ds = ProbeDataset(test_filename, args.activations_name, args.target_layer, args.target_head, tag=test_tag)
        test_loader = DataLoader(test_ds, args.batch_size)

        result = evaluate(args, test_ds, test_loader, model, grouped_acc=args.compute_grouped_acc, retraction_acc=args.compute_retraction_acc, save_predictions=True)
        results[test_tag] = result
        logging.info(
            f"{test_tag} Loss: {result['loss']}, Accuracy: {result['acc']}, Balanced acc: {result['balanced_acc']}, Positive acc: {result['pos_acc']}, Negative acc: {result['neg_acc']}")

    wandb.run.summary["best_test_loss"] = best_test['loss']
    wandb.run.summary["best_test_acc"] = best_test['acc']
    wandb.run.summary["best_test_balanced_acc"] = best_test['balanced_acc']
    wandb.run.summary["best_epoch"] = best_test['epoch']
    logging.info(f"Best test loss: {best_test['loss']}, Best test accuracy: {best_test['acc']}, Best test balanced acc: {best_test['balanced_acc']}, Best epoch: {best_test['epoch']}")

    wandb.finish()

    # Save model
    if args.save_checkpoints:
        model_name = f"{wandb_expname}.pth"
        model_save_path = os.path.join(args.save_dir, 'checkpoints', model_name)
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"Model saved at {model_save_path}")

    return results


def hparamsearch(args):
    return


def main(args):
    print(args)
    assert len(args.test_tag) == len(args.test_filename)
    for test_tag, test_filename in zip(args.test_tag, args.test_filename):
        print(f"Test tag: {test_tag}, Test filename: {test_filename}")

    if args.do_inference:
        args.disable_wandb = True
    if args.accelerator == "cuda":
        assert torch.cuda.is_available()

    set_seed(args.seed)

    if args.target_head != -1:
        raise NotImplementedError
    else:
        results = train(args)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input/Output
    parser.add_argument("--train_filename", type=str, required=True)
    parser.add_argument("--test_filename", nargs='+', type=str, required=True)
    parser.add_argument("--test_tag", nargs='+', type=str, required=True)
    parser.add_argument("--activations_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--load_model_path", type=str, help='For loading the model for inference.', default=None)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument('--save_checkpoints', action='store_true')

    # Probe setup
    parser.add_argument("--nonlinear", action="store_true")
    parser.add_argument("--singlelinear", action="store_true")
    parser.add_argument("--target_layer", type=int, default=25)
    parser.add_argument("--target_head", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--num_layers", type=int, default=32)

    # Experiment setup
    parser.add_argument('--wandb_prefix', type=str, required=True)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--wandb_hparam_search", action="store_true")
    parser.add_argument("--wandb_expnamesuffix", type=str)
    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do_inference", action="store_true")

    parser.add_argument('--compute_grouped_acc', action='store_true')
    parser.add_argument('--compute_retraction_acc', action='store_true')

    args = parser.parse_args()

    print(args.num_layers)
    print(args.num_layers // 2)
    print(args.num_layers // 4 * 3)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    if args.do_inference:
        assert args.load_model_path is not None
        main(args)
    else:
        results = []
        for target_layer in range(args.num_layers):
            print(f"Probing layer {target_layer}")
            if target_layer >= args.num_layers // 2:
                args.lr = 5e-4
                args.weight_decay = 1e-2
                args.num_epochs = 50
            if target_layer >= args.num_layers // 4 * 3:
                args.lr = 5e-4
                args.weight_decay = 1e-2
                args.num_epochs = 30
            args.target_layer = target_layer
            l_results = main(args)
            l_results['target_layer'] = target_layer
            results.append(l_results)

        if not args.save_checkpoints:
            output_filename = os.path.join(args.save_dir, 'probe_results.jsonl')
            with open(output_filename, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            print(f"Results saved at {output_filename}")