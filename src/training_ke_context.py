import math
import os
import time
from collections import defaultdict

import dill
import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import multilabel_margin_loss
from torch.optim import Adam


def llprint(message):
    print(message, end="", flush=True)


def get_n_params(model):
    return sum(param.numel() for param in model.parameters())


def ddi_rate_score(record, path="../data/mimic3/output/ddi_A_final.pkl"):
    ddi_A = dill.load(open(path, "rb"))
    all_cnt, dd_cnt = 0, 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    return 0 if all_cnt == 0 else dd_cnt / all_cnt


def average_precision(y_true, y_score):
    positives = np.where(y_true == 1)[0]
    if len(positives) == 0:
        return 0
    order = np.argsort(y_score)[::-1]
    hits, precision_sum = 0, 0.0
    positive_set = set(positives.tolist())
    for rank, idx in enumerate(order, start=1):
        if idx in positive_set:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(positive_set)


def multi_label_metric(y_gt, y_pred, y_prob):
    def jaccard():
        score = []
        for b in range(y_gt.shape[0]):
            target = set(np.where(y_gt[b] == 1)[0].tolist())
            out = set(np.where(y_pred[b] == 1)[0].tolist())
            union = target | out
            score.append(0 if len(union) == 0 else len(target & out) / len(union))
        return np.mean(score)

    avg_prc, avg_recall, avg_f1, fp_rates, fn_rates, prauc = [], [], [], [], [], []
    for b in range(y_gt.shape[0]):
        target = set(np.where(y_gt[b] == 1)[0].tolist())
        out = set(np.where(y_pred[b] == 1)[0].tolist())
        actual_negatives = set(np.where(y_gt[b] == 0)[0].tolist())
        inter = target & out

        prc = 0 if len(out) == 0 else len(inter) / len(out)
        recall = 0 if len(target) == 0 else len(inter) / len(target)
        f1 = 0 if prc + recall == 0 else 2 * prc * recall / (prc + recall)

        fp = len(out - target)
        fn = len(target - out)
        tp = len(inter)
        tn = len(actual_negatives - out)

        avg_prc.append(prc)
        avg_recall.append(recall)
        avg_f1.append(f1)
        fp_rates.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        fn_rates.append(fn / (fn + tp) if (fn + tp) > 0 else 0)
        prauc.append(average_precision(y_gt[b], y_prob[b]))

    return (
        jaccard(),
        np.mean(prauc),
        np.mean(avg_prc),
        np.mean(avg_recall),
        np.mean(avg_f1),
        np.mean(fp_rates),
        np.mean(fn_rates),
    )


def tensor_to_array(tensor):
    return np.asarray(tensor.detach().cpu().tolist())


def eval_one_epoch(model, data_eval, voc_size):
    model.eval()
    smm_record, ja, prauc, avg_p, avg_r, avg_f1, avg_fp, avg_fn = [[] for _ in range(8)]
    med_cnt, visit_cnt = 0, 0
    with torch.no_grad():
        for step, input_seq in enumerate(data_eval):
            y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
            for adm_idx, adm in enumerate(input_seq):
                output, _ = model(input_seq[:adm_idx + 1])

                y_gt_tmp = np.zeros(voc_size[2])
                y_gt_tmp[adm[2]] = 1
                y_gt.append(y_gt_tmp)

                output = tensor_to_array(torch.sigmoid(output))[0]
                y_pred_prob.append(output)

                y_pred_tmp = output.copy()
                y_pred_tmp[y_pred_tmp >= 0.5] = 1
                y_pred_tmp[y_pred_tmp < 0.5] = 0
                y_pred.append(y_pred_tmp)

                y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
                y_pred_label.append(sorted(y_pred_label_tmp.tolist()))
                visit_cnt += 1
                med_cnt += len(y_pred_label_tmp)

            smm_record.append(y_pred_label)
            adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1, fp, fn = multi_label_metric(
                np.asarray(y_gt), np.asarray(y_pred), np.asarray(y_pred_prob)
            )
            ja.append(adm_ja)
            prauc.append(adm_prauc)
            avg_p.append(adm_avg_p)
            avg_r.append(adm_avg_r)
            avg_f1.append(adm_avg_f1)
            avg_fp.append(fp)
            avg_fn.append(fn)
            llprint("\reval step: {} / {}".format(step + 1, len(data_eval)))

    ddi_rate = ddi_rate_score(smm_record)
    avg_med = med_cnt / max(visit_cnt, 1)
    print(
        "\nDDI Rate: {:.4f}, Jaccard: {:.4f}, PRAUC: {:.4f}, "
        "AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}, "
        "AVG_FP: {:.4f}, AVG_FN: {:.4f}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            avg_med,
            np.mean(avg_fp),
            np.mean(avg_fn),
        )
    )
    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        avg_med,
        np.mean(avg_fp),
        np.mean(avg_fn),
    )


def TrainKEContext(model, device, data_train, data_eval, voc_size, args):
    optimizer = Adam(model.parameters(), lr=args.lr)
    history = defaultdict(list)
    best = {
        "epoch": 0,
        "ja": 0,
        "ddi": 0,
        "prauc": 0,
        "f1": 0,
        "med": 0,
        "model_state": None,
    }

    log_dir = getattr(
        args,
        "output_dir",
        os.path.abspath(os.path.join("..", "saved", args.dataset, args.run_name)),
    )
    os.makedirs(log_dir, exist_ok=True)

    total_train_time, ddi_losses, ddi_values = 0, [], []
    epochs = min(args.epochs, 3) if args.debug else args.epochs

    for epoch in range(epochs):
        print(f"----------------Epoch {epoch + 1}------------------")
        model.train()
        tic, ddi_losses_epoch = time.time(), []

        for step, input_seq in enumerate(data_train):
            for adm_idx, adm in enumerate(input_seq):
                bce_target = torch.zeros((1, voc_size[2]), device=device)
                bce_target[:, adm[2]] = 1

                multi_target = -torch.ones((1, voc_size[2]), dtype=torch.long, device=device)
                for idx, item in enumerate(adm[2]):
                    multi_target[0][idx] = item

                result, loss_ddi = model(input_seq[:adm_idx + 1])
                sigmoid_res = torch.sigmoid(result)

                loss_bce = binary_cross_entropy_with_logits(result, bce_target)
                loss_multi = multilabel_margin_loss(sigmoid_res, multi_target)

                result_np = tensor_to_array(sigmoid_res)[0]
                result_np[result_np >= 0.5] = 1
                result_np[result_np < 0.5] = 0
                y_label = np.where(result_np == 1)[0]
                current_ddi_rate = ddi_rate_score([[y_label]])

                if current_ddi_rate <= args.target_ddi:
                    task_loss = 0.95 * loss_bce + 0.05 * loss_multi
                    loss = task_loss
                else:
                    beta = args.coef * (1 - (current_ddi_rate / args.target_ddi))
                    beta = min(math.exp(beta), 1)
                    task_loss = 0.95 * loss_bce + 0.05 * loss_multi
                    loss = beta * task_loss + (1 - beta) * loss_ddi

                ddi_losses_epoch.append(loss_ddi.detach().cpu().item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            llprint("\rtraining step: {} / {}".format(step + 1, len(data_train)))

        ddi_loss_epoch = sum(ddi_losses_epoch) / max(len(ddi_losses_epoch), 1)
        ddi_losses.append(ddi_loss_epoch)

        train_time = time.time() - tic
        total_train_time += train_time
        print(f"\nddi_loss : {ddi_loss_epoch}\n")

        tic = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med, avg_fp, avg_fn = eval_one_epoch(
            model, data_eval, voc_size
        )
        print(f"training time: {train_time}, eval time: {time.time() - tic}")

        ddi_values.append(ddi_rate)
        history["ja"].append(ja)
        history["ddi_rate"].append(ddi_rate)
        history["avg_p"].append(avg_p)
        history["avg_r"].append(avg_r)
        history["avg_f1"].append(avg_f1)
        history["prauc"].append(prauc)
        history["med"].append(avg_med)
        history["avg_fp"].append(avg_fp)
        history["avg_fn"].append(avg_fn)

        model_name = f"Epoch_{epoch}_JA_{ja:.4f}_DDI_{ddi_rate:.4f}_ke_context.pt"
        torch.save(model.state_dict(), os.path.join(log_dir, model_name))

        if epoch == 0 or best["ja"] < ja:
            best.update(
                {
                    "epoch": epoch,
                    "ja": ja,
                    "ddi": ddi_rate,
                    "prauc": prauc,
                    "f1": avg_f1,
                    "med": avg_med,
                    "model_state": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                }
            )

        with open(os.path.join(log_dir, "best.txt"), "a") as f:
            f.write(f"{best['epoch']}\n")

        with open(os.path.join(log_dir, "ddi_losses.txt"), "w") as f:
            for dloss, dvalue in zip(ddi_losses, ddi_values):
                f.write(f"{dloss}\t{dvalue}\n")

        with open(os.path.join(log_dir, "history.pkl"), "wb") as f:
            dill.dump(history, f)

        print("best_epoch: {}, best_ja: {:.4f}".format(best["epoch"], best["ja"]))

    print("avg training time/epoch: {:.4f}".format(total_train_time / epochs))
    print("parameters", get_n_params(model))

    if best["model_state"] is not None:
        model.load_state_dict(best["model_state"])
    torch.save(model.state_dict(), os.path.join(log_dir, f"best_model_{best['ja']:.4f}.pt"))
    return model


def TestKEContext(model, device, data_test, voc_size):
    model = model.to(device).eval()
    print("--------------------Begin Testing--------------------")
    tic, result = time.time(), []
    np.random.seed(0)
    for _ in range(10):
        selected_indices = np.random.choice(len(data_test), size=round(len(data_test) * 0.8), replace=True)
        test_sample = [data_test[i] for i in selected_indices.tolist()]
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med, avg_fp, avg_fn = eval_one_epoch(
            model, test_sample, voc_size
        )
        result.append([ja, ddi_rate, avg_f1, prauc, avg_med, avg_fp, avg_fn])

    result = np.asarray(result)
    mean, std = result.mean(axis=0), result.std(axis=0)
    metric_list = ["ja", "ddi_rate", "avg_f1", "prauc", "med", "avg_fp", "avg_fn"]
    outstring = "".join(
        [
            "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
            for idx, (m, s) in enumerate(zip(mean, std))
        ]
    )

    print(outstring)
    print("average test time: {}".format((time.time() - tic) / 10))
    print("parameters", get_n_params(model))
