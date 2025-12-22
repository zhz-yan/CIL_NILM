import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np
from utils.json_logger import ResultsLogger


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    repeats = int(args.get("repeats", 1))

    for exp_idx in range(repeats):
        args["exp_idx"] = exp_idx
        for seed in seed_list:
            args["seed"] = seed
            args["device"] = device
            _train(args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]

    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])
    os.makedirs(logs_name, exist_ok=True)
    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"], args["dataset"], init_cls, args["increment"],
        args["prefix"], args["seed"], args["convnet_type"]
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[logging.FileHandler(filename=logfilename + ".log"),
                  logging.StreamHandler(sys.stdout)],
    )

    _set_random()
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"], args["shuffle"], args["seed"],
        args["init_cls"], args["increment"], args.get("aug", 1),
        data_dir=f'D:/Workspace/PyCharm/CILNILM/CIL_NILM_benchmark/PyCIL/data/{args["dataset"]}/',
        train_ratio=0.8, no_transform=True
    )
    model = factory.get_model(args["model_name"], args)

    results_dir = os.path.join("results", args["model_name"], "model_size")
    os.makedirs(results_dir, exist_ok=True)
    json_name = f"{args['model_name']}_case2_ba_{init_cls}_in_{args['increment']}_exp_{args.get('exp_idx', 0)}.json"
    json_path = os.path.join(results_dir, json_name)

    clean_args = copy.deepcopy(args)
    if isinstance(clean_args.get("device"), list):
        clean_args["device"] = [str(d) for d in clean_args["device"]]

    logger = ResultsLogger(config=clean_args, save_path=json_path)
    logger.start_run(extra={"exp_idx": args.get("exp_idx", 0)})

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info("Trainable params: {}".format(count_parameters(model._network, True)))

        known = model._known_classes
        total = known + data_manager.get_task_size(task)
        logger.start_task(task_id=task, known_classes=known, total_classes=total)

        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.report_model_and_memory()

        y_pred_topk, y_true = model._eval_cnn(model.test_loader)   # shape: (N, topk), (N,)
        y_pred = y_pred_topk[:, 0] if y_pred_topk.ndim == 2 else y_pred_topk

        logger.end_task(y_true=y_true, y_pred=y_pred, model=model._network, include_confusion=False)

        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_keys_sorted = sorted(nme_keys)
            nme_values = [nme_accy["grouped"][key] for key in nme_keys_sorted]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"]); cnn_curve["top5"].append(cnn_accy["top5"])
            nme_curve["top1"].append(nme_accy["top1"]); nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))
            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"]); cnn_curve["top5"].append(cnn_accy["top5"])
            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))
            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    if len(cnn_matrix) > 0:
        n_rows = len(cnn_matrix)
        n_cols = max(len(line) for line in cnn_matrix)  # 关键：取最长行
        np_acctable = np.zeros((n_rows, n_cols), dtype=float)

        for idxx, line in enumerate(cnn_matrix):
            L = len(line)
            np_acctable[idxx, :L] = np.array(line, dtype=float)  # 保证形状匹配

        end_col = min(task + 1, np_acctable.shape[1])
        curr = np_acctable[:, :end_col]

        if end_col > 1:
            last = curr[:, end_col - 1]  # 当前轮
            prev_max = np.max(curr[:, :end_col - 1], axis=1)  # 之前各轮的最好
            forgetting = np.mean((prev_max - last)[:end_col - 1])
        else:
            forgetting = 0.0
        print('Accuracy Matrix (CNN):'); print(np_acctable)
        print('Forgetting (CNN):', forgetting)
        logging.info('Forgetting (CNN): {}'.format(forgetting))
        logger.payload["summary"]["forgetting_cnn"] = float(forgetting)

    if len(nme_matrix) > 0:
        n_rows = len(nme_matrix)
        n_cols = max(len(line) for line in nme_matrix)

        np_acctable = np.zeros((n_rows, n_cols), dtype=float)
        for idxx, line in enumerate(nme_matrix):
            L = len(line)
            np_acctable[idxx, :L] = np.asarray(line, dtype=float)

        print('Accuracy Matrix (NME):');
        print(np_acctable.T)

        cur_col = n_cols - 1
        prev_best = np.max(np_acctable[:, :cur_col], axis=1) if cur_col > 0 else np.zeros(n_rows)
        cur_vals = np_acctable[:, cur_col]
        forgetting = float(np.mean(prev_best - cur_vals))

        print('Forgetting (NME):', forgetting)
        logging.info('Forgetting (NME): {}'.format(forgetting))
        logger.payload["summary"]["forgetting_nme"] = forgetting

    if len(cnn_curve["top1"]) > 0:
        logger.payload["summary"]["avg_cnn_top1"] = float(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
        logger.payload["summary"]["cnn_top1_curve"] = cnn_curve["top1"]
        logger.payload["summary"]["cnn_top5_curve"] = cnn_curve["top5"]
    if len(nme_curve["top1"]) > 0:
        logger.payload["summary"]["avg_nme_top1"] = float(sum(nme_curve["top1"]) / len(nme_curve["top1"]))
        logger.payload["summary"]["nme_top1_curve"] = nme_curve["top1"]
        logger.payload["summary"]["nme_top5_curve"] = nme_curve["top5"]

    logger.payload["summary"]["num_tasks"] = int(data_manager.nb_tasks)
    logger.finish_run()
    logger.save()

def _set_device(args):
    device_type = args["device"]
    gpus = []
    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)
    args["device"] = gpus

def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
