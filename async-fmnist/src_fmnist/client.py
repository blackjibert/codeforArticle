import csv
import sys
import time
from threading import Thread, Lock
from multiprocessing import get_context
from datetime import datetime
import logging

from numpy import array

console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
console_handler.setLevel(logging.DEBUG)

logger = logging.getLogger("client")
logger.setLevel(level=logging.DEBUG)
logger.addHandler(console_handler)

logger_f1_epoch = logging.getLogger("client_f1_epoch")
logger_f1_epoch.setLevel(level=logging.DEBUG)
logger_f1_epoch.addHandler(console_handler)

import torch
from torchvision import datasets, transforms
import syft as sy
from syft.frameworks.torch.fl import utils
from syft.workers.websocket_client import WebsocketClientWorker
from sklearn.metrics import f1_score
import yappi
from socket import timeout
from websocket._exceptions import WebSocketTimeoutException

from model.base_model import Net, loss_fn
import numpy as np

model = torch.jit.trace(Net(), torch.zeros([1, 1, 28, 28], dtype=torch.float))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device) #移动模型到cuda
lock = Lock()

epochs = {}
last_smallest_epoch = 0
largest_stale = 0


def gaussian_mech_RDP_vec(sensitivity, alpha, epsilon_bar, weights_shape):
    sigma = np.sqrt((sensitivity ** 2 * alpha) / (2 * epsilon_bar))
    return np.random.normal(0, sigma, weights_shape)


def fed_dp_Allparam(model):
    with torch.no_grad():
        params = model.named_parameters()
        dict_params = dict(params)
        dict_params_list = list(dict_params)
        # print(dict_params_list)
        for parms in dict_params_list:
            data = dict_params[parms].data
            # noise = gaussian_mech_RDP_vec(2, 0.1, 0.1, dict_params["fc2.bias"].shape)
            gussNoise = gaussian_mech_RDP_vec(0.001, 0.1, 0.01, dict_params[parms].shape)
            # print(345, type(fc2_biasNoise))
            # print("key:", fc2_biasNoise)
            nu = array(gussNoise, dtype=float)
            final_tensor = torch.from_numpy(nu)
            final_tensor = final_tensor.float()
            final_tensor = final_tensor + data
            dict_params[parms].set_(final_tensor)
            # dict_params["fc2.bias"] = final_tensor + parm
            print("**********************************************************************************")
        print("----------------------------------------------success---------------------------------")
    return model


def main(n_server, staleness_threshold, eval_interval=15, eval_pool_size=1, training_duration=3600, log_path=None,
         yappi_log_path=None, f1_epoch_log_path=None):
    global epochs
    print("***************************fmnist******************")
    client_threads = []
    eval_results = {}
    stop_flag = False
    try:
        hook = sy.TorchHook(torch)

        # Connect to servers, keep trying indefinetly if failed
        kwargs_websocket = {"hook": hook, "host": "127.0.0.1", "verbose": True}
        servers = []
        for i in range(n_server):
            while True:
                try:
                    server = WebsocketClientWorker(id=f"dataserver-{i}", port=8777 + i, **kwargs_websocket)
                    servers.append(server)
                    epochs[server.id] = 0
                    break
                except (ConnectionRefusedError, timeout):
                    continue

        logger.debug("Training starts!")
        ctx = get_context("spawn")
        evaluator_q = ctx.Queue()
        client_threads = [Thread(name=server.id, target=train_loop,
                                 args=(server, staleness_threshold, n_server, evaluator_q, lambda: stop_flag)) for
                          server in servers]
        yappi.set_clock_type("wall")
        yappi.start()
        for thread in client_threads:
            thread.start()

        # The original thread becomes evaluator. Uses multiprocessing to eval at a constant rate
        start_time = datetime.now()

        p = ctx.Process(target=evaluator, args=(evaluator_q, log_path, f1_epoch_log_path))
        p.start()

        while True:
            dur = datetime.now() - start_time
            if dur.seconds > training_duration:
                logger.debug(f"No further evaluation needed")
                break
            evaluator_q.put((snapshot_model(), dur))
            logger.debug(f"Snapshoted model at {dur}, will evaluate soon...")
            time.sleep(eval_interval)

    except (KeyboardInterrupt, SystemExit):
        logger.debug("Gracefully shutting client down...")
    finally:
        stop_flag = True
        for thread in client_threads:
            thread.join()

        yappi.stop()
        logger.handlers = logger.handlers[:1]
        logger_file_initializer(yappi_log_path)
        logger.info("thread_name,train_loop_ttot,train_ttot,stale_ttot,train_loop_tavg,train_tavg,stale_tavg")
        for thread in yappi.get_thread_stats():
            func_stats = yappi.get_func_stats(ctx_id=thread.id, filter_callback=lambda x: yappi.func_matches(x, [
                train_loop, stale, train]))
            if not func_stats or thread.name == "_MainThread":
                continue
            logger.info(format_func_stats(thread, func_stats))
        logger.info(f"largest_stale: {largest_stale}")

        try:
            while not evaluator_q.empty():
                time.sleep(1)
            p.terminate()
            p.join()
        except NameError:
            pass


def train_loop(server, staleness_threshold, n_server, evaluator_q, should_stop):
    global epochs, last_smallest_epoch

    epoch = 1
    now = datetime.now()
    while True:
        if should_stop():
            break

        # Check staleness here
        stale(server, epoch, staleness_threshold, should_stop)

        # Train
        try:
            local_model_params = []
            loss = train(server, n_server, epoch, local_model_params, staleness_threshold)
        except (WebSocketTimeoutException, TimeoutError):
            logger.debug(f"{server.id} timeout, reconnecting...")
            while True:
                if should_stop():
                    break
                try:
                    server.close()
                    server.connect()
                    break
                except timeout:
                    continue
            continue
        logger.debug(f"{server.id} {epoch} {loss}")

        writer.writerow([server.id, epoch, loss])

        # Update global state
        epochs[server.id] = epoch
        smallest_epoch = min(epochs.values())
        if last_smallest_epoch != smallest_epoch:
            logger.debug(f"Smallest epoch changed, evaluating epoch {epoch}...")
            last_smallest_epoch = smallest_epoch
            evaluator_q.put((snapshot_model(), smallest_epoch))

        epoch += 1
    server.close()


# We express staleness in its own function to ease profiling
def stale(server, epoch, staleness_threshold, should_stop):
    global largest_stale
    staleness = epoch - min(epochs.values())
    if staleness > largest_stale:
        largest_stale = staleness
    while epoch - min(epochs.values()) > staleness_threshold + 1:
        if should_stop():
            break
        if staleness_threshold != 0:
            logging.debug(f"{server.id} is at {epoch}, while min epoch is at {min(epochs.values())}")
        time.sleep(1)  # Not busy wait


def train(server, n_server, epoch, local_model_params, staleness_threshold):
    global model, lock

    # Clone model
    old_model = snapshot_model()

    train_config = sy.TrainConfig(
        model=model,
        loss_fn=loss_fn,
        epochs=1,
        batch_size=64,
        max_nr_batches=1,  # report back after a batch is complete
        shuffle=True,
        optimizer="SGD",  # Adam doesn't work properly because momentum doesn't work?
        optimizer_args={"lr": 0.5}
    )
    train_config.send(server)
    loss = server.fit(dataset_key="fmnist", return_ids=[0])
    # model.eval()
    new_model = train_config.get_model().obj

    # 我的异步
    # Asynchronous federated averaging: model += 1/n * (new_model - old_model), the stateful way
    weight = 1 / n_server
    if epoch < 80:
        sigma = 0.49
    elif 80 <= epoch < 150:
        sigma = 0.5
    else:
        sigma = 0.51
    new_model = utils.scale_model(new_model, sigma)
    old_model = utils.scale_model(old_model, sigma)
    grad = utils.add_model(new_model, utils.scale_model(old_model, -1))
    scaled_grad = utils.scale_model(grad, weight)
    scaled_grad = fed_dp_Allparam(scaled_grad)
    staleness = 1 / staleness_threshold
    if min(epochs.values()) - epoch >= staleness_threshold:
        scaled_grad = utils.scale_model(scaled_grad, staleness)
    with lock:
        model = utils.add_model(model, scaled_grad)
    return loss.data

    # yibu Asynchronous federated averaging: model += 1/n * (new_model - old_model), the stateful way
    # weight = 1/n_server
    # grad = utils.add_model(new_model, utils.scale_model(old_model, -1))
    # scaled_grad = utils.scale_model(grad, weight)
    # # 加噪声
    # scaled_grad = fed_dp_Allparam(scaled_grad)
    
    # with lock:
    #     model = utils.add_model(model, scaled_grad)

    # return loss.data

def snapshot_model():
    global model, lock
    cloned_model = utils.scale_model(Net(), 0)
    with lock:
        cloned_model = utils.add_model(cloned_model, model)
    return cloned_model


def evaluator(evaluator_q, log_path, f1_epoch_log_path):
    # Reinitialize log
    logger_file_initializer(log_path)
    logger_f1_epoch_file_initializer(f1_epoch_log_path)

    eval_dataset = datasets.FashionMNIST(
        root="./dataf",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
        # transform=trans,
    )
    eval_dataset_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=64, shuffle=False)

    while True:
        snapshoted_model, marker = evaluator_q.get()
        logger.debug(f"Evaluating the model snapshoted at {marker}...")

        y_pred = []
        for data, _ in eval_dataset_loader:
            y_pred.extend(snapshoted_model(data).detach().numpy().argmax(axis=1))
        f1 = f1_score(eval_dataset.targets, y_pred, average='micro')

        if type(marker) is int:  # if it's epoch-wise evaluation
            logger_f1_epoch.info(f"{marker},{f1}")
        else:  # if it's duration-wise evaluation
            logger.info(f"{marker},{f1}")


# Need a function to init file handler in main and evaluator processes, because filename is not a constant
def logger_file_initializer(path):
    if not path:
        return
    file_handler = logging.FileHandler(path)  # Log f1 evaluation to file for analysis later
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)


def logger_f1_epoch_file_initializer(path):
    if not path:
        return
    file_handler = logging.FileHandler(path)  # Log f1 evaluation to file for analysis later
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    file_handler.setLevel(logging.INFO)
    logger_f1_epoch.addHandler(file_handler)


def format_func_stats(thread_stat, func_stats):
    # thread_name, train_loop_ttot, train_ttot, stale_ttot, train_loop_tavg, train_tavg, stale_tavg
    train_loop_stat, train_stat, stale_stat = func_stats
    return f"dataserver-{int(train_loop_stat.ctx_id) - 1}, {train_loop_stat[6]}, {train_stat[6]}, {stale_stat[6]}, {train_loop_stat[14]}, {train_stat[14]}, {stale_stat[14]}"


if __name__ == "__main__":

    csv_file = open('loss.csv', 'a+', newline='', encoding='gbk')
    writer = csv.writer(csv_file)

    try:
        N_SERVER = int(sys.argv[1])
        # N_SERVER = 4
        STALENESS_THRESHOLD = int(sys.argv[2])
        # STALENESS_THRESHOLD = 6  # 异步
        F1_LOG_PATH = sys.argv[3]
        # F1_LOG_PATH = "out10fmnist/f1_time.csv"
        F1_EPOCH_LOG_PATH = sys.argv[4]
        # F1_EPOCH_LOG_PATH = "out10fmnist/f1_epoch.csv"
        YAPPI_LOG_PATH = sys.argv[5]
        # YAPPI_LOG_PATH = "out10fmnist/yappi.csv"

        logger.debug(f"Will start client (model owner) that will connect to {N_SERVER} server(s)")
    except Exception as e:
        logger.error(e)
        sys.exit()

    logger_file_initializer(F1_LOG_PATH)
    logger.info("time,f1")

    logger_f1_epoch_file_initializer(F1_EPOCH_LOG_PATH)
    logger_f1_epoch.info("epoch,f1")

    main(N_SERVER, STALENESS_THRESHOLD, log_path=F1_LOG_PATH, yappi_log_path=YAPPI_LOG_PATH,
         f1_epoch_log_path=F1_EPOCH_LOG_PATH)
