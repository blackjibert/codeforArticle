import sys
import time
from threading import Thread, Lock
from multiprocessing import get_context
from datetime import datetime
import logging
from torch import optim

from numpy import array

# from model.resnet import resnetmode
from model.alexnet import  MyAlexNet
from model.lenet import LeNetRGB
from model.lenet import loss_fn
# from model.resnet import ResNet18
from model.vgg16 import VGG

from model.lenet_Regular import LeNetRGB_L2

from model.lenet_dropout import LeNetRGB_Dropout

from model.base_model import NetRGB_Dropout

from model.resnet import ResNet18
from model.simpleCNN import SimpleCNN
from model.newbase_model import NewDropoutNet
from model.base_model import NetRGB_Dropout


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
from torch import nn, optim
from torchvision import datasets, transforms
import syft as sy
from syft.frameworks.torch.fl import utils
from syft.workers.websocket_client import WebsocketClientWorker
from sklearn.metrics import f1_score, log_loss, accuracy_score
import yappi
from socket import timeout
from websocket._exceptions import WebSocketTimeoutException

from model.base_model import NetRGB
# from model.base_model import loss_fn, Net
import numpy as np
import torch.nn.functional as F

# net = AlexNettest()
# net = LeNetRGB()
# nets=LeNetRGB()
# nets = MyAlexNet()
# nets = LeNetRegularized()
# net = ResNet18()
model = torch.jit.trace(NetRGB_Dropout(), torch.zeros([64, 3, 32, 32], dtype=torch.float))
# model = torch.jit.trace(SimpleCNN(), torch.zeros([64, 3, 32, 32], dtype=torch.float))

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

            # print(234, type(data))
            # print("{}:".format(parms), dict_params[parms].shape)
            # noise = gaussian_mech_RDP_vec(2, 0.1, 0.1, dict_params["fc2.bias"].shape)
            # gussNoise = gaussian_mech_RDP_vec(0.01, 0.1, 0.2, dict_params[parms].shape)
            
            gussNoise = gaussian_mech_RDP_vec(0.005, 0.0001, 1, dict_params[parms].shape)
            # print(345, type(fc2_biasNoise))
            # print("key:", fc2_biasNoise)
            nu = array(gussNoise, dtype=float)
            final_tensor = torch.from_numpy(nu)
            final_tensor = final_tensor.float()
            final_tensor = final_tensor + data
            # print("--before clip--final_tensor.grad-----", final_tensor)
            # print("type(final_tensor)", type(final_tensor) )# tensor type
            # nn.utils.clip_grad_value_(final_tensor, clip_value=0.5)
            # nn.utils.clip_grad_norm_(final_tensor, max_norm=0.01208, norm_type=2)
            # print("--after clip--final_tensor.grad-----", final_tensor)

            dict_params[parms].set_(final_tensor)
            # dict_params["fc2.bias"] = final_tensor + parm
            # print("**********************************************************************************")
        # print("----------------------------------------------success---------------------------------")
        return model


def main(n_server, staleness_threshold, eval_interval=15, eval_pool_size=1, training_duration=7200, log_path=None,
         yappi_log_path=None, f1_epoch_log_path=None):
    global epochs

    client_threads = []
    eval_results = {}
    stop_flag = False
    try:
        hook = sy.TorchHook(torch)
        # Connect to servers, keep trying indefinetly if failed
        kwargs_websocket = {"hook": hook, "host": "0.0.0.0", "verbose": True}
        # kwargs_websocket = {"hook": hook, "host": "127.0.0.1", "verbose": True}
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
                time.sleep(10)
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


def train(server, n_server, epoch, local_model_params,staleness_threshold):
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
        optimizer_args={"lr": 0.5} # 原本为0.5的学习率
    )
    train_config.send(server)
    loss = server.fit(dataset_key="cifar10", return_ids=[0])
    new_model = train_config.get_model().obj

    # 1、
    # Asynchronous federated averaging: model += 1/n * (new_model - old_model), the stateful way
    # weight = 1 / n_server
    # if epoch < 300:
    #     sigma = 0.49
    # elif 300 <= epoch < 600:
    #     sigma = 0.5
    # else:
    #     sigma = 0.51
    # # new_model = fed_dp_Allparam(new_model)
    # new_model = utils.scale_model(new_model, sigma)
    # old_model = utils.scale_model(old_model, sigma)
    # grad = utils.add_model(new_model, utils.scale_model(old_model, -1))
    # # nn.utils.clip_grad_norm_(grad, max_norm=20, norm_type=2)
    # # grad = fed_dp_Allparam(grad)
    # scaled_grad = utils.scale_model(grad, weight)
    # # 添加噪声
    # scaled_grad = fed_dp_Allparam(scaled_grad)
    # if staleness_threshold == 0:
    #     staleness_threshold = 1
    # staleness = 1 / staleness_threshold
    # if min(epochs.values()) - epoch >= staleness_threshold:
    #     scaled_grad = utils.scale_model(scaled_grad, staleness)
    # with lock:
    #     model = utils.add_model(model, scaled_grad)
    # return loss.data

    # 2、
    # yibu Asynchronous federated averaging: model += 1/n * (new_model - old_model), the stateful way
    weight = 1/n_server
    grad = utils.add_model(new_model, utils.scale_model(old_model, -1))
    scaled_grad = utils.scale_model(grad, weight)
    # # 加噪声
    # scaled_grad = fed_dp_Allparam(scaled_grad)
    
    with lock:
        model = utils.add_model(model, scaled_grad)

    return loss.data

def snapshot_model():
    global model, lock

    cloned_model = utils.scale_model(NetRGB_Dropout(), 0)  # Empty model
    # cloned_model = utils.scale_model(SimpleCNN(), 0)  # Empty model
    with lock:
        cloned_model = utils.add_model(cloned_model, model)
    # time.sleep(15)
    return cloned_model


def evaluator(evaluator_q, log_path, f1_epoch_log_path):
    # Reinitialize log
    logger_file_initializer(log_path)
    logger_f1_epoch_file_initializer(f1_epoch_log_path)

    eval_dataset = datasets.CIFAR10(
        root="./datacifar10",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    )
    eval_dataset_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=100, shuffle=False)

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
    try:
        N_SERVER = int(sys.argv[1])
        # N_SERVER = 4  #数据服务器数量
        STALENESS_THRESHOLD = int(sys.argv[2])
        # STALENESS_THRESHOLD = 100    # 异步程度
        F1_LOG_PATH = sys.argv[3]
        # F1_LOG_PATH = "out11cifar10/f1_time.csv"
        F1_EPOCH_LOG_PATH = sys.argv[4]
        # F1_EPOCH_LOG_PATH = "out11cifar10/f1_epoch.csv"
        YAPPI_LOG_PATH = sys.argv[5]
        # YAPPI_LOG_PATH = "out11cifar10/yappi.csv"

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
