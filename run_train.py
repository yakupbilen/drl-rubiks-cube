"""
For running: python run_train.py -p "ini_files/train.ini"
"""
if __name__ == "__main__":
    import json
    import logging
    import argparse
    from train.train_utils import generate_states_by_ADI, update_generator
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR
    from torch.nn import MSELoss
    import torch
    from networks.getNetwork import getNetwork
    from configs.train_conf import TrainConfig
    from torch.utils.tensorboard import SummaryWriter
    import os
    parser = argparse.ArgumentParser(description="Evaluate checkpoints")
    parser.add_argument("-p", "--ini_path", type=str, metavar="", required=True, help="Path of config file. Extension of file must be .ini")
    args = parser.parse_args()

    config = TrainConfig(args.ini_path)
    device = torch.device(config.device)
    name = f"arch{config.network_type}_lr{config.learning_rate}_iter{config.max_iter}" \
           f"_batch_size{config.batch_size}_decay{config.lr_decay_gamma}_decay_interval{config.lr_decay_interval}" \
           f"_weight{config.weight_type}_include_first{config.include_first}_scramble_depth{config.scramble_depth}_tau{config.tau}" \
           f"_tau_interval{config.generator_interval}_checkpoints{config.checkpoint}"

    if not os.path.exists(f"models/{name}"):
        os.makedirs(f"models/{name}/tmp")
        os.makedirs(f"models/{name}/checkpoints")
        info = {"iter":0,"lr":config.learning_rate}
        with open(f"models/{name}/tmp/info.json","w") as json_file:
            json.dump(info,json_file)

    with open(f"models/{name}/tmp/info.json", "r") as json_file:
        info = json.load(json_file)



    step_idx = 0
    writer = SummaryWriter()

    nnet = getNetwork(config.network_type)
    nnet = nnet(54*6).to(device)

    log_mode = "w"
    if info["iter"] != 0:
        log_mode = "a"
        nnet.load_state_dict(torch.load(f"models/{name}/tmp/model_last.dat"))
        generator = getNetwork(config.network_type)
        generator = generator(54*6).to(device)
        generator.load_state_dict(torch.load(f"models/{name}/tmp/generator_last.dat"))
        step_idx = info["iter"]
    else:
        generator = nnet.clone().to(device)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
    file_handler = logging.FileHandler(f'models/{name}/tmp/logs.log', mode=log_mode)
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    loss_fn = MSELoss(reduction='none')
    opt = optim.Adam(nnet.parameters(), lr=info["lr"])
    scheduler = StepLR(opt, 1, config.lr_decay_gamma)
    while True:

        """torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()"""

        with torch.no_grad():
            x, y, weights = generate_states_by_ADI(config,generator,device)
        opt.zero_grad()
        yhat = nnet(x).squeeze()
        loss = loss_fn(yhat, y)*weights
        loss = torch.mean(loss)
        loss.backward()
        opt.step()

        step_idx += 1
        if step_idx % config.checkpoint == 0:
            info["iter"] = step_idx
            model_path = os.path.join(f"models/{name}/checkpoints", f"model{info['iter']}.dat")
            torch.save(nnet.state_dict(), model_path)
            logger.info(f"Epoch : {step_idx}_Checkpoint Saved..")

        if step_idx % config.report_batches == 0:
            writer.add_scalar("Loss", loss, step_idx)
            logger.info(f"Epoch : {step_idx}__loss : {loss}")

        if step_idx % config.generator_interval == 0:
            with torch.no_grad():
                generator = update_generator(generator, nnet, config.tau, device)

        if step_idx % config.lr_decay_interval == 0:
            scheduler.step()
            logger.info(f"Optimizer new lr : {scheduler.get_last_lr()}")

        if step_idx % config.max_iter == 0:
            logger.info("Finished")
            info["iter"] = step_idx
            info["lr"] = scheduler.get_last_lr()[0]
            with open(f"models/{name}/tmp/info.json", "w") as json_file:
                json.dump(info, json_file)
            torch.save(nnet.state_dict(), f"models/{name}/tmp/model_last.dat")
            torch.save(generator.state_dict(), f"models/{name}/tmp/generator_last.dat")
            break

    writer.flush()
    writer.close()
