import torch

from core.nets.additive_net import AdditiveNet
from core.nets.additive_net_attention import AdditiveNetAttention
from core.nets.additive_net_exchangeable import AdditiveNetExchangeable
from core.trainer.trainer import Trainer, DistillationTrainer, TrainerCrossVal
from core.utils import get_objects


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(setting, seed=0):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"DEVICE: {device}")

    cfg, clip_op, generator, setting_name = get_objects(setting)
    cfg.setting = cfg.get('setting', setting_name)
    if cfg.save_data is not None:
        cfg.save_data = cfg.save_data.split("/setting")[0]
        cfg.save_data = cfg.save_data + f"/setting_{cfg.setting}/seed_{seed}"

    if cfg.regret_type in ["standard", "distillation"]:
        if cfg.architecture == "RegretNet":
            net = AdditiveNet(cfg, device).to(device)
        elif cfg.architecture == "RegretFormer":
            net = AdditiveNetAttention(cfg, device).to(device)
        elif cfg.architecture == "EquivariantNet":
            net = AdditiveNetExchangeable(cfg, device).to(device)
        print("number of parameters, net =", count_parameters(net))
        generators = [generator(cfg, "train"), generator(cfg, "val")]

        if cfg.regret_type == "standard":
            trainer = Trainer(cfg, net, clip_op, device)

        elif cfg.regret_type == "distillation":
            if cfg.distill.architecture == "RegretNet":
                target_net = AdditiveNet(cfg, device).to(device)
            elif cfg.distill.architecture == "RegretFormer":
                target_net = AdditiveNetAttention(cfg, device).to(device)
            elif cfg.distill.architecture == "EquivariantNet":
                target_net = AdditiveNetExchangeable(cfg, device).to(device)
            state_dict = torch.load(f'target_nets/{cfg.distill.architecture}/setting_{cfg.setting}/seed_{seed}/'
                                    f'model_200000', map_location=device)
            target_net.load_state_dict(state_dict)
            print("number of parameters, target_net =", count_parameters(target_net))
            trainer = DistillationTrainer(cfg, net, target_net, clip_op, device)

    elif cfg.regret_type == 'cross_val':
        nets = {'RegretNet': AdditiveNet(cfg, device).to(device),
                'EquivariantNet': AdditiveNetExchangeable(cfg, device).to(device),
                'RegretFormer': AdditiveNetAttention(cfg, device).to(device)}
        state_dict = torch.load(f'target_nets/RegretNet/setting_{cfg.setting}/seed_{seed}/'
                                f'model_200000', map_location=device)
        nets['RegretNet'].load_state_dict(state_dict)

        state_dict = torch.load(f'target_nets/EquivariantNet/setting_{cfg.setting}/seed_{seed}/'
                                f'model_200000', map_location=device)
        nets['EquivariantNet'].load_state_dict(state_dict)

        state_dict = torch.load(f'target_nets/RegretFormer/setting_{cfg.setting}/seed_{seed}/'
                                f'model_200000', map_location=device)
        nets['RegretFormer'].load_state_dict(state_dict)

        generators = [generator(cfg, "train"), generator(cfg, "val")]
        trainer = TrainerCrossVal(cfg, nets, clip_op, device)
    else:
        raise NotImplementedError("This type of regret is not implemented")

    trainer.train(generators)
