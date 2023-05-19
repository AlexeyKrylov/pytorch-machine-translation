import sys
sys.path.append("C:/Users/krilo/PycharmProjects/pytorch-machine-translation/src")
import torch
import yaml
from models import seq2seq_t5, trainer
from data.datamodule_t5 import DataManager
from txt_logger import TXTLogger



if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'
    print(DEVICE)
    data_config = yaml.load(open("C:/Users/krilo/PycharmProjects/pytorch-machine-translation/configs/data_config_t5.yaml", 'r'),   Loader=yaml.Loader)
    dm = DataManager(data_config, DEVICE)
    train_dataloader, dev_dataloader = dm.prepare_data()
    model_config = yaml.load(open("C:/Users/krilo/PycharmProjects/pytorch-machine-translation/configs/model_config_t5.yaml", 'r'),   Loader=yaml.Loader)

    model = seq2seq_t5.Seq2SeqT5(model_config=model_config, tokenizer=dm.tokenizer, device=DEVICE)

    logger = TXTLogger('C:/Users/krilo/PycharmProjects/pytorch-machine-translation/training_logs')
    trainer_cls = trainer.Trainer(model=model, model_config=model_config, logger=logger)


    if model_config['try_one_batch']:
        train_dataloader = [list(train_dataloader)[0]]
        dev_dataloader = [list(train_dataloader)[0]]

    trainer_cls.train(train_dataloader, dev_dataloader)




