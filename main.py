from src.train import Trainer
from data_loader.data_loader import get_loader
from config.config import get_config


def main(config):
    # make directory not existed
    if config.checkpoint_dir is None:
        config.checkpoint_dir = 'checkpoints'
    print(f"Image Compression start")

    data_loader, val_data_loader = get_loader(config.train_data_dir, config.test_data_dir, config.image_size,
                                              config.batch_size)
    trainer = Trainer(config, data_loader)
    trainer.train()


if __name__ == "__main__":
    config = get_config()
    main(config)
