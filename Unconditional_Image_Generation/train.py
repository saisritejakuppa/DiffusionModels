from models.unet_diffusion import train


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = '/home/saiteja/detectwork/helmetdetection/completedataset/helmet_classification'
    args.device = "cuda"
    args.lr = 3e-4
    train(args)
    

if __name__ == '__main__':
    launch()
    