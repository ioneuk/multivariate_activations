import torch
import torchvision
from autoattack import AutoAttack
import torch.utils.data as data
from robustbench.data import get_preprocessing, load_clean_dataset
from robustbench.loaders import CustomImageFolder
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import clean_accuracy
from timm.models import load_checkpoint
from torchvision.transforms import transforms

from model.vit import VisionTransformer


def load_default_robust_vit():
    model = VisionTransformer(use_mega=False)
    load_checkpoint(model, '/home/user/robust_training_experiments/default/checkpoints/100090_iter_9_epoch.pt', use_ema=True)
    model = model.eval()
    return model

def load_mega_mli2d_vit():
    model = VisionTransformer(use_mega=True, act_layer='mli2d-gated')
    load_checkpoint(model, '/home/user/robust_training_experiments/default/checkpoints/100090_iter_9_epoch.pt', use_ema=True)
    model = model.eval()
    return model


def load_robust_mega_raf2d_vit():
    model = VisionTransformer(use_mega=True, act_layer='raf2d-1degree')
    load_checkpoint(model, '/home/user/robust_training_experiments/mega_raf2d/checkpoints/100090_iter_8_epoch.pt', use_ema=True)
    model = model.eval()
    return model


def preprocessing_pipeline():
    def imagenet_normalization():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            imagenet_normalization(),
        ]
    )


def eval_adv_robustness(model, x_test, y_test, batch_size, device, threat_model):
    model=model.to(device)
    adversary = AutoAttack(model, norm=threat_model, eps=8 / 255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.n_restarts = 1
    accuracy = clean_accuracy(model,
                              clean_x_test,
                              clean_y_test,
                              batch_size=batch_size,
                              device=device)
    print(f'Clean accuracy: {accuracy:.2%}')
    _ = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)


def load_imagenet(
    data_dir,
    transforms_test
):
    # imagenet = CustomImageFolder(data_dir + '/val', transforms_test)
    dataset = torchvision.datasets.ImageNet(data_dir, split='val', transform=transforms_test)
    test_loader = data.DataLoader(dataset,
                                  batch_size=50,
                                  shuffle=True,
                                  num_workers=4)


    return test_loader


def calculate_clean_accuracy(model, imagenet_val_loader):
    model = model.to(device)
    total_acc = 0
    total_size = 0
    for _x, _y in imagenet_val_loader:
        x_size = _x.shape[0]
        accuracy = clean_accuracy(model,
                                  _x,
                                  _y,
                                  batch_size=x_size,
                                  device=device)
        total_acc += accuracy * x_size
        total_size += x_size
    return total_acc / total_size


if __name__ == '__main__':
    dataset_ = BenchmarkDataset('imagenet')

    device = torch.device('cuda')
    model_name = 'vit'
    n_examples = 5000
    data_dir = '/home/user/datasets/imagenet'
    batch_size = 32

    print(f'Calculating clean accuracy:')
    prepr = preprocessing_pipeline()
    imagenet_val_loader = load_imagenet('/home/user/datasets/imagenet', prepr)
    default_acc = calculate_clean_accuracy(load_default_robust_vit(), imagenet_val_loader)
    print(f'Default ViT accuracy: {default_acc}')
    mega_raf_acc = calculate_clean_accuracy(load_robust_mega_raf2d_vit(), imagenet_val_loader)
    print(f'ViT + MEGA + RAF2d accuracy: {mega_raf_acc}')

    print(f'=================================================================================================')
    print(f'Evaluation for Linf threat model')
    prepr = get_preprocessing(dataset_, ThreatModel('Linf'), model_name, preprocessing_pipeline())
    clean_x_test, clean_y_test = load_clean_dataset(dataset_, n_examples, data_dir, prepr)

    eval_adv_robustness(model=load_default_robust_vit(), x_test=clean_x_test, y_test=clean_y_test, batch_size=batch_size, device=device, threat_model='Linf')
    eval_adv_robustness(model=load_robust_mega_raf2d_vit(), x_test=clean_x_test, y_test=clean_y_test, batch_size=batch_size, device=device, threat_model='Linf')

    print(f'=================================================================================================')
    print(f'Evaluation for L2 threat model')
    prepr = get_preprocessing(dataset_, ThreatModel('L2'), model_name, preprocessing_pipeline())
    clean_x_test, clean_y_test = load_clean_dataset(dataset_, n_examples, data_dir, prepr)

    eval_adv_robustness(model=load_default_robust_vit(), x_test=clean_x_test, y_test=clean_y_test, batch_size=batch_size, device=device, threat_model='L2')
    eval_adv_robustness(model=load_robust_mega_raf2d_vit(), x_test=clean_x_test, y_test=clean_y_test, batch_size=batch_size, device=device, threat_model='L2')
