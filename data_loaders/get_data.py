from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate
from pdb import set_trace as st

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "wlasl30":
        from .a2m.WLASL30 import WLASL30
        return WLASL30
    elif name == "wlasl_smplx100":
        from .a2m.WLASL_SMPLX100 import WLASL_SMPLX100
        return WLASL_SMPLX100
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train'):
    DATA = get_dataset_class(name)
    
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
        print(f'get {split} dataset.')
    return dataset


def get_dataset_loader_evl(name, batch_size, num_frames, split='train', hml_mode='train'):

    train_dataset = get_dataset(name, num_frames, 'train', hml_mode)  # wlasl30  split train or test
    test_dataset = get_dataset(name, num_frames, 'test', hml_mode)
    datasets = {"train": train_dataset,
                "test": test_dataset}
    
    collate = get_collate_fn(name, hml_mode)
    
    loader = DataLoader(
        datasets, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True, collate_fn=collate
    )

    return loader

def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train'):
    train_dataset = get_dataset(name, num_frames, 'train', hml_mode)
    test_dataset = get_dataset(name, num_frames, 'test', hml_mode)

    if split == 'train':
        dataset = train_dataset
    elif split == 'test':
        dataset = test_dataset
    else:
        raise ValueError(f"Invalid split: {split}")

    collate = get_collate_fn(name, hml_mode)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate
    )
    
    return loader