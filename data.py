import io, json, os, random
from collections import defaultdict

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from datasets import load_dataset_builder, load_dataset, Image as HFImage
from tqdm import tqdm


MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  


def class_mapping():
    names = load_dataset_builder("clane9/imagenet-100").info.features["label"].names
    path = hf_hub_download("huggingface/label-files", "imagenet-1k-id2label.json", repo_type="dataset")
    label2id = {v: int(k) for k, v in json.load(open(path)).items()}
    idx_1k = [label2id[n] for n in names]
    return names, idx_1k


def _download_subset(split, n, seed):
    dataset = load_dataset("clane9/imagenet-100", split=split, streaming=True)
    dataset = dataset.cast_column("image", HFImage(decode=False)) 

    rng = random.Random(seed)
    samples, seen = defaultdict(list), defaultdict(int)

    for obj in tqdm(dataset, desc=split):
        c, img = obj["label"], obj["image"]["bytes"]
        seen[c] += 1

        if len(samples[c]) < n:
            samples[c].append(img)
        else:
            j = rng.randrange(seen[c])
            if j < n:
                samples[c][j] = img

    images = [img for c in sorted(samples) for img in samples[c]]
    labels = [c for c in sorted(samples) for _ in samples[c]]

    return images, labels


class ImageNet100Subset(Dataset):

    def __init__(self, root, split, transform, n, seed=0):
        os.makedirs(root, exist_ok=True)
        cache = os.path.join(root, f"{split}_{n}pc_seed{seed}.pt")
        
        if not os.path.exists(cache):
            images, labels = _download_subset(split, n, seed)
            torch.save({"images": images, "labels": labels}, cache)
        
        d = torch.load(cache)
        self.images, self.labels, self.transform = d["images"], d["labels"], transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.open(io.BytesIO(self.images[i])).convert("RGB")
        return self.transform(img), self.labels[i]


def get_transforms():

    to_tensor = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
    
    train = v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.5, 1.0)), 
        v2.RandomHorizontalFlip(), *to_tensor])
    
    val = v2.Compose([
        v2.Resize((224, 224)), *to_tensor])
    return train, val


def get_loaders(root="data", batch_size=32, num_workers=4, seed=0):
  
    train_tf, val_tf = get_transforms()
    
    train_set = ImageNet100Subset(root, "train", train_tf, 20, seed)       
    val_set = ImageNet100Subset(root, "validation", val_tf, 5, seed)      

    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader




def _download_subset_gsm8k(split, n, seed):
    dataset = load_dataset("openai/gsm8k", 'main', streaming=True)
 
    rng = random.Random(seed)
    samples = []

    for i, obj in enumerate(tqdm(dataset[split], desc=split)):
        sample = obj["question"], obj["answer"]

        if len(samples) < n:
            samples.append(sample)
        else:
            j = rng.randrange(i)
            if j < n:
                samples[j] = sample

    questions = [q for q,_ in samples]
    answers = [a for _,a in samples]

    return questions, answers


class GSM8KSubset(Dataset):

    def __init__(self, root, split, n, seed=0):
        os.makedirs(root, exist_ok=True)
        cache = os.path.join(root, f"{split}_{n}gsm8k_seed{seed}.pt")
        
        if not os.path.exists(cache):
            questions, answers = _download_subset_gsm8k(split, n, seed)
            torch.save({"questions": questions, "answers": answers }, cache)
        
        d = torch.load(cache)
        self.questions, self.answers = d["questions"], d["answers"]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, i):
        return self.questions[i], self.answers[i]


CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def get_cifar_loaders(root=".", n_train=5000, n_val=1000, batch_size=64, num_workers=2, seed=0):

    to_tensor = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(CIFAR_MEAN, CIFAR_STD)]
    train_tf = v2.Compose([v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip(), *to_tensor])
    val_tf = v2.Compose(to_tensor)

    # both subsets come from the official training split (expects root/cifar-10-batches-py)
    train_full = CIFAR10(root, train=True, transform=train_tf, download=False)
    val_full = CIFAR10(root, train=True, transform=val_tf, download=False)

    by_class = defaultdict(list)
    for i, c in enumerate(train_full.targets):
        by_class[c].append(i)

    rng = random.Random(seed)
    n_tr, n_va = n_train // len(by_class), n_val // len(by_class)
    train_idx, val_idx = [], []

    for c in sorted(by_class):
        idx = rng.sample(by_class[c], n_tr + n_va)   # disjoint, stratified
        train_idx += idx[:n_tr]
        val_idx += idx[n_tr:]

    train_loader = DataLoader(Subset(train_full, train_idx), batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(Subset(val_full, val_idx), batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
