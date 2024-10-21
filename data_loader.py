import os
import torch
from torch.utils.data import Dataset, DataLoader
import collections
from PIL import Image
from torchvision import transforms


def get_feature_channel_num(feature_extractor, input_img_shape):
    img = torch.rand(size = input_img_shape)
    feature_extractor.eval()
    with torch.no_grad():
        sample_feature = feature_extractor(img)
    _, c, h, w = sample_feature.shape
    
    return c

class ImgCaptionDataset(Dataset):
    def __init__(self, img_dir, annotations_file, img_transform = None):

        self.flickr8k_list = self.flickr8k_info(img_dir, annotations_file)
        self.img_transform =  img_transform if img_transform is not None else transforms.ToTensor()
    
    def __len__(self):
        return len(self.flickr8k_list)
    
    def __getitem__(self, idx):
        flickr8k = self.flickr8k_list[idx]
        img_path, caption = next(iter(flickr8k.items()))
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        return img, caption
    
    def flickr8k_info(self, img_dir, annotations_file):
        cap_dict = collections.defaultdict(list)
        captions = open(annotations_file).read().splitlines()
        captions = (line.split("\t") for line in captions)
        captions = ((fname.split("#")[0], caption) for (fname, caption) in captions)


        for fname, caption in captions:
            cap_dict[fname].append(caption)
        
        info = [{os.path.join(img_dir, key) : v} for key, value in cap_dict.items() for v in value if os.path.exists(os.path.join(img_dir, key))]
   
        return info

class ImgCapDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, tokenizer, max_len, shuffle):
        super().__init__(dataset = dataset,
                         batch_size = batch_size,
                         shuffle = shuffle,
                         collate_fn = self.collate_fn)
        
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def prepare_tokens(self, txts):
        txts = tuple("</s>" + txt + "</s>" for txt in txts)
        tokens = self.tokenizer(txts,
                           padding = True, #"max_length",
                           truncation = True,
                           max_length = self.max_len,
                           return_tensors = "pt"
                           )["input_ids"]

        input_tokens = tokens[..., :-1]
        label_tokens = tokens[..., 1:]
        
        return input_tokens, label_tokens
    
    def collate_fn(self, batch):
        imgs, txts = zip(*batch)

        input_tokens, label_tokens = self.prepare_tokens(txts)

        return (torch.stack(imgs), input_tokens), label_tokens

