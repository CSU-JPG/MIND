import torch
from utils.utils import clip_transform_Image
import torch.nn.functional as F

def visual_quality_metric(images, imaging_model, aesthetic_model, clip_model, batch_size=8):
    result_dict = {}
    scores = []
    for i in range(0, len(images), batch_size):
        frame = images[i:i+batch_size]
        imaging_batch = imaging_model(frame).cpu().tolist()
        imaging_batch = [item[0] for item in imaging_batch]
        scores.extend(imaging_batch)
    scores = [i/100 for i in scores]
    result_dict["imaging"] = scores
    result_dict["avg_imaging"] = sum(scores) / len(scores)

    scores = []
    image_transform = clip_transform_Image(224)
    for i in range(0, len(images), batch_size):
        batch = images[i: i+batch_size]
        batch = image_transform(batch)
        image_feats = clip_model.encode_image(batch).to(torch.float32)
        image_feats = F.normalize(image_feats, dim=-1, p=2)
        aesthetic_scores = aesthetic_model(image_feats).squeeze(dim=-1)
        scores.extend(aesthetic_scores.cpu().tolist())
    scores = [i/10 for i in scores]
    result_dict["aesthetic"] = scores
    result_dict["avg_aesthetic"] = sum(scores) / len(scores)
    return result_dict