"""Local CLIP backend. Images and labels are embedded independently; no truth input."""
class ClipMatcher:
    def __init__(self, model, revision, cache_dir=None):
        from pathlib import Path
        cache_dir = cache_dir or str(Path(__file__).resolve().parents[1]/'models')
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        torch.manual_seed(0)
        torch.set_num_threads(2)
        self.processor = CLIPProcessor.from_pretrained(model, revision=revision, cache_dir=cache_dir)
        self.model = CLIPModel.from_pretrained(model, revision=revision, cache_dir=cache_dir).eval().cpu()

    def text_features(self, texts, batch_size=32):
        t = self.torch
        rows = []
        with t.inference_mode():
            for i in range(0, len(texts), batch_size):
                tokens = self.processor(text=texts[i:i+batch_size], return_tensors='pt', padding=True, truncation=True, max_length=77)
                features = self.model.get_text_features(**tokens)
                rows.append(features/features.norm(dim=-1,keepdim=True))
        return t.cat(rows)

    def image_features(self, candidates):
        from PIL import Image
        t = self.torch
        rows, sizes = [], {}
        with t.inference_mode():
            for c in candidates:
                with Image.open(c.image) as source:
                    sizes[c.image] = source.size
                    x, y, w, h = c.bbox
                    if x+w > source.width or y+h > source.height:
                        raise ValueError('Box outside image')
                    crop = source.convert('RGB').crop((x, y, x+w, y+h))
                pixels = self.processor(images=crop, return_tensors='pt')
                visual = self.model.get_image_features(**pixels)
                rows.append(visual/visual.norm(dim=-1,keepdim=True))
        return t.cat(rows), sizes

    def scores(self, candidates, vocabulary):
        text = self.text_features(['a photo of a '+v for v in vocabulary])
        visual, sizes = self.image_features(candidates)
        return (visual @ text.T).tolist(), sizes
