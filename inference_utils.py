import torch
from PIL import Image
from einops import rearrange


def translation(model, tokenizer, img_transform, img_path, max_len, device):
    model.eval()
    with torch.no_grad():
        img = Image.open(img_path).convert("RGB")
        img = img_transform(img).to(device)
        img = img.unsqueeze(0)
        enc_out = model.feature_extractor(img)
        enc_out = rearrange(enc_out, "B C H W -> B (H W) C")
        enc_out = model.feature_projection(enc_out)

        pred = tokenizer.encode('</s>', return_tensors='pt', add_special_tokens=False).to(device) # 1x1
        for _ in range(1, max_len - 1): # <sos> 가 한 토큰이기 때문에 최대 99 번까지만 loop을 돌아야 함
            dec_mask = model.make_dec_mask(pred).to(device)
            enc_dec_mask = None

            pos = torch.arange(pred.shape[1]).repeat(pred.shape[0], 1).to(device)
            x = model.scale * model.input_embedding(pred) + model.pos_embedding(pos)
            x = model.dropout(x)

            for layer in model.layers:
                x = layer(x, enc_out, dec_mask, enc_dec_mask)
            out = model.fc_out(x)

            pred_word = out.argmax(dim=2)[:,-1].unsqueeze(0) # shape = (1,1)
            pred = torch.cat([pred, pred_word], dim=1) # 1x단 (단은 하나씩 늘면서)

            if tokenizer.decode(pred_word.item()) == '</s>':
                break

        translated_text = tokenizer.decode(pred[0])

    return translated_text