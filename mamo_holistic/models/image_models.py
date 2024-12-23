import torch
import torch.nn as nn

from models.modules import Gray2RGBadaptor, SEBlock, CBAMImagePooling
import timm


class PatchFusionAttention(torch.nn.Module):
    def __init__(self, nfeatures):
        super().__init__()

        self.fc = torch.nn.Linear(nfeatures, 2)
        self.fc_final = torch.nn.Linear(nfeatures, 2)
        
    def init_from_patchmodel(self, patch_model):
        weights = patch_model[1].head.fc[1].weight[5:,:].data.cpu()
        bias = patch_model[1].head.fc[1].bias[5:].data.cpu()
        self.fc.weight.data = weights
        self.fc_final.weight.data = weights
        self.fc.bias.data = bias
        self.fc_final.bias.data = bias
    
    def forward(self, x):
        # Reshape input for batch processing
        tokens = x.reshape(x.shape[0], -1, x.shape[-1])  # (b, t, c)
        #print("tokens.shape:", tokens.shape)
        
        # Normalize tokens
        normalized_tokens = tokens / tokens.norm(dim=-1, keepdim=True)  # (b, t, c)
        
        # Compute attention
        attention = torch.einsum('b t f, b T f -> b t T', normalized_tokens, normalized_tokens)  # (b, t, T)
        attention = torch.nn.functional.softmax(attention, dim=-1)  # Softmax over T
        #print("attention.shape:", attention.shape)
        
        # Compute cancer probabilities
        cancer_probs = torch.softmax(self.fc(tokens), dim=-1)[:, :, 1]  # (b, t)
        #print("cancer_probs:", cancer_probs.shape)
        
        # Find index of max cancer probability for each batch
        idx = torch.argmax(cancer_probs, dim=1)  # (b)
        #print("idx:", idx)
        
        # Use gather to select the attention vectors corresponding to idx
        batch_indices = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)  # (b, 1)
        attention_idx = attention[batch_indices, idx.unsqueeze(-1)].squeeze(1)  # (b, t)
        attention_idx = attention_idx * cancer_probs  # Weight by cancer_probs
        attention_idx = attention_idx / attention_idx.sum(dim=-1, keepdim=True)  # Normalize
        
        # Compute fused representation
        x_fused = torch.einsum('b t, b t f -> b f', attention_idx, tokens)  # (b, c)
        #print("x_fused.shape:", x_fused.shape)
        
        # Final output
        return self.fc_final(x_fused)
    

import torch.distributions as dist

class RelevantTokenSelector(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.scorer = nn.Linear(embedding_dim, 1)  # Scoring relevance for each token
        
    def init_scorer_from_patchmodel(self, patch_model):
        weights = patch_model[1].head.fc[1].weight[5:,:].data.cpu().reshape(1,-1)
        bias = patch_model[1].head.fc[1].bias[5:].data.cpu()
        
        self.scorer.weight.data = weights
        self.scorer.bias.data = bias

    def forward(self, token_embeddings):
        # Compute relevance scores (shape: batch_size, seq_len, nfeatures)
        relevance_logits = self.scorer(token_embeddings).squeeze(-1)  # (batch_size, seq_len)
        

        if self.training:
            # During training, sample tokens based on the relevance scores
            categorical_dist = dist.Categorical(logits=relevance_logits)  # Create a Categorical distribution
            
            # Sample token indices (shape: batch_size)
            selected_token_indices = categorical_dist.sample()  # (batch_size,)
        else:
            # During evaluation, select the token with the highest relevance score
            selected_token_indices = torch.argmax(relevance_logits, dim=-1)        
        
        # Gather the embeddings of the selected tokens
        selected_token_embeddings = token_embeddings[torch.arange(token_embeddings.size(0)), selected_token_indices]  # (batch_size, embedding_dim) 
        
        return selected_token_embeddings, selected_token_indices
 

import numpy as np
class RelevantTokenAttention(nn.Module):
    def __init__(self, window_size_x, window_size_y, qv_dim = 256, token_dim = 1024):
        super().__init__()
        self.create_positional_biases(window_size_x, window_size_y)
        self.token_selector = RelevantTokenSelector(token_dim, window_size_x*window_size_y)
        self.kv = nn.Linear(token_dim, qv_dim)
        self.layer_norm = nn.LayerNorm(token_dim)
        self.create_positional_biases(window_size_x, window_size_y)
        self.norm_attention = np.sqrt(qv_dim)
        self.fc_final = torch.nn.Linear(token_dim, 2)
        
    def create_positional_biases(self, window_size_x, window_size_y):
        mesh = np.meshgrid(range(window_size_x), range(window_size_y))

        indices = np.array([np.array([x,y]) for x,y in zip(mesh[0].flatten(), mesh[1].flatten())])

        pos_dict = {}
        relative_positions = indices[np.newaxis,:] - indices[:,np.newaxis]
        relative_indexes = []
        #print(relative_positions.shape)
        for pos in (relative_positions.reshape(-1,2)):
            pos = (pos[0], pos[1])

            if not pos in pos_dict:
                pos_dict[pos] = len(pos_dict)
            
            relative_indexes.append(pos_dict[pos])
                
        relative_indexes = np.array(relative_indexes).reshape(window_size_y*window_size_x, window_size_y*window_size_x)
        #print(relative_indexes)   
        self.register_buffer('relative_indexes', torch.tensor(relative_indexes, dtype=torch.long))
        self.unique_bias_param = nn.Parameter(torch.zeros(len(pos_dict), dtype=torch.float))
        
    def init_scorer_from_patchmodel(self, patch_model):
        print("Token selector init scorer intializing")
        self.token_selector.init_scorer_from_patchmodel(patch_model)

  
    def forward(self, x):
        # x B x N x token_dim        
        if len(x.shape) == 4: # B x H x W x token_dim 
            x = x.reshape(x.shape[0], -1, x.shape[-1]) # B x N x token_dim
        
        selected_token, selected_idx = self.token_selector(x) # B x token_dim, B 
        
        xnorm = self.layer_norm(x)
        
        key = self.kv(xnorm) # B x N x qv_dim
        query = self.kv(self.layer_norm(selected_token)) # B x  qv_dim
        
        attn = torch.einsum('bd,bnd->bn', query, key) / self.norm_attention   # B x N
        
        self.bias_param = self.unique_bias_param[self.relative_indexes]
        bias = self.bias_param[ selected_idx, :] # B x N
        
        attn = attn + bias
        
        attn = torch.softmax(attn, dim=-1)
        y = torch.einsum('bn,bnd->bd', attn, xnorm) # B x token_dim
    
        y = y + selected_token
        
        y = self.fc_final(y)
        
        
        
        return y




class SwinBreastCancer(torch.nn.Module):
    def __init__(self, num_classes=2, image_size=(2240,1792), **kwargs):
        super().__init__()
    
        swin_model_name = kwargs.get("swin_model_name", "swin_base_patch4_window7_224")
        swin = timm.create_model(swin_model_name, pretrained=False)
        swin.head.fc = nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(swin.head.fc.in_features, 6))
        
        self.patch_model = nn.Sequential(
            Gray2RGBadaptor(), 
            swin)
        
        patch_checkpoint = kwargs.get("patch_checkpoint", None)
        if patch_checkpoint is not None:
            checkpoint = torch.load(patch_checkpoint, map_location='cpu', weights_only=True)
            new_state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
            self.patch_model.load_state_dict(new_state_dict)
        
        image_size_swin = (image_size[0]//2, image_size[1]//2)
        self.patch_model[1].set_input_size(image_size_swin)
        
        self.pooling = nn.AvgPool2d(7, stride=7)
        
        if kwargs.get("patch_fusion", "PatchFusionAttention") == "PatchFusionAttention":
            self.fusion = PatchFusionAttention(self.patch_model[1].head.fc[1].in_features)
            self.fusion.init_from_patchmodel(self.patch_model)
        elif kwargs.get("patch_fusion", "PatchFusionAttention") == "RelevantTokenAttention":
            print("Using RelevantTokenAttention")
            window_size_x = kwargs.get("window_size_x", 4)
            window_size_y = kwargs.get("window_size_y", 5)
            self.fusion = RelevantTokenAttention(window_size_x, window_size_y, qv_dim = 256, token_dim = self.patch_model[1].head.fc[1].in_features)
            self.fusion.init_scorer_from_patchmodel(self.patch_model)
        
        #self.freeze_patch_model()
        if kwargs.get("LoRA", False):
            self.set_LoRA(**kwargs.get("LoRA"))
        else:
            print("Not using LoRA, Unfreezing layer 3")
            self.freeze_patch_model()
            self.unfreeze_layers_patch_model(3)
        
        
    def set_LoRA(self, **kwargs):
            from peft import get_peft_model, LoraConfig, TaskType
            task_type=TaskType.FEATURE_EXTRACTION
            print("Using LoRA")
            # r = kwargs["LoRA"].get("r", 64)
            # lora_alpha = kwargs["LoRA"].get("alpha", 32)
            # lora_dropout = kwargs["LoRA"].get("dropout", 0.2)
            # target_modules = kwargs["LoRA"].get("target_modules", ["qkv"])
            
            r = kwargs.get("r", 128)
            lora_alpha = kwargs.get("alpha", 32)
            lora_dropout = kwargs.get("dropout", 0.2)
            target_modules = kwargs.get("target_modules", ["qkv"])
        
            lora_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,  # Default for transformer models
            )
            
            self.patch_model[1] = get_peft_model(self.patch_model[1], lora_config)
            
    
    def freeze_patch_model(self):
        for param in self.patch_model.parameters():
            param.requires_grad = False
            
    def unfreeze_layers_patch_model(self, layers_to_unfreeze):
        
        if isinstance(layers_to_unfreeze,str) and  layers_to_unfreeze == "all":
            layers_to_unfreeze = list(range(len(self.patch_model.layers)))
        elif isinstance(layers_to_unfreeze,int):
            layers_to_unfreeze = [layers_to_unfreeze]
        
        for layer in layers_to_unfreeze:
            for param in self.patch_model[1].layers[layer].parameters():
                param.requires_grad = True    
        
    def forward(self, x):
        x_rgb = self.patch_model[0](x) # Convert to RGB
        patch_features = self.patch_model[1].forward_features(x_rgb) # b h w c
        patch_features = patch_features.permute(0, 3, 1, 2) # b c h w
        pooled_features = self.pooling(patch_features)
        pooled_features = pooled_features.permute(0, 2, 3, 1) # b h w c
        
        out  = self.fusion(pooled_features)
        
        return out