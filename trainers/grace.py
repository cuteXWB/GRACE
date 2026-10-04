import torch
import torch.nn as nn
from torch.nn import functional as F
from open_clip_lib import create_model_and_transforms, HFTokenizer, get_mean_std
from typing import Optional
from .layers import PVL_Adapter
from .scale_block import ScaleBlock
import os


def _resolve_repo_path(*parts: str) -> str:

    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(this_dir)
    return os.path.join(repo_root, *parts)


def _resolve_pretrained_path(cfg, default_rel: str) -> str:





    configured = getattr(cfg.MODEL, "UNIMEDCLIP_PRETRAINED_PATH", None)
    candidate = configured if configured else default_rel


    if os.path.isabs(candidate):
        return candidate


    cwd_path = os.path.abspath(candidate)
    if os.path.exists(cwd_path):
        return cwd_path


    rel_norm = candidate.replace("\\", "/").lstrip("./")
    repo_path = _resolve_repo_path(*rel_norm.split("/"))
    return repo_path

def get_text_encoder_name():
    return os.environ.get(
        "GRACE_TEXT_ENCODER_PATH",
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    )

def load_unimedclip_to_device(cfg):
    if cfg.MODEL.BACKBONE != "ViT-B/16":
        raise NotImplementedError(f"Backbone {cfg.MODEL.BACKBONE} not implemented.")

    model_name = "ViT-B-16-quickgelu"
    pretrained_weights = _resolve_pretrained_path(
        cfg, default_rel="checkpoints/unimed_clip_vit_b16.pt"
    )

    if not os.path.isfile(pretrained_weights):
        raise FileNotFoundError(
            "UniMedCLIP pretrained checkpoint not found.\n"
            f"Expected path: {pretrained_weights}\n"
            "Set MODEL.UNIMEDCLIP_PRETRAINED_PATH to a valid .pt file, e.g.\n"
            "  checkpoints/unimed_clip_vit_b16.pt"
        )

    text_encoder_name = get_text_encoder_name()
    mean, std = get_mean_std()
    device=cfg.MODEL.DEVICE

    model, _, _ = create_model_and_transforms(
        model_name,
        pretrained_weights,
        precision='amp',
        device=device,
        force_quick_gelu=True,
        mean=mean, std=std,
        inmem=True,
        text_encoder_name=text_encoder_name,)

    return model.to(device).eval()


class GRACEModel(nn.Module):
    def __init__(self, cfg, clip_model, output_hidden_states=False):
        super(GRACEModel, self).__init__()

        self.cfg = cfg
        self.vision_model = clip_model.visual
        self.text_model = clip_model.text_encoder
        self.logit_scale = clip_model.logit_scale
        self.temperature = cfg.MODEL.TEMPERATURE
        self.fusion_stages = cfg.MODEL.FUSION_LAYERS

        if(cfg.MODEL.BACKBONE == "ViT-B/16"):
            self.embed_dim = 768
            self.patch_size = 16
            self.text_proj_dim = 512

        elif(cfg.MODEL.BACKBONE == "ViT-L/14"):
            self.embed_dim = 1024
            self.patch_size = 14
            self.text_proj_dim = 768
            raise NotImplementedError("ViT-L/14 not implemented yet.")

        self.output_hidden_states = output_hidden_states
        self.dtype = self.text_model.transformer.dtype
        self.im_size = cfg.DATASET.SIZE
        self.device = cfg.MODEL.DEVICE

        self.tokenizer = HFTokenizer(
                            get_text_encoder_name(),
                            context_length=256,
                            **{},)

        adapter_channels = cfg.MODEL.ADAPTER_DIM
        self.num_upscale = cfg.MODEL.NUM_UPSCALE
        self.beta = cfg.MODEL.ATTENTION_BETA
        self.beta_mode = str(
            getattr(cfg.MODEL, "ATTENTION_BETA_MODE", "adaptive")
        ).lower()
        self.beta_min = float(getattr(cfg.MODEL, "ATTENTION_BETA_MIN", 1.0))
        self.beta_max = float(getattr(cfg.MODEL, "ATTENTION_BETA_MAX", 3.0))
        self.beta_adapt_scale = float(
            getattr(cfg.MODEL, "ATTENTION_BETA_ADAPT_SCALE", 1.0)
        )
        self.gate_init = cfg.MODEL.GATE_INIT

        self.mask_head = nn.Sequential(
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
            nn.GELU(),
            nn.Linear(self.text_proj_dim, self.text_proj_dim),
        )

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.text_proj_dim) for _ in range(self.num_upscale)],
        )

        self.pvl_adapters = nn.ModuleList([
            PVL_Adapter(in_channels_vis=self.embed_dim, in_channels_txt=self.embed_dim, adapter_channels=adapter_channels,
                            beta=self.beta, gate_init=self.gate_init,
                            beta_mode=self.beta_mode, beta_min=self.beta_min, beta_max=self.beta_max,
                            beta_adapt_scale=self.beta_adapt_scale)
            for _ in range(len(self.fusion_stages))
        ])

    def encode_text_image(self,  tokenized_prompts, text_prompts, image,
                            attention_mask: Optional[torch.LongTensor] = None):

        if attention_mask is None:
            attention_mask = (tokenized_prompts != self.text_model.config.pad_token_id).long()

        x_txt = self.text_model.transformer.embeddings(
            inputs_embeds=text_prompts
        )

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        x_img = self.vision_model.conv1(image)
        x_img = x_img.reshape(x_img.shape[0], x_img.shape[1], -1)
        x_img = x_img.permute(0, 2, 1)
        x_img = torch.cat(
            [self.vision_model.class_embedding.to(x_img.dtype) + torch.zeros(x_img.shape[0], 1, x_img.shape[-1], dtype=x_img.dtype, device=x_img.device),
             x_img], dim=1)

        x_img = x_img + self.vision_model.positional_embedding.to(x_img.dtype)
        x_img = self.vision_model.ln_pre(x_img)
        x_img = x_img.permute(1, 0, 2)

        hidden_states = []

        for i, (block, layer) in enumerate(zip(self.vision_model.transformer.resblocks,self.text_model.transformer.encoder.layer)):

            if(i in self.fusion_stages):

                vis_pvl, txt_pvl = self.pvl_adapters[self.fusion_stages.index(i)](x_img.transpose(1,0), x_txt)

                x_txt = x_txt + txt_pvl
                x_img = x_img + vis_pvl.transpose(1,0)

            x_img = block(x_img)
            x_txt = layer(x_txt, attention_mask=extended_attention_mask)

            hidden_states.append(x_img)
            x_txt = x_txt[0]

        x_img = x_img.permute(1, 0, 2)

        x_img = self.vision_model.ln_post(x_img)

        if self.vision_model.proj is not None:
            x_img = x_img @ self.vision_model.proj

        pooled_out = x_txt[:, 0, :]
        projected = self.text_model.proj(pooled_out)
        x_txt = self.text_model.proj(x_txt)

        if self.output_hidden_states:
            return x_img, hidden_states, projected
        else:
            return x_img, projected


    def compute_seg_logits(self, image_features, text_features, B, H, W):

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        cls_token = image_features[:, 0, :]
        cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)

        seg_feats = image_features[:, 1:, :]
        seg_feats = seg_feats / seg_feats.norm(dim=-1, keepdim=True)

        h_patch = H // self.patch_size
        w_patch = W // self.patch_size
        seg_feats = seg_feats.reshape(B, h_patch, w_patch, -1).permute(0,3,1,2)

        seg_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(text_features).unsqueeze(1), self.upscale(seg_feats)
        )
        seg_logits = F.interpolate(seg_logits, self.im_size, mode="bilinear", align_corners=False).squeeze(1)

        return seg_logits, cls_token

    def soft_cross_entropy(self, pred_logits, soft_targets):
        log_probs = F.log_softmax(pred_logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        return loss

    def prepare_text(self, text):
        tokenized_prompts = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            prompts = self.text_model.transformer.embeddings.word_embeddings(
                tokenized_prompts
            ).type(self.dtype)
        return tokenized_prompts, prompts

    def extract_features(self, image, text):
        _, _, H, W = image.shape
        tokenized_prompts, prompts = self.prepare_text(text)
        image_features, text_features = self.encode_text_image(
            tokenized_prompts, prompts, image
        )
        return image_features, text_features, H, W

    def decode_from_features(self, image_features, text_features, H, W):
        B = image_features.shape[0]
        seg_logits, cls_token = self.compute_seg_logits(
            image_features, text_features, B, H, W
        )
        return seg_logits, cls_token

    def compute_clip_loss_from_features(self, image_features, text_features):
        patch_logits = image_features[:, 1:, :]
        patch_logits = patch_logits / patch_logits.norm(dim=-1, keepdim=True)
        patch_mean = patch_logits.mean(dim=1)

        logits_per_image = (patch_mean @ text_features.T) / self.temperature
        logits_per_text = (text_features @ patch_mean.T) / self.temperature

        with torch.no_grad():
            text_sim = (text_features @ text_features.T) / self.temperature
            text_sim = text_sim / text_sim.norm(dim=-1, keepdim=True)
            soft_targets = F.softmax(text_sim, dim=-1)

        loss_i2t = self.soft_cross_entropy(logits_per_image, soft_targets)
        loss_t2i = self.soft_cross_entropy(logits_per_text, soft_targets.T)
        clip_loss = (loss_i2t + loss_t2i) / 2

        return clip_loss

    def generate_virtual_features(
        self,
        z_src,
        text_features,
        mask,
        H,
        W,
        seg_loss_fn,
        steps=2,
        step_size=5e-3,
        eps=2e-2,
        patch_only=True,
        random_start=True,
        random_start_scale=0.5,
    ):
        z0 = z_src.detach()
        z_virt = z0.clone().detach()

        if random_start:
            rand_scale = max(float(random_start_scale), 0.0)
            delta0 = torch.empty_like(z_virt).uniform_(-eps, eps) * rand_scale
            delta0 = torch.clamp(delta0, min=-eps, max=eps)
            if patch_only:
                delta0[:, 0, :] = 0.0
            z_virt = (z0 + delta0).detach()

        for _ in range(steps):
            z_virt.requires_grad_(True)

            seg_logits_virt, _ = self.decode_from_features(
                z_virt, text_features.detach(), H, W
            )
            loss_task = seg_loss_fn(seg_logits_virt, mask)

            grad = torch.autograd.grad(
                loss_task, z_virt, retain_graph=False, create_graph=False
            )[0]

            if patch_only:
                grad[:, 0, :] = 0.0

            grad_step = grad.sign()

            with torch.no_grad():
                z_virt = z_virt + step_size * grad_step
                delta = z_virt - z0
                delta = torch.clamp(delta, min=-eps, max=eps)

                if patch_only:
                    delta[:, 0, :] = 0.0

                z_virt = (z0 + delta).detach()

        return z_virt

    def split_cls_patch_tokens(self, z):
        cls_token = z[:, :1, :]
        patch_tokens = z[:, 1:, :]
        return cls_token, patch_tokens

    def sample_local_probes(
        self,
        z,
        num_probes=3,
        noise_std=0.008,
        patch_only=True,
    ):
        if num_probes < 1:
            raise ValueError("num_probes must be >= 1")

        neighbors = []
        for _ in range(num_probes):
            noise = torch.randn_like(z) * noise_std
            if patch_only:
                noise[:, 0, :] = 0.0
            neighbors.append(z + noise)
        return neighbors

    def compute_segmentation_probe(
        self,
        seg_logits,
        foreground_threshold=0.7,
    ):

        probabilities = torch.sigmoid(seg_logits)
        foreground = probabilities > float(foreground_threshold)
        flat_probabilities = probabilities.reshape(probabilities.shape[0], -1)
        flat_foreground = foreground.reshape(foreground.shape[0], -1).float()
        foreground_count = flat_foreground.sum(dim=1)
        foreground_mean = (
            (flat_probabilities * flat_foreground).sum(dim=1)
            / foreground_count.clamp_min(1.0)
        )
        full_mean = flat_probabilities.mean(dim=1)
        return torch.where(
            foreground_count > 0,
            foreground_mean,
            full_mean,
        ).mean()

    def compute_probe_gradients(
        self,
        z,
        text_features,
        H,
        W,
        num_probes=3,
        noise_std=0.008,
        foreground_threshold=0.7,
        patch_only=True,
    ):
        z0 = z.detach().clone()
        neighbors = self.sample_local_probes(
            z0,
            num_probes=num_probes,
            noise_std=noise_std,
            patch_only=patch_only,
        )

        grads_c = []
        for z_nb in neighbors:
            with torch.enable_grad():
                z_nb = z_nb.detach().clone().requires_grad_(True)
                seg_logits_probe, _ = self.decode_from_features(
                    z_nb, text_features.detach(), H, W
                )


                probe_score = self.compute_segmentation_probe(
                    seg_logits_probe,
                    foreground_threshold=foreground_threshold,
                )
                probe_grad = torch.autograd.grad(
                    probe_score, z_nb, retain_graph=False, create_graph=False
                )[0]

            grad_patch = probe_grad[:, 1:, :]

            grad_channel = grad_patch.abs().mean(dim=1)
            grad_channel = grad_channel / (
                grad_channel.norm(dim=-1, keepdim=True) + 1e-6
            )
            grads_c.append(grad_channel)

        return torch.stack(grads_c, dim=0)

    def compute_channel_consistency(self, grads_c):
        var_c = grads_c.var(dim=0, unbiased=False)
        return var_c

    def compute_consistency_stats(
        self,
        z,
        text_features,
        H,
        W,
        num_probes=3,
        noise_std=0.008,
        foreground_threshold=0.7,
        patch_only=True,
    ):
        grads_c = self.compute_probe_gradients(
            z,
            text_features,
            H,
            W,
            num_probes=num_probes,
            noise_std=noise_std,
            foreground_threshold=foreground_threshold,
            patch_only=patch_only,
        )
        return self.compute_channel_consistency(grads_c)

    def build_instability_score(self, var_c):
        var_min = var_c.amin(dim=-1, keepdim=True)
        var_max = var_c.amax(dim=-1, keepdim=True)

        var_range = var_max - var_min + 1e-12
        instability_score = (var_c - var_min) / var_range
        return instability_score

    def build_retention_weights(
        self,
        instability_score,
        tau=3.0,
        min_keep=0.55,
        max_keep=1.0,
        blend_beta=1.0,
        center=0.5,
    ):
        if min_keep < 0.0 or max_keep > 1.0 or min_keep > max_keep:
            raise ValueError(
                f"Invalid keep range: min_keep={min_keep}, max_keep={max_keep}"
            )

        beta = float(max(0.0, min(1.0, blend_beta)))
        unstable = torch.sigmoid(tau * (instability_score - center))
        base_weight = max_keep - (max_keep - min_keep) * unstable
        base_weight = torch.clamp(base_weight, min=min_keep, max=max_keep)


        return 1.0 - beta * (1.0 - base_weight)

    def estimate_retention_weights(
        self,
        z,
        text_features,
        H,
        W,
        num_probes=3,
        noise_std=0.008,
        foreground_threshold=0.7,
        tau=3.0,
        min_keep=0.55,
        max_keep=1.0,
        blend_beta=1.0,
        patch_only=True,
    ):
        var_c = self.compute_consistency_stats(
            z,
            text_features,
            H,
            W,
            num_probes=num_probes,
            noise_std=noise_std,
            foreground_threshold=foreground_threshold,
            patch_only=patch_only,
        )

        instability_score = self.build_instability_score(var_c)
        retention_w = self.build_retention_weights(
            instability_score,
            tau=tau,
            min_keep=min_keep,
            max_keep=max_keep,
            blend_beta=blend_beta,
        )

        aux_dict = {
            "var_c": var_c,
            "instability_score_c": instability_score,
            "mask_c": retention_w,
            "var_range_mean": (var_c.amax(dim=-1) - var_c.amin(dim=-1)).mean().item(),
            "grad_source": "probe_score_dz",
        }
        return retention_w, aux_dict

    def estimate_shared_retention_weights(
        self,
        z_src,
        z_virt,
        text_features,
        H,
        W,
        num_probes=3,
        noise_std=0.008,
        foreground_threshold=0.7,
        tau=3.0,
        min_keep=0.55,
        max_keep=1.0,
        blend_beta=1.0,
        patch_only=True,
    ):

        grads_src = self.compute_probe_gradients(
            z_src,
            text_features,
            H,
            W,
            num_probes=num_probes,
            noise_std=noise_std,
            foreground_threshold=foreground_threshold,
            patch_only=patch_only,
        )
        grads_virt = self.compute_probe_gradients(
            z_virt,
            text_features,
            H,
            W,
            num_probes=num_probes,
            noise_std=noise_std,
            foreground_threshold=foreground_threshold,
            patch_only=patch_only,
        )
        grads_joint = torch.cat([grads_src, grads_virt], dim=0)
        var_c = self.compute_channel_consistency(grads_joint)
        instability_score = self.build_instability_score(var_c)
        retention_w = self.build_retention_weights(
            instability_score,
            tau=tau,
            min_keep=min_keep,
            max_keep=max_keep,
            blend_beta=blend_beta,
        )

        aux_dict = {
            "var_c": var_c,
            "instability_score_c": instability_score,
            "mask_c": retention_w,
            "var_range_mean": (var_c.amax(dim=-1) - var_c.amin(dim=-1)).mean().item(),
            "joint_mask_shared": 1.0,
            "joint_grad_groups": int(grads_joint.shape[0]),
            "joint_source": "source_virt_2m",
            "grad_source": "probe_score_dz",
        }
        return retention_w, aux_dict

    def apply_retention_weights(
        self,
        z,
        retention_w,
    ):
        cls_token, patch_tokens = self.split_cls_patch_tokens(z)
        w_patch = retention_w.unsqueeze(1)
        patch_tokens_filtered = patch_tokens * w_patch
        aux = {
            "rule": "shared_weight",
            "effective_keep_mean": w_patch.mean().item(),
        }
        return torch.cat([cls_token, patch_tokens_filtered], dim=1), aux

    def purify_features(
        self,
        z,
        text_features,
        H,
        W,
        num_probes=3,
        noise_std=0.008,
        foreground_threshold=0.7,
        tau=3.0,
        min_keep=0.55,
        max_keep=1.0,
        blend_beta=1.0,
        patch_only=True,
    ):


        z_for_mask = z.detach()
        retention_w, aux_dict = self.estimate_retention_weights(
            z_for_mask,
            text_features,
            H,
            W,
            num_probes=num_probes,
            noise_std=noise_std,
            foreground_threshold=foreground_threshold,
            tau=tau,
            min_keep=min_keep,
            max_keep=max_keep,
            blend_beta=blend_beta,
            patch_only=patch_only,
        )
        z_filtered, rule_aux = self.apply_retention_weights(
            z,
            retention_w,
        )
        aux_dict.update(rule_aux)
        return z_filtered, aux_dict

    def purify_source_and_virtual(
        self,
        z_src,
        z_virt,
        text_features,
        H,
        W,
        num_probes=3,
        noise_std=0.008,
        foreground_threshold=0.7,
        tau=3.0,
        min_keep=0.55,
        max_keep=1.0,
        blend_beta=1.0,
        patch_only=True,
    ):

        retention_w, aux_joint = self.estimate_shared_retention_weights(
            z_src=z_src.detach(),
            z_virt=z_virt.detach(),
            text_features=text_features,
            H=H,
            W=W,
            num_probes=num_probes,
            noise_std=noise_std,
            foreground_threshold=foreground_threshold,
            tau=tau,
            min_keep=min_keep,
            max_keep=max_keep,
            blend_beta=blend_beta,
            patch_only=patch_only,
        )

        z_src_filtered, source_rule_aux = self.apply_retention_weights(
            z_src,
            retention_w,
        )
        z_virt_filtered, virt_rule_aux = self.apply_retention_weights(
            z_virt,
            retention_w,
        )

        source_aux = dict(aux_joint)
        source_aux.update(source_rule_aux)
        virt_aux = dict(aux_joint)
        virt_aux.update(virt_rule_aux)
        return z_src_filtered, z_virt_filtered, source_aux, virt_aux

    def forward(self, image, text, num_samples=30):
        if self.training:
            image_features, text_features, H, W = self.extract_features(image, text)
            seg_logits, _ = self.decode_from_features(
                image_features, text_features, H, W
            )
            clip_loss = self.compute_clip_loss_from_features(
                image_features, text_features
            )
            return seg_logits, clip_loss

        _, _, H, W = image.shape
        tokenized_prompts, prompts = self.prepare_text(text)
        use_eval_gcfp = bool(getattr(self.cfg.MODEL, "USE_GCFP", False)) and bool(
            getattr(self.cfg.MODEL, "GCFP_ON_EVAL", False)
        )

        num_probes = int(getattr(self.cfg.MODEL, "GCFP_NUM_PROBES", 3))
        noise_std = float(getattr(self.cfg.MODEL, "GCFP_NOISE_STD", 0.008))
        foreground_threshold = float(
            getattr(self.cfg.MODEL, "GCFP_FOREGROUND_THRESHOLD", 0.7)
        )
        tau = float(
            getattr(
                self.cfg.MODEL,
                "GCFP_TAU_EVAL",
                getattr(self.cfg.MODEL, "GCFP_TAU", 3.0),
            )
        )
        min_keep = float(
            getattr(
                self.cfg.MODEL,
                "GCFP_MIN_KEEP_EVAL",
                getattr(self.cfg.MODEL, "GCFP_MIN_KEEP", 0.55),
            )
        )
        max_keep = float(
            getattr(
                self.cfg.MODEL,
                "GCFP_MAX_KEEP_EVAL",
                getattr(self.cfg.MODEL, "GCFP_MAX_KEEP", 1.0),
            )
        )
        blend_beta = float(
            getattr(
                self.cfg.MODEL,
                "GCFP_BLEND_BETA_EVAL",
                getattr(self.cfg.MODEL, "GCFP_BLEND_BETA", 1.0),
            )
        )
        patch_only = bool(getattr(self.cfg.MODEL, "GVOS_PATCH_ONLY", True))

        seg_samples = []
        for _ in range(max(1, int(num_samples))):
            image_features, text_features = self.encode_text_image(
                tokenized_prompts, prompts, image
            )
            if use_eval_gcfp:
                image_features, _ = self.purify_features(
                    image_features,
                    text_features,
                    H,
                    W,
                    num_probes=num_probes,
                    noise_std=noise_std,
                    foreground_threshold=foreground_threshold,
                    tau=tau,
                    min_keep=min_keep,
                    max_keep=max_keep,
                    blend_beta=blend_beta,
                    patch_only=patch_only,
                )
            seg_logits, _ = self.decode_from_features(
                image_features, text_features, H, W
            )
            seg_samples.append(seg_logits)

        return torch.stack(seg_samples, dim=0)

def build_grace(cfg):
    print(f"Loading UniMedCLIP (backbone: {cfg.MODEL.BACKBONE})")
    clip_model = load_unimedclip_to_device(cfg)

    clip_model.float()

    print("Building GRACE")
    model = GRACEModel(cfg, clip_model)

    print("Turning off gradients in both the image and the text encoder")

    for name, param in model.named_parameters():

        if "pvl_adapters" in name:
            param.requires_grad_(True)
        elif "mask_head" in name:
            param.requires_grad_(True)
        elif "upscale" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return model
