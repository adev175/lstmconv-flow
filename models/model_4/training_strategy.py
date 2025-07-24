# models/model_4/training_strategy.py
import torch.optim as optim


class EATrainingStrategy:
    """Training strategies for EA-Net integration"""

    @staticmethod
    def get_stage1_optimizer(model, lr=1e-4):
        """Stage 1: Train only edge components"""
        edge_params = []

        # Get edge-related parameters
        if hasattr(model, 'edge_weight'):
            edge_params.append(model.edge_weight)
        if hasattr(model, 'edge_extractor'):
            edge_params.extend(model.edge_extractor.parameters())

        # Filter parameters that require gradients
        trainable_params = [p for p in edge_params if p.requires_grad]

        if not trainable_params:
            print("Warning: No trainable edge parameters found!")
            return None

        optimizer = optim.Adam(trainable_params, lr=lr)
        print(f"Stage 1: Training {len(trainable_params)} edge parameters")
        return optimizer

    @staticmethod
    def get_stage2_optimizer(model, interp, lr_edge=1e-4, lr_decoder=1e-5, lr_interp=1e-4):
        """Stage 2: Fine-tune with different learning rates"""
        # Unfreeze decoder
        if hasattr(model, 'unfreeze_decoder'):
            model.unfreeze_decoder()

        # Separate parameter groups
        edge_params = []
        decoder_params = []
        interp_params = list(interp.parameters())

        # Edge parameters
        if hasattr(model, 'edge_weight'):
            edge_params.append(model.edge_weight)

        # Decoder parameters
        decoder_modules = ['up1', 'up2', 'up3', 'flow_pred']
        for name, param in model.named_parameters():
            if any(module in name for module in decoder_modules) and param.requires_grad:
                decoder_params.append(param)

        # Create optimizer with different learning rates
        param_groups = []
        if edge_params:
            param_groups.append({'params': edge_params, 'lr': lr_edge})
        if decoder_params:
            param_groups.append({'params': decoder_params, 'lr': lr_decoder})
        if interp_params:
            param_groups.append({'params': interp_params, 'lr': lr_interp})

        optimizer = optim.Adam(param_groups)
        print(
            f"Stage 2: Training {len(edge_params)} edge + {len(decoder_params)} decoder + {len(interp_params)} interp params")
        return optimizer