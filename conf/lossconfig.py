# === LossConfigManager: for controlled ablation experiments ===

class LossConfigManager:
    """
    Manages configurable weighting for multi-component loss functions.
    Supports ablation studies and easy config switching.
    """
    def __init__(self, recon=1.0, perceptual=0.0, structure=0.0, edge=0.0):
        self.weights = {
            'recon': recon,
            'perceptual': perceptual,
            'structure': structure,
            'edge': edge
        }
        self._validate()

    def _validate(self):
        total = sum(self.weights.values())
        if not (0.99 <= total <= 1.01):
            print(f"⚠️ Warning: Loss weights do not sum to 1.0 (sum={total:.3f})")

    def __getitem__(self, key):
        return self.weights.get(key, 0.0)

    def __str__(self):
        return ' + '.join([f"{v:.2f}*{k}" for k, v in self.weights.items() if v > 0])


# === Example ablation config usage ===
if __name__ == '__main__':
    configs = [
        LossConfigManager(recon=1.0),
        LossConfigManager(recon=0.9, perceptual=0.1),
        LossConfigManager(recon=0.8, perceptual=0.1, structure=0.1),
        LossConfigManager(recon=0.7, perceptual=0.1, structure=0.1, edge=0.1),
    ]

    for cfg in configs:
        print(f"Testing config: {cfg}")
