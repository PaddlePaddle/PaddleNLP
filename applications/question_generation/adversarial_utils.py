import copy
import paddle


class FGM():
    """
        Fast Gradient Method to add disturbance. 
    """

    def __init__(self, model, emb_name='embeddings.', epsilon=3):
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.model = model
        self.backup = {}

    def attack(self, ):
        """Add interference and backup parameters for the embedding layer."""
        for name, param in self.model.named_parameters():
            if (not param.stop_gradient) and self.emb_name in name:
                self.backup[name] = copy.deepcopy(param.numpy())
                norm = paddle.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.add(r_at)

    def restore(self, ):
        """Restore the original embedding layer parameters."""
        for name, param in self.model.named_parameters():
            if (not param.stop_gradient) and self.emb_name in name:
                assert name in self.backup
                param.set_value(self.backup[name])
        self.backup = {}


class PGD():
    """
        Projected Gradient Descent to add disturbance. 
    """

    def __init__(self,
                 model,
                 emb_name='embeddings.',
                 epsilon=1.,
                 K=3,
                 alpha=0.3):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.K = K
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        """
            Add interference and backup parameters for the embedding layer. 
            The intensity of interference is limited within epsilon.
        """
        for name, param in self.model.named_parameters():
            if (not param.stop_gradient) and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = copy.deepcopy(param.numpy())
                norm = paddle.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    if paddle.norm(r_at) > self.epsilon:
                        r_at = self.epsilon * r_at / paddle.norm(r_at)
                    param.add(r_at)

    def restore_emb(self):
        """Restore the original embedding layer parameters."""
        for name, param in self.model.named_parameters():
            if (not param.stop_gradient) and self.emb_name in name:
                assert name in self.emb_backup
                param.set_value(self.emb_backup[name])
        self.emb_backup = {}

    def backup_grad(self):
        """Backup the original embedding layer gradients."""
        for name, param in self.model.named_parameters():
            if (not param.stop_gradient) and param.grad is not None:
                self.grad_backup[name] = copy.deepcopy(param.grad.numpy())

    def restore_grad(self):
        """Restore the original embedding layer gradients."""
        for name, param in self.model.named_parameters():
            if (not param.stop_gradient) and param.grad is not None:
                param.grad.set_value(self.grad_backup[name])
