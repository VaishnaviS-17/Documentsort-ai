import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    """
    Grad-CAM implementation: highlights which regions of an input image
    most influenced the model's prediction for a given class.

    Works by hooking into a target convolutional layer to capture:
    - the activations (forward pass output of that layer)
    - the gradients (how much the loss w.r.t. the predicted class changes
      with respect to those activations, from the backward pass)
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Hook to capture the activations during the forward pass
        target_layer.register_forward_hook(self._save_activations)
        # Hook to capture the gradients during the backward pass
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: preprocessed image tensor, shape [1, C, H, W]
        class_idx: which class to explain. If None, uses the model's
                   own top prediction.
        """
        self.model.eval()
        
        # Ensure gradients can flow back to this input, even if the model's
        # own parameters are frozen (e.g. transfer learning with a frozen backbone).
        # Without this, frozen layers produce no backward graph, and our
        # gradient hook never fires.
        input_tensor = input_tensor.clone().requires_grad_(True)


        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Zero gradients, then backprop from the target class score only
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        # Global-average-pool the gradients across spatial dims -> importance weight per channel
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight each activation channel by its corresponding gradient importance
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        # Average across channels -> single heatmap, then apply ReLU
        # (we only care about features that positively influenced the class)
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)  # normalize to [0,1]

        return heatmap, class_idx


def overlay_heatmap(original_image_pil, heatmap, alpha=0.5):
    """
    Resizes the heatmap to match the original image size and overlays it
    as a color heatmap on top of the (grayscale) original image.
    Uses PIL/NumPy/Matplotlib only -- no OpenCV dependency needed.
    """
    from PIL import Image
    import matplotlib.cm as cm

    original = np.array(original_image_pil.convert('RGB'))

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_resized = heatmap_img.resize((original.shape[1], original.shape[0]), Image.BILINEAR)
    heatmap_array = np.array(heatmap_resized) / 255.0

    colormap = cm.get_cmap('jet')
    heatmap_colored = colormap(heatmap_array)[:, :, :3]
    heatmap_colored = np.uint8(255 * heatmap_colored)

    overlayed = np.uint8(original * (1 - alpha) + heatmap_colored * alpha)
    return overlayed


def visualize_gradcam(model, target_layer, image_pil, input_tensor, class_names, class_idx=None):
    """
    Full pipeline: generate Grad-CAM heatmap and display it side-by-side
    with the original image.
    """
    gradcam = GradCAM(model, target_layer)
    heatmap, predicted_class = gradcam.generate(input_tensor, class_idx)
    overlayed = overlay_heatmap(image_pil, heatmap)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_pil, cmap='gray')
    axes[0].set_title("Original Document")
    axes[0].axis('off')

    axes[1].imshow(overlayed)
    axes[1].set_title(f"Grad-CAM: {class_names[predicted_class]}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    return overlayed, predicted_class