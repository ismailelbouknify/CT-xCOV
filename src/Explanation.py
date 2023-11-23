from tensorflow.keras.models import Model
from tf_explain.core.integrated_gradients import IntegratedGradients
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom



class GradCAM:
    def __init__(self, model, layer_name="block5_conv3"):
        self.model = model
        self.layer_name = layer_name

    def generate_heatmap(self, img, label_name=None, category_id=None):
        img_tensor = np.expand_dims(img, axis=0)
        conv_layer = self.model.get_layer(self.layer_name)
        heatmap_model = Model([self.model.inputs], [conv_layer.output, self.model.output])

        with tf.GradientTape() as gtape:
            conv_output, predictions = heatmap_model(img_tensor)
            if category_id is None:
                category_id = np.argmax(predictions[0])
            if label_name:
                print(label_name[category_id])
            output = predictions[:, category_id]
            grads = gtape.gradient(output, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat

        return cv2.resize(np.squeeze(heatmap), dsize=(256, 256))


class GradCAMPlus:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name

    def generate_heatmap(self, img, H=256, W=256):
        cls = np.argmax(self.model.predict(img))
        y_c = self.model.output[0, cls]
        
        conv_output = self.model.get_layer(self.layer_name).output
        grads = K.gradients(y_c, conv_output)[0]

        first = K.exp(y_c) * grads
        second = K.exp(y_c) * grads * grads
        third = K.exp(y_c) * grads * grads * grads

        gradient_function = K.function([self.model.input], [y_c, first, second, third, conv_output, grads])
        y_c, conv_first_grad, conv_second_grad, conv_third_grad, conv_output, grads_val = gradient_function([img])
        global_sum = np.sum(conv_output[0].reshape((-1, conv_first_grad[0].shape[2])), axis=0)

        alpha_num = conv_second_grad[0]
        alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum.reshape((1, 1, conv_first_grad[0].shape[2]))
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num / alpha_denom

        weights = np.maximum(conv_first_grad[0], 0.0)

        alpha_normalization_constant = np.sum(np.sum(alphas, axis=0), axis=0)

        alphas /= alpha_normalization_constant.reshape((1, 1, conv_first_grad[0].shape[2]))

        deep_linearization_weights = np.sum((weights * alphas).reshape((-1, conv_first_grad[0].shape[2])), axis=0)
        grad_CAM_map = np.sum(deep_linearization_weights * conv_output[0], axis=2)

        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)
        cam = zoom(cam, H / cam.shape[0])
        cam = cam / np.max(cam)  # scale 0 to 1.0

        return cam



class IntegratedGradientsExplainer:
    def __init__(self, model):
        self.model = model

    def explain(self, image, target):
        # Convert the image to the appropriate format (you can customize this based on your model's input requirements)
        img_array = preprocess_image(image)

        # Compute integrated gradients
        integrated_gradients = self.integrated_gradients(img_array, target)

        # Post-process and normalize the explanation
        explanation = self.post_process(integrated_gradients)

        return explanation

    def integrated_gradients(self, img_array, target):
        # Create IntegratedGradients instance
        explainer = IntegratedGradients()

        # Compute integrated gradients
        integrated_gradients = explainer.explain((img_array, None), self.model, target)

        return integrated_gradients



class LimeImageExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = lime_image.LimeImageExplainer()

    def explain(self, image, label, num_samples=1000, num_features=500):
        explanation = self.explainer.explain_instance(
            image.astype('double'),
            classifier_fn=self.model.predict,
            top_labels=None,
            hide_color=0,
            num_features=num_features,
            num_samples=num_samples
        )
        temp, mask = explanation.get_image_and_mask(label, positive_only=True, hide_rest=True)

        return temp, mask

# Example of usage
model = tf.keras.models.load_model("your_model_path")  
image = load_image("your_image_path")

gradcam_explainer = GradCAM(model)
gradcam_heatmap = gradcam_explainer.generate_heatmap(image)

gradcam_plus_explainer = GradCAMPlus(model, layer_name="some_layer")
gradcam_plus_heatmap = gradcam_plus_explainer.generate_heatmap(image)

integrated_gradients_explainer = IntegratedGradientsExplainer(model)
integrated_gradients_result = integrated_gradients_explainer.explain(image, target_label)

lime_explainer = LimeImageExplainer(model)
lime_temp, lime_mask = lime_explainer.explain(image, label=0, num_samples=1000, num_features=500)
