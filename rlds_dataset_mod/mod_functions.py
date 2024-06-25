import torch
from abc import ABC, abstractmethod

import sys
import os
import dlimp as dl
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import math

class TfdsModFunction(ABC):
    @classmethod
    @abstractmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        """
        Modifies the data builder feature dict to reflect feature changes of ModFunction.
        """
        ...

    @classmethod
    @abstractmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        """
        Perform arbitrary modifications on the dataset that comply with the modified feature definition.
        """
        ...


def mod_obs_features(features, obs_feature_mod_function):
    """Utility function to only modify keys in observation dict."""
    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(
                        {
                            key: obs_feature_mod_function(
                                key, features["steps"]["observation"][key]
                            )
                            for key in features["steps"]["observation"].keys()
                        }
                    ),
                    **{
                        key: features["steps"][key]
                        for key in features["steps"].keys()
                        if key not in ("observation",)
                    },
                }
            ),
            **{key: features[key] for key in features.keys() if key not in ("steps",)},
        }
    )

def add_obs_key(features, obs_key, feature_size):
    """Utility function to only modify keys in observation dict."""
    observations = {
                    key: features["steps"]["observation"][key]
                    for key in features["steps"]["observation"].keys()
                }
    observations[obs_key] = feature_size

    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(observations),
                    **{
                        key: features["steps"][key]
                        for key in features["steps"].keys()
                        if key not in ("observation",)
                    },
                }
            ),
            **{key: features[key] for key in features.keys() if key not in ("steps",)},
        }
    )


class ResizeAndJpegEncode(TfdsModFunction):
    MAX_RES: int = 256

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        def downsize_and_jpeg(key, feat):
            """Downsizes image features, encodes as jpeg."""
            if len(feat.shape) >= 2 and feat.shape[0] >= 64 and feat.shape[1] >= 64:  # is image / depth feature
                should_jpeg_encode = (
                    isinstance(feat, tfds.features.Image) and "depth" not in key
                )
                if len(feat.shape) > 2:
                    new_shape = (ResizeAndJpegEncode.MAX_RES, ResizeAndJpegEncode.MAX_RES, feat.shape[2])
                else:
                    new_shape = (ResizeAndJpegEncode.MAX_RES, ResizeAndJpegEncode.MAX_RES)

                if isinstance(feat, tfds.features.Image):
                    return tfds.features.Image(
                        shape=new_shape,
                        dtype=feat.dtype,
                        encoding_format="jpeg" if should_jpeg_encode else "png",
                        doc=feat.doc,
                    )
                else:
                    return tfds.features.Tensor(
                        shape=new_shape,
                        dtype=feat.dtype,
                        doc=feat.doc,
                    )

            return feat

        return mod_obs_features(features, downsize_and_jpeg)

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def resize_image_fn(step):
            # resize images
            for key in step["observation"]:
                if len(step["observation"][key].shape) >= 2 and (
                    step["observation"][key].shape[0] >= 64
                    or step["observation"][key].shape[1] >= 64
                ):
                    size = (ResizeAndJpegEncode.MAX_RES,
                            ResizeAndJpegEncode.MAX_RES)
                    if "depth" in key:
                        step["observation"][key] = tf.cast(
                            dl.utils.resize_depth_image(
                                tf.cast(step["observation"][key], tf.float32), size
                            ),
                            step["observation"][key].dtype,
                        )
                    else:
                        step["observation"][key] = tf.cast(
                            dl.utils.resize_image(step["observation"][key], size),
                            tf.uint8,
                        )
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(resize_image_fn)
            return episode

        return ds.map(episode_map_fn)


class FilterSuccess(TfdsModFunction):
    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        return features  # no feature changes

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.filter(lambda e: e["success"])


class FlipImgChannels(TfdsModFunction):
    FLIP_KEYS = ["image"]

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        return features  # no feature changes

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def flip(step):
            for key in cls.FLIP_KEYS:
                if key in step["observation"]:
                    step["observation"][key] = step["observation"][key][..., ::-1]
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(flip)
            return episode

        return ds.map(episode_map_fn)
    

class FlipWristImgChannels(FlipImgChannels):
    FLIP_KEYS = ["wrist_image", "hand_image"]


# New mod_functions for diffusion augmentations
# TODO: Should be set via args but can add defaults in case
device = 'cuda:3'
source_robot = "Franka"
target_robot = "Franka"

class ViewAugmentationMod:
    view_augmenter = None
    traj_idx = 0
    batch_size = 80

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        return features  # no feature changes

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        if ViewAugmentationMod.view_augmenter is None:
            from mirage2.view_augmentation.view_augmentation import ViewAugmentation
            from mirage2.view_augmentation.sampler.uniform_view_sampler import UniformViewSampler
            view_sampler = UniformViewSampler(device=device)

            sys.path.append(os.path.expanduser('/home/kdharmarajan/mirage2/viewpoint-robust-control/ZeroNVS'))
            import threestudio.utils.misc as misc
            misc.EXT_DEVICE = device

            from threestudio.models.guidance import zero123_guidance

            ViewAugmentationMod.view_augmenter = ViewAugmentation(
                view_sampler,
                sample_img_path="/home/kdharmarajan/mirage2/viewpoint-robust-control/original_view.png",
                checkpoint_path="/home/kdharmarajan/mirage2/viewpoint-robust-control/checkpoint/zeronvs.ckpt",
                zeronvs_config_path="/home/kdharmarajan/mirage2/viewpoint-robust-control/ZeroNVS/zeronvs_config.yaml",
                zero123_guidance_module=zero123_guidance,
                original_size=256,
                device=device,
            )

        def augment_view(step):
            def process_images(trajectory_images):
                for i in range(0, math.ceil(len(trajectory_images)//ViewAugmentationMod.batch_size) + 1):
                    start = i*ViewAugmentationMod.batch_size
                    end = min((i+1)*ViewAugmentationMod.batch_size, len(trajectory_images))
                    camera_obs_batch = torch.from_numpy(trajectory_images[start:end]).float().to(ViewAugmentationMod.view_augmenter.device)
                    augmented_batch = ViewAugmentationMod.view_augmenter(camera_obs_batch, traj_id=ViewAugmentationMod.traj_idx, batch_id=i)
                    trajectory_images[start:end] = augmented_batch.cpu().numpy()
                ViewAugmentationMod.traj_idx += 1
                return trajectory_images

            step["observation"]["robot_aug_imgs"] = tf.numpy_function(process_images, [step["observation"]["merged_robot_aug"]], tf.uint8)
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].batch(ViewAugmentationMod.batch_size).map(augment_view).unbatch()
            return episode

        return ds.map(episode_map_fn)

class RobotMaskMod:
    mask_generator = None
    batch_size = 256

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        img_size = None
        for key in features["steps"]["observation"]:
            if "image" == key or "agentview_rgb" == key or "front_rgb" == key:
                img_size = features["steps"]["observation"][key].shape[0]
                break
        first_added_feature_dict = add_obs_key(features, "masks", tfds.features.Tensor(shape=(img_size, img_size, 3), dtype=tf.uint8))
        return add_obs_key(first_added_feature_dict, "masked_imgs", tfds.features.Tensor(shape=(img_size, img_size, 3), dtype=tf.uint8))

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        if RobotMaskMod.mask_generator is None:
            sys.path.append("/home/kdharmarajan/mirage2/SAMed_h/SAMed_h")
            from inference_factorized import InferenceManager
            lora_ckpt = os.path.join("/home/kdharmarajan/mirage2/weights/mask", source_robot.lower() + ".pth")
            mask_args = {
                "num_classes": 1,
                "img_size": 256,
                "input_size": 256,
                "seed": 1234,
                "deterministic": 1,
                "ckpt": "/home/kdharmarajan/mirage2/weights/mask/sam_vit_h_4b8939.pth",
                "lora_ckpt":lora_ckpt,
                "vit_name": "vit_h",
                "rank": 4,
                "module": "sam_lora_image_encoder",
                "mask_threshold": 0.5,
                "device": device,
            }
            RobotMaskMod.mask_generator = InferenceManager(mask_args)

        def generate_masks(step):
            def process_images(trajectory_images):
                masked_images, output_masks = RobotMaskMod.mask_generator.inference(torch.from_numpy(trajectory_images).permute(0, 3, 1, 2))
                masked_images *= 255
                masked_images = masked_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                output_masks = output_masks.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                return np.concatenate([masked_images, output_masks], axis=-1)

            img_key = "image"
            if "agentview_rgb" in step["observation"]:
                img_key = "agentview_rgb"
            if "front_rgb" in step["observation"]:
                img_key = "front_rgb"
            processed_output = tf.numpy_function(process_images, [step["observation"][img_key]], tf.uint8)
            step["observation"]["masked_imgs"] = processed_output[..., :3]
            step["observation"]["masks"] = processed_output[..., 3:]

            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].batch(RobotMaskMod.batch_size).map(generate_masks).unbatch()
            return episode

        return ds.take(1).map(episode_map_fn)

class R2RMod:
    r2r_augmentor = None
    batch_size = 200

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        img_size = None
        for key in features["steps"]["observation"]:
            if "image" == key or "agentview_rgb" == key or "front_rgb" == key:
                img_size = features["steps"]["observation"][key].shape[0]
                break
        return add_obs_key(features, "robot_aug_imgs", tfds.features.Tensor(shape=(img_size, img_size, 3), dtype=tf.uint8))

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        if R2RMod.r2r_augmentor is None:
            sys.path.append('/home/kdharmarajan/mirage2/r2r/examples/controlnet')
            from refactorized_inference import ImageProcessor
            controlnet_path = os.path.join("/home/kdharmarajan/mirage2/weights/r2r", source_robot.lower() + "_to_" + target_robot.lower())
            R2RMod.r2r_augmentor = ImageProcessor(
                base_model_path="runwayml/stable-diffusion-v1-5",
                controlnet_path=controlnet_path,
                batch_size=R2RMod.batch_size,
                device=device,
                target_robot=target_robot,
            )

        def augment_view(step):
            def process_images(trajectory_images):
                with torch.no_grad():
                    return R2RMod.r2r_augmentor.process_folders(trajectory_images)

            step["observation"]["robot_aug_imgs"] = tf.numpy_function(process_images, [step["observation"]["masked_imgs"]], tf.uint8)
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].batch(R2RMod.batch_size).map(augment_view).unbatch()
            return episode

        return ds.take(5).map(episode_map_fn)

class VideoInpaintMod:
    video_inpainter = None

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        img_size = None
        for key in features["steps"]["observation"]:
            if "image" == key or "agentview_rgb" == key or "front_rgb" == key:
                img_size = features["steps"]["observation"][key].shape[0]
                break
        return add_obs_key(features, "inpainted_background", tfds.features.Tensor(shape=(img_size, img_size, 3), dtype=tf.uint8))

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        if VideoInpaintMod.video_inpainter is None:
            sys.path.append('/home/kdharmarajan/mirage2/video-inpaint')
            from refractorized_inference import VideoProcessor
            args = {
                "model": "e2fgvi_hq",
                "width": 256,
                "height": 256,
                "step": 10,
                "num_ref": -1,
                "neighbor_stride": 5,
                "savefps": 24,
                "set_size": True,
                "save_frame": "./",
                "video.split": "test.mp4",
                "use_mp4": True,
                "ckpt": "/home/kdharmarajan/mirage2/weights/video-inpaint/E2FGVI-HQ-CVPR22.pth"
            }
            VideoInpaintMod.video_inpainter = VideoProcessor(args)

        def augment_view(step):
            def process_images(trajectory_images, masked_images):
                try:
                    non_stacked_imgs = VideoInpaintMod.video_inpainter.main_worker(trajectory_images, masked_images)
                    return non_stacked_imgs
                except Exception as e:
                    print(e)
                    return trajectory_images
                # return np.stack(non_stacked_imgs, axis=0).astype(np.uint8)

            img_key = "image"
            if "agentview_rgb" in step["observation"]:
                img_key = "agentview_rgb"
            if "front_rgb" in step["observation"]:
                img_key = "front_rgb"

            step["observation"]["inpainted_background"] = tf.numpy_function(process_images, [step["observation"][img_key], step["observation"]["masks"]], tf.uint8)
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].batch(400).map(augment_view).unbatch()
            return episode

        return ds.map(episode_map_fn)

class AugMergeMod:
    aug_merger = None

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        # img_size = None
        # for key in features["steps"]["observation"]:
        #     if "image" == key or "agentview_rgb" == key or "front_rgb" == key:
        #         img_size = features["steps"]["observation"][key].shape[0]
        #         break
        # return add_obs_key(features, "merged_robot_aug", tfds.features.Tensor(shape=(img_size, img_size, 3), dtype=tf.uint8))
        return features

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        if AugMergeMod.aug_merger is None:
            sys.path.append('/home/kdharmarajan/mirage2/video-inpaint')
            from merge_two_images_refactoriezed import ImageProcessor
            AugMergeMod.aug_merger = ImageProcessor('.')

        def augment_view(step):
            def process_images(background_images, objects):
                augmented_img = AugMergeMod.aug_merger.paste_objects(background_images, objects)
                # from PIL import Image
                # a = Image.fromarray(augmented_img)
                # a.save('test_aug.png')
                # b = Image.fromarray(test_img)
                # b.save('source.png')
                # import pdb; pdb.set_trace()
                return augmented_img
            
            # key_to_use = "merged_robot_aug"
            key_to_use = "eye_in_hand_rgb"
            step["observation"][key_to_use] = tf.numpy_function(process_images, [step["observation"]["inpainted_background"], step["observation"]["robot_aug_imgs"]], tf.uint8)
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(augment_view)
            return episode

        return ds.map(episode_map_fn)
    
class AugMergeMod2:
    aug_merger = None

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        # img_size = None
        # for key in features["steps"]["observation"]:
        #     if "image" == key or "agentview_rgb" == key or "front_rgb" == key:
        #         img_size = features["steps"]["observation"][key].shape[0]
        #         break
        # return add_obs_key(features, "merged_robot_aug", tfds.features.Tensor(shape=(img_size, img_size, 3), dtype=tf.uint8))
        return features

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        if AugMergeMod.aug_merger is None:
            sys.path.append('/home/kdharmarajan/mirage2/video-inpaint')
            from merge_two_images_refactoriezed import ImageProcessor
            AugMergeMod.aug_merger = ImageProcessor('.')

        def augment_view(step):
            def process_images(background_images, objects):
                augmented_img = AugMergeMod.aug_merger.paste_objects(background_images, objects)
                return augmented_img
            
            # key_to_use = "merged_robot_aug"
            key_to_use = "masks"
            step["observation"][key_to_use] = tf.numpy_function(process_images, [step["observation"]["inpainted_background"], step["observation"]["robot_aug_imgs"]], tf.uint8)
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(augment_view)
            return episode

        return ds.map(episode_map_fn)

class AugMergeMod3:
    aug_merger = None

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        # img_size = None
        # for key in features["steps"]["observation"]:
        #     if "image" == key or "agentview_rgb" == key or "front_rgb" == key:
        #         img_size = features["steps"]["observation"][key].shape[0]
        #         break
        # return add_obs_key(features, "merged_robot_aug", tfds.features.Tensor(shape=(img_size, img_size, 3), dtype=tf.uint8))
        return features

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        if AugMergeMod.aug_merger is None:
            sys.path.append('/home/kdharmarajan/mirage2/video-inpaint')
            from merge_two_images_refactoriezed import ImageProcessor
            AugMergeMod.aug_merger = ImageProcessor('.')

        def augment_view(step):
            def process_images(background_images, objects):
                augmented_img = AugMergeMod.aug_merger.paste_objects(background_images, objects)
                return augmented_img
            
            # key_to_use = "merged_robot_aug"
            key_to_use = "masks"
            step["observation"][key_to_use] = tf.numpy_function(process_images, [step["observation"]["inpainted_background"], step["observation"]["robot_aug_imgs"]], tf.uint8)
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(augment_view)
            return episode

        return ds.map(episode_map_fn)

TFDS_MOD_FUNCTIONS = {
    "resize_and_jpeg_encode": ResizeAndJpegEncode,
    "filter_success": FilterSuccess,
    "flip_image_channels": FlipImgChannels,
    "flip_wrist_image_channels": FlipWristImgChannels,
    "view_augmentation": ViewAugmentationMod,
    "robot_mask_generator": RobotMaskMod,
    "r2r": R2RMod,
    "video_inpaint": VideoInpaintMod,
    "aug_merge": AugMergeMod,
    "aug_merge2": AugMergeMod2,
    "aug_merge3": AugMergeMod3
}