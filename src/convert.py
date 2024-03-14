import os
import shutil

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from supervisely.imaging.color import get_predefined_colors
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # Possible structure for bbox case. Feel free to modify as you needs.

    images_path = "/home/alex/DATASETS/TODO/VOST/JPEGImages"
    split_path = "/home/alex/DATASETS/TODO/VOST/ImageSets"
    batch_size = 30

    def get_unique_colors(img):
        unique_colors = []
        img = img.astype(np.int32)
        h, w = img.shape[:2]
        colhash = img[:, :, 0] * 256 * 256 + img[:, :, 1] * 256 + img[:, :, 2]
        unq, unq_inv, unq_cnt = np.unique(colhash, return_inverse=True, return_counts=True)
        indxs = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))
        col2indx = {unq[i]: indxs[i][0] for i in range(len(unq))}
        for col, indx in col2indx.items():
            unique_colors.append((col // (256**2), (col // 256) % 256, col % 256))

        return unique_colors

    def create_ann(image_path):
        labels = []

        mask_path = image_path.replace("JPEGImages", "Annotations").replace(".jpg", ".png")

        split_path_data = image_path.split("/")[-2].split("_")

        seq_id = split_path_data[0]
        action_val = split_path_data[1]
        # class_name = ("_").join(split_path_data[2:])
        class_name = image_path.split("/")[-2].split("_")[-1]

        # seq_id, action_val, class_name = image_path.split("/")[-2].split("_")

        action = sly.Tag(action_meta, value=action_val)
        seq = sly.Tag(seq_meta, value=int(seq_id))
        obj_class = meta.get_obj_class(class_name)

        mask_np = sly.imaging.image.read(mask_path)
        img_height = mask_np.shape[0]
        img_wight = mask_np.shape[1]
        unique_colors = get_unique_colors(mask_np)
        for color in unique_colors[1:]:
            mask = np.all(mask_np == color, axis=2)
            ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
            for i in range(1, ret):
                obj_mask = curr_mask == i
                curr_bitmap = sly.Bitmap(obj_mask)
                if curr_bitmap.area > 25:
                    curr_label = sly.Label(curr_bitmap, obj_class)
                    labels.append(curr_label)

        return sly.Annotation(
            img_size=(img_height, img_wight), labels=labels, img_tags=[action, seq]
        )

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    classes_names = []
    for folder in os.listdir(images_path):
        if folder.split("_")[-1] not in classes_names:
            classes_names.append(folder.split("_")[-1])

    obj_classes = [
        sly.ObjClass(name, sly.Bitmap, color)
        for name, color in zip(classes_names, get_predefined_colors(len(classes_names)))
    ]

    action_meta = sly.TagMeta("action", sly.TagValueType.ANY_STRING)
    seq_meta = sly.TagMeta("sequence", sly.TagValueType.ANY_NUMBER)

    meta = sly.ProjectMeta(tag_metas=[action_meta, seq_meta], obj_classes=obj_classes)

    api.project.update_meta(project.id, meta.to_json())

    for split in os.listdir(split_path):
        ds_name = get_file_name(split)
        if ds_name == "test":
            continue

        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        curr_split_path = os.path.join(split_path, split)

        with open(curr_split_path) as f:
            folders = f.read().split("\n")

        for folder in folders:
            if len(folder) > 1:
                curr_images_path = os.path.join(images_path, folder)
                if dir_exists(curr_images_path):
                    images_names = os.listdir(curr_images_path)

                    progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

                    for images_names_batch in sly.batched(images_names, batch_size=batch_size):
                        im_names_batch = []
                        img_pathes_batch = []
                        for image_name in images_names_batch:
                            img_pathes_batch.append(os.path.join(curr_images_path, image_name))
                            im_names_batch.append(folder + "_" + image_name)

                        img_infos = api.image.upload_paths(
                            dataset.id, im_names_batch, img_pathes_batch
                        )
                        img_ids = [im_info.id for im_info in img_infos]

                        anns = [create_ann(image_path) for image_path in img_pathes_batch]
                        api.annotation.upload_anns(img_ids, anns)

                        progress.iters_done_report(len(images_names_batch))

    return project
