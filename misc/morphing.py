"""
Adapted from https://github.com/andrewdcampbell/face-movie/blob/master/face-movie/face_morph.py
"""

import copy
from typing import Dict

import cv2
import numpy as np
from scipy.spatial import Delaunay


def warp_im(im, src_landmarks, dst_landmarks, triangulation, flags=cv2.INTER_LINEAR):
    im_out = im.copy()

    for i in range(len(triangulation)):
        src_tri = src_landmarks[triangulation[i]]
        dst_tri = dst_landmarks[triangulation[i]]
        morph_triangle(im, im_out, src_tri, dst_tri, flags)

    return im_out


def morph_triangle(im, im_out, src_tri, dst_tri, flags):
    sr = cv2.boundingRect(np.float32([src_tri]))
    dr = cv2.boundingRect(np.float32([dst_tri]))
    cropped_src_tri = [(src_tri[i][0] - sr[0], src_tri[i][1] - sr[1]) for i in range(3)]
    cropped_dst_tri = [(dst_tri[i][0] - dr[0], dst_tri[i][1] - dr[1]) for i in range(3)]

    mask = np.zeros((dr[3], dr[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(cropped_dst_tri), (1.0, 1.0, 1.0), 16, 0)

    cropped_im = im[sr[1] : sr[1] + sr[3], sr[0] : sr[0] + sr[2]]

    size = (dr[2], dr[3])
    warpImage1 = affine_transform(cropped_im, cropped_src_tri, cropped_dst_tri, size, flags)

    im_out[dr[1] : dr[1] + dr[3], dr[0] : dr[0] + dr[2]] = (
        im_out[dr[1] : dr[1] + dr[3], dr[0] : dr[0] + dr[2]] * (1 - mask) + warpImage1 * mask
    )


def affine_transform(src, src_tri, dst_tri, size, flags=cv2.INTER_LINEAR):
    M = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, M, size, flags=flags, borderMode=cv2.BORDER_REPLICATE)
    return dst


def morph_seq(
    source_img: np.ndarray,
    target_img: np.ndarray,
    source_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    comp_list: Dict[int, float],
):
    source_img = np.float32(source_img)
    target_img = np.float32(target_img)
    source_mask = np.repeat(np.float32(source_mask[..., None]), 3, axis=-1)

    triangulation = Delaunay(source_landmarks).simplices
    warped_source_mask = warp_im(
        source_mask, source_landmarks, target_landmarks, triangulation, flags=cv2.INTER_NEAREST
    )[..., 0]
    warped_source = warp_im(source_img, source_landmarks, target_landmarks, triangulation)
    # warped_source[warped_source == 0] = target_img[warped_source == 0]

    un_covered_mask = np.zeros_like(target_mask, dtype="bool")
    blended = copy.deepcopy(target_img)
    for comp, alpha in comp_list.items():
        if comp == 9 or comp == 13:
            target_comp_mask = np.logical_or(target_mask == 9, target_mask == 13)
            source_comp_mask = np.logical_or(warped_source_mask == 9, warped_source_mask == 13)
        else:
            target_comp_mask = target_mask == comp
            source_comp_mask = warped_source_mask == comp

        union_comp_mask = target_comp_mask & source_comp_mask

        blended[union_comp_mask] = (
            alpha * warped_source[union_comp_mask] + (1 - alpha) * target_img[union_comp_mask]
        )
        target_comp_mask[union_comp_mask] = 0
        un_covered_mask[target_comp_mask] = 1

    return blended.astype("uint8"), un_covered_mask
