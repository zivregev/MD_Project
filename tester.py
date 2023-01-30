import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import os
from time import time
import gt_masks

IN_PATH = "imgs/in/"
OUT_PATH = "imgs/out/"

NUCLEI_ROUNDNESS_TOLERANCE = 0.7
MOMENT_CALCULATION_RADIUS = 3
K_PIXEL_TYPES = 3

def save_img(img, base_name, append="", convertBGR=True):
    img_name, img_type = base_name.split(".")
    if convertBGR:
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite(OUT_PATH + img_name + "_" + append + "." + img_type, img)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


def local_first_moment(img, radius=MOMENT_CALCULATION_RADIUS):
    # EV is a normalized boxfilter
    return cv.boxFilter(img, -1, (radius, radius), borderType=cv.BORDER_REFLECT, normalize=True)


def local_second_moment(img, radius=MOMENT_CALCULATION_RADIUS):
    # using VAR(X) = E[X^2] - E[X]^2
    E_X = local_first_moment(img, radius)
    E_X_squared = local_first_moment(img * img, radius)
    return np.sqrt(E_X_squared - (E_X * E_X))


def apply_k_means_to_channels(channels, k=K_PIXEL_TYPES, normalize=True):
    values = []
    for channel in channels:
        values.append(channel)
        values.append(local_first_moment(channel))
        # values.append(local_second_moment(channel))
    values = np.dstack(values).reshape((-1, 2 * len(channels)))
    values = np.float32(values)
    if normalize:
        scaler = MinMaxScaler()
        values = scaler.fit_transform(values)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv.kmeans(values, k, None, criteria, 10, cv.KMEANS_PP_CENTERS)
    return ret, labels, centers


def find_suspected_nuclei_contours(contours, tolerance=NUCLEI_ROUNDNESS_TOLERANCE):
    suspected_nucs = []
    suspected_non_nucs = []
    for cnt in contours:
        roundness = calculate_contour_roundness(cnt)
        if roundness > 0:
            if roundness >= tolerance:
                # cv.drawContours(revised_nuc_mask, [cnt], 0, 1, -1)
                suspected_nucs.append(cnt)
            else:
                suspected_non_nucs.append(cnt)
    return suspected_nucs, suspected_non_nucs


def calculate_contour_roundness(contour):
    perimeter = cv.arcLength(contour, True)
    area = cv.contourArea(contour)
    if perimeter > 0:
        roundness = 4.0 * np.pi * area / (perimeter ** 2)
    else:
        roundness = 0
    return roundness



def find_suspected_nuclei_in_cell_hulls(bg_mask, nuc_mask):
    bg_contours, _ = cv.findContours(bg_mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    bg_hulls_mask = np.zeros((bg_mask.shape[0], bg_mask.shape[1]), dtype=np.uint8)
    hulls = []
    for cnt in bg_contours:
        hulls.append(cv.convexHull(cnt))
    cv.drawContours(bg_hulls_mask, hulls, -1, 1, -1)
    nucs_in_bg_hulls = cv.bitwise_and(nuc_mask, nuc_mask, mask=bg_hulls_mask)
    nuc_contours, nuc_hierarchy = cv.findContours(nucs_in_bg_hulls, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    suspected_nucs, _ = find_suspected_nuclei_contours(nuc_contours)
    revised_nuc_mask = np.zeros((bg_mask.shape[0], bg_mask.shape[1]), dtype=np.uint8)
    cv.drawContours(revised_nuc_mask, suspected_nucs, -1, 1, -1)
    return revised_nuc_mask


def apply_k_means_to_lab(img, equalize_l, k=K_PIXEL_TYPES):
    l, a, b = cv.split(cv.cvtColor(img, cv.COLOR_RGB2LAB))
    if equalize_l:
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
    ret, labels, centers = apply_k_means_to_channels([l, a, b], k=k)
    return ret, labels, centers

def apply_k_means_to_l(img, equalize=False, k=K_PIXEL_TYPES):
    l, _, _ = cv.split(cv.cvtColor(img, cv.COLOR_RGB2LAB))
    if equalize:
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
    ret, labels, centers = apply_k_means_to_channels([l], k=k)
    return ret, labels, centers


def create_mask_from_label(clustered_img, label, color=1):
    mask = np.zeros(shape=(clustered_img.shape[0], clustered_img.shape[1]), dtype=np.uint8)
    mask[clustered_img == label] = color
    return mask


def create_bg_nuc_masks_from_k_means(img, labels, centers):
    clustered_img = labels.reshape((img.shape[0], img.shape[1]))
    bg_label = np.argsort(centers[:, 0])[-1]
    bg_mask = create_mask_from_label(clustered_img, bg_label, color=1)
    nuc_label = np.argsort(centers[:, 0])[0]
    nuc_mask = create_mask_from_label(clustered_img, nuc_label, color=1)
    return bg_mask, nuc_mask


def extend_revised_nuclei_mask(bg_mask, revised_nuc_mask, size_dilation=2, size_close=3):
    extened_bg_mask = bg_mask.copy()
    dilation_kernel_size = (size_dilation, size_dilation)
    dilation_kernel = cv.getStructuringElement(cv.MORPH_RECT, dilation_kernel_size)
    revised_nuc_mask = cv.dilate(revised_nuc_mask, kernel=dilation_kernel)
    extened_bg_mask[revised_nuc_mask == 1] = 1
    # save_img(extened_bg_mask, img_file_name, "revised_bw_bg_after_dilate_" + str(dilation_kernel_size))
    close_kernel_size = (size_close, size_close)
    close_kernel = cv.getStructuringElement(cv.MORPH_RECT, close_kernel_size)
    extened_bg_mask = cv.morphologyEx(extened_bg_mask, cv.MORPH_CLOSE, kernel=close_kernel)
    return extened_bg_mask
    # save_img(extened_bg_mask, img_file_name, "closed_revised_bw_bg_after_dilate_" + str(close_kernel_size))


def remove_internal_nuclei_contours(contours, hierarchy):
    internal_bg_contours = []
    external_bg_contours = []
    for cnt, relations in zip(contours, hierarchy[0]):
        if relations[2] == -1 and relations[3] >= 0:
            internal_bg_contours.append(cnt)
        else:
            external_bg_contours.append(cnt)
            # cv.drawContours(img, [cnt], 0, (255, 0, 0), -1)
    suspected_nucs, suspected_non_nucs = find_suspected_nuclei_contours(internal_bg_contours)
    return external_bg_contours + suspected_non_nucs


def calculate_fat_mask(img, k=K_PIXEL_TYPES, return_nuc_mask=False):
    _, labels, clusters = apply_k_means_to_lab(img, equalize_l=True, k=k)
    bg_mask, nuc_mask = create_bg_nuc_masks_from_k_means(img, labels, clusters)
    revised_nuc_mask = find_suspected_nuclei_in_cell_hulls(bg_mask, nuc_mask)
    bg_mask = extend_revised_nuclei_mask(bg_mask, revised_nuc_mask)
    revised_bg_contours, revised_bg_hierarchy = cv.findContours(bg_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    fatty_contours = remove_internal_nuclei_contours(revised_bg_contours, revised_bg_hierarchy)
    fat_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv.drawContours(fat_mask, fatty_contours, -1, 1, -1)
    if return_nuc_mask:
        return fat_mask, revised_nuc_mask
    else:
        return fat_mask


def clean_up_fat_mask(fat_mask, remove_internal_contours=False):
    remove_isolated_contour_points_from_mask(fat_mask)
    morphological_kernel = np.ones((3, 3))
    fat_mask = cv.morphologyEx(fat_mask, cv.MORPH_OPEN, morphological_kernel)
    if remove_internal_contours:
        remove_interal_contours(fat_mask)
    return fat_mask


def remove_interal_contours(mask):
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contour, relationships in zip(contours, hierarchy[0]):
        if relationships[3] >= 0:
            cv.drawContours(mask, [contour], 0, 1, -1)



def remove_isolated_contour_points_from_mask(mask):
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    isolated_points = find_isolated_contour_points(mask, contours)
    mask[isolated_points > 0] = 0


def apply_watershed_to_mask(mask):
    dist = cv.distanceTransform(mask, cv.DIST_L2, 3)
    peak_idx = peak_local_max(dist, min_distance=10, labels=mask)
    peaks = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    peaks[tuple(peak_idx.T)] = 1
    unknown = cv.subtract(mask, peaks)
    _, markers = cv.connectedComponents(peaks)
    markers += 1
    markers[unknown == np.max(unknown)] = 0
    ws = watershed(-dist, markers, watershed_line=True)
    return ws


# def segment_fat_cells(img):
#     fat_mask = calculate_fat_mask(img)
#     ws = apply_watershed_to_mask(fat_mask)
#     return ws


class ContourROI:
    def __init__(self, contour, img, fill=False, mask=False, pad=True):
        self.box = ContourROI.create_bounding_box_for_contour(contour)
        self.original_h = img.shape[0]
        self.original_w = img.shape[1]
        lo_bound, hi_bound, r_bound, l_bound = self.create_roi_bounds_from_box(pad)
        self.x_offset = hi_bound
        self.y_offset = l_bound
        if fill:
            shifted_contour = self.shift_contour(contour)
            self.roi = np.zeros((r_bound - l_bound, lo_bound - hi_bound), dtype=img.dtype)
            cv.drawContours(self.roi, [shifted_contour], 0, 1, -1)
        else:
            self.roi = np.copy(img[l_bound:r_bound, hi_bound:lo_bound])

        if mask:
            shifted_contour = self.shift_contour(contour)
            mask_roi = np.zeros((r_bound - l_bound, lo_bound - hi_bound), dtype=np.uint8)
            cv.drawContours(mask_roi, [shifted_contour], 0, 1, -1)
            self.roi = cv.bitwise_and(self.roi, self.roi, mask=mask_roi)

    def shift_contour(self, contour):
        return contour - [self.x_offset, self.y_offset]
        # shifted_contour = []
        # if len(contour) > 0:
        #     for pnt in contour:
        #         pnt = pnt[0]
        #         pnt[0] -= self.x_offset
        #         pnt[1] -= self.y_offset
        #         pnt = pnt.reshape(contour[0].shape)
        #         shifted_contour.append(pnt)
        # return np.asarray(shifted_contour)


    @staticmethod
    def create_bounding_box_for_contour(contour):
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        return box

    def create_roi_bounds_from_box(self, pad=True):
        padding = 2 if pad else 0
        lo_bound = min(np.max(self.box[:, 0]) + padding, self.original_w)
        hi_bound = max(np.min(self.box[:, 0]) - padding, 0)
        r_bound = min(np.max(self.box[:, 1]) + padding, self.original_h)
        l_bound = max(np.min(self.box[:, 1]) - padding, 0)
        return lo_bound, hi_bound, r_bound, l_bound

    def apply_other_roi_as_mask(self, other):
        self.roi = cv.bitwise_and(self.roi, self.roi, mask=other.roi)


def find_isolated_contour_points(mask, contours):
    # fatty_contours, _ = cv.findContours(fat_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    mask = mask.astype(int)
    mask[mask > 0] = -20
    cv.drawContours(mask, contours, -1, 1)
    isolated_kernel = np.asarray([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    isolated_counts = signal.convolve2d(mask, isolated_kernel, mode='same')
    isolated_points = np.zeros((mask.shape[0], mask.shape[1]))
    isolated_points[isolated_counts >= 10] = 1
    return isolated_points


def find_mask_elongations(mask, contours):
    elongations = []
    for cnt in contours:
        elongation = calc_contour_elongation(cnt, mask)
        elongations.append(elongation)
    return elongations


def calc_contour_elongation(contour, mask):
    thickness = calc_contour_thickness(contour, mask)
    if thickness > 0:
        elongation = cv.contourArea(contour) / (2.0 * thickness ** 2)
    else:
        elongation = 0
    return elongation


def calc_contour_thickness(contour, mask):
    contour_roi = ContourROI(contour, mask, fill=True)
    roi = contour_roi.roi
    return calc_solitary_mask_thickness(roi)


def calc_solitary_mask_thickness(mask_of_shape):
    mask = np.copy(mask_of_shape)
    thickness = 0
    erosion_kernel = np.ones((3, 3), np.uint8)
    before_erosion = np.count_nonzero(mask)
    after_erosion = 0
    while after_erosion < before_erosion:
        before_erosion = np.count_nonzero(mask)
        mask = cv.erode(mask, kernel=erosion_kernel, iterations=1)
        thickness += 1
        after_erosion = np.count_nonzero(mask)
    thickness = 2 * thickness
    return thickness

def calc_shape_thickness(mask, marker):
    marker_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    marker_mask[mask == marker] = 1
    return calc_solitary_mask_thickness(marker_mask)


def calc_all_segmented_thicknesses(segmented):
    segmented_mask = np.zeros(segmented.shape[:2], dtype=np.uint8)
    segmented_mask[segmented > 1] = 1
    remaining_shapes = []
    erosion_kernel = np.ones((3, 3), np.uint8)
    while np.count_nonzero(segmented_mask) > 0:
        segmented_remaining = cv.bitwise_and(segmented, segmented, mask=segmented_mask)
        remaining_shapes.append(np.unique(segmented_remaining[segmented_remaining > 1]))
        segmented_mask = cv.erode(segmented_mask, kernel=erosion_kernel, iterations=1)
    thicknesses = {}
    appearances = np.concatenate(remaining_shapes)
    shapes = np.unique(segmented[segmented > 1])
    for shape in shapes:
        thicknesses[shape] = np.count_nonzero(appearances == shape)
    return thicknesses


def calc_number_of_segments_in_contour(contour, mask, segmented):
    original_contour_roi = ContourROI(contour, mask)
    segmented_contour_roi = ContourROI(contour, segmented)
    segmented_contour_roi.apply_other_roi_as_mask(original_contour_roi)
    return len(np.unique(segmented_contour_roi.roi))


def get_new_watershed_boundaries(pre_segmented_mask, segmented, pre_segmented_contours):
    contour_mask = np.zeros((pre_segmented_mask.shape[0], pre_segmented_mask.shape[1]))
    cv.drawContours(contour_mask, pre_segmented_contours, -1, 1, -1)
    cv.drawContours(contour_mask, pre_segmented_contours, -1, 0)
    watershed_boundaries = np.zeros((pre_segmented_mask.shape[0], pre_segmented_mask.shape[1]), dtype=np.uint8)
    watershed_boundaries[np.where((contour_mask > 0) & (segmented == 0))] = 1
    return watershed_boundaries


def get_internal_watersheds(segmented):
    kernel = np.ones((3, 3), dtype=np.uint8)
    ws_bounds = np.zeros((segmented.shape[0], segmented.shape[1]), dtype=np.uint8)
    ws_bounds[segmented == 1] = 1
    ws_bounds[segmented == 0] = 1
    ws_bounds_open = cv.morphologyEx(ws_bounds, cv.MORPH_OPEN, kernel)
    internal_ws = np.zeros(segmented.shape[:2], dtype=np.uint8)
    internal_ws[(ws_bounds == 1) & (ws_bounds_open == 0)] = 1
    return internal_ws



def find_bad_watersheds_based_on_thickness(segmented, tolerance=0.5):
    thicknesses = calc_all_segmented_thicknesses(segmented)
    new_ws_boundaries = get_internal_watersheds(segmented)
    _, ws_new_boundaries_marked = cv.connectedComponents(new_ws_boundaries)
    segmented_mask = np.zeros((segmented.shape[0], segmented.shape[1]), dtype=np.uint8)
    segmented_mask[segmented > 1] = 1

    kernel = np.ones((3, 3), dtype=np.uint8)
    new_ws_boundaries_dilated = cv.dilate(new_ws_boundaries, kernel, iterations=1)
    _, new_ws_boundaries_dilated_marked = cv.connectedComponents(new_ws_boundaries_dilated)
    markers, counts = np.unique(ws_new_boundaries_marked, return_counts=True)
    bad_markers = []
    for i in range(len(markers)):
        marker = markers[i]
        if marker == 0:
            pass
        else:
            marker_in_dilated = np.unique(new_ws_boundaries_dilated_marked[ws_new_boundaries_marked == marker])[0]
            marker_in_segmented = np.unique(segmented[new_ws_boundaries_dilated_marked == marker_in_dilated])
            marker_in_segmented = marker_in_segmented[marker_in_segmented > 1]
            marker_count = counts[i]
            bordering_thicknesses = [thicknesses[shape] for shape in marker_in_segmented]
            thicker = [thickness for thickness in bordering_thicknesses if thickness > marker_count]
            thinner = [thickness for thickness in bordering_thicknesses if thickness <= marker_count]
            if len(thinner) > 0:
                if len(thicker) > 0:
                    if min(thicker) <= (1.0 + tolerance) * max(thinner):
                        bad_markers.append(marker)
                else:
                    bad_markers.append(marker)
    bad_watersheds = np.zeros(segmented.shape[:2], dtype=np.uint32)
    new_marker = 1
    for marker in bad_markers:
        bad_watersheds[ws_new_boundaries_marked == marker] = new_marker
        new_marker += 1
    return bad_watersheds


def find_and_remove_bad_watersheds(base_mask, segmented, thicken=True):
    bad_watersheds = find_bad_watersheds_based_on_thickness(segmented)
    # lone_watersheds = find_lone_bad_watersheds(base_mask, bad_watersheds)
    updated_segmented = np.zeros(base_mask.shape[:2], dtype=np.uint8)
    updated_segmented[segmented > 1] = 1
    updated_segmented[bad_watersheds > 0] = 1
    fix_bad_ws_endspoints(updated_segmented, bad_watersheds)
    # for marker in lone_watersheds:
    #     updated_segmented[bad_watersheds == marker] = 0
    thicken_watersheds(segmented, updated_segmented)
    return updated_segmented


def find_lone_bad_watersheds(base_mask, bad_watersheds):
    base_contours, _ = cv.findContours(base_mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    not_bad_markers = []
    for cnt in base_contours:
        base_roi = ContourROI(cnt, base_mask, fill=True)
        bad_ws_roi = ContourROI(cnt, bad_watersheds, fill=False)
        bad_ws_roi.apply_other_roi_as_mask(base_roi)
        bad_markers_in_roi = np.unique(bad_ws_roi.roi)
        if len(bad_markers_in_roi) == 2:
            not_bad_markers.append(bad_markers_in_roi[-1])
    return not_bad_markers


def fix_bad_ws_endspoints(updated_segmented, bad_watersheds):
    ends_kernel = np.asarray([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    bad_ws_mask = np.zeros((updated_segmented.shape[0], updated_segmented.shape[1]), dtype=np.uint8)
    bad_ws_mask[bad_watersheds > 0] = 1
    ends_counts = signal.convolve2d(bad_ws_mask, ends_kernel, mode='same')
    end_points = np.zeros((updated_segmented.shape[0], updated_segmented.shape[1]), dtype=np.uint8)
    end_points[ends_counts == 11] = 1
    updated_segmented[end_points > 0] = 1


def calc_max_inscribed_circle(cnt, mask):
    mask_roi = ContourROI(cnt, mask, fill=True)
    dist = cv.distanceTransform(mask_roi.roi, cv.DIST_L2, 3)
    _, max_val, _, max_idx = cv.minMaxLoc(dist)
    radius = int(max_val)
    x, y = max_idx
    x += mask_roi.x_offset
    y += mask_roi.y_offset
    return x, y, radius


def find_non_trivial_fatty_contours(mask, size_tolerance=250):
    all_contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]
    relevant_contours = []
    for cnt, relations in zip(all_contours, hierarchy):
        size = cv.contourArea(cnt)
        if size > size_tolerance:
            cnt_roi = ContourROI(cnt, mask, mask=True)
            fat_in_cnt = np.count_nonzero(cnt_roi.roi)
            if fat_in_cnt / size > 0.5:
                relevant_contours.append(cnt)
    return relevant_contours


def calculate_roundness_from_max_inscribed_circle(cnt, mask):
    _, _, radius = calc_max_inscribed_circle(cnt, mask)
    cnt_area = cv.contourArea(cnt)
    return calc_circle_area(radius) / cnt_area


def calc_roundness_of_contours_from_max_inscrible_circle(contours, mask):
    vals = []
    for cnt in contours:
        vals.append(calculate_roundness_from_max_inscribed_circle(cnt, mask))
    return vals


def guess_macrosteatosis_contours(mask, contours, roundness_thresh=0.7, area_thresh=500):
    macrosteatotic_contours = []
    for cnt in contours:
        roundness = calculate_roundness_from_max_inscribed_circle(cnt, mask)
        area = cv.contourArea(cnt)
        if roundness >= roundness_thresh and area >= area_thresh:
            macrosteatotic_contours.append(cnt)
    sizes = [cv.contourArea(cnt) for cnt in macrosteatotic_contours]
    idxs = remove_gross_outliers(sizes)
    macrosteatotic_contours = [macrosteatotic_contours[i] for i in idxs]
    return macrosteatotic_contours


def decompose_cnt_as_circles(mask, cnt, overlap_tolerance=0.8, return_circles=False):
    mask_roi = ContourROI(cnt, mask, mask=True)
    dist = cv.distanceTransform(mask_roi.roi, cv.DIST_L2, 3)
    peak_idx = peak_local_max(dist, min_distance=10, labels=mask_roi.roi)
    if return_circles:
        circles = []
    for peak in peak_idx:
        y, x = peak
        radius = int(dist[y, x])
        if mask_roi.roi[y, x] > 1:
            pass
        else:
            test_roi = np.copy(mask_roi.roi)
            area_covered_before = np.count_nonzero(mask_roi.roi >= 2)
            cv.circle(test_roi, (x, y), radius, 2, -1)
            area_covered_after = np.count_nonzero(test_roi >= 2)
            new_coverage = area_covered_after - area_covered_before
            expected_coverage = calc_circle_area(radius)
            if radius > 0 and new_coverage / expected_coverage >= overlap_tolerance:
                cv.circle(mask_roi.roi, (x, y), radius, 2, -1)
                mask_roi.roi[y, x] = 3
                if return_circles:
                    circles.append((x + mask_roi.x_offset, y + mask_roi.y_offset, radius))
    if return_circles:
        return mask_roi, circles
    else:
        return mask_roi


def find_cnt_as_circles_properties(mask, cnt, overlap_tolerance=0.8, return_circles=False):
    circle_decomp = decompose_cnt_as_circles(mask, cnt, overlap_tolerance, return_circles)
    if return_circles:
        circle_decomp, circles = circle_decomp
    circled_area = np.count_nonzero(circle_decomp.roi > 1)
    cnt_area = np.count_nonzero(circle_decomp.roi > 0)
    num_of_circles = np.count_nonzero(circle_decomp.roi == 3)
    if return_circles:
        return circled_area, cnt_area, num_of_circles, circles
    else:
        return circled_area, cnt_area, num_of_circles


def calc_circle_area(radius):
    return (radius ** 2) * np.pi


def remove_gross_outliers(data, m=5., fail_quietly=True):
    d_to_median = np.abs(data - np.median(data))
    median_of_d_to_median = np.median(d_to_median)
    dev_from_median = d_to_median / median_of_d_to_median if median_of_d_to_median else 0
    good_idxs = np.where(dev_from_median < m)[0]
    if len(good_idxs) <= 0.9 * len(data) or len(data) - len(good_idxs) > 5:
        if fail_quietly:
            return np.where(dev_from_median >= 0)[0]
    return good_idxs


def make_bins(min_size, max_size, step_size):
    bins = np.arange(min_size + step_size, max_size + step_size, step_size)
    return bins


def bin_array(arr, bins):
    cum = 0
    binned = []
    for bin in bins:
        tot = np.count_nonzero(arr <= bin)
        in_bin = tot - cum
        binned.append(in_bin)
        cum += in_bin
    return binned


def setup_bayes(good_sizes, all_sizes, step_size):
    good_sizes = remove_gross_outliers(good_sizes)
    max_size = max(all_sizes)
    min_size = min(good_sizes)
    all_sizes = all_sizes[np.where(all_sizes >= min_size)]
    bins = make_bins(min_size, max_size, step_size)
    good_sizes_binned = bin_array(good_sizes, bins)
    all_sizes_binned = bin_array(all_sizes, bins)
    return good_sizes_binned, all_sizes_binned, bins


def calc_bayes(s, F_bins, all_bins, bins):
    s_bin = np.argmax(bins >= s)
    p_s_F = F_bins[s_bin] / sum(F_bins)
    p_F = sum(F_bins) / sum(all_bins)
    p_s = all_bins[s_bin] / sum(all_bins)
    p_F_s = (p_s_F * p_F) / p_s
    return p_F_s

def find_suspected_contours_based_on_macrosteatosis(fat_mask, macrosteatotic, overlap_tolerance=0.5):
    cluster_mask, external_mask = make_internal_external_clustered_masks(fat_mask, macrosteatotic)
    non_trivial_external_contours = find_non_trivial_fatty_contours(external_mask)
    decomp_properties, circles = get_circle_decomposition_of_mask(cluster_mask, non_trivial_external_contours, overlap_tolerance)
    decomp_properties = np.asarray(decomp_properties)

    contours_suspected_by_bayes_idxs = find_suspect_contours_by_circles_bayes(macrosteatotic, circles, non_trivial_external_contours, decomp_properties)
    contours_suspected_by_ratio_idxs = find_suspect_contours_by_ratio(decomp_properties)

    contours_suspected_by_bayes = [non_trivial_external_contours[i] for i in contours_suspected_by_bayes_idxs]
    contours_suspected_by_ratio = []

    for i in range(len(contours_suspected_by_ratio_idxs)):
        contours_suspected_by_ratio.extend([non_trivial_external_contours[i] for i in contours_suspected_by_ratio_idxs[i][0]])

    return contours_suspected_by_bayes, contours_suspected_by_ratio


def find_suspect_contours_by_ratio(decomp_properties):
    ratios = decomp_properties[:, 0] / decomp_properties[:, 1]
    num_of_comps = decomp_properties[:, 2]
    x = np.vstack((ratios, num_of_comps)).T
    suspected = [np.where((x[:, 1] >= i) & (x[:, 0] >= 1 - (i * 0.05))) for i in [2, 3, 4, 5]]
    return suspected


def find_suspect_contours_by_circles_bayes(macrosteatotic, circles, contours, decomp_properties):
    circle_sizes = unravel_circle_sizes(circles)
    macro_sizes = [cv.contourArea(cnt) for cnt in macrosteatotic]
    macro_sizes = np.asarray(macro_sizes)
    all_sizes = np.concatenate([macro_sizes, circle_sizes])
    macro_sizes_binned, all_sizes_binned, bins = setup_bayes(macro_sizes, all_sizes, step_size=100)
    suspected = []
    for i in range(len(contours)):
        for circle in circles[i]:
            x, y, radius = circle
            size = calc_circle_area(radius)
            prob = calc_bayes(size, macro_sizes_binned, all_sizes_binned, bins)
            props = decomp_properties[i]
            if prob >= 0.5 and size / props[0] >= 0.5:
                suspected.append(i)
    return suspected

def get_circle_decomposition_of_mask(mask, contours, overlap_tolerance=0.5):
    decomp_properties_and_circles = [find_cnt_as_circles_properties(mask, cnt, overlap_tolerance, True) for cnt in contours]
    decomp_properties = []
    circles = []

    for tup in decomp_properties_and_circles:
        decomp_properties.append(tup[:-1])
        circles.append(tup[-1])

    return decomp_properties, circles


def unravel_circle_sizes(circles):
    circle_sizes = []
    for cnt_circles in circles:
        for circle in cnt_circles:
            circle_sizes.append(calc_circle_area(circle[-1]))
    circle_sizes = np.asarray(circle_sizes)
    return circle_sizes


def make_internal_external_clustered_masks(mask, macrosteatotic_contours):
    bg_mask = np.copy(mask)
    _ = cv.drawContours(bg_mask, macrosteatotic_contours, -1, 0, -1)
    kernel = np.ones((3, 3), dtype=np.uint8)
    clopen_mask = cv.morphologyEx(cv.morphologyEx(bg_mask, cv.MORPH_CLOSE, kernel), cv.MORPH_OPEN, kernel)

    external_contours, _ = cv.findContours(clopen_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    external_contours_mask = np.zeros(clopen_mask.shape[:2], dtype=np.uint8)
    _ = cv.drawContours(external_contours_mask, external_contours, -1, 1, -1)
    external_watershed = apply_watershed_to_mask(external_contours_mask)
    external_watershed_revised = find_and_remove_bad_watersheds(external_contours_mask, external_watershed)

    mask_watershed = np.copy(clopen_mask)
    mask_watershed[external_watershed_revised == 0] = 0
    # external_watershed_mask = np.copy(external_contours_mask)
    # external_watershed_mask[external_watershed == 1] = 0
    # eroded_external_watershed_mask = cv.erode(external_watershed_mask, kernel, iterations=1)

    return mask_watershed, external_watershed_revised


def find_clustered_fat_cells(mask, external_mask):
    non_trivial_external_contours = find_non_trivial_fatty_contours(external_mask)
    for overlap_tolerance in [0.5, 0.6, 0.7]:
        decomp_properties = [find_cnt_as_circles_properties(mask, cnt, overlap_tolerance) for cnt in
                             non_trivial_external_contours]
        decomp_properties = np.asarray(decomp_properties)
        ratios = decomp_properties[:, 0] / decomp_properties[:, 1]
        img_copy = np.copy(img)
        for i in range(len(non_trivial_external_contours)):
            if ratios[i] >= 0.7:
                cv.drawContours(img_copy, non_trivial_external_contours, i, (255, 0, 0))
        save_img(img_copy, img_file_name, "single" + str(overlap_tolerance))


def check_if_macrosteatotic(contours, min_size=1000, min_num=5):
    count = 0
    for cnt in contours:
        if cv.contourArea(cnt) >= min_size:
            count += 1
            if count >= min_num:
                return True
    return False


def find_nucleated_fat_cells(contours, bcgk_mask, nucs_mask, nuceli_containment_tolerance, roundness_thresh):
    good_contours = find_contours_with_nuceli(contours, nucs_mask, nuceli_containment_tolerance)
    good_contours = filter_contours_by_roundness(good_contours, bcgk_mask, roundness_thresh)
    return good_contours


def find_lipid_laden_hepatocytes(bckg_mask, nucs_mask, nuceli_containment_tolerance=0.8, roundness_thresh=0.4):
    relevant_contours_ll = find_non_trivial_fatty_contours(bckg_mask)
    # print(len(relevant_contours_ll))
    good_contours = find_nucleated_fat_cells(relevant_contours_ll, bckg_mask, nucs_mask, nuceli_containment_tolerance, roundness_thresh)
    # print(len(good_contours))
    if len(good_contours) > 10:
        good_sizes = [cv.contourArea(cnt) for cnt in good_contours]
        idxs = remove_gross_outliers(good_sizes)
        # print(len(idxs))
        good_sizes = [good_sizes[i] for i in idxs]
        extension_coeff = (np.median(good_sizes) / 2)
        suspected_contours = []
        min_size = max(0, min(good_sizes) - extension_coeff)
        max_size = max(good_sizes) + extension_coeff
        for cnt in relevant_contours_ll:
            size = cv.contourArea(cnt)
            roundness = calculate_roundness_from_max_inscribed_circle(cnt, bckg_mask)
            if min_size <= size <= max_size and roundness >= roundness_thresh:
                suspected_contours.append(cnt)
        return suspected_contours
    else:
        return []


def filter_contours_by_roundness(contours, mask, roundness_thresh):
    cnts_good_roundness = []
    for cnt in contours:
        roundness = calculate_roundness_from_max_inscribed_circle(cnt, mask)
        if roundness >= roundness_thresh:
            cnts_good_roundness.append(cnt)
    return cnts_good_roundness


def find_contours_with_nuceli(contours, nucs_mask, nuceli_containment_tolerance):
    _, nuclei_labelled = cv.connectedComponents(nucs_mask)
    all_markers, all_counts = np.unique(nuclei_labelled, return_counts=True)
    contours_with_nuclei = []
    for cnt in contours:
        nuclei_mask_roi = ContourROI(cnt, nuclei_labelled, mask=True)
        for marker, count_in_cnt in zip(*np.unique(nuclei_mask_roi.roi, return_counts=True)):
            if marker == 0:
                pass
            count_total = all_counts[tuple(np.argwhere(all_markers == marker)[0])]
            if count_in_cnt / count_total >= nuceli_containment_tolerance:
                contours_with_nuclei.append(cnt)
                break
    return contours_with_nuclei


def make_correct_folder_structure(base_folder):
    in_path = IN_PATH + "/" + base_folder
    out_path = OUT_PATH + "/" + base_folder
    for folder, subs, files in os.walk(in_path):
        rel_dir = os.path.relpath(folder, in_path)
        Path(out_path + "/" + rel_dir).mkdir(exist_ok=True)


def get_all_imgs(base_folder):
    in_path = IN_PATH + "/" + base_folder
    imgs = []
    for folder, _, files in os.walk(in_path):
        rel_dir = os.path.relpath(folder, in_path)
        for file_name in files:
            rel_file = os.path.join(rel_dir, file_name)
            imgs.append(os.path.join(base_folder, rel_file))
    return imgs


def thicken_watersheds(segmented, updated_mask):
    dilation_kernel = np.ones((3, 3), dtype=np.uint8)
    all_watersheds = get_internal_watersheds(segmented)
    bad_watersheds = np.zeros(segmented.shape[:2], dtype=np.uint8)
    bad_watersheds[(segmented == 0) & (updated_mask == 1)] = 1
    good_watersheds = all_watersheds - bad_watersheds
    dilated_good_watersheds = cv.dilate(good_watersheds, dilation_kernel, iterations=1)
    updated_mask[dilated_good_watersheds == 1] = 0


def draw_shaded_contours(img, contours, color, alpha=0.25):
    img_copy = np.copy(img)
    _ = cv.drawContours(img_copy, contours, -1, color, -1)
    res = cv.addWeighted(img, 1 - alpha, img_copy, alpha, 0)
    return res


if __name__ == "__main__":
    test_dir = "test"
    # make_correct_folder_structure(test_dir)
    # img_file_names = get_all_imgs(test_dir)
    # img_file_names = ["0079.jpg", "A1M1 10X_0052_cropped.tif", "0085.jpg", "0086.jpg", "a7_3.jpg"]
    img_file_names = ["a7_3_test.jpg"]
    start_time = time()
    for img_file_name in img_file_names:
        print("Working on " + img_file_name + "...")
        # print(img_file_name.split("\\")[-1].split(".")[0])
        # print(img_file_name.split("\\")[-1])
        img = cv.imread(IN_PATH + img_file_name, 1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        fat_mask, revised_nuc_mask = calculate_fat_mask(img, return_nuc_mask=True)
        fat_mask = clean_up_fat_mask(fat_mask)
        fatty_contours, _ = cv.findContours(fat_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        segmented = apply_watershed_to_mask(fat_mask)
        updated_mask = find_and_remove_bad_watersheds(fat_mask, segmented)
        relevant_contours = find_non_trivial_fatty_contours(updated_mask)
        macrosteatotic = guess_macrosteatosis_contours(updated_mask, relevant_contours)
        macrosteatotic_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        total_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if len(macrosteatotic) > 0:
            cv.drawContours(macrosteatotic_mask, macrosteatotic, -1, 1, -1)
            cv.drawContours(total_mask, macrosteatotic, -1, 1, -1)
        # plt.imsave(OUT_PATH + "test/" + gt_masks.FOLDER_TYPES[1] + "/" + img_file_name.split("\\")[-1].split(".")[0] + ".png", macrosteatotic_mask)
        plt.imsave(OUT_PATH + "test/" + gt_masks.FOLDER_TYPES[1] + "/" + img_file_name.split("\\")[-1].split(".")[0] + ".png", macrosteatotic_mask)

        # img_copy = np.copy(img)
        microsteatotic_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if check_if_macrosteatotic(macrosteatotic):
            # cv.drawContours(img_copy, macrosteatotic, -1, (255, 0, 0))
            contours_suspected_by_bayes, contours_suspected_by_ratio = find_suspected_contours_based_on_macrosteatosis(updated_mask, macrosteatotic)
            if len(contours_suspected_by_bayes) > 0:
                cv.drawContours(microsteatotic_mask, contours_suspected_by_bayes, -1, 1, -1)
                cv.drawContours(total_mask, contours_suspected_by_bayes, -1, 1, -1)
            if len(contours_suspected_by_ratio) > 0:
                cv.drawContours(microsteatotic_mask, contours_suspected_by_ratio, -1, 1, -1)
                cv.drawContours(total_mask, contours_suspected_by_ratio, -1, 1, -1)
            cv.drawContours(updated_mask, macrosteatotic, -1, 0, -1)
        plt.imsave(OUT_PATH + "test/" + gt_masks.FOLDER_TYPES[2] + "/" + img_file_name.split("\\")[-1].split(".")[0] + ".png",
                   microsteatotic_mask)
        # plt.imsave("imgs/out/updated_mask_" + img_file_name, updated_mask)
        lipid_laden = find_lipid_laden_hepatocytes(updated_mask, revised_nuc_mask)
        fatty_hepatoctyes_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if len(lipid_laden) > 0:
            cv.drawContours(fatty_hepatoctyes_mask, lipid_laden, -1, 1, -1)
            cv.drawContours(total_mask, lipid_laden, -1, 1, -1)
        plt.imsave(OUT_PATH + "test/" + gt_masks.FOLDER_TYPES[0] + "/" + img_file_name.split("\\")[-1].split(".")[0] + ".png",
                       fatty_hepatoctyes_mask)
        plt.imsave(OUT_PATH + "test/" + gt_masks.FOLDER_TYPES[3] + "/" + img_file_name.split("\\")[-1].split(".")[0] + ".png",
                       fatty_hepatoctyes_mask)
        # save_img(img_copy, img_file_name, "all_colored_kmeans_pp")
        end_time = time()
        print(str(end_time - start_time))
        start_time = end_time