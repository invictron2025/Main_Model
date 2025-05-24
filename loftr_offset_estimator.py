import torch
import cv2
import numpy as np
from kornia.feature import LoFTR

def load_image_as_tensor(path, size=(480, 360)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    img_tensor = torch.from_numpy(img / 255.0).float().unsqueeze(0).unsqueeze(0).cuda()
    return img_tensor, img

def estimate_rotation_sift(img1, img2, threshold_deg=5):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    if des1 is None or des2 is None:
        raise ValueError("No SIFT descriptors found in one or both images.")

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]

    if len(good_matches) < 4:
        raise ValueError("Not enough matches.")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    M, _ = cv2.estimateAffine2D(dst_pts, src_pts, method=cv2.RANSAC)

    angle_rad = np.arctan2(M[1, 0], M[0, 0])
    angle_deg = np.degrees(angle_rad)

    if abs(angle_deg) < threshold_deg:
        return img2
    else:
        h, w = img2.shape[:2]
        center = (w // 2, h // 2)
        undo_M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
        corrected = cv2.warpAffine(img2, undo_M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return corrected

def calculate_loftr_offset(img1_path, img2_path, GSD, size=(480, 360)):
    matcher = LoFTR(pretrained='outdoor').cuda()

    img1_tensor, _ = load_image_as_tensor(img1_path, size=size)
    img1_color = cv2.imread(img1_path)
    img2_color = cv2.imread(img2_path)

    corrected_img2 = estimate_rotation_sift(img1_color, img2_color)

    img2_gray = cv2.cvtColor(corrected_img2, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.resize(img2_gray, (size[0], size[1]))
    img2_tensor = torch.from_numpy(img2_gray / 255.0).float().unsqueeze(0).unsqueeze(0).cuda()

    with torch.no_grad():
        input_dict = {"image0": img1_tensor, "image1": img2_tensor}
        correspondences = matcher(input_dict)

    keypoints0 = correspondences["keypoints0"].cpu().numpy()
    keypoints1 = correspondences["keypoints1"].cpu().numpy()

    if len(keypoints0) < 10:
        raise ValueError("Not enough matches found.")

    M, _ = cv2.estimateAffinePartial2D(keypoints0, keypoints1, method=cv2.RANSAC)
    dx_pix, dy_pix = M[0, 2], M[1, 2]
    dx_m = dx_pix * GSD
    dy_m = dy_pix * GSD

    return  dx_m, dy_m

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate pixel and meter offset between two aerial images.")
    parser.add_argument("img1", help="Path to the first image")
    parser.add_argument("img2", help="Path to the second image")
    parser.add_argument("GSD", type=float, help="Ground Sampling Distance (in meters per pixel)")

    args = parser.parse_args()

    try:
        dx_pix, dy_pix, dx_m, dy_m = calculate_loftr_offset(args.img1, args.img2, args.GSD)
        print(f"Pixel Offset: dx = {dx_pix:.2f}, dy = {dy_pix:.2f}")
        print(f"Distance Offset: dx = {dx_m:.2f} m, dy = {dy_m:.2f} m")
    except Exception as e:
        print("[✗] Error:", e)
